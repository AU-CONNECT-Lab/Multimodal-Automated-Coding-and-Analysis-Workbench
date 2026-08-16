"""Ties config, recordings, DAG, cache and backend together. The one entry point `macaw run` uses."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from macaw.config.hashing import hash_extractor_config
from macaw.config.schema import MacawConfig
from macaw.exceptions import MacawError
from macaw.execution.backends.base import JobSpec, get_backend
from macaw.execution.dag import build_dag
from macaw.execution.job_db import JobDB, JobRecord, now
from macaw.execution.output_checker import completed_tasks, load_rules
from macaw.extractors import get_extractor
from macaw.extractors.base import QCWarning
from macaw.project.layout import layout_for, new_execution_id
from macaw.project.manifest import FeatureManifest, merge_manifests, write_manifest
from macaw.project.recording import build_recording, detect_recordings, parse_recordings_input
from macaw.store import annotations as annotations_store
from macaw.store import body_model as body_model_store
from macaw.store import cache as cache_store
from macaw.store import keypoints as keypoints_store
from macaw.store import longtable as longtable_store


@dataclass
class RunSummary:
    execution_id: str
    n_tasks: int
    n_done: int
    n_cached: int
    n_failed: int
    warnings: list[QCWarning]
    output_dir: Path


def task_id_for(recording_id: str, extractor: str) -> str:
    return f"{recording_id}:{extractor}"


def resolve_recordings(config: MacawConfig, requested: str | None) -> list[str]:
    """A comma list, a file of ids, or None meaning every detected recording minus exclusions."""
    if requested:
        found = parse_recordings_input(requested)
    else:
        found = detect_recordings(
            config.project.raw_dir,
            config.project.recording_pattern,
            config.project.recording_prefix,
        )
    excluded = set(config.project.excluded_recordings)
    return [recording_id for recording_id in found if recording_id not in excluded]


def plan(config: MacawConfig, recording_ids: list[str], extractors: list[str]) -> list[str]:
    """What would run, in order, without running it."""
    dag = build_dag(config, extractors or None)
    order = dag.topological_order()
    return [task_id_for(recording_id, name) for name in order for recording_id in recording_ids]


def _cache_root(config: MacawConfig, layout) -> Path:
    return Path(config.execution.cache_dir) if config.execution.cache_dir else layout.cache_dir


def _output_rules(config: MacawConfig):
    return load_rules(config.execution.output_rules) if config.execution.output_rules else []


def run(
    config: MacawConfig,
    *,
    recordings: list[str] | None = None,
    extractors: list[str] | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> RunSummary:
    layout = layout_for(config)
    execution_id = new_execution_id()

    recording_ids = recordings if recordings is not None else resolve_recordings(config, None)
    dag = build_dag(config, extractors or None)
    order = dag.topological_order()

    already = completed_tasks(layout.output_dir, _output_rules(config)) if resume else set()
    specs: list[JobSpec] = []
    for name in order:
        for recording_id in recording_ids:
            task_id = task_id_for(recording_id, name)
            if task_id in already:
                continue
            specs.append(
                JobSpec(
                    task_id=task_id,
                    recording_id=recording_id,
                    extractor=name,
                    command=_task_command(config, recording_id, name, execution_id),
                    resources=None,
                    log_path=layout.logs_dir / execution_id / f"{_slug(task_id)}.log",
                    depends_on=[
                        task_id_for(recording_id, dependency)
                        for dependency in sorted(dag.dependencies(name))
                        if task_id_for(recording_id, dependency) not in already
                    ],
                )
            )

    if dry_run:
        return RunSummary(
            execution_id=execution_id,
            n_tasks=len(specs),
            n_done=0,
            n_cached=len(already),
            n_failed=0,
            warnings=[],
            output_dir=layout.output_dir,
        )

    from macaw.config.loader import archive_resolved

    archive_resolved(config, layout.output_dir)

    database = JobDB(layout.job_db_path)
    database.initialise()
    for spec in specs:
        database.upsert(
            JobRecord(
                task_id=spec.task_id,
                recording_id=spec.recording_id,
                extractor=spec.extractor,
                execution_id=execution_id,
                status="pending",
                log_path=str(spec.log_path),
            )
        )

    backend = get_backend(config.execution.backend, config.execution, job_db=database)
    backend.wait(backend.submit(specs))

    counts = database.summary(execution_id)
    _merge_manifest_fragments(layout, recording_ids)
    return RunSummary(
        execution_id=execution_id,
        n_tasks=len(specs),
        n_done=counts.get("done", 0),
        n_cached=counts.get("cached", 0) + len(already),
        n_failed=counts.get("failed", 0) + counts.get("cancelled", 0),
        warnings=_collect_warnings(layout, recording_ids),
        output_dir=layout.output_dir,
    )


def _slug(task_id: str) -> str:
    return task_id.replace(":", "__").replace("/", "_")


def _task_command(
    config: MacawConfig, recording_id: str, extractor: str, execution_id: str
) -> list[str]:
    # python -m rather than the console script: a cluster job script activates an
    # environment but the bin directory is not always on PATH by then
    return [
        sys.executable, "-m", "macaw",
        "run-task",
        "--config", str(layout_for(config, create=False).resolved_config_path),
        "--recording", recording_id,
        "--extractor", extractor,
        "--execution-id", execution_id,
    ]


def run_task(
    config: MacawConfig, recording_id: str, extractor_name: str, execution_id: str
) -> None:
    """The single-task entry point cluster array elements call back into."""
    layout = layout_for(config)
    database = JobDB(layout.job_db_path)
    database.initialise()
    task_id = task_id_for(recording_id, extractor_name)

    cls = get_extractor(extractor_name)
    config_hash = hash_extractor_config(config, extractor_name)
    database.upsert(
        JobRecord(
            task_id=task_id,
            recording_id=recording_id,
            extractor=extractor_name,
            execution_id=execution_id,
            status="running",
            started_at=now(),
            config_hash=config_hash,
        )
    )

    try:
        extractor = cls(config)
        extractor.check_available()
        recording = build_recording(recording_id, config)

        key = cache_store.make_key(
            extractor=extractor_name,
            extractor_version=cls.version,
            inputs=extractor.inputs_for(recording),
            config_hash=config_hash,
        )
        cache_root = _cache_root(config, layout)
        entry = cache_store.lookup(cache_root, key) if config.execution.cache else None
        if entry is not None:
            cache_store.restore(entry, layout.recording_dir(recording_id))
            database.set_status(task_id, "cached", finished_at=now())
            return

        with tempfile.TemporaryDirectory(
            prefix=f"macaw_{_slug(task_id)}_", dir=config.project.scratch_dir or None
        ) as workspace:
            result = extractor.run(recording, Path(workspace))
            payload = Path(workspace) / "_payload"
            _write_result(layout, recording_id, extractor_name, result, execution_id, config_hash)
            _stage_payload(layout, recording_id, extractor_name, result, payload)
            if config.execution.cache:
                cache_store.store(cache_root, key, payload)

        database.set_status(task_id, "done", finished_at=now())
    except Exception as error:
        database.set_status(
            task_id, "failed", finished_at=now(), message=f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()
        raise


def _write_result(layout, recording_id, extractor, result, execution_id, config_hash) -> None:
    if result.rows is not None and not result.rows.empty:
        # provenance is stamped here so no extractor has to be told which execution it is in
        rows = result.rows.copy()
        rows["execution_id"] = rows["execution_id"].fillna(execution_id)
        rows["config_hash"] = rows["config_hash"].fillna(config_hash)
        longtable_store.write_long_table(rows, layout.long_table_path(recording_id))
    if result.keypoints is not None and not result.keypoints.empty:
        keypoints_store.write_keypoints(result.keypoints, layout.keypoints_path(recording_id))
    if result.body_model is not None and not result.body_model.empty:
        body_model_store.write_body_model(result.body_model, layout.body_model_path(recording_id))
    if result.annotations is not None and not result.annotations.empty:
        annotations_store.write_annotations(
            result.annotations, layout.annotations_path(recording_id)
        )

    manifest = FeatureManifest()
    manifest.extend(list(result.specs))
    write_manifest(manifest, layout.manifest_fragment_path(recording_id, extractor))

    warnings_path = layout.warnings_path(recording_id, extractor)
    warnings_path.parent.mkdir(parents=True, exist_ok=True)
    warnings_path.write_text(
        json.dumps([asdict(warning) for warning in result.warnings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, source in result.artifacts.items():
        destination = layout.artifacts_dir(recording_id, extractor) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move, not os.replace: the workspace sits in scratch_dir, which on a cluster
        # is a different filesystem from output_dir and a rename across those fails
        shutil.move(str(source), str(destination))


def _stage_payload(layout, recording_id: str, extractor: str, result, payload: Path) -> None:
    """Copy exactly what this extractor produced into a cache payload, so a later hit can be
    replayed and look identical to a real run."""
    payload.mkdir(parents=True, exist_ok=True)
    pieces = [
        ("features", layout.long_table_path(recording_id) / f"{extractor}.parquet"),
        ("keypoints", layout.keypoints_path(recording_id) / f"{extractor}.parquet"),
        ("body_model", layout.body_model_path(recording_id) / f"{extractor}.parquet"),
        ("manifest", layout.manifest_fragment_path(recording_id, extractor)),
        ("warnings", layout.warnings_path(recording_id, extractor)),
    ]
    # annotations are partitioned by layer, not by extractor, so the file name cannot be
    # derived from the extractor name the way the others can
    if result.annotations is not None and not result.annotations.empty:
        for layer in result.annotations["layer"].dropna().unique():
            pieces.append(
                ("annotations", layout.annotations_path(recording_id) / f"{layer}.parquet")
            )

    for group, source in pieces:
        if source.is_file():
            target = payload / group / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    artifacts = layout.artifacts_dir(recording_id, extractor)
    if artifacts.is_dir():
        # keep the per-extractor level, otherwise a restored hit lands one directory too high
        shutil.copytree(artifacts, payload / "artifacts" / extractor, dirs_exist_ok=True)


def _merge_manifest_fragments(layout, recording_ids: list[str]) -> None:
    manifests = []
    for recording_id in recording_ids:
        directory = layout.recording_dir(recording_id) / "manifest"
        if not directory.is_dir():
            continue
        from macaw.project.manifest import read_manifest

        manifests.extend(read_manifest(path) for path in sorted(directory.glob("*.json")))
    if manifests:
        write_manifest(merge_manifests(manifests), layout.manifest_path)


def _collect_warnings(layout, recording_ids: list[str]) -> list[QCWarning]:
    warnings: list[QCWarning] = []
    for recording_id in recording_ids:
        directory = layout.recording_dir(recording_id) / "warnings"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                entries = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise MacawError(f"{path} is not readable: {error}") from error
            warnings.extend(QCWarning(**entry) for entry in entries)
    return warnings
