"""Ties config, recordings, DAG, cache and backend together. The one entry point `macaw run` uses."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from macaw.config.hashing import hash_extractor_config
from macaw.config.schema import MacawConfig
from macaw.exceptions import MacawError
from macaw.execution.backends.base import JobSpec, get_backend
from macaw.execution.dag import build_dag
from macaw.execution.job_db import JobDB, JobRecord, now
from macaw.execution.lock import run_lock
from macaw.execution.output_checker import completed_tasks, load_rules
from macaw.extractors import get_extractor
from macaw.extractors.base import Phase, QCWarning
from macaw.project.layout import layout_for, new_execution_id
from macaw.project.manifest import FeatureManifest, merge_manifests, write_manifest
from macaw.project.recording import build_recording, detect_recordings, parse_recordings_input
from macaw.store import annotations as annotations_store
from macaw.store import body_model as body_model_store
from macaw.store import cache as cache_store
from macaw.store import keypoints as keypoints_store
from macaw.store import longtable as longtable_store


@dataclass
class GateWait:
    recording_id: str
    gate: str
    message: str
    # waiting for steps of this run rather than for a person
    deferred: bool = False


@dataclass
class RunSummary:
    execution_id: str
    n_tasks: int
    n_done: int
    n_cached: int
    n_failed: int
    warnings: list[QCWarning]
    output_dir: Path
    waiting: list[GateWait] = field(default_factory=list)


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


def plan(
    config: MacawConfig,
    recording_ids: list[str],
    extractors: list[str],
    finished: set[str] | None = None,
) -> list[str]:
    """What would run, in order, without running it. Gates are checked, not listed; finished tasks are left out."""
    dag = build_dag(config, extractors or None)
    gates = _gates(dag)
    _, blocked, _ = check_gates(config, dag, recording_ids, finished)
    skip = blocked | (finished or set())
    return [
        task_id_for(recording_id, name)
        for name in dag.topological_order()
        if name not in gates
        for recording_id in recording_ids
        if task_id_for(recording_id, name) not in skip
    ]


def _job_dependencies(dag, name: str, gates: set[str]) -> set[str]:
    """A gate is not a job, so whatever waits on it waits on what the gate itself needs."""
    found: set[str] = set()
    for dependency in dag.dependencies(name):
        found |= _job_dependencies(dag, dependency, gates) if dependency in gates else {dependency}
    return found


def _gates(dag) -> list[str]:
    return [name for name in dag.topological_order() if dag.nodes[name].phase == Phase.INTERACTIVE]


def check_gates(
    config: MacawConfig, dag, recording_ids: list[str], finished: set[str] | None = None
) -> tuple[list[GateWait], set[str], dict[tuple[str, str], list[QCWarning]]]:
    """Gates are checked here, in-process; one not yet cleared holds back everything after it for that recording.
    With finished given, a gate whose upstream tasks are not all in it is deferred until they are."""
    waiting: list[GateWait] = []
    blocked: set[str] = set()
    notes: dict[tuple[str, str], list[QCWarning]] = {}
    gates = _gates(dag)
    upstream = {gate: sorted(dag.ancestors(gate) - set(gates)) for gate in gates}
    for recording_id in recording_ids if gates else ():
        recording, unreadable = None, None
        try:
            recording = build_recording(recording_id, config)
        except MacawError as error:
            unreadable = str(error)
        for gate in gates:
            missing = [
                name for name in upstream[gate] if task_id_for(recording_id, name) not in finished
            ] if finished is not None else []
            extractor = get_extractor(gate)(config)
            try:
                if finished is not None and recording is not None and not missing:
                    extractor.clear_by_itself(recording)
                reason = (
                    unreadable if recording is None
                    else f"checked once {', '.join(missing)} has run" if missing
                    else extractor.pending(recording)
                )
            except MacawError as error:
                reason = str(error)
            if reason is None:
                notes[(recording_id, gate)] = extractor.notes(recording)
                continue
            waiting.append(
                GateWait(
                    recording_id=recording_id,
                    gate=gate,
                    message=reason,
                    deferred=bool(missing) and recording is not None,
                )
            )
            blocked.add(task_id_for(recording_id, gate))
            blocked.update(task_id_for(recording_id, name) for name in dag.descendants(gate))
    return waiting, blocked, notes


def completed_on_disk(config: MacawConfig, dag, recording_ids: list[str]) -> set[str]:
    """Tasks whose outputs are there and were made from what is there now: the same version,
    config and inputs as when they ran, and nothing upstream about to run again."""
    layout = layout_for(config, create=False)
    on_disk = completed_tasks(layout.output_dir, _output_rules(config))
    gates = set(_gates(dag))
    current: set[str] = set()
    for recording_id in recording_ids:
        try:
            recording = build_recording(recording_id, config)
        except MacawError:
            continue
        for name in dag.topological_order():
            task_id = task_id_for(recording_id, name)
            if name in gates or task_id not in on_disk:
                continue
            needed = _job_dependencies(dag, name, gates)
            if any(task_id_for(recording_id, dependency) not in current for dependency in needed):
                continue
            stamp = layout.completion_path(recording_id, name)
            try:
                if stamp.is_file() and stamp.read_text(encoding="utf-8") == task_key(config, name, recording):
                    current.add(task_id)
            except Exception:
                # a key that cannot be worked out means run it again, which is where the error shows
                continue
    return current


def task_states(config: MacawConfig, dag, recording_ids: list[str]) -> dict[str, str]:
    """Each task as a resumed run would find it: current, held behind a gate waiting for a person,
    stale (outputs made from something that has changed since) or new."""
    layout = layout_for(config, create=False)
    on_disk = completed_tasks(layout.output_dir, _output_rules(config))
    current = completed_on_disk(config, dag, recording_ids)
    waiting, _, _ = check_gates(config, dag, recording_ids, current)
    held = {
        task_id_for(wait.recording_id, name)
        for wait in waiting if not wait.deferred
        for name in dag.descendants(wait.gate)
    }
    gates = set(_gates(dag))
    states = {}
    for name in dag.topological_order():
        for recording_id in recording_ids if name not in gates else ():
            task_id = task_id_for(recording_id, name)
            states[task_id] = (
                "current" if task_id in current else "held" if task_id in held
                else "stale" if task_id in on_disk else "new"
            )
    return states


def task_key(config: MacawConfig, extractor_name: str, recording) -> str:
    cls = get_extractor(extractor_name)
    return _make_key(config, cls, cls(config), recording, hash_extractor_config(config, extractor_name)).as_dirname()


def _make_key(config, cls, extractor, recording, config_hash):
    return cache_store.make_key(
        extractor=cls.name,
        extractor_version=cls.version,
        inputs=extractor.inputs_for(recording),
        config_hash=config_hash,
    )


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
    if dry_run:
        return _run(config, recordings, extractors, resume, dry_run)
    with run_lock(layout_for(config).output_dir):
        return _run(config, recordings, extractors, resume, dry_run)


def _run(config, recordings, extractors, resume, dry_run) -> RunSummary:
    layout = layout_for(config)
    execution_id = new_execution_id()

    recording_ids = recordings if recordings is not None else resolve_recordings(config, None)
    dag = build_dag(config, extractors or None)
    order = dag.topological_order()
    gates = set(_gates(dag))

    in_scope = {task_id_for(r, name) for r in recording_ids for name in order if name not in gates}
    already = completed_on_disk(config, dag, recording_ids) & in_scope if resume else set()
    finished = set(already)
    attempted: set[str] = set()
    n_tasks = 0
    database = None
    # a gate that reads what this run produces is checked between waves, once that is there
    while True:
        waiting, blocked, notes = check_gates(config, dag, recording_ids, finished)
        specs = _wave(config, layout, dag, order, gates, recording_ids, execution_id,
                      finished | attempted | blocked, finished, attempted)
        if dry_run:
            return RunSummary(
                execution_id=execution_id,
                n_tasks=len(specs),
                n_done=0,
                n_cached=len(already),
                n_failed=0,
                warnings=[],
                output_dir=layout.output_dir,
                waiting=waiting,
            )
        if database is None:
            from macaw.config.loader import archive_resolved

            archive_resolved(config, layout.output_dir)
            database = JobDB(layout.job_db_path)
            database.initialise()
        for (recording_id, gate), gate_notes in notes.items():
            _write_warnings(layout, recording_id, gate, gate_notes)
        if not specs:
            break

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
        n_tasks += len(specs)
        attempted |= {spec.task_id for spec in specs}
        finished |= {
            record.task_id
            for record in database.query(execution_id=execution_id)
            if record.status in ("done", "cached")
        }

    counts = database.summary(execution_id)
    _merge_manifest_fragments(layout, recording_ids)
    return RunSummary(
        execution_id=execution_id,
        n_tasks=n_tasks,
        n_done=counts.get("done", 0),
        n_cached=counts.get("cached", 0) + len(already),
        n_failed=counts.get("failed", 0) + counts.get("cancelled", 0),
        warnings=_collect_warnings(layout, recording_ids),
        output_dir=layout.output_dir,
        waiting=[wait for wait in waiting if not wait.deferred],
    )


def _wave(config, layout, dag, order, gates, recording_ids, execution_id, skip, finished, attempted):
    """Tasks that can run now; one whose upstream already failed in this run is left out."""
    specs: list[JobSpec] = []
    for name in order:
        if name in gates:
            continue
        for recording_id in recording_ids:
            task_id = task_id_for(recording_id, name)
            if task_id in skip:
                continue
            needed = [
                task_id_for(recording_id, dependency)
                for dependency in sorted(_job_dependencies(dag, name, gates))
            ]
            if any(task in attempted and task not in finished for task in needed):
                continue
            specs.append(
                JobSpec(
                    task_id=task_id,
                    recording_id=recording_id,
                    extractor=name,
                    command=_task_command(config, recording_id, name, execution_id),
                    resources=None,
                    log_path=layout.logs_dir / execution_id / f"{_slug(task_id)}.log",
                    depends_on=[task for task in needed if task not in finished],
                )
            )
    return specs


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

    stamp = layout.completion_path(recording_id, extractor_name)
    stamp.unlink(missing_ok=True)
    try:
        extractor = cls(config)
        extractor.check_available()
        recording = build_recording(recording_id, config)

        key = _make_key(config, cls, extractor, recording, config_hash)
        cache_root = _cache_root(config, layout)
        caching = config.execution.cache and cls.cached
        entry = cache_store.lookup(cache_root, key) if caching else None
        if entry is not None:
            shutil.rmtree(layout.artifacts_dir(recording_id, extractor_name), ignore_errors=True)
            cache_store.restore(entry, layout.recording_dir(recording_id))
            _write_stamp(stamp, key)
            database.set_status(task_id, "cached", finished_at=now())
            return

        with tempfile.TemporaryDirectory(
            prefix=f"macaw_{_slug(task_id)}_", dir=config.project.scratch_dir or None
        ) as workspace:
            result = extractor.run(recording, Path(workspace))
            payload = Path(workspace) / "_payload"
            _write_result(layout, recording_id, extractor_name, result, execution_id, config_hash)
            if caching:
                _stage_payload(layout, recording_id, extractor_name, result, payload)
                cache_store.store(cache_root, key, payload)

        _write_stamp(stamp, key)
        database.set_status(task_id, "done", finished_at=now())
    except Exception as error:
        database.set_status(
            task_id, "failed", finished_at=now(), message=f"{type(error).__name__}: {error}"
        )
        traceback.print_exc()
        raise


def _write_stamp(path: Path, key) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key.as_dirname(), encoding="utf-8")


def _stamp(frame, execution_id: str, config_hash: str):
    """Provenance is stamped here so no extractor has to be told which execution it is in.
    Every table that carries the two columns gets it, not just the long table."""
    stamped = frame.copy()
    stamped["execution_id"] = stamped["execution_id"].fillna(execution_id)
    stamped["config_hash"] = stamped["config_hash"].fillna(config_hash)
    return stamped


def _write_result(layout, recording_id, extractor, result, execution_id, config_hash) -> None:
    if result.rows is not None and not result.rows.empty:
        longtable_store.write_long_table(
            _stamp(result.rows, execution_id, config_hash), layout.long_table_path(recording_id)
        )
    if result.keypoints is not None and not result.keypoints.empty:
        keypoints_store.write_keypoints(result.keypoints, layout.keypoints_path(recording_id))
    elif result.keypoints is not None:
        keypoints_store.write_no_keypoints(layout.keypoints_path(recording_id), extractor)
    if result.body_model is not None and not result.body_model.empty:
        body_model_store.write_body_model(result.body_model, layout.body_model_path(recording_id))
    if result.annotations is not None and not result.annotations.empty:
        annotations_store.write_annotations(
            _stamp(result.annotations, execution_id, config_hash),
            layout.annotations_path(recording_id),
        )

    manifest = FeatureManifest()
    manifest.extend(list(result.specs))
    write_manifest(manifest, layout.manifest_fragment_path(recording_id, extractor))

    _write_warnings(layout, recording_id, extractor, result.warnings)

    # an artifact an earlier run left would otherwise pass for this run's
    shutil.rmtree(layout.artifacts_dir(recording_id, extractor), ignore_errors=True)
    for name, source in result.artifacts.items():
        destination = layout.artifacts_dir(recording_id, extractor) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move, not os.replace: the workspace sits in scratch_dir, which on a cluster
        # is a different filesystem from output_dir and a rename across those fails
        shutil.move(str(source), str(destination))


def _write_warnings(layout, recording_id: str, extractor: str, warnings: list[QCWarning]) -> None:
    path = layout.warnings_path(recording_id, extractor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(warning) for warning in warnings], indent=2, ensure_ascii=False), encoding="utf-8")


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
