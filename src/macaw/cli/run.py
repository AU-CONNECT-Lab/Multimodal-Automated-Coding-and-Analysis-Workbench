from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution import runner
from macaw.execution.dag import build_dag
from macaw.execution.job_db import JobDB
from macaw.extractors import (
    get_backend,
    get_extractor,
    list_backend_groups,
    list_backends,
    list_extractors,
)
from macaw.extractors.video.pose.backends import BACKEND_GROUP as POSE_BACKENDS
from macaw.logging_setup import get_logger
from macaw.project.layout import layout_for
from macaw.project.setup import require_setup
from macaw.store import cache

app = typer.Typer()
logger = get_logger("run")


@app.command("run")
def run(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    extractor: list[str] = typer.Option(
        None,
        "--extractor",
        "-e",
        help="Extractor or group name, repeatable. Defaults to pipeline.run, else everything installed.",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    workers: int = typer.Option(None, "--workers", "-j"),
    output_dir: Path = typer.Option(
        None, "--output", "-o", help="Overrides project.output_dir. Also read from $MACAW_OUTPUT_DIR."
    ),
    work_dir: Path = typer.Option(
        None, "--work", "-w", help="Logs, job records and cache. Also read from $MACAW_WORK_DIR."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and exit."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Run the extraction pipeline. Archives the resolved config into the output directory."""
    overrides = list(set_ or ())
    # -j goes through the same merge as everything else; no extractor declares the execution
    # section, so it cannot move a config hash and invalidate a cache
    if workers is not None:
        overrides.append(f"execution.n_workers={workers}")

    # the two directories bypass --set: a Windows path is not always a valid YAML scalar,
    # and paths never enter a config hash anyway
    resolved = resolve_config(
        config, cli_overrides=overrides, output_dir=output_dir, work_dir=work_dir
    )
    require_setup(resolved)
    recording_ids = runner.resolve_recordings(resolved, recordings)
    if not recording_ids:
        raise MacawError(f"No recordings found under {resolved.project.raw_dir}")
    wanted = list(extractor or ())
    dag = build_dag(resolved, wanted or None)
    _echo_reporting(resolved, dag, recording_ids)

    if dry_run:
        finished = runner.completed_on_disk(resolved, dag, recording_ids)
        for task_id in runner.plan(resolved, recording_ids, wanted, finished):
            typer.echo(task_id)
        _echo_waiting(runner.check_gates(resolved, dag, recording_ids, finished)[0])
        runner.check_ready(runner.recording_configs(resolved, recording_ids), runner.pending_steps(dag, recording_ids, finished))
        return

    logger.info("%d recordings, extractors: %s", len(recording_ids), ", ".join(wanted) or "all")
    summary = runner.run(
        resolved, recordings=recording_ids, extractors=wanted or None, resume=resume
    )
    typer.echo(f"execution_id {summary.execution_id}")
    typer.echo(
        f"tasks {summary.n_tasks}  done {summary.n_done}  "
        f"cached {summary.n_cached}  failed {summary.n_failed}"
    )
    typer.echo(f"output {summary.output_dir}")
    _echo_waiting(summary.waiting)
    if summary.warnings:
        by_severity: dict[str, int] = {}
        for warning in summary.warnings:
            by_severity[warning.severity] = by_severity.get(warning.severity, 0) + 1
        typer.echo(
            "qc " + "  ".join(f"{severity} {count}" for severity, count in sorted(by_severity.items()))
        )
    if summary.n_failed:
        raise typer.Exit(code=1)


def _echo_reporting(config, dag, recording_ids: list[str]) -> None:
    from macaw.coding.llm.notes import notes_plan
    from macaw.extractors.video.face.backends import BACKEND_GROUP as FACE_BACKENDS
    from macaw.project.recording import recording_entities

    def said_by_entity(name: str) -> str | None:
        where, llm = get_extractor(name).notes_settings(config)
        if not llm.notes_by:
            return None
        return notes_plan(name, where, llm, {rid: recording_entities(rid, config) for rid in recording_ids})

    notices = []
    if "pose.detect" in dag.nodes and config.pose.backend is not None:
        notices.append(get_backend(POSE_BACKENDS, config.pose.backend).reporting)
    if "face.au" in dag.nodes:
        notices.append(get_backend(FACE_BACKENDS, config.face.backend).reporting)
    for name in sorted(node for node in dag.nodes if node == "coding.judge" or node.startswith("coding.judge.")):
        from macaw.coding.llm.backends import BACKEND_GROUP as LLM_BACKENDS
        from macaw.coding.llm.judge import cloud_notice, units_per_request

        _, llm = get_extractor(name).settings(config)
        notices.append(cloud_notice(get_backend(LLM_BACKENDS, llm.provider)(llm), llm))
        notices.append(
            f"{name} sends {llm.n_samples} x ceil(units / {units_per_request(llm)}) requests per recording "
            f"to {llm.provider}, units being the codebook's utterances or turns"
        )
        notices.append(said_by_entity(name))
    for name in sorted(node for node in dag.nodes if node.startswith("coding.rate.")):
        from macaw.coding.llm.backends import BACKEND_GROUP as LLM_BACKENDS
        from macaw.coding.llm.rate import rating_notice, scales_asked

        own, llm = get_extractor(name).settings(config)
        backend = get_backend(LLM_BACKENDS, llm.provider)(llm)
        notices.append(rating_notice(name, backend) if backend.reporting else None)
        scales = scales_asked(config, get_extractor(name).rating)
        where = f"each of {len(own.segments)} segments of each recording" if own.segments else "each recording"
        notices.append(f"{name} sends {llm.n_samples} x {scales if scales is not None else 'its'} scales requests for {where} to {llm.provider}")
        notices.append(said_by_entity(name))
    if "pose.listening" in dag.nodes:
        from macaw.extractors.video.pose.listening import body_size_notice

        notices.append(body_size_notice(config))
    look = config.qc.look
    if "pose.keypoints" in dag.nodes and config.pose.suggest_names and look.model and look.people:
        from macaw.qc.look import notice

        notices.append(notice(look, "pictures of the people on cameras nobody clicked, for pose.suggest_names,"))
    from macaw.execution import models

    fetching = models.fetching_steps(dag.topological_order())
    if fetching and not models.offline():
        notices.append(f"model weights for {', '.join(fetching)} are fetched or checked before anything runs; the steps then run offline")
    for notice in notices:
        if notice:
            typer.echo(f"note: {notice}")


def _echo_waiting(waiting) -> None:
    later = [wait for wait in waiting if wait.deferred]
    if later:
        typer.echo("decided during the run, once what they read exists:")
        for wait in later:
            typer.echo(f"  {wait.recording_id}  {wait.gate}: {wait.message}")
    waiting = [wait for wait in waiting if not wait.deferred]
    if not waiting:
        return
    typer.echo(f"waiting for a person ({len({wait.recording_id for wait in waiting})} recording(s)):")
    for wait in waiting:
        typer.echo(f"  {wait.recording_id}  {wait.gate}: {wait.message}")
    typer.echo("everything after these steps runs on the next `macaw run` once they are cleared")


@app.command("run-task")
def run_task(
    config: Path = typer.Option(..., "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    extractor: str = typer.Option(..., "--extractor", "-e"),
    execution_id: str = typer.Option(..., "--execution-id"),
) -> None:
    """Run one extractor for one recording, as each task of a run does."""
    resolved = resolve_config(config, recording_id=recording)
    runner.run_task(resolved, recording, extractor, execution_id)


@app.command("fetch-models", hidden=True)
def fetch_models_command(
    config: Path = typer.Option(..., "--config", "-c"),
    recordings: str = typer.Option("", "--recordings", "-r", help="Comma-separated ids."),
    extractor: list[str] = typer.Option(..., "--extractor", "-e"),
) -> None:
    """What a run does before its steps: their model weights, fetched while the network is allowed."""
    from macaw.execution.models import fetch_models

    ids = [part for part in recordings.split(",") if part]
    if fetch_models(config, extractor, ids, strict=False, echo=typer.echo):
        raise typer.Exit(code=1)


@app.command("status")
def status(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    execution_id: str = typer.Option(None, "--execution-id"),
    failed_only: bool = typer.Option(False, "--failed"),
    work_dir: Path = typer.Option(
        None, "--work", "-w", help="Where the job records are. Also read from $MACAW_WORK_DIR."
    ),
) -> None:
    """Show task states from the job database."""
    resolved = resolve_config(config, work_dir=work_dir)
    layout = layout_for(resolved, create=False)
    database = JobDB(layout.job_db_path)

    known = database.executions()
    if not known:
        typer.echo("no executions recorded")
        return
    # execution ids start with a timestamp, so the last one sorted is the most recent
    wanted = execution_id or known[-1]

    if failed_only:
        for record in database.failed(wanted):
            typer.echo(f"{record.task_id}\t{record.exit_code}\t{record.message or ''}")
        return

    typer.echo(wanted)
    counts = database.summary(wanted)
    typer.echo("  ".join(f"{name} {count}" for name, count in counts.items() if count))


@app.command("list-extractors")
def list_registered(
    group: str = typer.Option(None, "--group"),
    backends: bool = typer.Option(False, "--backends", help="Also show registered backends."),
    config: Path = typer.Option(None, "--config", "-c", help="Also report which ones can run here."),
) -> None:
    """List registered extractors and, with a config, which of them are installed."""
    resolved = resolve_config(config) if config is not None else None
    for name in list_extractors(group):
        cls = get_extractor(name)
        line = f"{name}\t{cls.group}\t{cls.phase.value}\t{cls.version}"
        if resolved is not None:
            try:
                cls(resolved).check_available()
                line += "\tok"
            except MacawError as error:
                # install instructions run to several lines; one extractor stays one row
                line += "\t" + " ".join(str(error).split())
        typer.echo(line)

    if backends:
        for backend_group in list_backend_groups():
            typer.echo(f"{backend_group}\t{', '.join(list_backends(backend_group))}")


@app.command("clean")
def clean(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    extractor: str = typer.Option(None, "--extractor", "-e"),
    cache_only: bool = typer.Option(
        True, "--cache-only/--all", help="--all also deletes the computed output, keeping what people did by hand."
    ),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Also read from $MACAW_OUTPUT_DIR."),
    work_dir: Path = typer.Option(
        None, "--work", "-w", help="Where the cache is. Also read from $MACAW_WORK_DIR."
    ),
) -> None:
    """Invalidate cached results."""
    resolved = resolve_config(config, output_dir=output_dir, work_dir=work_dir)
    layout = layout_for(resolved, create=False)
    cache_root = (
        Path(resolved.execution.cache_dir) if resolved.execution.cache_dir else layout.cache_dir
    )

    removed = cache.invalidate(cache_root, extractor=extractor) if cache_root.exists() else 0
    typer.echo(f"cache entries removed {removed}")
    if cache_only:
        return

    targets = (
        sorted(path for path in layout.recordings_dir.glob("*") if path.is_dir())
        if layout.recordings_dir.exists()
        else []
    )
    if not targets:
        return
    typer.confirm(
        f"--all also deletes the computed output in the folders of {len(targets)} recordings under "
        f"{layout.output_dir}. What people did there by hand stays: offsets, voices and cameras named or "
        "accepted, measured scales, labelled frames, corrected transcripts, review decisions and imported "
        "coding. Continue?",
        abort=True,
    )
    kept: Counter[str] = Counter()
    for path in targets:
        made = {item for item in layout.human_made(path.name) if item.is_file()}
        _remove_all_but(path, made)
        kept.update(item.relative_to(path).as_posix() for item in made)
    typer.echo(f"computed output removed from {len(targets)} recording folders")
    if kept:
        typer.echo("kept, made by hand: " + ", ".join(f"{name} in {count}" for name, count in sorted(kept.items())))
        typer.echo("to remove those too, delete the folders yourself")


def _remove_all_but(folder: Path, keep: set[Path]) -> None:
    for path in folder.iterdir():
        if path in keep:
            continue
        if path.is_dir() and any(path in item.parents for item in keep):
            _remove_all_but(path, keep)
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if not any(folder.iterdir()):
        folder.rmdir()
