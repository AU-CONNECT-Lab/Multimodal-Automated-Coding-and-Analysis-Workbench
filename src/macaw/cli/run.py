from __future__ import annotations

import shutil
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution import runner
from macaw.execution.job_db import JobDB
from macaw.extractors import get_extractor, list_backend_groups, list_backends, list_extractors
from macaw.logging_setup import get_logger
from macaw.project.layout import layout_for
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
    recording_ids = runner.resolve_recordings(resolved, recordings)
    if not recording_ids:
        raise MacawError(f"No recordings found under {resolved.project.raw_dir}")
    wanted = list(extractor or ())

    if dry_run:
        for task_id in runner.plan(resolved, recording_ids, wanted):
            typer.echo(task_id)
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
    if summary.warnings:
        by_severity: dict[str, int] = {}
        for warning in summary.warnings:
            by_severity[warning.severity] = by_severity.get(warning.severity, 0) + 1
        typer.echo(
            "qc " + "  ".join(f"{severity} {count}" for severity, count in sorted(by_severity.items()))
        )
    if summary.n_failed:
        raise typer.Exit(code=1)


@app.command("run-task")
def run_task(
    config: Path = typer.Option(..., "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    extractor: str = typer.Option(..., "--extractor", "-e"),
    execution_id: str = typer.Option(..., "--execution-id"),
) -> None:
    """Run one extractor for one recording. This is what a cluster array element calls."""
    resolved = resolve_config(config, recording_id=recording)
    runner.run_task(resolved, recording, extractor, execution_id)


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
    cache_only: bool = typer.Option(True, "--cache-only/--all"),
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
        f"--all also deletes computed output for {len(targets)} recordings under "
        f"{layout.output_dir}. Continue?",
        abort=True,
    )
    for path in targets:
        shutil.rmtree(path)
    typer.echo(f"recording directories removed {len(targets)}")
