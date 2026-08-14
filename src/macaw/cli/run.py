from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("run")
def run(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    extractor: list[str] = typer.Option(None, "--extractor", "-e", help="Defaults to all enabled."),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    workers: int = typer.Option(None, "--workers", "-j"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and exit."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Run the extraction pipeline. Archives the resolved config into the output directory."""
    raise NotImplementedError


@app.command("run-task")
def run_task(
    config: Path = typer.Option(..., "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    extractor: str = typer.Option(..., "--extractor", "-e"),
    execution_id: str = typer.Option(..., "--execution-id"),
) -> None:
    """Run one extractor for one recording. This is what a cluster array element calls."""
    raise NotImplementedError


@app.command("status")
def status(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    execution_id: str = typer.Option(None, "--execution-id"),
    failed_only: bool = typer.Option(False, "--failed"),
) -> None:
    """Show task states from the job database."""
    raise NotImplementedError


@app.command("list-extractors")
def list_extractors(
    group: str = typer.Option(None, "--group"),
    backends: bool = typer.Option(False, "--backends", help="Also show available backends."),
) -> None:
    """List registered extractors and which backends are installed."""
    raise NotImplementedError


@app.command("clean")
def clean(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    extractor: str = typer.Option(None, "--extractor", "-e"),
    cache_only: bool = typer.Option(True, "--cache-only/--all"),
) -> None:
    """Invalidate cached results."""
    raise NotImplementedError
