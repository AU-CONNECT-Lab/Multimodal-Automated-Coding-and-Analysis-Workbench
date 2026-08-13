from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("qc")
def qc(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
    check: list[str] = typer.Option(None, "--check", help="Run only these checks."),
    report: bool = typer.Option(True, "--report/--no-report"),
    open_report: bool = typer.Option(False, "--open"),
) -> None:
    """Run quality checks and write a self-contained HTML report."""
    raise NotImplementedError


@app.command("identify")
def identify_speakers(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
    interactive: bool = typer.Option(True, "--interactive/--accept-votes"),
) -> None:
    """Three-source speaker identity vote, then confirm each session interactively."""
    raise NotImplementedError
