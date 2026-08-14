from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("align")
def align(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    recut: bool = typer.Option(False, "--recut", help="Physically re-encode instead of storing offsets."),
    drift_check: bool = typer.Option(True, "--drift-check/--no-drift-check"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Estimate inter-source offsets by cross-correlation and flag low-confidence results."""
    raise NotImplementedError


@app.command("set-offset")
def set_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    media: str = typer.Option(..., "--media", help="Media file name to adjust."),
    offset: float = typer.Option(..., "--offset", help="Seconds."),
) -> None:
    """Manually override one offset after inspecting it."""
    raise NotImplementedError
