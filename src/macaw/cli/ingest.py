from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("ingest")
def ingest(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    session: list[str] = typer.Option(None, "--session", "-s", help="Limit to these session ids."),
    rescan: bool = typer.Option(False, "--rescan", help="Rebuild the session index from raw_dir."),
    set_: list[str] = typer.Option(None, "--set", help="key.path=value override."),
) -> None:
    """Discover sessions, probe media, and write the session index."""
    raise NotImplementedError


@app.command("sessions")
def list_sessions(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    show_media: bool = typer.Option(False, "--media"),
) -> None:
    """List known sessions and their media."""
    raise NotImplementedError
