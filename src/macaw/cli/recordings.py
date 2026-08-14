from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("recordings")
def list_recordings(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    prefix: str = typer.Option(None, "--prefix", "-p", help="Directory prefix. Overrides the config."),
    out: Path = typer.Option(None, "--out", help="Write the ids to a file, one per line."),
) -> None:
    """List recording ids found under raw_dir."""
    raise NotImplementedError
