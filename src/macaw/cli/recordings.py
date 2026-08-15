from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.logging_setup import get_logger
from macaw.project.recording import detect_recordings

app = typer.Typer()
logger = get_logger("recordings")


@app.command("recordings")
def list_recordings(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    prefix: str = typer.Option(None, "--prefix", "-p", help="Directory prefix. Overrides the config."),
    out: Path = typer.Option(None, "--out", help="Write the ids to a file, one per line."),
) -> None:
    """List recording ids found under raw_dir."""
    resolved = resolve_config(config)
    project = resolved.project
    found = detect_recordings(
        project.raw_dir,
        project.recording_pattern,
        project.recording_prefix if prefix is None else prefix,
    )
    excluded = set(project.excluded_recordings)
    ids = [recording_id for recording_id in found if recording_id not in excluded]

    logger.info(
        "%d recordings under %s, %d excluded", len(ids), project.raw_dir, len(found) - len(ids)
    )
    for recording_id in ids:
        typer.echo(recording_id)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(f"{recording_id}\n" for recording_id in ids), encoding="utf-8")
