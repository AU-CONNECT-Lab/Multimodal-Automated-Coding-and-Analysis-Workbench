from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.project.previews import current_preview, make_camera_preview
from macaw.project.recording import build_recording

app = typer.Typer()


@app.command("previews")
def previews_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: list[str] = typer.Option(None, "--recording", "-r", help="Recordings to make copies of; every one by default."),
) -> None:
    """Make the small copies of each camera, with sound, that the Review page plays; copies still current are kept."""
    resolved = resolve_config(config)
    for rid in resolve_recordings(resolved, ",".join(recordings) if recordings else None):
        own = resolve_config(config, recording_id=rid)
        try:
            cameras = [item.name for item in build_recording(rid, own).cameras()]
        except MacawError as error:
            typer.echo(f"{rid}: {error}")
            continue
        for camera in cameras:
            if current_preview(own, rid, camera) is not None:
                typer.echo(f"{rid} {camera}: already made")
                continue
            path = make_camera_preview(own, rid, camera)
            typer.echo(f"{rid} {camera}: {path}")
