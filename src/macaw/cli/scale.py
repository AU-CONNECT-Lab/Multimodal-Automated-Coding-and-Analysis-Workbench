from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.scale import HAND, Measurement, read_measurements, recording_scale, write_measurements

app = typer.Typer()


@app.command("set-scale")
def set_scale(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    media: str = typer.Option(None, "--media", help="Video the yardstick was measured on."),
    points: str = typer.Option(None, "--points", help="Its two ends in pixels of the frame: x1,y1,x2,y2."),
    length_cm: float = typer.Option(None, "--length-cm", help="Its real length in centimetres."),
    at: float = typer.Option(0.0, "--at", help="Recording time of the frame, in seconds."),
    object_: str = typer.Option(None, "--object", help="What was measured, such as a table edge."),
    by: str = typer.Option(None, "--by", help="Who measured. Default: the login name."),
    clear: bool = typer.Option(False, "--clear", help="Remove this recording's measurements."),
) -> None:
    """Measure an object of known length on a video frame, so exports add centimetres; measure again to average."""
    resolved = resolve_config(config, recording_id=recording)
    layout = layout_for(resolved)
    path = layout.scale_path(recording)
    if clear:
        path.unlink(missing_ok=True)
        typer.echo(f"{recording}: measurements removed")
        return
    if media is None or points is None or length_cm is None:
        raise MacawError("give --media, --points and --length-cm, or --clear")

    videos = [item.path.name for item in build_recording(recording, resolved).media if item.kind == "video"]
    if media not in videos:
        raise MacawError(f"{media} is not a video of {recording}; its videos are {', '.join(videos) or 'none'}")
    try:
        x1, y1, x2, y2 = (float(value) for value in points.split(","))
    except ValueError:
        raise MacawError(f"--points takes four numbers, x1,y1,x2,y2, not {points}") from None
    if length_cm <= 0:
        raise MacawError("--length-cm must be above 0")
    measurement = Measurement(
        video=media, t=at, points=[[x1, y1], [x2, y2]], length_cm=length_cm, object=object_,
        by=by or getpass.getuser(), at=datetime.now().isoformat(timespec="seconds"),
    )
    if measurement.length_px == 0:
        raise MacawError("the two ends are the same point")

    measurements = read_measurements(path)
    # one scale per recording, so every measurement is on the same picture
    elsewhere = sorted({item.video for item in measurements} - {media})
    if elsewhere:
        raise MacawError(f"{recording} was measured on {', '.join(elsewhere)}; remove that with --clear to measure on {media}")
    write_measurements([*measurements, measurement], path)
    scale = recording_scale(resolved, layout, recording)
    typer.echo(f"{recording}: {measurement.length_cm:g} cm over {measurement.length_px:.1f} px; now {scale.describe()}")
    if scale.source == HAND:
        typer.echo("scale.cm_per_px is set in the settings, so exports use that number instead")
