from __future__ import annotations

import getpass
import tempfile
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.scale import (
    CALIBRATION,
    DEPTH_NOTE,
    HAND,
    MEASURED,
    Measurement,
    clip_frame,
    from_measurements,
    read_measurements,
    recording_scale,
    write_measurements,
)

app = typer.Typer()


@app.command("set-scale")
def set_scale(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(None, "--recording", "-r", help="Measure on this recording's own video."),
    calibration: str = typer.Option(None, "--calibration", help="Measure on this clip in scale.calibration_dir instead."),
    media: str = typer.Option(None, "--media", help="The recording's video measured on; needed when it has several."),
    points: str = typer.Option(None, "--points", help="The object's two ends in pixels of the frame: x1,y1,x2,y2."),
    length_cm: float = typer.Option(None, "--length-cm", help="Its real length in centimetres."),
    person: str = typer.Option(None, "--person", help="Role at whose place the object stood. Default: everyone."),
    at: float = typer.Option(0.0, "--at", help="Time of the frame in seconds, on the recording's clock or the clip's."),
    object_: str = typer.Option(None, "--object", help="What was measured, such as a ruler."),
    by: str = typer.Option(None, "--by", help="Who measured. Default: the login name."),
    clear: bool = typer.Option(False, "--clear", help="Remove the measurements of this recording or clip."),
) -> None:
    """Measure an object of known length on a frame, so exports add centimetres; measure again to average.
    A scale is right only for people as far from the camera as the object."""
    if (recording is None) == (calibration is None):
        raise MacawError("give --recording to measure on a recording's video, or --calibration for a clip")
    resolved = resolve_config(config, recording_id=recording) if recording else resolve_config(config)
    layout = layout_for(resolved)
    path = layout.scale_path(recording) if recording else layout.calibration_scale_path(calibration)
    name = recording or f"calibration clip {calibration}"
    if clear:
        path.unlink(missing_ok=True)
        typer.echo(f"{name}: measurements removed")
        return
    if points is None or length_cm is None:
        raise MacawError("give --points and --length-cm, or --clear")
    try:
        x1, y1, x2, y2 = (float(value) for value in points.split(","))
    except ValueError:
        raise MacawError(f"--points takes four numbers, x1,y1,x2,y2, not {points}") from None
    if length_cm <= 0:
        raise MacawError("--length-cm must be above 0")
    if person is not None and person not in resolved.project.roles:
        raise MacawError(f"--person takes a role, one of {', '.join(resolved.project.roles)}")

    frame_size = None
    if recording:
        media = _video(resolved, recording, media)
    else:
        with tempfile.TemporaryDirectory() as folder:
            _, size = clip_frame(resolved, calibration, at, Path(folder) / "frame.jpg")
        frame_size, media = list(size), calibration
    measurement = Measurement(
        video=media, t=at, points=[[x1, y1], [x2, y2]], length_cm=length_cm, person=person, object=object_,
        by=by or getpass.getuser(), at=datetime.now().isoformat(timespec="seconds"), frame_size=frame_size,
    )
    if measurement.length_px == 0:
        raise MacawError("the two ends are the same point")

    measurements = read_measurements(path)
    # every measurement of a recording is on the same picture
    elsewhere = sorted({item.video for item in measurements} - {media})
    if elsewhere:
        raise MacawError(f"{recording} was measured on {', '.join(elsewhere)}; remove that with --clear to measure on {media}")
    measurements.append(measurement)
    write_measurements(measurements, path)
    typer.echo(f"{name}: {length_cm:g} cm over {measurement.length_px:.1f} px for {person or 'everyone'}")
    typer.echo(f"now {from_measurements(measurements, MEASURED if recording else CALIBRATION, media).values()}; {DEPTH_NOTE}")
    if recording and recording_scale(resolved, layout, recording).source == HAND:
        typer.echo("scale.cm_per_px or scale.cm_per_px_by_role is set in the settings, so exports use that instead")
    if calibration and resolved.scale.calibration != calibration:
        typer.echo(f"to use it, write scale: {{calibration: {calibration}}} in project.yaml, or in recordings/<id>.yaml for some recordings")


def _video(config, recording_id: str, media: str | None) -> str:
    videos = [item.path.name for item in build_recording(recording_id, config).media if item.kind == "video"]
    if media is None:
        if len(videos) != 1:
            raise MacawError(f"{recording_id} has {len(videos)} videos; name the one measured on with --media")
        return videos[0]
    if media not in videos:
        raise MacawError(f"{media} is not a video of {recording_id}; its videos are {', '.join(videos) or 'none'}")
    return media
