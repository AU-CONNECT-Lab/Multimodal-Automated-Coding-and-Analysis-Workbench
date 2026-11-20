from __future__ import annotations

import getpass
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.extractors.video.objects.decisions import ABSENT, KEPT, STARTS, add_start, clear, keep, read_decisions, set_absent, write_decisions
from macaw.extractors.video.objects.review import start_of, waiting
from macaw.extractors.video.objects.track import FINDER, load_starts, object_cameras
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

app = typer.Typer()


def describe_start(entry: dict) -> str:
    """One thing on one camera in a few words: where it starts, by whom, and how long it was in view."""
    if "frame" not in entry:
        return entry["from"]
    said = [f"starts at {entry['t']:.1f} s, {entry['from']}"]
    if entry["from"] == FINDER:
        said[0] += f" (score {entry['score']:.2f}" + (f", {entry['candidates']} candidates)" if entry.get("candidates", 1) > 1 else ")")
    if entry.get("frames"):
        said.append(
            f"in view {entry['found'] / entry['frames']:.0%} of the recording ({entry['share_after_start']:.0%} after the start), "
            f"at most {entry['longest_gap_s']:g} s out of view at once"
        )
    return "; ".join(said)


def _point(text: str) -> tuple[float, float]:
    try:
        x, y = (float(value) for value in text.split(","))
    except ValueError:
        raise MacawError("--point takes X,Y in the video's pixels, e.g. 640,360") from None
    return x, y


@app.command("objects", rich_help_panel="Review")
def objects(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recording: str = typer.Option(..., "--recording", "-r", help="The recording's id."),
    keep_found: bool = typer.Option(False, "--keep", help="Keep the starts found by name that wait for a person, as they are."),
    start: str = typer.Option(None, "--start", help="A thing from objects.names that a person starts at --point, --at, on --camera."),
    not_here: str = typer.Option(None, "--not-here", help="A thing from objects.names that is not on --camera."),
    forget: str = typer.Option(None, "--clear", help="Forget what people said about this thing on --camera."),
    camera: str = typer.Option(None, "--camera", help="The camera; with --keep, only its starts. Default for --keep: every camera."),
    at: float = typer.Option(None, "--at", help="Time of the frame in seconds, on the recording's clock."),
    point: str = typer.Option(None, "--point", help="X,Y of a point on the thing, in the video's pixels."),
    by: str = typer.Option(None, "--by", help="Who looked. Default: the login name."),
) -> None:
    """List where each thing in objects.names starts on each camera and what waits for a person, or keep, give or
    forget a start. A start given here is followed from the next `macaw run`."""
    resolved = resolve_config(config, recording_id=recording)
    names = list(resolved.objects.names)
    if not names:
        raise MacawError("objects.names names no things to follow")
    path = layout_for(resolved, create=False).objects_path(recording)
    said = read_decisions(path)
    starts = load_starts(resolved, recording)
    who = by or getpass.getuser()
    chosen = [flag for flag, value in (("--start", start), ("--not-here", not_here), ("--clear", forget), ("--keep", keep_found)) if value]
    if len(chosen) > 1:
        raise MacawError(f"give one of {', '.join(chosen)}")
    if start or not_here or forget:
        name = start or not_here or forget
        if name not in names:
            raise MacawError(f"{name} is not in objects.names ({', '.join(names)})")
        cameras = [item.name for item in object_cameras(build_recording(recording, resolved))]
        if camera not in cameras:
            raise MacawError(f"--camera takes one of {', '.join(cameras) or 'no camera: nobody is read from one'}")
        if start:
            if at is None or point is None:
                raise MacawError("--start needs --at and --point")
            x, y = _point(point)
            add_start(said, camera, name, at, x, y, who)
            message = f"{name} on {camera} is at {x:g},{y:g} at {at:g} s; the next `macaw run` follows it from there"
        elif not_here:
            set_absent(said, camera, name, who)
            message = f"{name} is not on {camera}; the next `macaw run` stops looking for it there"
        else:
            clear(said, camera, name)
            message = f"what people said about {name} on {camera} is forgotten; the next `macaw run` looks for it by name"
        write_decisions(path, said)
        typer.echo(f"{recording}: {message}")
        return
    if keep_found:
        kept = []
        for item in waiting(resolved, recording, review="always"):
            entry = starts.get(item.camera, {}).get(item.name, {})
            spoken = item.name in said[STARTS].get(item.camera, {}) or item.name in said[ABSENT].get(item.camera, {})
            if entry.get("from") != FINDER or spoken or (camera and item.camera != camera):
                continue
            keep(said, item.camera, item.name, start_of(entry), who)
            kept.append(f"{item.name} on {item.camera}")
        if not kept:
            typer.echo(f"{recording}: no start found by name waits to be kept" + (f" on {camera}" if camera else ""))
            return
        write_decisions(path, said)
        typer.echo(f"{recording}: kept the starts of {', '.join(kept)} as found by name")
        return
    if not starts:
        typer.echo(f"{recording}: no thing followed yet; `macaw run` finds them")
        return
    open_items = {(item.camera, item.name): item.why for item in waiting(resolved, recording)}
    for name_of_camera, entries in starts.items():
        for name, entry in entries.items():
            kept_by = said[KEPT].get(name_of_camera, {}).get(name, {}).get("by")
            state = f"waits: {open_items[name_of_camera, name]}" if (name_of_camera, name) in open_items else f"kept by {kept_by}" if kept_by else "clear"
            typer.echo(f"{recording} {name_of_camera} {name}: {describe_start(entry)} [{state}]")
