from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.gaze_labels import ELSEWHERE, UNSURE, GazeLabel, add_label, read_labels, write_labels
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

app = typer.Typer()


@app.command("label-gaze")
def label_gaze(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    at: float = typer.Option(None, "--at", help="Time of the frame in seconds, on the recording's clock."),
    person: str = typer.Option(None, "--person", help="The role who looks."),
    looks_at: str = typer.Option(None, "--looks-at", help=f"A role, a name from gaze.targets, {ELSEWHERE}, or {UNSURE}."),
    by: str = typer.Option(None, "--by", help="Who labelled. Default: the login name."),
    clear: bool = typer.Option(False, "--clear", help="Remove this recording's labels."),
) -> None:
    """Say whom or what a person looks at in one frame. The quality page compares gaze.target with these labels, and its
    angles backend reads where each target lies from them. Label frames picked at random, not only clear ones."""
    resolved = resolve_config(config, recording_id=recording)
    path = layout_for(resolved).gaze_labels_path(recording)
    if clear:
        path.unlink(missing_ok=True)
        typer.echo(f"{recording}: gaze labels removed")
        return
    if at is None or person is None or looks_at is None:
        raise MacawError("give --at, --person and --looks-at, or --clear")
    roles = list(resolved.project.roles)
    if person not in roles:
        raise MacawError(f"--person takes a role, one of {', '.join(roles)}")
    allowed = [role for role in roles if role != person] + list(resolved.gaze.targets) + [ELSEWHERE, UNSURE]
    if looks_at not in allowed:
        raise MacawError(f"--looks-at takes one of {', '.join(allowed)}")
    fps = [item.fps for item in build_recording(recording, resolved).media if item.kind == "video" and item.fps]
    label = GazeLabel(t=at, subject=person, target=looks_at, by=by or getpass.getuser(), at=datetime.now().isoformat(timespec="seconds"))
    labels = add_label(read_labels(path), label, 1.0 / max(fps) if fps else 0.04)
    write_labels(labels, path)
    mine = [item for item in labels if item.subject == person and item.target != UNSURE]
    typer.echo(f"{recording}: {person} looks at {looks_at} at {at:g} s; {len(mine)} labelled frames for {person}")
