from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.project.segments import Segment, remove_segment, seconds_text, set_segment

app = typer.Typer()


@app.command("set-segment")
def set_segment_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    name: str = typer.Option(..., "--name", help="What the segment is called; its rows in the export carry this name."),
    start: float = typer.Option(None, "--start", help="Where it begins, in seconds on the recording's clock."),
    end: float = typer.Option(None, "--end", help="Where it ends, in seconds on the recording's clock."),
    clear: bool = typer.Option(False, "--clear", help="Take the segment out of the table."),
) -> None:
    """Name a part of a recording whose measures are worked out again on their own, in the project's segments table."""
    resolved = resolve_config(config)
    if recording not in resolve_recordings(resolved, None):
        raise MacawError(f"{recording} is not one of this project's recordings")
    if clear:
        path = remove_segment(resolved, recording, name)
        typer.echo(f"{recording}: {name} taken out of {path}")
        return
    if start is None or end is None:
        raise MacawError("give --start and --end, or --clear")
    path = set_segment(resolved, recording, Segment(name.strip(), start, end))
    typer.echo(f"{recording}: {name.strip()} from {seconds_text(start)} to {seconds_text(end)} s in {path}; macaw run works out its measures")
