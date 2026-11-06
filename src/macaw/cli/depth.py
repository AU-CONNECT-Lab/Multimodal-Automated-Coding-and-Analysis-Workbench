from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Depth cameras.")


@app.command("depth-import")
def depth_import(
    recording: Path = typer.Argument(..., help="A RealSense .bag, or an Azure Kinect or Orbbec Femto .mkv."),
    out: Path = typer.Argument(..., help="The camera's device folder in one recording, such as raw/dyad01/kinect."),
) -> None:
    """Write a depth camera's colour video, and beside it a depth folder with one depth image per frame aligned with
    the colour picture, as a camera marked depth: true in the setup reads them. The alignment is the camera maker's:
    pyrealsense2 reads a .bag, and pyk4a, with the Azure Kinect or Orbbec SDK installed, reads a .mkv."""
    from macaw.media.depth_import import import_recording

    done = import_recording(recording, out)
    typer.echo(
        f"{done['video']}: {done['frames']} frames at {done['fps']:g} fps, {done['size'][0]} x {done['size'][1]}\n"
        f"{done['depth']}: one depth image per frame"
    )
    if done["repeated"]:
        typer.echo(f"{done['repeated']} frame(s) missing from the recording repeat the picture before them, with no depth")
    if done["empty_depth"]:
        typer.echo(f"{done['empty_depth']} depth image(s) are empty in the recording itself")
    for note in done["notes"]:
        typer.echo(note)
