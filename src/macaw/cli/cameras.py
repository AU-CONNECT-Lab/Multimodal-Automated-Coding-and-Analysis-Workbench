from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.project.recording import camera_clips

app = typer.Typer()


@app.command("calibration-board")
def calibration_board(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    paper: str = typer.Option("A2", "--paper", help="A4 | A3 | A2 | A1, landscape."),
    out: Path = typer.Option(None, "--out", help="Where to write the PNG. Default: the calibration folder."),
) -> None:
    """Draw the ChArUco board pose3d.board describes, to print at 100% and glue flat."""
    from macaw.extractors.video.pose.board import PAPERS_MM, board_page, write_png

    resolved = resolve_config(config)
    if paper not in PAPERS_MM:
        raise MacawError(f"--paper is one of {', '.join(PAPERS_MM)}")
    target = write_png(board_page(resolved.pose3d.board, paper), out or Path(resolved.project.calibration_dir) / f"board_{paper}.png")
    typer.echo(str(target))
    typer.echo(
        "Print it at 100% (actual size), glue it flat on a stiff board, measure one square with a ruler and write "
        "that length as pose3d.board.square_mm in project.yaml."
    )


@app.command("calibrate-cameras")
def calibrate_cameras(
    setup: str = typer.Argument(..., help="Folder in project.calibration_dir holding a folder per camera with its board clip."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    until: float = typer.Option(None, "--until", help="Read the clips only up to this many seconds."),
) -> None:
    """Find each camera's lens and where the cameras stand from a ChArUco board filmed by all of them at once;
    writes the calibration pose3d.calibration reads. Name each camera's folder as in the recordings."""
    from macaw.extractors.video.pose.board import calibrate, describe_board, find_board, make_board
    from macaw.extractors.video.pose.cameras import write_setup

    resolved = resolve_config(config)
    folder = Path(resolved.project.calibration_dir) / setup
    by_camera = camera_clips(folder)
    if len(by_camera) < 2:
        raise MacawError(f"{folder} holds {len(by_camera)} camera folder(s) with a clip; put each camera's board clip in a folder named after it, two or more")
    clips = list(by_camera.values())
    names = {clip: camera for camera, clip in by_camera.items()}
    offsets, notes = _offsets(clips, resolved)
    board = make_board(resolved.pose3d.board)
    found = []
    for clip in clips:
        typer.echo(f"reading {clip.name} as camera {names[clip]}")
        found.append(find_board(
            clip, board, name=names[clip], every_s=resolved.pose3d.board_every_s, offset_s=offsets[clip],
            width_factor=_width_factor(clip, resolved), end_s=until,
        ))
    fit = calibrate(found, board)
    fit.setup.metadata.update(
        made_by="macaw calibrate-cameras", made_at=datetime.now().isoformat(timespec="seconds"),
        board=describe_board(resolved.pose3d.board),
        error_px={name: round(value, 3) for name, value in fit.error_px.items()},
        lens_error_px={name: round(value, 3) for name, value in fit.lens_error_px.items()},
        views=dict(fit.views),
        together={f"{a} and {b}": count for (a, b), count in fit.shared.items()},
        clips={names[clip]: clip.name for clip in clips},
        offsets_s={names[clip]: round(offsets[clip], 4) for clip in clips},
        notes=[*notes, *fit.notes],
    )
    path = write_setup(fit.setup, layout_for(resolved).camera_setup_path(setup))
    for camera in fit.setup.cameras:
        x, y, z = camera.position() / 1000.0
        typer.echo(
            f"{camera.name}: board seen well on {fit.views[camera.name]} frames; corners miss by {fit.lens_error_px[camera.name]:.2f} px "
            f"with its own lens fit, {fit.error_px[camera.name]:.2f} px with all cameras together; stands at ({x:.2f}, {y:.2f}, {z:.2f}) m"
        )
    for (a, b), count in fit.shared.items():
        typer.echo(f"{a} and {b} saw the board together {count} times")
    for note in [*notes, *fit.notes]:
        typer.echo(f"note: {note}")
    worst = max(fit.error_px.values())
    if worst > 2.0:
        typer.echo(f"note: corners miss by up to {worst:.1f} px; under 1 or 2 px is good. Film the board again, slower and nearer")
    typer.echo(str(path))


def _offsets(clips: list[Path], config) -> tuple[dict[Path, float], list[str]]:
    """Seconds to add to each clip's clock to reach the first sounding clip's, found from their sound."""
    from macaw.media.align import align_signals, load_normalised
    from macaw.media.probe import probe

    sounding = [clip for clip in clips if probe(clip).audio_streams()]
    offsets = {clip: 0.0 for clip in clips}
    notes = []
    silent = [clip.name for clip in clips if clip not in sounding]
    if silent:
        notes.append(f"{', '.join(silent)} have no sound, so they are taken to start at the same moment as the others")
    if len(sounding) >= 2:
        reference = load_normalised(sounding[0], limit_s=None)
        for clip in sounding[1:]:
            result = align_signals(reference, load_normalised(clip, limit_s=None), config.align, names=(sounding[0].name, clip.name))
            offsets[clip] = result.offset_s
            if result.confidence is not None and result.confidence < config.align.min_confidence:
                notes.append(f"{clip.name} lines up with {sounding[0].name} only weakly by sound; a clap at the start helps")
    return offsets, notes


def _width_factor(clip: Path, config) -> float:
    """How much wider a clip's pixels are than tall, when prep.video makes the recordings' pixels square."""
    from macaw.media.probe import probe

    if not config.video.normalise:
        return 1.0
    streams = probe(clip).video_streams()
    aspect = streams[0].pixel_aspect if streams else None
    return float(aspect) if aspect else 1.0
