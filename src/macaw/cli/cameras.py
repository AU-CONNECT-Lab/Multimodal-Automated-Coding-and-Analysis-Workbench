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
    target = write_png(board_page(resolved.pose3d.board.in_mm(resolved.project.units), paper), out or Path(resolved.project.calibration_dir) / f"board_{paper}.png")
    typer.echo(str(target))
    typer.echo(
        "Print it at 100% (actual size), glue it flat on a stiff board, measure one square with a ruler and write "
        "that length, in project.units, as pose3d.board.square in project.yaml."
    )


@app.command("calibrate-cameras")
def calibrate_cameras(
    setup: str = typer.Argument(..., help="Folder in project.calibration_dir holding a folder per camera with its board clip."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    until: float = typer.Option(None, "--until", help="Read the clips only up to this many seconds."),
) -> None:
    """Find each camera's lens and where the cameras stand from a ChArUco board filmed by all of them at once;
    writes the calibration pose3d.calibration reads. Name each camera's folder as in the recordings."""
    from macaw.extractors.video.pose.board import calibrate, describe_board, find_board, make_board, width_factor
    from macaw.extractors.video.pose.cameras import write_setup

    resolved = resolve_config(config)
    folder = Path(resolved.project.calibration_dir) / setup
    by_camera = camera_clips(folder)
    if len(by_camera) < 2:
        raise MacawError(f"{folder} holds {len(by_camera)} camera folder(s) with a clip; put each camera's board clip in a folder named after it, two or more")
    clips = list(by_camera.values())
    names = {clip: camera for camera, clip in by_camera.items()}
    offsets, notes = _offsets(clips, resolved)
    board = make_board(resolved.pose3d.board.in_mm(resolved.project.units))
    found = []
    for clip in clips:
        typer.echo(f"reading {clip.name} as camera {names[clip]}")
        found.append(find_board(
            clip, board, name=names[clip], every_s=resolved.pose3d.board_every_s, offset_s=offsets[clip],
            width_factor=width_factor(clip, resolved.video.normalise), end_s=until,
        ))
    fit = calibrate(found, board)
    fit.setup.metadata.update(
        made_by="macaw calibrate-cameras", made_at=datetime.now().isoformat(timespec="seconds"),
        board=describe_board(resolved.pose3d.board.in_mm(resolved.project.units)),
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


@app.command("calibration-floor")
def calibration_floor(
    calibration: str = typer.Argument(..., help="Folder in project.calibration_dir holding the board clips."),
    at: float = typer.Option(..., "--at", help="Second of the board clips, on the first camera's clock, at which the board lay flat."),
    on: str = typer.Option("floor", "--on", help="floor, or the thing of the room the board lay on, such as table."),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Say when the board lay flat in a calibration's board clips, on the floor or on a thing of the room; placing the
    cameras in the room then takes which way is up and how high they are from it."""
    from macaw.extractors.video.pose.board import make_board
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.extractors.video.pose.triangulate import calibration_path
    from macaw.project.placing import find_floor, write_floor

    resolved = resolve_config(config)
    folder = Path(resolved.project.calibration_dir) / calibration
    if not folder.is_dir():
        raise MacawError(f"{folder} is not a folder of board clips; the board lying flat is read from them")
    path = calibration_path(resolved, calibration)
    if not path.is_file():
        raise MacawError(f"the cameras of {calibration} are not found yet: run macaw calibrate-cameras {calibration} first")
    setup = read_setup(path, resolved.pose3d.units)
    board = make_board(resolved.pose3d.board.in_mm(resolved.project.units))
    floor = find_floor(setup, camera_clips(folder), board, at, on, normalise=resolved.video.normalise)
    seen = floor.seen_by if floor is not None else {}
    for camera in setup.cameras:
        count = seen.get(camera.name, 0)
        typer.echo(f"{camera.name}: {f'board found, {count} corners' if count else 'board not found'}")
    if floor is None:
        raise MacawError(f"no camera finds the board at {at:g} s; choose a moment when it lies flat and still in view of a camera")
    typer.echo(str(write_floor(folder, at, on)))


@app.command("place-cameras")
def place_cameras_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    setup: str = typer.Option(None, "--setup", help="Which setup. Default: setup.name."),
) -> None:
    """Fill each calibrated camera's place, height, facing, tilt and lens angle in a setup's room from pose3d.calibration,
    tied to the room by what it already gives and the board lying flat; writes setups/<setup>.yaml."""
    import yaml

    from macaw.config.schema import MacawConfig
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.extractors.video.pose.triangulate import calibration_path
    from macaw.project.placing import check_rows, floor_of, place_cameras, remarks, summary
    from macaw.project.setup import write_setup as write_devices

    resolved = resolve_config(config)
    name = setup or resolved.setup.name
    path = Path(resolved.project.setups_dir) / f"{name}.yaml"
    if not path.is_file():
        raise MacawError(f"setup {name} is not saved yet ({path})")
    calibration = resolved.pose3d.calibration
    if not calibration:
        raise MacawError("no calibration is chosen: set pose3d.calibration, or press Use for this project on the Setup page")
    source = calibration_path(resolved, calibration)
    if not source.is_file():
        raise MacawError(f"the cameras of {calibration} are not found yet: run macaw calibrate-cameras {calibration} first")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    laid = MacawConfig.model_validate({**resolved.model_dump(), "setup": {**data, "name": name}}).setup
    cameras = read_setup(source, resolved.pose3d.units)
    floor, problem = floor_of(resolved, calibration, cameras)
    if problem:
        typer.echo(f"note: {problem}")
    placing = place_cameras(cameras, laid.room, resolved.project.units, floor)
    for camera, values in placing.values.items():
        placed = laid.room.cameras[camera]
        placed.lens = values["lens"]
        if placing.tied:
            placed.at = [values["x"], values["y"]]
            placed.height, placed.facing, placed.tilt, placed.calibration = values["height"], values["facing"], values["tilt"], calibration
    if placing.values:
        write_devices(path, laid)
    typer.echo(summary(placing, calibration))
    for camera, what, given, fitted, off, agrees in check_rows(placing, resolved.project.units):
        typer.echo(f"{camera} {what}: given {given}, placed at {fitted}, off by {off}" + ("" if agrees == "yes" else ", does not agree"))
    for remark in remarks(placing):
        typer.echo(f"note: {remark}")
    if placing.values:
        typer.echo(str(path))


@app.command("draw-room")
def draw_room_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    setup: str = typer.Option(None, "--setup", help="Which setup. Default: setup.name."),
    out: Path = typer.Option(None, "--out", help="Where to write the PNG. Default: setups/<setup>.png."),
) -> None:
    """Draw the room a setup lays out, front and side above and beside a plan, at 300 dpi for a paper."""
    import yaml

    from macaw.config.schema import MacawConfig
    from macaw.project.room_figure import draw_room
    from macaw.qc.figures import require_matplotlib

    require_matplotlib()
    resolved = resolve_config(config)
    name = setup or resolved.setup.name
    path = Path(resolved.project.setups_dir) / f"{name}.yaml"
    if not path.is_file():
        raise MacawError(f"setup {name} is not saved yet ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    laid = MacawConfig.model_validate({**resolved.model_dump(), "setup": {**data, "name": name}}).setup
    reads = {camera: device.people() for camera, device in laid.cameras.items()}
    try:
        target = draw_room(laid.room, list(resolved.project.roles), resolved.project.units, out or path.with_suffix(".png"), reads=reads)
    except ValueError as error:
        raise MacawError(f"setup {name}: {error}; place people, devices or things on the Setup page first") from None
    typer.echo(str(target))
