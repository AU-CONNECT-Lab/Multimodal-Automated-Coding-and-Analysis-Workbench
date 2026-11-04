"""The Setup page's section on cameras for 3D: print the board, find the cameras of each calibration from its board
clips, see how well they fit and where they stand, and choose the calibration the project uses. Every button runs the
macaw command shown with it."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from scipy.spatial.transform import Rotation

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.extractors.video.pose.triangulate import calibration_path
from macaw.interface import cli_bridge, project_file, ui
from macaw.interface.files import file_url
from macaw.interface.pages.results import _shown, axes_3d, in_first_camera
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording, camera_clips
from macaw.project.room import CM_PER_UNIT, DECIMALS
from macaw.project.scale import CM_PER_METRE

PATH = "/setup"
PAPERS = ("A4", "A3", "A2", "A1")
# one calibration at a time per project, like a run
_CURRENT: dict[str, cli_bridge.LaunchedCommand] = {}
# the command flags corners missing by more than this, and pairs that saw the board fewer times together
_GOOD_PX = 2.0
_FEW_TOGETHER = 15
# how far each camera's line of sight is drawn, in metres
_SIGHT_M = 0.3


def wanted(config) -> bool:
    """Whether a project can use it: a calibration named or present already, or a recording filmed by two cameras."""
    if config.pose3d.calibration or setups(config):
        return True
    for rid in resolve_recordings(config, None):
        try:
            if sum(item.kind == "video" for item in build_recording(rid, config).media) >= 2:
                return True
        except MacawError:
            continue
    return False


def setups(config) -> list[str]:
    """Folders of board clips and calibration files in project.calibration_dir."""
    folder = Path(config.project.calibration_dir)
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir() if (path.is_dir() and _clips(path)) or path.suffix.lower() == ".toml")


def _clips(folder: Path) -> dict[str, Path]:
    try:
        return camera_clips(folder)
    except MacawError:
        return {}


def section(config_path: Path):
    """Shown for projects that can use it: a calibration named or present already, or a recording filmed by two cameras."""
    config = resolve_config(config_path)
    if not wanted(config):
        return None
    names = setups(config)
    chosen = config.pose3d.calibration if config.pose3d.calibration in names else (names[0] if names else None)
    return html.Div(
        [
            html.H2("Cameras for three dimensions", className="section-title"),
            html.Div(id="cameras-message"),
            ui.card(
                "Board",
                html.Div(id="cameras-board"),
                html.Div(
                    [
                        ui.field("Paper", dcc.Dropdown(list(PAPERS), "A2", id="cameras-paper", clearable=False, className="picker")),
                        ui.field(f"Measured square ({config.project.units})", dcc.Input(id="cameras-square", type="number", value=measured_square(config), min=0, step="any")),
                    ],
                    className="toolbar",
                ),
                html.Div(id="cameras-board-result"),
                desc="Print at 100%, mount flat, measure one square.",
                actions=[ui.button("Save the square", "cameras-save-square"), ui.button("Draw the board", "cameras-draw", icon_name="download")],
            ),
            html.Div(ui.field("Calibration", dcc.Dropdown(names, chosen, id="cameras-setup", clearable=False, className="picker")), className="toolbar"),
            ui.card(
                "Calibration",
                dcc.Loading(html.Div(id="cameras-view")),
                desc="A folder per camera with its board clip, or a calibration file from another tool.",
                actions=[ui.button("Find the cameras", "cameras-find", icon_name="play"), ui.button("Use for this project", "cameras-use", primary=True)],
            ),
            ui.card(
                "Board lying flat",
                html.Div(
                    [
                        html.Div(dcc.Slider(0, 60, 0.1, value=0, id="cameras-floor-at", marks=None, updatemode="mouseup",
                                            tooltip={"placement": "bottom", "always_visible": True}), className="floor-slider"),
                        ui.field("On", dcc.Dropdown(["floor"], "floor", id="cameras-floor-on", clearable=False, className="picker")),
                    ],
                    className="toolbar",
                ),
                dcc.Loading(html.Div(id="cameras-floor-view")),
                html.Div(id="cameras-floor-result"),
                desc="Drag to when the board lay flat; the room takes the floor from it.",
                actions=[ui.button("Save the moment", "cameras-floor-save")],
            ),
            ui.card("Progress", html.Div(id="cameras-progress"), dcc.Interval(id="cameras-poll", interval=2000), dcc.Store(id="cameras-state")),
        ]
    )


def measured_square(config) -> float:
    """The board's square in project.units, as written or as drawn."""
    from macaw.project.room import CM_PER_UNIT

    per_mm = 10.0 * CM_PER_UNIT[config.project.units]
    return round(config.pose3d.board.in_mm(config.project.units).square_mm / per_mm, 3)


def board_view(config_path: Path):
    from macaw.extractors.video.pose.board import fits_on

    config = resolve_config(config_path)
    board = config.pose3d.board.in_mm(config.project.units)
    fits = [paper for paper in PAPERS if fits_on(board, paper)]
    return [
        html.P(
            f"{board.squares_x} x {board.squares_y} squares of {measured_square(config):g} {config.project.units}, markers "
            f"{round(measured_square(config) * board.marker_mm / board.square_mm, 3):g} {config.project.units}, "
            f"dictionary {board.dictionary}; " + (f"fits {', '.join(fits)}." if fits else "too big for A1: set smaller squares in pose3d.board."),
        ),
        ui.command(_board_command(config_path, "A2")),
    ]


def _board_command(config_path: Path, paper: str) -> list[str]:
    return cli_bridge.build_command(["calibration-board"], config_path=config_path, extra_flags=["--paper", paper])


def _find_command(config_path: Path, name: str) -> list[str]:
    return cli_bridge.build_command(["calibrate-cameras", name], config_path=config_path)


def setup_view(config_path: Path, name: str | None):
    from macaw.exceptions import ConfigError
    from macaw.extractors.video.pose.board import describe_board
    from macaw.extractors.video.pose.cameras import read_setup

    config = resolve_config(config_path)
    if name is None:
        return ui.note(f"No calibration yet: make a folder in {config.project.calibration_dir} holding a folder per camera with its board clip, then Refresh.")
    source = Path(config.project.calibration_dir) / name
    parts = [ui.pill("used by this project", "ok")] if config.pose3d.calibration == name else []
    clips = _clips(source) if source.is_dir() else {}
    if source.is_dir():
        parts += [
            ui.table(["Camera", "Clip"], [[camera, clip.name] for camera, clip in clips.items()]),
            html.Span("The same in a terminal", className="hint"),
            ui.command(_find_command(config_path, name)),
        ]
    path = calibration_path(config, name)
    if not path.is_file():
        return [*parts, ui.note("The cameras of this calibration are not found yet: press Find the cameras.")]
    try:
        setup = read_setup(path, config.pose3d.units)
    except ConfigError as error:
        return [*parts, ui.note(str(error), "bad")]
    meta = setup.metadata
    made_here = meta.get("made_by") == "macaw calibrate-cameras"
    status = f"Found on {meta.get('made_at', '?')} from the clips above." if made_here else f"Read from {path.name}, made with another tool."
    notes = [ui.note(status, "ok")]
    if made_here and clips and max(clip.stat().st_mtime for clip in clips.values()) > path.stat().st_mtime:
        notes.append(ui.note("A clip changed since the cameras were found: find them again.", "warn"))
    if made_here and meta.get("board") not in (None, describe_board(config.pose3d.board.in_mm(config.project.units))):
        notes.append(ui.note(f"The board was {meta['board']} then, and pose3d.board differs now: find the cameras again.", "warn"))
    notes += [ui.note(note, "warn") for note in meta.get("notes", [])]
    errors = meta.get("error_px", {})
    if errors and max(errors.values()) > _GOOD_PX:
        notes.append(ui.note(
            f"Corners miss by up to {max(errors.values()):.1f} px; under 1 or 2 px is good. Film the board again, slower and nearer.", "warn",
        ))
    shared = meta.get("together", {})
    tables = [ui.table(
        ["Camera", "Picture", "Board seen", "Own lens fit (px)", "All together (px)", f"Across, ahead, up of the first camera ({config.project.units})"],
        [_camera_row(camera, setup, meta, config.project.units) for camera in setup.cameras],
        numeric={"Board seen", "Own lens fit (px)", "All together (px)"},
    )]
    if shared:
        tables.append(ui.table(
            ["Cameras", "Saw the board together"],
            [[pair, f"{count}" + (" (few: film more where both see it)" if count < _FEW_TOGETHER else "")] for pair, count in shared.items()],
        ))
    return [*parts, *notes, *tables, dcc.Graph(figure=cameras_figure(setup, name, config.project.units), config={"displaylogo": False})]


def _camera_row(camera, setup, meta: dict, units: str) -> list[str]:
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES

    where = np.stack(_shown(in_first_camera(setup, camera.position() * UNITS_TO_METRES[setup.units]) * CM_PER_METRE / CM_PER_UNIT[units]))
    places = DECIMALS[units]
    # adding zero turns the -0.00 of a rounding residue into 0.00
    x, y, z = np.round(where, places) + 0.0
    number = lambda key, form: format(meta[key][camera.name], form) if camera.name in meta.get(key, {}) else ""
    return [
        camera.name, f"{camera.size[0]} x {camera.size[1]}", number("views", "d"), number("lens_error_px", ".2f"),
        number("error_px", ".2f"), f"{x:.{places}f}, {y:.{places}f}, {z:.{places}f}",
    ]


def cameras_figure(setup, name: str, units: str) -> go.Figure:
    """Where the cameras stand and which way each looks, turned like the Results page's 3D view."""
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES

    per = CM_PER_METRE / CM_PER_UNIT[units]
    where = np.vstack([camera.position() for camera in setup.cameras]) * UNITS_TO_METRES[setup.units]
    ahead = np.vstack([Rotation.from_rotvec(camera.rotation).inv().apply([0.0, 0.0, 1.0]) for camera in setup.cameras])
    figure = go.Figure()
    sights = np.vstack([np.vstack([start, start + _SIGHT_M * direction, [np.nan] * 3]) for start, direction in zip(where, ahead)])
    x, y, z = _shown(in_first_camera(setup, sights) * per)
    figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": "#8b8b90", "width": 4}, name="looking", hoverinfo="skip"))
    x, y, z = _shown(in_first_camera(setup, where) * per)
    figure.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="markers+text", text=[camera.name for camera in setup.cameras], textposition="top center",
        marker={"size": 5, "symbol": "diamond", "color": "#1d4e89"}, name="cameras",
    ))
    axis = {"gridcolor": "rgba(128,128,128,0.25)", "zerolinecolor": "rgba(128,128,128,0.4)", "backgroundcolor": "rgba(0,0,0,0)"}
    figure.update_layout(
        height=420, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90"},
        showlegend=False,
        scene={
            # a little further out than the default, so cameras at the edge of the room stay in view
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.1}},
            **axes_3d(units, axis),
        },
        uirevision=name,
    )
    return figure


def _offsets(config, name: str) -> dict[str, float]:
    """Seconds to add to each camera's clip to reach the first one's clock, as finding the cameras measured them."""
    from macaw.exceptions import ConfigError
    from macaw.extractors.video.pose.cameras import read_setup

    path = calibration_path(config, name)
    try:
        found = read_setup(path, config.pose3d.units).metadata.get("offsets_s") if path.is_file() else None
    except ConfigError:
        found = None
    return {camera: float(value) for camera, value in (found or {}).items()}


def floor_things(config, setup_name: str | None) -> list[str]:
    """The floor, and each thing of a setup's room with a height, which the board may have lain on."""
    import yaml

    path = Path(config.project.setups_dir) / f"{setup_name}.yaml" if setup_name else None
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path is not None and path.is_file() else {}
    things = (data.get("room") or {}).get("things") or {}
    return ["floor", *(name for name, thing in things.items() if (thing or {}).get("height") is not None)]


def floor_tools(config_path: Path, name: str | None, setup_name: str | None) -> tuple[float, float, list[str], str, list]:
    """The slider's length and place, what the board may lie on and what it lay on, and the moment saved so far."""
    from macaw.extractors.video.pose.board import clip_seconds
    from macaw.project.placing import floor_said

    config = resolve_config(config_path)
    things = floor_things(config, setup_name)
    folder = Path(config.project.calibration_dir) / name if name else None
    if folder is None or not folder.is_dir():
        return 60.0, 0.0, things, "floor", []
    offsets = _offsets(config, name)
    length = max([clip_seconds(clip) + offsets.get(camera, 0.0) for camera, clip in _clips(folder).items()] or [60.0])
    said = floor_said(folder)
    if said is None:
        return round(length, 1), 0.0, things, "floor", []
    on = said["on"] if said["on"] in things else "floor"
    return round(length, 1), said["at_s"], things, on, [ui.pill(f"saved: {said['at_s']:g} s, on {said['on']}", "ok")]


def floor_view(config_path: Path, name: str | None, at) -> list:
    """Each camera's frame at that moment of the board clips, with the board corners it finds marked."""
    from macaw.extractors.video.pose.board import corners_on, frame_at, make_board, marked_jpeg, width_factor

    config = resolve_config(config_path)
    if name is None:
        return []
    folder = Path(config.project.calibration_dir) / name
    if not folder.is_dir():
        return [ui.note(f"{name} is a calibration file, with no board clips to show the board lying flat.")]
    offsets = _offsets(config, name)
    board = make_board(config.pose3d.board.in_mm(config.project.units))
    figures = []
    for camera, clip in _clips(folder).items():
        frame = frame_at(clip, float(at or 0.0) - offsets.get(camera, 0.0), width_factor(clip, config.video.normalise))
        if frame is None:
            figures.append(html.Figure(html.Figcaption(f"{camera}: no frame at this moment")))
            continue
        sighting = corners_on(frame, board)
        uri = "data:image/jpeg;base64," + base64.b64encode(marked_jpeg(frame, sighting)).decode()
        caption = f"{camera}: board found, {len(sighting.ids)} corners" if sighting is not None else f"{camera}: board not found"
        figures.append(html.Figure([html.Img(src=uri), html.Figcaption(caption)]))
    return [html.Div(figures, className="thumbs")]


def save_floor(config_path: Path, name: str | None, at, on: str):
    from macaw.extractors.video.pose.board import make_board
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.project.placing import find_floor, write_floor

    config = resolve_config(config_path)
    if name is None:
        return ui.note("Choose a calibration first.", "warn")
    folder = Path(config.project.calibration_dir) / name
    path = calibration_path(config, name)
    if not folder.is_dir() or not path.is_file():
        return ui.note(f"Find the cameras of {name} first.", "warn")
    at = float(at or 0.0)
    setup = read_setup(path, config.pose3d.units)
    board = make_board(config.pose3d.board.in_mm(config.project.units))
    if find_floor(setup, _clips(folder), board, at, on, normalise=config.video.normalise) is None:
        return ui.note("No camera finds the board at this moment; choose one where it lies flat and still in view.", "warn")
    write_floor(folder, at, on)
    argv = cli_bridge.build_command(["calibration-floor", name, "--at", f"{at:g}", "--on", on], config_path=config_path)
    surface = "the floor" if on == "floor" else on
    return [
        ui.note(f"Saved: the board lay on {surface} at {at:g} s. From the calibration on the setup now uses it.", "ok"),
        html.Span("The same in a terminal", className="hint"), ui.command(argv),
    ]


def progress(config_path: Path):
    current = _CURRENT.get(str(config_path))
    if current is None:
        return html.P("Nothing started from this window yet.", className="none")
    code = cli_bridge.exit_code(current)
    state = ui.pill("running", "warn") if code is None else ui.pill("finished" if code == 0 else f"stopped, exit {code}", "ok" if code == 0 else "bad")
    return [
        html.P([state, f" started {current.started_at}"]), ui.command(current.argv),
        html.Pre("\n".join(cli_bridge.tail_log(current.log_path, n_lines=60)) or "(no output yet)", className="log"),
    ]


def use_setup(config_path: Path, name: str) -> str:
    """Writes pose3d.calibration into project.yaml once the text still reads as a valid project."""
    text = project_file.with_values(config_path.read_text(encoding="utf-8"), {"pose3d.calibration": name})
    project_file.check(config_path, text)
    project_file.save(config_path, text)
    return text


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("cameras-setup", "options"), Output("cameras-setup", "value"), Output("cameras-board", "children"),
        Input("url", "pathname"), Input("app-refresh", "n_clicks"), State("cameras-setup", "value"),
    )
    def fill(pathname, _refresh, chosen):
        if (pathname or "/").rstrip("/") != PATH:
            return no_update, no_update, no_update
        config = resolve_config(config_path)
        names = setups(config)
        if chosen not in names:
            chosen = config.pose3d.calibration if config.pose3d.calibration in names else (names[0] if names else None)
        return names, chosen, board_view(config_path)

    @app.callback(Output("cameras-view", "children"), Input("cameras-setup", "value"), Input("cameras-state", "data"), Input("cameras-message", "children"))
    def show(name, _state, _message):
        return setup_view(config_path, name)

    @app.callback(
        Output("cameras-message", "children"), Output("cameras-board-result", "children"),
        Input("cameras-draw", "n_clicks"), Input("cameras-save-square", "n_clicks"), Input("cameras-find", "n_clicks"), Input("cameras-use", "n_clicks"),
        State("cameras-paper", "value"), State("cameras-square", "value"), State("cameras-setup", "value"),
        prevent_initial_call=True,
    )
    def act(_draw, _save, _find, _use, paper, square, name):
        config = resolve_config(config_path)
        if ctx.triggered_id == "cameras-draw":
            argv = _board_command(config_path, paper)
            code, text = cli_bridge.run_now(argv, cwd=config_path.parent)
            drawn = Path(config.project.calibration_dir) / f"board_{paper}.png"
            link = [html.A(f"Open {drawn.name}", href=file_url(drawn), target="_blank")] if code == 0 and drawn.is_file() else []
            return no_update, [ui.command(argv), ui.output(code, text), *link]
        if ctx.triggered_id == "cameras-save-square":
            if not square or float(square) <= 0:
                return ui.note(f"Type the length of one printed square in {config.project.units}.", "warn"), no_update
            try:
                text = project_file.with_values(config_path.read_text(encoding="utf-8"), {"pose3d.board.square": float(square)})
                project_file.check(config_path, text)
            except (MacawError, ValueError) as error:
                return ui.note(str(error), "bad"), no_update
            project_file.save(config_path, text)
            return ui.note(f"Saved pose3d.board.square: {float(square):g} in project.yaml. Calibrations found before need finding again.", "ok"), board_view(config_path)
        if name is None:
            return ui.note("Choose a calibration first.", "warn"), no_update
        if ctx.triggered_id == "cameras-use":
            if not calibration_path(config, name).is_file():
                return ui.note(f"Find the cameras of {name} first.", "warn"), no_update
            try:
                use_setup(config_path, name)
            except (MacawError, ValueError) as error:
                return ui.note(str(error), "bad"), no_update
            return ui.note(f"Saved pose3d.calibration: {name} in project.yaml; the next run places people in 3D with it.", "ok"), no_update
        source = Path(config.project.calibration_dir) / name
        if not source.is_dir():
            return ui.note(f"{name} is a calibration file; there is nothing to find.", "warn"), no_update
        if len(_clips(source)) < 2:
            return ui.note(f"{source} holds {len(_clips(source))} camera folder(s) with a clip; put each camera's board clip in a folder named after it, two or more.", "warn"), no_update
        current = _CURRENT.get(str(config_path))
        if current is not None and cli_bridge.is_running(current):
            return ui.note("The cameras of a calibration are still being found; wait for it to finish.", "warn"), no_update
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log = layout_for(config, create=False).work_dir / "gui" / f"cameras_{name}_{stamp}.log"
        _CURRENT[str(config_path)] = cli_bridge.launch(_find_command(config_path, name), log, cwd=config_path.parent)
        return ui.note(f"Finding the cameras of {name}; this reads every clip and takes a few minutes. The output goes to {log}.", "ok"), no_update

    @app.callback(
        Output("cameras-floor-at", "max"), Output("cameras-floor-at", "value"), Output("cameras-floor-on", "options"),
        Output("cameras-floor-on", "value"), Output("cameras-floor-result", "children"),
        Input("cameras-setup", "value"), Input("setup-pick", "value"),
    )
    def floor_choices(name, setup_name):
        try:
            return floor_tools(config_path, name, setup_name)
        except MacawError as error:
            return no_update, no_update, no_update, no_update, ui.note(str(error), "bad")

    @app.callback(Output("cameras-floor-view", "children"), Input("cameras-floor-at", "value"), State("cameras-setup", "value"))
    def floor_frames(at, name):
        try:
            return floor_view(config_path, name, at)
        except MacawError as error:
            return ui.note(str(error), "bad")

    @app.callback(
        Output("cameras-floor-result", "children", allow_duplicate=True),
        Input("cameras-floor-save", "n_clicks"), State("cameras-setup", "value"), State("cameras-floor-at", "value"), State("cameras-floor-on", "value"),
        prevent_initial_call=True,
    )
    def floor_save(clicks, name, at, on):
        if not clicks:
            return no_update
        try:
            return save_floor(config_path, name, at, on)
        except MacawError as error:
            return ui.note(str(error), "bad")

    @app.callback(Output("cameras-progress", "children"), Output("cameras-state", "data"), Input("cameras-poll", "n_intervals"), State("cameras-state", "data"))
    def poll(_, known):
        current = _CURRENT.get(str(config_path))
        state = None if current is None else f"{current.started_at}:{cli_bridge.exit_code(current)}"
        return progress(config_path), state if state != known else no_update
