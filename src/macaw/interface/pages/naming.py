"""The Setup page's card where a person says who is who on a camera: a frame of the camera's own file, a click on
each person. Save runs macaw name-people."""

from __future__ import annotations

import base64
from pathlib import Path

from dash import Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.project.layout import layout_for
from macaw.project.names import read_names
from macaw.project.recording import build_recording

PATH = "/setup"
# the frame is shown at most this wide; clicks are kept as shares of it, so its size does not matter
SHOWN_SIDE = 1280
_MARK = "#e8590c"


def card(config):
    rids = resolve_recordings(config, None)
    return ui.card(
        "Who is who on each camera",
        html.Div(
            [
                ui.field("Recording", dcc.Dropdown(rids, rids[0] if rids else None, id="names-recording", clearable=False, className="picker")),
                ui.field("Camera", dcc.Dropdown([], None, id="names-camera", clearable=False, className="picker")),
                ui.field("At (s)", dcc.Input(id="names-at", type="number", min=0, step="any", value=10)),
                ui.button("Show frame", "names-show"),
            ],
            className="toolbar",
        ),
        html.Div(id="names-saved"),
        html.Div(ui.field("Click on", dcc.RadioItems([], None, id="names-person", inline=True)), className="toolbar"),
        html.Div(id="names-note"),
        dcc.Graph(id="names-graph", figure=marks_figure(None, {}), config={"displayModeBar": False}),
        dcc.Store(id="names-marks", data={}),
        dcc.Store(id="names-frame", data=None),
        html.Div(
            [
                # kept in this browser, as on the Review page
                ui.field("Your name", dcc.Input(id="names-by", value="", placeholder="needed to save", persistence=True, persistence_type="local")),
                ui.button("Save the names", "names-save", primary=True), ui.button("Clear this camera's names", "names-clear"),
            ],
            className="toolbar",
        ),
        html.Div(id="names-result"),
        desc="Click each person on a frame where all are in view.",
    )


def cameras_of(config_path: Path, rid: str | None) -> dict[str, list[str]]:
    """camera -> the people the setup reads it for, in a recording; empty when it cannot be read."""
    from macaw.extractors.video.pose.keypoints import camera_people

    if not rid:
        return {}
    config = resolve_config(config_path, recording_id=rid)
    try:
        recording = build_recording(rid, config)
    except MacawError:
        return {}
    roles = list(config.project.roles)
    return {item.name: sorted(camera_people(item), key=lambda person: (person not in roles, roles.index(person) if person in roles else 0, person)) for item in recording.cameras()}


def saved(config_path: Path, rid: str | None, camera: str | None):
    if not rid or not camera:
        return None
    config = resolve_config(config_path, recording_id=rid)
    clicks = read_names(layout_for(config, create=False).names_path(rid)).get(camera, [])
    if not clicks:
        return ui.note(f"Nobody is named on {camera} yet.")
    rows = [
        [click["person"], f"{float(click['at']):g}", f"{float(click['x']):.0%} across, {float(click['y']):.0%} down", click.get("by", "")]
        for click in sorted(clicks, key=lambda click: (float(click["at"]), click["person"]))
    ]
    return ui.table(["Person", "At (s)", "Where", "By"], rows, numeric={"At (s)"}, fit=True)


def frame_of(config_path: Path, rid: str, camera: str, at_s: float) -> tuple[dict | None, str | None]:
    """The frame of the camera's own file at_s from its first picture, shaped as the prepared copy is; or why not."""
    from macaw.media.ffmpeg import extract_frame

    config = resolve_config(config_path, recording_id=rid)
    try:
        item = build_recording(rid, config).media_named(camera)
    except (MacawError, ValueError) as error:
        return None, str(error)
    dest = layout_for(config, create=False).work_dir / "gui" / "names" / f"{rid}_{camera}.jpg"
    try:
        extract_frame(item.path, at_s, dest, square_pixels=True, long_side=SHOWN_SIDE)
    except Exception as error:
        return None, f"No frame at {at_s:g} s of {camera}: {error}"
    width, height = jpeg_size(dest.read_bytes())
    uri = "data:image/jpeg;base64," + base64.b64encode(dest.read_bytes()).decode()
    return {"uri": uri, "width": width, "height": height, "rid": rid, "camera": camera, "at": at_s}, None


def jpeg_size(data: bytes) -> tuple[int, int]:
    """(width, height) from the frame header of a JPEG file."""
    index = 2
    while index + 9 < len(data):
        marker = data[index + 1]
        if data[index] != 0xFF or marker == 0xFF:
            index += 1
            continue
        # start-of-frame markers; C4, C8 and CC are tables, not frames
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return int.from_bytes(data[index + 7:index + 9], "big"), int.from_bytes(data[index + 5:index + 7], "big")
        index += 2 + int.from_bytes(data[index + 2:index + 4], "big")
    raise MacawError("the frame written is not a JPEG picture")


def marks_figure(frame: dict | None, marks: dict[str, list[float]]):
    import plotly.graph_objects as go

    figure = go.Figure()
    if frame:
        # no label on hover, but the picture still reports clicks
        figure.add_trace(go.Image(source=frame["uri"], hoverinfo="none"))
        if marks:
            people = sorted(marks)
            figure.add_trace(go.Scatter(
                x=[marks[person][0] * frame["width"] for person in people], y=[marks[person][1] * frame["height"] for person in people],
                mode="markers+text", text=people, textposition="top center", textfont={"color": _MARK, "size": 15},
                marker={"size": 14, "color": _MARK, "line": {"color": "white", "width": 2}}, hoverinfo="skip",
            ))
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False, dragmode=False,
        xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=520 if frame else 40,
    )
    return figure


def marked(marks: dict, person: str, click: dict, frame: dict) -> dict:
    point = click["points"][0]
    x = min(max(float(point["x"]) / frame["width"], 0.0), 1.0)
    y = min(max(float(point["y"]) / frame["height"], 0.0), 1.0)
    return {**marks, person: [round(x, 4), round(y, 4)]}


def next_person(people: list[str], marks: dict, current: str) -> str:
    """The next person not clicked yet, else the one just clicked."""
    waiting = [person for person in people if person not in marks]
    return waiting[0] if waiting else current


def name_command(config_path: Path, frame: dict, marks: dict, by: str) -> list[str]:
    people = [flag for person in sorted(marks) for flag in ("--person", f"{person}={marks[person][0]:g},{marks[person][1]:g}")]
    return cli_bridge.build_command(
        ["name-people"], config_path=config_path, recordings=[frame["rid"]],
        extra_flags=["--camera", frame["camera"], "--at", f"{frame['at']:g}", *people, "--by", by],
    )


def clear_command(config_path: Path, rid: str, camera: str) -> list[str]:
    return cli_bridge.build_command(["name-people"], config_path=config_path, recordings=[rid], extra_flags=["--camera", camera, "--clear"])


def register(app, config_path: Path) -> None:
    @app.callback(Output("names-recording", "value"), Input("url", "pathname"), Input("url", "search"))
    def linked_recording(pathname, search):
        rid = ui.asked(search).get("recording")
        if (pathname or "/").rstrip("/") != PATH or rid not in resolve_recordings(resolve_config(config_path), None):
            return no_update
        return rid

    @app.callback(
        Output("names-camera", "options"), Output("names-camera", "value"),
        Input("names-recording", "value"), Input("url", "search"), State("names-camera", "value"),
    )
    def cameras(rid, search, chosen):
        found = list(cameras_of(config_path, rid))
        asked = ui.asked(search).get("camera")
        if asked in found and ui.asked(search).get("recording") == rid:
            return found, asked
        return found, chosen if chosen in found else (found[0] if found else None)

    @app.callback(
        Output("names-person", "options"), Output("names-person", "value"), Output("names-saved", "children"),
        Input("names-camera", "value"), Input("names-result", "children"), State("names-recording", "value"),
    )
    def people(camera, _result, rid):
        found = cameras_of(config_path, rid).get(camera, [])
        return found, found[0] if found else None, saved(config_path, rid, camera)

    @app.callback(
        Output("names-graph", "figure"), Output("names-marks", "data"), Output("names-frame", "data"),
        Output("names-note", "children"), Output("names-person", "value", allow_duplicate=True),
        Input("names-show", "n_clicks"), Input("names-graph", "clickData"),
        State("names-recording", "value"), State("names-camera", "value"), State("names-at", "value"),
        State("names-person", "value"), State("names-person", "options"), State("names-marks", "data"), State("names-frame", "data"),
        prevent_initial_call=True,
    )
    def frame(_show, click, rid, camera, at, person, people, marks, shown):
        if ctx.triggered_id == "names-show":
            if not rid or not camera:
                return no_update, no_update, no_update, ui.note("Choose a recording and a camera first.", "warn"), no_update
            found, problem = frame_of(config_path, rid, camera, float(at or 0.0))
            if problem:
                return marks_figure(None, {}), {}, None, ui.note(problem, "warn"), no_update
            note = f"Frame at {found['at']:g} s of {camera}. Click on each person: {', '.join(people)}."
            return marks_figure(found, {}), {}, found, ui.note(note), people[0] if people else None
        if not shown or not click or not person:
            return no_update, no_update, no_update, no_update, no_update
        marks = marked(marks or {}, person, click, shown)
        left = [name for name in people if name not in marks]
        note = f"Now click {left[0]}." if left else "Everyone is clicked: save the names, or click a person again to move them."
        return marks_figure(shown, marks), marks, no_update, ui.note(note), next_person(people, marks, person)

    @app.callback(
        Output("names-result", "children"),
        Input("names-save", "n_clicks"), Input("names-clear", "n_clicks"),
        State("names-frame", "data"), State("names-marks", "data"), State("names-recording", "value"), State("names-camera", "value"),
        State("names-by", "value"),
        prevent_initial_call=True,
    )
    def keep(_save, _clear, shown, marks, rid, camera, by):
        if ctx.triggered_id == "names-save":
            if not shown or not marks:
                return ui.note("Show a frame and click on the people first.", "warn")
            if not (by or "").strip():
                return ui.note("Type your name first: the names are kept with who clicked them.", "warn")
            argv = name_command(config_path, shown, marks, by.strip())
        else:
            if not rid or not camera:
                return ui.note("Choose a recording and a camera first.", "warn")
            argv = clear_command(config_path, rid, camera)
        code, text = cli_bridge.run_now(argv, cwd=config_path.parent)
        return [ui.command(argv), ui.output(code, text)]
