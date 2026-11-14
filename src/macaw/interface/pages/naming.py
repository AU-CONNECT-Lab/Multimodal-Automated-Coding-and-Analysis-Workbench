"""The Review page's editor where a person says who is who on a camera: a frame of the camera's own file, a click on
each person. Where pose.detect has run, the people it found are drawn, and a click shows whom it lands on."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.interface import cli_bridge, ui
from macaw.project.layout import layout_for
from macaw.project.names import read_names
from macaw.project.recording import build_recording

# the frame is shown at most this wide; clicks are kept as shares of it, so its size does not matter
SHOWN_SIDE = 1280
# one colour per person, in the order of project.roles, on the frame and beside the names
COLOURS = ("#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f59f00", "#0c8599", "#c2255c", "#5c940d")
_UNNAMED = "#adb5bd"
AT, TIMELINE, FROM_PLAYER, PERSON, GRAPH, NOTE = "names-at", "names-timeline", "names-from-player", "names-person", "names-graph", "names-note"
WINDOW = "names-window"
# the moment shown first before the analysed part is known, or the middle of a shorter file
FIRST_AT_S = 10.0
# otherwise this share of the way into the analysed part
FIRST_SHARE = 0.2
# the Review page's save reads these by pattern, as a recording without cameras has none
MARKS, FRAME = {"type": "names-marks", "key": "one"}, {"type": "names-frame", "key": "one"}


def colour_of(person: str, people: list[str]) -> str:
    return COLOURS[people.index(person) % len(COLOURS)] if person in people else _UNNAMED


def swatch(person: str, people: list[str]):
    return html.Span([html.Span(className="swatch", style={"background": colour_of(person, people)}), person], className="person")


def cameras_of(config, rid: str | None) -> dict[str, list[str]]:
    """camera -> the people the setup reads it for, in a recording; empty when it cannot be read."""
    from macaw.extractors.video.pose.keypoints import camera_people

    if not rid:
        return {}
    try:
        recording = build_recording(rid, config)
    except MacawError:
        return {}
    roles = list(config.project.roles)
    return {item.name: sorted(camera_people(item), key=lambda person: (person not in roles, roles.index(person) if person in roles else 0, person)) for item in recording.cameras()}


def saved(config, rid: str, camera: str, people: list[str], part: tuple[float, float] | None = None, take_back=None):
    """Every click kept for the camera, those outside the analysed part marked; take_back(target) draws a row's button."""
    clicks = read_names(layout_for(config, create=False).names_path(rid)).get(camera, [])
    if not clicks:
        return ui.note(f"Nobody is named on {camera} yet.")
    rows = []
    for click in sorted(clicks, key=lambda click: (float(click["at"]), click["person"])):
        at = float(click["at"])
        used = "yes" if outside(at, part) is None else "no, outside the analysed part"
        row = [swatch(click["person"], people), f"{at:g}", f"{float(click['x']):.0%} across, {float(click['y']):.0%} down", click.get("by", "").split(" ")[0]]
        rows.append(row + ([used] if part else []) + ([take_back(f"{at:g}|{click['person']}")] if take_back else []))
    columns = ["Person", "At (s)", "Where", "By"] + (["Used"] if part else []) + ([""] if take_back else [])
    return html.Div([html.P(f"Clicks kept on {camera}; each one counts until taken back."), ui.table(columns, rows, numeric={"At (s)"}, fit=True)])


def analysed(config, rid: str, camera: str) -> tuple[float, float] | None:
    from macaw.extractors.video.pose.keypoints import analysed_part

    try:
        return analysed_part(config, rid, camera)
    except (MacawError, ValueError):
        return None


def outside(at_s: float, part) -> str | None:
    """Why a moment names nobody in a run, or None when it is inside the analysed part."""
    if not part or part[0] <= at_s <= part[1]:
        return None
    return f"{at_s:g} s is outside the analysed part of this file ({part[0]:.1f} to {part[1]:.1f} s): a run leaves clicks here out."


def file_length(config, rid: str, camera: str) -> float | None:
    from macaw.media.probe import probe

    try:
        return probe(build_recording(rid, config).media_named(camera).path).duration_s
    except Exception:   # a file that cannot be read gets a typed moment only
        return None


def first_moment(part: tuple[float, float] | None, length: float | None) -> float:
    if part is not None and part[1] > part[0]:
        return round(part[0] + FIRST_SHARE * (part[1] - part[0]), 1)
    return FIRST_AT_S if not length or length > FIRST_AT_S else round(length / 2, 1)


def editor(config, rid: str, camera: str, people: list[str], take_back=None):
    """A frame of the camera at the moment on the timeline, typed, or taken from the player; Save and Clear are the
    Review page's buttons."""
    options = [{"label": swatch(person, people), "value": person} for person in people]
    length = file_length(config, rid, camera)
    part = analysed(config, rid, camera)
    first = first_moment(part, length)
    edges = {edge: f"{edge:g} s" for edge in part if 0 <= edge <= length} if part and length else None
    timeline = dcc.Slider(
        0, round(length, 1), value=first, id={"type": TIMELINE, "key": "one"}, updatemode="mouseup", marks=edges or None,
        tooltip={"placement": "bottom", "always_visible": False},
    ) if length else None
    return html.Div(
        [
            saved(config, rid, camera, people, part, take_back),
            ui.note(f"Analysed: {part[0]:.1f} to {part[1]:.1f} s of this file; click at a moment inside it.") if part else None,
            html.Div(
                [
                    ui.field("At (s) of its file", dcc.Input(id=AT, type="number", min=0, step="any", value=first, debounce=True)),
                    ui.button("The player's time", FROM_PLAYER),
                ],
                className="toolbar",
            ),
            html.Div(timeline, className="timeline"),
            html.Div(ui.field("Click on", dcc.RadioItems(options, people[0] if people else None, id=PERSON, inline=True)), className="toolbar"),
            html.Div(id=NOTE),
            dcc.Graph(id=GRAPH, figure=marks_figure(None, {}, people), config={"displayModeBar": False}),
            dcc.Store(id=MARKS, data={}),
            dcc.Store(id=FRAME, data=None),
            dcc.Store(id=WINDOW, data=list(part) if part else None),
        ],
        className="naming",
    )


def frame_of(config_path: Path, rid: str, camera: str, at_s: float) -> tuple[dict | None, str | None]:
    """The frame of the camera's own file at_s from its first picture, shaped as the prepared copy is, with whoever
    pose.detect found in it; or why not."""
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
    try:
        found = detected_at(config, rid, camera, at_s, (width, height))
    except Exception:   # the people found are a help; the frame can be named without them
        found = None
    return {"uri": uri, "width": width, "height": height, "rid": rid, "camera": camera, "at": at_s,
            "detected": None if found is None else found.to_dict("records")}, None


def detected_at(config, rid: str, camera: str, at_s: float, shown: tuple[int, int]) -> pd.DataFrame | None:
    """Everyone pose.detect found in the frame nearest a moment of the camera's file, in the pixels of the frame shown;
    None before it has run."""
    from macaw.extractors.prep import apply_prep_timing, load_video_prep
    from macaw.extractors.video.pose.keypoints import PoseExtractor, load_detections

    try:
        detections, _ = load_detections(config, rid)
    except MacawError:
        return None
    recording = build_recording(rid, config)
    apply_prep_timing(config, recording)
    item = recording.media_named(camera)
    video = load_video_prep(config, rid).get(camera)
    t = PoseExtractor._click(item, video, {"person": "", "at": at_s, "x": 0.0, "y": 0.0}).t
    piece = detections[(detections["video"] == camera) & detections["track_id"].notna() & detections["x"].notna()]
    if piece.empty:
        return piece[["track_id", "frame", "t", "joint", "x", "y"]]
    times = piece.groupby("frame")["t"].first()
    frame = (times - t).abs().idxmin()
    if abs(times[frame] - t) > (1.0 / item.fps if item.fps else 0.1):
        return piece.iloc[:0][["track_id", "frame", "t", "joint", "x", "y"]]
    here = piece[piece["frame"] == frame][["track_id", "frame", "t", "joint", "x", "y"]].copy()
    width, height = (video.get("prepared_size") or video["size"]) if video else shown
    here["x"] = here["x"] * shown[0] / width
    here["y"] = here["y"] * shown[1] / height
    here["track_id"] = here["track_id"].astype(int)
    return here.reset_index(drop=True)


def landed(detected: list[dict] | None, point: list[float], frame: dict) -> tuple[int | None, str]:
    """The person found under a click, as pose.keypoints will read it, or why none."""
    from macaw.extractors.video.pose.track_repair import CLICK_REACH, Click, _clicked_track, _track_boxes

    if not detected:
        return None, ""
    here = pd.DataFrame(detected)
    x, y = point[0] * frame["width"], point[1] * frame["height"]
    boxes = _track_boxes(here)
    pad = CLICK_REACH * boxes["size"]
    inside = boxes[(boxes["x0"] - pad <= x) & (x <= boxes["x1"] + pad) & (boxes["y0"] - pad <= y) & (y <= boxes["y1"] + pad)]
    if inside.empty:
        return None, "nobody was found there"
    track = _clicked_track(here, here["frame"].iloc[0], inside["track_id"].astype(int).tolist(), Click("", 0.0, 0.0, x, y))
    return (track, "") if track is not None else (None, "this falls on two people; click where they do not overlap")


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


def marks_figure(frame: dict | None, marks: dict[str, list[float]], people: list[str]):
    import plotly.graph_objects as go

    figure = go.Figure()
    if frame:
        # no label on hover, but the picture still reports clicks
        figure.add_trace(go.Image(source=frame["uri"], hoverinfo="none"))
        owners = {}
        for person, point in marks.items():
            track, _ = landed(frame.get("detected"), point, frame)
            if track is not None:
                owners[track] = person
        detected = pd.DataFrame(frame.get("detected") or [], columns=["track_id", "x", "y"])
        for track, joints in detected.groupby("track_id"):
            colour = colour_of(owners[track], people) if track in owners else _UNNAMED
            figure.add_trace(go.Scatter(
                x=joints["x"], y=joints["y"], mode="markers", hoverinfo="skip",
                marker={"size": 7, "color": colour, "line": {"color": "white", "width": 1}},
            ))
        for person, (x, y) in marks.items():
            figure.add_trace(go.Scatter(
                x=[x * frame["width"]], y=[y * frame["height"]], mode="markers+text", text=[person], textposition="top center",
                textfont={"color": colour_of(person, people), "size": 15}, hoverinfo="skip",
                marker={"size": 16, "color": colour_of(person, people), "line": {"color": "white", "width": 2}},
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


def where_they_landed(frame: dict, marks: dict) -> str:
    if not frame.get("detected"):
        return "" if frame.get("detected") is None else " Nobody was found in this frame; try another moment."
    said = [f"{person}: {why}" for person, point in marks.items() if (why := landed(frame["detected"], point, frame)[1])]
    return (" " + "; ".join(said) + ".") if said else ""


def name_command(config_path: Path, frame: dict, marks: dict, by: str) -> list[str]:
    people = [flag for person in sorted(marks) for flag in ("--person", f"{person}={marks[person][0]:g},{marks[person][1]:g}")]
    return cli_bridge.build_command(
        ["name-people"], config_path=config_path, recordings=[frame["rid"]],
        extra_flags=["--camera", frame["camera"], "--at", f"{frame['at']:g}", *people, "--by", by],
    )


def take_back_command(config_path: Path, rid: str, camera: str, target: str) -> list[str]:
    at, _, person = target.partition("|")
    return cli_bridge.build_command(
        ["name-people"], config_path=config_path, recordings=[rid], extra_flags=["--camera", camera, "--remove", "--at", at, "--person", person],
    )


def clear_command(config_path: Path, rid: str, camera: str) -> list[str]:
    return cli_bridge.build_command(["name-people"], config_path=config_path, recordings=[rid], extra_flags=["--camera", camera, "--clear"])


def register(app, config_path: Path, player: str, recording: str, camera: dict) -> None:
    app.clientside_callback(
        "function (clicks) { const video = document.getElementById('" + player + "');"
        " if (!clicks || !video) { return window.dash_clientside.no_update; }"
        " return Math.round(video.currentTime * 100) / 100; }",
        Output(AT, "value"), Input(FROM_PLAYER, "n_clicks"), prevent_initial_call=True,
    )

    @app.callback(
        Output(GRAPH, "figure"), Output(MARKS, "data"), Output(FRAME, "data"), Output(NOTE, "children"),
        Output(AT, "value"), Output({"type": TIMELINE, "key": ALL}, "value"),
        Input(AT, "value"), Input({"type": TIMELINE, "key": ALL}, "value"), Input(GRAPH, "clickData"),
        State(recording, "value"), State(camera, "value"),
        State(PERSON, "value"), State(PERSON, "options"), State(MARKS, "data"), State(FRAME, "data"), State(WINDOW, "data"),
    )
    def frame(at, moved, click, rid, chosen, person, options, marks, shown, part):
        people = [option["value"] for option in options or []]
        if ctx.triggered_id != GRAPH:
            # the timeline and the typed moment follow each other; either shows its frame
            at = moved[0] if isinstance(ctx.triggered_id, dict) and moved else at
            if at is None or not rid or not chosen:
                return no_update, no_update, no_update, no_update, no_update, [no_update] * len(moved)
            found, problem = frame_of(config_path, rid, chosen, float(at))
            if problem:
                return marks_figure(None, {}, people), {}, None, ui.note(problem, "warn"), at, [at] * len(moved)
            hint = " The people found are drawn grey until a click names them." if found["detected"] else ""
            away = outside(found["at"], part)
            note = ui.note(away, "warn") if away else ui.note(f"Frame at {found['at']:g} s of {chosen}. Choose a person, then click on them.{hint}")
            return marks_figure(found, {}, people), {}, found, note, at, [at] * len(moved)
        if not shown or not click or not person:
            return no_update, no_update, no_update, no_update, no_update, [no_update] * len(moved)
        marks = marked(marks or {}, person, click, shown)
        left = [name for name in people if name not in marks]
        note = (f"Still to click: {', '.join(left)}." if left else "Everyone is clicked: save the names, or click again to move one.")
        away = outside(shown["at"], part)
        said = ui.note(f"{away} {note}{where_they_landed(shown, marks)}", "warn") if away else ui.note(note + where_they_landed(shown, marks))
        return marks_figure(shown, marks, people), marks, no_update, said, no_update, [no_update] * len(moved)
