"""The Setup page's card that lays what a camera should see, by the room as laid out, half transparent over a frame of
its own file, so its height, facing, tilt and lens angle can be set by eye."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from pydantic import ValidationError

from macaw.config.loader import resolve_config
from macaw.config.schema import PlacedCamera, RoomSection
from macaw.execution.runner import resolve_recordings
from macaw.interface import ui
from macaw.interface.pages import naming
from macaw.interface.pages.room import NUM, _COLOURS, _HEAD_CM, _THING_CM, _colour, _extent, room_of, rows_from
from macaw.project import room as geometry

SLIDERS = ("height", "facing", "tilt", "lens")
# floor lines every this many centimetres
_GRID_CM = 50.0
_RING = np.linspace(0, 2 * np.pi, 25)


def card(config):
    rids = resolve_recordings(config, None)
    units = config.project.units
    # up to three metres, a centimetre at a time, in the project's unit
    top, step = geometry.to_unit(300.0, units), geometry.to_unit(1.0, units)
    sliders = {
        "height": (f"Height ({units})", 0, top, step),
        "facing": ("Facing (°)", 0, 360, 1),
        "tilt": ("Tilt down (°)", -30, 90, 1),
        "lens": ("Lens angle (°)", 20, 150, 1),
    }
    return ui.card(
        "What each camera should see",
        html.Div(
            [
                ui.field("Recording", dcc.Dropdown(rids, rids[0] if rids else None, id="sketch-recording", clearable=False, className="picker")),
                ui.field("Camera", dcc.Dropdown([], None, id="sketch-camera", clearable=False, className="picker")),
                ui.field("At (s)", dcc.Input(id="sketch-at", type="number", min=0, step="any", value=0)),
                ui.button("Show frame", "sketch-show"),
            ],
            className="toolbar",
        ),
        html.Div(
            [
                ui.field(label, dcc.Slider(low, high, step, value=None, id=f"sketch-{name}", marks=None, tooltip={"placement": "bottom", "always_visible": True}))
                for name, (label, low, high, step) in sliders.items()
            ],
            className="sketch-sliders",
        ),
        html.Div(id="sketch-note"),
        dcc.Graph(id="sketch-graph", figure=sketch_figure(None, None, None, [], units), config={"displayModeBar": False}),
        dcc.Store(id="sketch-frame", data=None),
        html.Div(ui.button("Use these for the camera", "sketch-use"), className="buttons"),
        desc=(
            "The room as laid out above, drawn from where the camera stands, over a frame of its own file: the floor in "
            "lines, each person up to their eyes with a ring for the head, and the things. Move the sliders until the "
            "drawing lies on the picture, then use the values for the camera and save the room. The frame is the start of "
            "the file unless another moment is given."
        ),
    )


def placed_with(room: RoomSection, camera: str, values: dict) -> PlacedCamera | None:
    """The camera as the room places it, with the sliders' values in place of its own; None until it has a place."""
    place = room.cameras.get(camera)
    if place is None or place.at is None:
        return None
    given = {name: value for name, value in values.items() if value is not None}
    return place.model_copy(update=given)


def outline(room: RoomSection, roles: list[str], units: str) -> list[tuple[np.ndarray, str, str | None]]:
    """(points in the room, colour, label) of each line to draw: the floor grid, each person and each thing."""
    lines = []
    across, along, _ = _extent(room, units)
    step = geometry.to_unit(_GRID_CM, units)
    for x in np.arange(np.ceil(across[0] / step) * step, across[1], step):
        lines.append((np.array([[x, along[0], 0], [x, along[1], 0]]), "#adb5bd", None))
    for y in np.arange(np.ceil(along[0] / step) * step, along[1], step):
        lines.append((np.array([[across[0], y, 0], [across[1], y, 0]]), "#adb5bd", None))
    head = geometry.to_unit(_HEAD_CM, units)
    for name, place in room.people.items():
        if place.at is None or place.eyes is None:
            continue
        colour = _colour("people", name, roles)
        x, y = place.at
        lines.append((np.array([[x, y, 0], [x, y, place.eyes - head]]), colour, None))
        # a ring standing upright, round the eyes, faced to no one in particular
        ring = np.stack([x + head * np.cos(_RING), np.full(_RING.size, y), place.eyes + head * np.sin(_RING)], axis=1)
        lines.append((ring, colour, name))
        ring = np.stack([np.full(_RING.size, x), y + head * np.cos(_RING), place.eyes + head * np.sin(_RING)], axis=1)
        lines.append((ring, colour, None))
    thing = geometry.to_unit(_THING_CM, units)
    for name, place in room.things.items():
        if place.at is None or place.height is None:
            continue
        width, depth = place.size or (thing, thing)
        x, y = place.at
        corners = np.array([[x + sx * width / 2, y + sy * depth / 2] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        bottom = np.column_stack([corners, np.zeros(4)])
        top = np.column_stack([corners, np.full(4, place.height)])
        lines.append((np.vstack([bottom, bottom[:1]]), _COLOURS["things"], None))
        lines.append((np.vstack([top, top[:1]]), _COLOURS["things"], name))
        for low, high in zip(bottom, top):
            lines.append((np.vstack([low, high]), _COLOURS["things"], None))
    return lines


def sketch_figure(frame: dict | None, room: RoomSection | None, camera: PlacedCamera | None, roles: list[str], units: str) -> go.Figure:
    figure = go.Figure()
    if frame:
        figure.add_trace(go.Image(source=frame["uri"], hoverinfo="none"))
        view = geometry.view_of(camera) if camera is not None and camera.lens else None
        if view is not None and room is not None:
            size = (frame["width"], frame["height"])
            for points, colour, label in outline(room, roles, units):
                # a line is drawn only where both its ends are in front of the camera
                ahead = view.local(points)[:, 1]
                pixels = view.pixels(points, size)
                pixels[ahead <= 0] = np.nan
                figure.add_trace(go.Scatter(
                    x=pixels[:, 0], y=pixels[:, 1], mode="lines", line={"color": colour, "width": 2.5}, opacity=0.6, hoverinfo="skip",
                    connectgaps=False,
                ))
                if label and np.isfinite(pixels).all():
                    top = pixels[np.argmin(pixels[:, 1])]
                    figure.add_trace(go.Scatter(x=[top[0]], y=[top[1]], mode="text", text=[label], textposition="top center",
                                                textfont={"color": colour, "size": 14}, hoverinfo="skip"))
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False, dragmode=False,
        xaxis={"visible": False, "range": [0, frame["width"]] if frame else None},
        yaxis={"visible": False, "scaleanchor": "x", "range": [frame["height"], 0] if frame else None},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=560 if frame else 40,
    )
    return figure


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("sketch-camera", "options"), Output("sketch-camera", "value"),
        Input("sketch-recording", "value"), State("sketch-camera", "value"),
    )
    def cameras(rid, chosen):
        found = list(naming.cameras_of(config_path, rid))
        return found, chosen if chosen in found else (found[0] if found else None)

    @app.callback(
        *[Output(f"sketch-{name}", "value") for name in SLIDERS],
        Input("sketch-camera", "value"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"), State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
    )
    def start_from_the_room(camera, values, ids):
        row = rows_from(values, ids)["cameras"].get(camera or "", {})
        return tuple(row.get(name) for name in SLIDERS)

    @app.callback(
        Output("sketch-graph", "figure"), Output("sketch-frame", "data"), Output("sketch-note", "children"),
        Input("sketch-show", "n_clicks"), *[Input(f"sketch-{name}", "value") for name in SLIDERS],
        Input({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
        State("sketch-recording", "value"), State("sketch-camera", "value"), State("sketch-at", "value"), State("sketch-frame", "data"),
        prevent_initial_call=True,
    )
    def show(_show, height, facing, tilt, lens, values, ids, rid, camera, at, shown):
        config = resolve_config(config_path)
        note = no_update
        if ctx.triggered_id == "sketch-show":
            if not rid or not camera:
                return no_update, no_update, ui.note("Choose a recording and a camera first.", "warn")
            shown, problem = naming.frame_of(config_path, rid, camera, float(at or 0.0))
            if problem:
                return sketch_figure(None, None, None, [], config.project.units), None, ui.note(problem, "warn")
        if not shown:
            return no_update, no_update, no_update
        try:
            room = RoomSection.model_validate(room_of(rows_from(values, ids)))
        except ValidationError:
            room = None
        placed = placed_with(room, shown["camera"], {"height": height, "facing": facing, "tilt": tilt, "lens": lens}) if room else None
        if placed is None or geometry.view_of(placed) is None or not placed.lens:
            note = ui.note(f"Place {shown['camera']} in the room and give its height, facing and lens angle to see the drawing.", "warn")
        elif ctx.triggered_id == "sketch-show":
            note = None
        return sketch_figure(shown, room, placed, list(config.project.roles), config.project.units), shown, note

    @app.callback(
        Output({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value", allow_duplicate=True),
        Input("sketch-use", "n_clicks"),
        *[State(f"sketch-{name}", "value") for name in SLIDERS],
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"), State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
        State("sketch-camera", "value"),
        prevent_initial_call=True,
    )
    def use(clicks, height, facing, tilt, lens, values, ids, camera):
        if not clicks or not camera:
            return [no_update] * len(ids)
        chosen = {"height": height, "facing": facing, "tilt": tilt, "lens": lens}
        return [
            chosen[identity["field"]] if identity["kind"] == "cameras" and identity["name"] == camera and identity["field"] in chosen and chosen[identity["field"]] is not None
            else value
            for value, identity in zip(values, ids)
        ]
