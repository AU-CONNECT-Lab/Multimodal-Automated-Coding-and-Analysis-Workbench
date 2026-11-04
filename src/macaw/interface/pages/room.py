"""The Setup page's room: where the people, cameras, microphones and things are, typed as numbers or dragged in three
views drawn as in a technical drawing (front above top, side beside front), with a turnable 3D preview and notes on
what each camera sees. Saved in the setup file; anything may be left blank."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from pydantic import ValidationError

from macaw.config.loader import resolve_config
from macaw.config.schema import DEVICE_NAME, MacawConfig, RoomSection, SetupSection
from macaw.exceptions import MacawError
from macaw.interface import ui
from macaw.project import room as geometry
from macaw.project.setup import write_setup

NUM, ADD, REMOVE = "room-num", "room-add", "room-remove"
KINDS = ("people", "cameras", "microphones", "things")
WALLS = "walls"
FIELDS = {
    WALLS: ("width", "length"),
    "people": ("x", "y", "eyes", "facing"),
    "cameras": ("x", "y", "height", "facing", "tilt", "lens"),
    "microphones": ("x", "y", "height"),
    "things": ("x", "y", "height", "width", "depth"),
}
HEADS = {
    "x": "Across", "y": "Along", "eyes": "Eye height", "height": "Height", "facing": "Facing (°)", "tilt": "Tilt down (°)",
    "lens": "Lens angle (°)", "width": "Width", "depth": "Depth", "length": "Length",
}
ANGLES = {"facing", "tilt", "lens"}
NOUN = {WALLS: "Walls", "people": "Person", "cameras": "Camera", "microphones": "Microphone", "things": "Thing"}
# sizes drawn, in centimetres, turned into the project's unit
_BODY_CM, _HEAD_CM, _DEVICE_CM, _THING_CM = 22.0, 11.0, 8.0, 20.0
_ROOM_CM = (400.0, 300.0, 250.0)
_COLOURS = {"people": ("#1d4e89", "#e8590c", "#2b8a3e", "#862e9c", "#0b7285", "#c92a2a"), "cameras": "#7048e8", "microphones": "#2b8a3e", "things": "#868e96"}
# the card's drawing area, in pixels across; the views are sized so a centimetre is as long in each
_WIDTH_PX, _MARGIN = 1180, {"l": 46, "r": 10, "t": 28, "b": 36}


def card(config):
    units = config.project.units
    return ui.card(
        "The room",
        views(*(go.Figure() for _ in range(4))),
        html.Div(id="room-notes"),
        html.Div(id="room-panel"),
        html.Div(
            [
                ui.field("New thing", dcc.Input(id="room-thing-name", placeholder="table")),
                ui.button("Add a thing", ADD, small=True),
            ],
            className="toolbar",
        ),
        html.Div([ui.button("Save the room", "room-save", primary=True), ui.button("Draw the room for a paper", "room-draw", icon_name="download")], className="buttons"),
        html.Div(id="room-message"),
        dcc.Store(id="room-shapes", data={}),
        desc=(
            f"Where the people, cameras, microphones and things are, in {geometry.UNIT_NAMES[units]}: across and along the "
            "room seen from above, from any corner chosen as zero, and heights from the floor. Type the numbers or drag "
            "a person, device or thing in any of the three views. Facing is in degrees clockwise from straight along the "
            "room as seen from above, tilt in degrees below level, and the lens angle is the width of the camera's picture "
            "as an angle. Anything may be left blank; whom people look at is read from the room once the camera reading "
            "their gaze and the people are placed."
        ),
    )


def _blank(kind: str) -> dict:
    return dict.fromkeys(FIELDS[kind])


def rows_of(config, name: str) -> dict[str, dict[str, dict]]:
    """kind -> name -> field -> number or None: everyone and every device of the setup, placed or not, and the things."""
    path = Path(config.project.setups_dir) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    data = data if isinstance(data, dict) else {}
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    names = {
        "people": list(config.project.roles),
        "cameras": list(data.get("cameras") or {}),
        "microphones": list(data.get("microphones") or {}),
        "things": list((room.get("things") or {})),
    }
    size = room.get("size") or [None, None]
    rows: dict[str, dict[str, dict]] = {WALLS: {"room": {"width": size[0], "length": size[1]}}}
    for kind in KINDS:
        rows[kind] = {}
        for item in names[kind]:
            placed = (room.get(kind) or {}).get(item) or {}
            row = _blank(kind)
            at = placed.get("at") or [None, None]
            row.update(x=at[0], y=at[1])
            for key in ("eyes", "height", "facing", "tilt", "lens"):
                if key in row:
                    row[key] = placed.get(key)
            if kind == "things":
                size = placed.get("size") or [None, None]
                row.update(width=size[0], depth=size[1])
            rows[kind][str(item)] = row
    return rows


def room_of(rows: dict[str, dict[str, dict]]) -> dict:
    """The room section the rows describe, leaving out what is blank."""
    room: dict = {}
    walls = rows.get(WALLS, {}).get("room", {})
    if walls.get("width") is not None and walls.get("length") is not None:
        room["size"] = [float(walls["width"]), float(walls["length"])]
    for kind, items in rows.items():
        if kind == WALLS:
            continue
        placed = {}
        for name, row in items.items():
            entry = {}
            if row.get("x") is not None and row.get("y") is not None:
                entry["at"] = [float(row["x"]), float(row["y"])]
            for key in ("eyes", "height", "facing", "tilt", "lens"):
                if row.get(key) is not None:
                    entry[key] = float(row[key])
            if kind == "things" and row.get("width") is not None and row.get("depth") is not None:
                entry["size"] = [float(row["width"]), float(row["depth"])]
            placed[name] = entry
        room[kind] = placed
    return room


def rows_from(values: list, ids: list[dict]) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {kind: {} for kind in (WALLS, *KINDS)}
    for value, identity in zip(values, ids):
        row = rows[identity["kind"]].setdefault(identity["name"], _blank(identity["kind"]))
        row[identity["field"]] = None if value in (None, "") else float(value)
    return rows


def panel(rows: dict[str, dict[str, dict]], units: str):
    cards = []
    for kind in (WALLS, *KINDS):
        if not rows.get(kind):
            continue
        heads = [NOUN[kind], *[f"{HEADS[field]}{'' if field in ANGLES else f' ({units})'}" for field in FIELDS[kind]]]
        body = [
            [
                name,
                *[
                    dcc.Input(id={"type": NUM, "kind": kind, "name": name, "field": field}, type="number", value=row.get(field),
                              step="any", debounce=True, className="number")
                    for field in FIELDS[kind]
                ],
                *([ui.button("Remove", {"type": REMOVE, "name": name}, small=True)] if kind == "things" else []),
            ]
            for name, row in rows[kind].items()
        ]
        cards.append(ui.table([*heads, *([""] if kind == "things" else [])], body))
    return html.Div(cards, className="room-numbers")


def _extent(room: RoomSection, units: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Ranges across, along and up that hold everything placed, with a margin, or an empty room."""
    xs, ys, zs = [0.0], [0.0], [0.0]
    if room.size is not None:
        xs.append(room.size[0])
        ys.append(room.size[1])
    for kind in KINDS:
        for place in getattr(room, kind).values():
            if place.at is not None:
                xs.append(place.at[0])
                ys.append(place.at[1])
            for key in ("eyes", "height"):
                if getattr(place, key, None) is not None:
                    zs.append(getattr(place, key))
    room_size = [geometry.to_unit(value, units) for value in _ROOM_CM]
    margin = geometry.to_unit(40.0, units)
    ranges = []
    for values, size in zip((xs, ys, zs), room_size):
        low, high = min(values), max(values)
        if high - low < size:
            high = low + size
        ranges.append((low - margin, high + margin))
    ranges[2] = (min(ranges[2][0], 0.0 - margin / 2), ranges[2][1])
    return tuple(ranges)


def _circle(x, y, radius, colour, name, width=2, fill=None):
    return dict(type="circle", x0=x - radius, x1=x + radius, y0=y - radius, y1=y + radius, line={"color": colour, "width": width},
                fillcolor=fill or colour, opacity=0.85, editable=True, name=name)


def _square(x, y, half, colour, name):
    return dict(type="rect", x0=x - half, x1=x + half, y0=y - half, y1=y + half, line={"color": colour, "width": 2},
                fillcolor=colour, opacity=0.9, editable=True, name=name)


def _colour(kind: str, name: str, roles: list[str]) -> str:
    if kind == "people":
        palette = _COLOURS["people"]
        return palette[roles.index(name) % len(palette)] if name in roles else palette[0]
    return _COLOURS[kind]


def _view_figure(axes: tuple[str, str], ranges, size: tuple[int, int], title: str, units: str) -> go.Figure:
    figure = go.Figure()
    grid = "rgba(128,128,128,0.22)"
    figure.update_layout(
        width=size[0], height=size[1], autosize=False, margin=_MARGIN, showlegend=False, dragmode=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90", "size": 11},
        title={"text": title, "x": 0.01, "y": 0.99, "font": {"size": 12}},
        xaxis={"range": list(ranges[0]), "title": {"text": f"{axes[0]} ({units})", "standoff": 2}, "gridcolor": grid, "zeroline": False},
        yaxis={"range": list(ranges[1]), "title": {"text": f"{axes[1]} ({units})", "standoff": 2}, "gridcolor": grid, "zeroline": False},
    )
    return figure


def figures(room: RoomSection, roles: list[str], units: str) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, dict]:
    """(front, side, top, 3D, shapes): shapes says which item and which numbers each draggable shape of a view carries."""
    across, along, up = _extent(room, units)
    span_x, span_y, span_z = (high - low for low, high in (across, along, up))
    inner = _WIDTH_PX - 2 * (_MARGIN["l"] + _MARGIN["r"])
    per = inner / (span_x + span_y)
    left_w, right_w = int(span_x * per) + _MARGIN["l"] + _MARGIN["r"], int(span_y * per) + _MARGIN["l"] + _MARGIN["r"]
    top_h, low_h = int(span_z * per) + _MARGIN["t"] + _MARGIN["b"], int(span_y * per) + _MARGIN["t"] + _MARGIN["b"]
    front = _view_figure(("across", "up"), (across, up), (left_w, top_h), "Front, seen from along = 0", units)
    side = _view_figure(("along", "up"), (along, up), (right_w, top_h), "Side, seen from the right", units)
    top = _view_figure(("across", "along"), (across, along), (left_w, low_h), "Top, seen from above", units)
    body, head, device, thing = (geometry.to_unit(value, units) for value in (_BODY_CM, _HEAD_CM, _DEVICE_CM, _THING_CM))
    reach = max(span_x, span_y)
    shapes: dict[str, list] = {"front": [], "side": [], "top": []}
    labels: dict[str, list] = {"front": [], "side": [], "top": []}
    lines: dict[str, list] = {"front": [], "side": [], "top": []}

    def add(view: str, shape: dict, item: tuple[str, str], label_at, text: str):
        {"front": front, "side": side, "top": top}[view].add_shape(**shape)
        shapes[view].append(list(item))
        labels[view].append((label_at, text))

    # things first, so a person or device standing at a table is drawn over it and dragged rather than the table
    for kind in reversed(KINDS):
        for name, place in getattr(room, kind).items():
            if place.at is None:
                continue
            x, y = place.at
            colour = _colour(kind, name, roles)
            item = (kind, name)
            if kind == "people":
                add("top", _circle(x, y, body, colour, name), item, (x, y + body), name)
                if place.facing is not None:
                    tip = np.array([x, y]) + 1.8 * body * geometry.heading(place.facing)[:2]
                    lines["top"].append(((x, tip[0]), (y, tip[1]), colour))
                if place.eyes is not None:
                    add("front", _circle(x, place.eyes, head, colour, name), item, (x, place.eyes + head), name)
                    add("side", _circle(y, place.eyes, head, colour, name), item, (y, place.eyes + head), name)
                    lines["front"].append(((x, x), (0, place.eyes - head), colour))
                    lines["side"].append(((y, y), (0, place.eyes - head), colour))
            elif kind == "cameras":
                add("top", _square(x, y, device, colour, name), item, (x, y + device), name)
                if place.facing is not None:
                    half = (place.lens or 0.0) / 2
                    for angle in ((place.facing - half, place.facing + half) if half else (place.facing,)):
                        tip = np.array([x, y]) + reach * geometry.heading(angle)[:2]
                        lines["top"].append(((x, tip[0]), (y, tip[1]), colour))
                if place.height is not None:
                    add("front", _square(x, place.height, device, colour, name), item, (x, place.height + device), name)
                    add("side", _square(y, place.height, device, colour, name), item, (y, place.height + device), name)
                    view = geometry.view_of(place)
                    if view is not None:
                        tip = view.origin + reach * view.ahead
                        lines["front"].append(((x, tip[0]), (place.height, tip[2]), colour))
                        lines["side"].append(((y, tip[1]), (place.height, tip[2]), colour))
            elif kind == "microphones":
                add("top", _circle(x, y, device / 1.5, colour, name), item, (x, y + device), name)
                if place.height is not None:
                    add("front", _circle(x, place.height, device / 1.5, colour, name), item, (x, place.height + device), name)
                    add("side", _circle(y, place.height, device / 1.5, colour, name), item, (y, place.height + device), name)
            else:
                width, depth = place.size or (thing, thing)
                add("top", dict(type="rect", x0=x - width / 2, x1=x + width / 2, y0=y - depth / 2, y1=y + depth / 2, line={"color": colour, "width": 2},
                                fillcolor=colour, opacity=0.35, editable=True, name=name), item, (x, y + depth / 2), name)
                if place.height is not None:
                    add("front", dict(type="rect", x0=x - width / 2, x1=x + width / 2, y0=0, y1=place.height, line={"color": colour, "width": 2},
                                      fillcolor=colour, opacity=0.35, editable=True, name=name), item, (x, place.height), name)
                    add("side", dict(type="rect", x0=y - depth / 2, x1=y + depth / 2, y0=0, y1=place.height, line={"color": colour, "width": 2},
                                     fillcolor=colour, opacity=0.35, editable=True, name=name), item, (y, place.height), name)
    if room.size is not None:
        # lines, not shapes, so the walls cannot be dragged
        width, length = room.size
        lines["top"].append(((0, width, width, 0, 0), (0, 0, length, length, 0), "#8b8b90"))
        lines["front"] += [((at, at), (0, up[1]), "#8b8b90") for at in (0, width)]
        lines["side"] += [((at, at), (0, up[1]), "#8b8b90") for at in (0, length)]
    for view, figure in (("front", front), ("side", side), ("top", top)):
        for (xs, ys, colour) in lines[view]:
            wall = colour == "#8b8b90"
            figure.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines", line={"color": colour, "width": 2 if wall else 1.5, "dash": None if wall else "dot"}, hoverinfo="skip"))
        if labels[view]:
            figure.add_trace(go.Scatter(
                x=[at[0] for at, _ in labels[view]], y=[at[1] for at, _ in labels[view]], text=[text for _, text in labels[view]],
                mode="text", textposition="top center", hoverinfo="skip", textfont={"size": 11},
            ))
    preview = preview_figure(room, roles, units, (right_w, low_h), (across, along, up))
    return front, side, top, preview, shapes


def preview_figure(room: RoomSection, roles: list[str], units: str, size: tuple[int, int], ranges) -> go.Figure:
    """The room in 3D, to turn by dragging: people as a line up to their eyes, cameras with what they see, things as boxes."""
    figure = go.Figure()
    reach = geometry.to_unit(80.0, units)
    for name, place in room.people.items():
        if place.at is None or place.eyes is None:
            continue
        colour = _colour("people", name, roles)
        x, y = place.at
        figure.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[0, place.eyes], mode="lines", line={"color": colour, "width": 8}, hoverinfo="skip"))
        figure.add_trace(go.Scatter3d(x=[x], y=[y], z=[place.eyes], mode="markers+text", text=[name], textposition="top center",
                                      marker={"size": 7, "color": colour}, hoverinfo="skip"))
    for name, place in room.cameras.items():
        view = geometry.view_of(place)
        if view is None:
            continue
        corners = _frustum(view, place.lens or 60.0, reach)
        xs, ys, zs = [], [], []
        for corner in corners:
            xs += [view.origin[0], corner[0], None]
            ys += [view.origin[1], corner[1], None]
            zs += [view.origin[2], corner[2], None]
        loop = corners + corners[:1]
        xs += [point[0] for point in loop]
        ys += [point[1] for point in loop]
        zs += [point[2] for point in loop]
        figure.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line={"color": _COLOURS["cameras"], "width": 3}, hoverinfo="skip"))
        figure.add_trace(go.Scatter3d(x=[view.origin[0]], y=[view.origin[1]], z=[view.origin[2]], mode="markers+text", text=[name],
                                      textposition="top center", marker={"size": 4, "color": _COLOURS["cameras"], "symbol": "diamond"}, hoverinfo="skip"))
    for name, place in room.microphones.items():
        if place.at is None or place.height is None:
            continue
        figure.add_trace(go.Scatter3d(x=[place.at[0]], y=[place.at[1]], z=[place.height], mode="markers+text", text=[name],
                                      textposition="top center", marker={"size": 4, "color": _COLOURS["microphones"]}, hoverinfo="skip"))
    thing = geometry.to_unit(_THING_CM, units)
    for name, place in room.things.items():
        if place.at is None or place.height is None:
            continue
        width, depth = place.size or (thing, thing)
        x, y = place.at
        bx = [x - width / 2, x + width / 2]
        by = [y - depth / 2, y + depth / 2]
        figure.add_trace(go.Mesh3d(
            x=[bx[i] for i in (0, 1, 1, 0, 0, 1, 1, 0)], y=[by[i] for i in (0, 0, 1, 1, 0, 0, 1, 1)], z=[0.0] * 4 + [place.height] * 4,
            i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3], j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4], k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
            color=_COLOURS["things"], opacity=0.35, hoverinfo="skip", name=name,
        ))
        figure.add_trace(go.Scatter3d(x=[x], y=[y], z=[place.height], mode="text", text=[name], textposition="top center", hoverinfo="skip"))
    axis = {"gridcolor": "rgba(128,128,128,0.25)", "zerolinecolor": "rgba(128,128,128,0.4)", "backgroundcolor": "rgba(0,0,0,0)"}
    figure.update_layout(
        width=size[0], height=size[1], autosize=False, margin={"l": 0, "r": 0, "t": 24, "b": 0}, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90", "size": 10}, title={"text": "In 3D: drag to turn", "x": 0.01, "y": 0.99, "font": {"size": 12}},
        scene={
            "aspectmode": "data", "camera": {"eye": {"x": -1.3, "y": -1.6, "z": 1.0}},
            "xaxis": {"title": f"across ({units})", "range": list(ranges[0]), **axis},
            "yaxis": {"title": f"along ({units})", "range": list(ranges[1]), **axis},
            "zaxis": {"title": f"up ({units})", "range": [0, ranges[2][1]], **axis},
        },
        uirevision="room",
    )
    return figure


def _frustum(view: geometry.View, lens: float, reach: float, aspect: float = 9 / 16) -> list[np.ndarray]:
    half_w = np.tan(np.radians(lens) / 2) * reach
    half_h = half_w * aspect
    centre = view.origin + reach * view.ahead
    return [centre + sx * half_w * view.right + sy * half_h * view.up for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]


def dragged(rows: dict, shapes: dict, view: str, change: dict, units: str) -> dict | None:
    """The rows with the item behind a dragged shape moved to where it was dropped; None when nothing was dragged."""
    places = geometry.DECIMALS[units] + 1
    moved = {}
    for key, value in change.items():
        found = re.fullmatch(r"shapes\[(\d+)\]\.([xy][01])", key)
        if found:
            moved.setdefault(int(found.group(1)), {})[found.group(2)] = float(value)
    if not moved:
        return None
    rows = {kind: {name: dict(row) for name, row in items.items()} for kind, items in rows.items()}
    for index, ends in moved.items():
        if index >= len(shapes.get(view, [])) or not {"x0", "x1", "y0", "y1"} <= set(ends):
            continue
        kind, name = shapes[view][index]
        row = rows[kind][name]
        middle_x = round((ends["x0"] + ends["x1"]) / 2, places)
        middle_y = round((ends["y0"] + ends["y1"]) / 2, places)
        if view == "top":
            row["x"], row["y"] = middle_x, middle_y
            continue
        row["x" if view == "front" else "y"] = middle_x
        level = "eyes" if kind == "people" else "height"
        # a thing stands on the floor, so where its top was dropped is its height
        row[level] = round(max(ends["y0"], ends["y1"]), places) if kind == "things" else max(middle_y, 0.0)
    return rows


def notes(setup: SetupSection, room: RoomSection, units: str) -> list[str]:
    """What each placed camera sees of the people it is read for: whether they are in its picture, how far away they
    are, and from how far round their face it looks; the numbers a person needs to judge a scale or a face reading."""
    found = []
    for name, device in setup.cameras.items():
        camera = room.cameras.get(name)
        if camera is None or geometry.view_of(camera) is None:
            continue
        distances = {}
        for person in device.people():
            place = room.people.get(person)
            if place is None or place.at is None or place.eyes is None:
                continue
            distance = geometry.distance(camera, place)
            distances[person] = distance
            seen = geometry.in_picture(camera, [*place.at, place.eyes])
            if seen is False:
                found.append(f"{name}: {person} is outside its picture as laid out; check the facing and the lens angle.")
            angle = geometry.angle_to(camera, place)
            if angle is not None and (person in device.face or person in device.gaze):
                side = "from behind, so the face is not seen" if angle > 90 else "from in front" if angle < 10 else "from the side"
                found.append(f"{name}: {person}'s face is seen {angle:.0f}° {side}.")
        body = [person for person in device.body if person in distances]
        if len(body) > 1:
            near, far = min(body, key=distances.get), max(body, key=distances.get)
            ratio = distances[far] / distances[near]
            listed = ", ".join(f"{person} {distances[person]:.{geometry.DECIMALS[units]}f} {units}" for person in body)
            found.append(
                f"{name}: {listed} away. One pixel spans {ratio:.2f} times as much at {far}'s place as at {near}'s, so a scale "
                "measured at one of them misstates the other by that much."
                if ratio > 1.005 else f"{name}: {listed} away, at the same distance."
            )
    return found


def save_room(config_path: Path, name: str, rows: dict) -> Path:
    """Writes the room into the setup file, beside its devices, once it reads as a valid setup."""
    config = resolve_config(config_path)
    path = Path(config.project.setups_dir) / f"{name}.yaml"
    if not path.is_file():
        raise MacawError(f"setup {name} is not saved yet: describe and save its cameras and microphones first")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data = {**data, "name": name, "room": room_of(rows)}
    try:
        setup = MacawConfig.model_validate({**config.model_dump(), "setup": data}).setup
    except ValidationError as error:
        raise MacawError("; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())) from None
    return write_setup(path, setup)


def added(rows: dict, name: str, roles: list[str]) -> dict:
    name = (name or "").strip()
    if not re.fullmatch(DEVICE_NAME, name):
        raise MacawError("a thing's name takes letters, digits, - and _ only")
    taken = {item for items in rows.values() for item in items} | set(roles)
    if name in taken:
        raise MacawError(f"{name} is already in the room")
    return {**rows, "things": {**rows["things"], name: _blank("things")}}


def drawn(config, rows: dict):
    """The four views and the notes for these rows, or why they cannot be drawn."""
    units = config.project.units
    try:
        room = RoomSection.model_validate(room_of(rows))
    except ValidationError as error:
        problem = "; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())
        return None, None, None, None, {}, [ui.note(problem, "bad")]
    front, side, top, preview, shapes = figures(room, list(config.project.roles), units)
    setup = config.setup
    try:
        listed = notes(setup, room, units)
    except Exception:   # notes are a help; a room that cannot be read for them still draws
        listed = []
    return front, side, top, preview, shapes, [ui.note(text, "warn" if "outside" in text or "behind" in text else "info") for text in listed]


def views(front, side, top, preview):
    edit = {"displayModeBar": False, "edits": {"shapePosition": True}}
    return html.Div(
        [
            dcc.Graph(id="room-front", figure=front, config=edit),
            dcc.Graph(id="room-side", figure=side, config=edit),
            dcc.Graph(id="room-top", figure=top, config=edit),
            dcc.Graph(id="room-3d", figure=preview, config={"displaylogo": False}),
        ],
        className="room-views",
    )


def register(app, config_path: Path) -> None:
    def setup_config(name: str):
        """The project's settings with this setup's file read as the setup, so the notes speak of its devices."""
        config = resolve_config(config_path)
        path = Path(config.project.setups_dir) / f"{name}.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        if isinstance(data, dict):
            try:
                config = MacawConfig.model_validate({**config.model_dump(), "setup": {**data, "name": name}})
            except ValidationError:
                pass
        return config

    @app.callback(
        Output("room-panel", "children"), Output("room-message", "children", allow_duplicate=True),
        Input("setup-pick", "value"), Input(ADD, "n_clicks"), Input({"type": REMOVE, "name": ALL}, "n_clicks"),
        Input("setup-files", "children"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"), State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
        State("room-thing-name", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def listed(name, _add, _remove, _saved, values, ids, thing):
        config = resolve_config(config_path)
        trigger = ctx.triggered_id
        if trigger == ADD and ctx.triggered[0]["value"]:
            rows = rows_from(values, ids)
            try:
                rows = added(rows, thing, list(config.project.roles))
            except MacawError as error:
                return no_update, ui.note(str(error), "bad")
            return panel(rows, config.project.units), None
        if isinstance(trigger, dict) and trigger.get("type") == REMOVE:
            if not ctx.triggered[0]["value"]:
                return no_update, no_update
            rows = rows_from(values, ids)
            rows["things"].pop(trigger["name"], None)
            return panel(rows, config.project.units), None
        return panel(rows_of(config, name), config.project.units), None

    @app.callback(
        Output("room-front", "figure"), Output("room-side", "figure"), Output("room-top", "figure"), Output("room-3d", "figure"),
        Output("room-notes", "children"), Output("room-shapes", "data"),
        Input({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"), State("setup-pick", "value"),
    )
    def draw(values, ids, name):
        config = setup_config(name)
        front, side, top, preview, shapes, said = drawn(config, rows_from(values, ids))
        if front is None:
            return no_update, no_update, no_update, no_update, said, no_update
        return front, side, top, preview, said, shapes

    @app.callback(
        Output({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"),
        Input("room-front", "relayoutData"), Input("room-side", "relayoutData"), Input("room-top", "relayoutData"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"), State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
        State("room-shapes", "data"),
        prevent_initial_call=True,
    )
    def drag(front, side, top, values, ids, shapes):
        view = {"room-front": "front", "room-side": "side", "room-top": "top"}.get(ctx.triggered_id)
        change = {"front": front, "side": side, "top": top}.get(view) or {}
        rows = dragged(rows_from(values, ids), shapes or {}, view, change, resolve_config(config_path).project.units)
        if rows is None:
            return [no_update] * len(ids)
        return [rows[identity["kind"]][identity["name"]][identity["field"]] for identity in ids]

    @app.callback(
        Output("room-message", "children"),
        Input("room-save", "n_clicks"), Input("room-draw", "n_clicks"),
        State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "value"), State({"type": NUM, "kind": ALL, "name": ALL, "field": ALL}, "id"),
        State("setup-pick", "value"),
        prevent_initial_call=True,
    )
    def keep(_save, _draw, values, ids, name):
        if ctx.triggered_id == "room-save":
            try:
                path = save_room(config_path, name, rows_from(values, ids))
            except MacawError as error:
                return ui.note(str(error), "bad")
            return ui.note(f"Saved the room in {path.name}.", "ok")
        return draw_figure(config_path, name)


def draw_figure(config_path: Path, name: str):
    from macaw.interface import cli_bridge
    from macaw.interface.files import file_url

    argv = cli_bridge.build_command(["draw-room"], config_path=config_path, extra_flags=["--setup", name])
    code, text = cli_bridge.run_now(argv, cwd=config_path.parent)
    config = resolve_config(config_path)
    path = Path(config.project.setups_dir) / f"{name}.png"
    link = [html.A(f"Open {path.name}", href=file_url(path), target="_blank")] if code == 0 and path.is_file() else []
    return [ui.command(argv), ui.output(code, text), *link]
