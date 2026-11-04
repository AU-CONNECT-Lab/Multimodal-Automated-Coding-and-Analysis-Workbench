"""A drawing of the room for a paper: a plan and a side view, without axes, at 300 dpi, a length as long in both."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.config.schema import RoomSection
from macaw.project import room as geometry

DPI = 300
WIDTH_IN = 7.0
_INK, _BODY, _CAMERA, _FAN, _MICROPHONE = "#212529", "#495057", "#4a72d4", "#c9d6f2", "#2f9e44"
_THING, _THING_EDGE, _WALL = "#e1e2e6", "#5c5f66", "#212529"
_PERSON_CM, _HEAD_CM, _DEVICE_CM, _THING_CM, _MARGIN_CM, _ARROW_CM = 23.0, 11.0, 9.0, 20.0, 30.0, 16.0
_SCALE = {"cm": (100.0, "100 cm"), "m": (1.0, "1 m"), "in": (36.0, "36 in")}


def _ranges(room: RoomSection, units: str):
    xs, ys, zs = [], [], [0.0]
    if room.size is not None:
        xs += [0.0, room.size[0]]
        ys += [0.0, room.size[1]]
    for kind in ("people", "cameras", "microphones", "things"):
        for place in getattr(room, kind).values():
            if place.at is not None:
                xs.append(place.at[0])
                ys.append(place.at[1])
            for key in ("eyes", "height"):
                if getattr(place, key, None) is not None:
                    zs.append(getattr(place, key))
    if not xs:
        raise ValueError("the room places nothing yet")
    # inside walls everything is drawn, so a narrow margin is enough across; above, a camera's name needs room
    margin = geometry.to_unit(_MARGIN_CM, units)
    side = margin / 3 if room.size is not None else margin
    return [(min(values) - side, max(values) + side) for values in (xs, ys)] + [(-margin / 3, max(zs) + margin)]


def _fan_reach(room: RoomSection, name: str, place, reads: dict[str, list[str]], fallback: float) -> float:
    """As far as the people the camera reads, and a little beyond."""
    found = [
        np.hypot(*np.subtract(room.people[person].at, place.at))
        for person in reads.get(name, []) if person in room.people and room.people[person].at is not None
    ]
    return 1.3 * max(found) if found else fallback


def draw_room(
    room: RoomSection, roles: list[str], units: str, path: Path, *, reads: dict[str, list[str]] | None = None,
) -> Path:
    """Writes the drawing as a PNG; what the room leaves blank is left out. reads gives the people each camera is read
    for, whose heads its sight lines go to in the side view."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge

    reads = reads or {}
    across, along, up = _ranges(room, units)
    span_x, span_y, span_z = (high - low for low, high in (across, along, up))
    per = (WIDTH_IN - 0.3) / (span_x + span_y)
    height_in = max(span_y, span_z) * per + 0.75
    figure = Figure(figsize=(WIDTH_IN, height_in), dpi=DPI)
    grid = figure.add_gridspec(2, 2, width_ratios=[span_x, span_y], height_ratios=[height_in - 0.45, 0.45], wspace=0.06, hspace=0.02)
    top, side, key = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])
    person, head, device, thing, arrow = (geometry.to_unit(value, units) for value in (_PERSON_CM, _HEAD_CM, _DEVICE_CM, _THING_CM, _ARROW_CM))
    for axes, horizontal, vertical, anchor, name in ((top, across, along, "C", "Top view"), (side, along, up, "S", "Side view, seen from the right")):
        axes.set_xlim(*horizontal)
        axes.set_ylim(*vertical)
        axes.set_aspect("equal", anchor=anchor)
        axes.axis("off")
        axes.set_title(name, fontsize=8.5, loc="left", color=_INK, pad=4)

    def arrow_from(axes, start, direction, length, colour, width=1.0):
        end = np.asarray(start) + length * np.asarray(direction)
        axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6, color=colour, linewidth=width, shrinkA=0, shrinkB=0))

    # the plan: walls, what each camera sees, things, people, devices
    if room.size is not None:
        top.add_patch(Rectangle((0, 0), *room.size, fill=False, edgecolor=_WALL, linewidth=1.4))
    fallback = 0.45 * np.hypot(span_x, span_y)
    for name, place in room.cameras.items():
        if place.at is not None and place.facing is not None and place.lens is not None:
            middle = 90 - place.facing
            top.add_patch(Wedge(place.at, _fan_reach(room, name, place, reads, fallback), middle - place.lens / 2, middle + place.lens / 2,
                                facecolor=_FAN, alpha=0.5, edgecolor="none"))
    for name, place in room.things.items():
        if place.at is None:
            continue
        width, depth = place.size or (thing, thing)
        x, y = place.at
        top.add_patch(Rectangle((x - width / 2, y - depth / 2), width, depth, facecolor=_THING, alpha=0.8, edgecolor=_THING_EDGE, linewidth=0.8))
        # in a corner, where people sitting at a table's sides leave it readable
        top.text(x - width / 2 + 0.05 * width, y - depth / 2 + 0.06 * depth, name, fontsize=6.5, color=_THING_EDGE, va="bottom", ha="left")
    for name, place in room.people.items():
        if place.at is None:
            continue
        top.add_patch(Circle(place.at, person, facecolor="white", edgecolor=_INK, linewidth=1.1, zorder=3))
        # the name inside the circle when it fits at a readable size, else beside it
        across_pt = 2 * person * per * 72 * 0.85
        size = min(6.5, across_pt / (0.6 * max(len(name), 1)))
        if size >= 4.5:
            top.text(*place.at, name, fontsize=size, ha="center", va="center", color=_INK, zorder=4)
        else:
            top.text(place.at[0] + 1.2 * person, place.at[1], name, fontsize=6, ha="left", va="center", color=_INK, zorder=4)
        if place.facing is not None:
            facing = geometry.heading(place.facing)[:2]
            arrow_from(top, np.asarray(place.at) + 1.15 * person * facing, facing, arrow, _INK)
    for name, place in room.cameras.items():
        if place.at is None:
            continue
        x, y = place.at
        top.add_patch(Rectangle((x - device / 2, y - device / 2), device, device, facecolor=_CAMERA, edgecolor="none", zorder=3))
        behind = -geometry.heading(place.facing)[:2] if place.facing is not None else np.array([0.0, -1.0])
        if place.facing is not None:
            arrow_from(top, np.asarray(place.at) + 0.7 * device * -behind, -behind, arrow, _CAMERA, 1.3)
        label = np.asarray(place.at) + 1.8 * device * behind
        near_wall = room.size is not None and (
            min(label[0], room.size[0] - label[0], label[1], room.size[1] - label[1]) < 0.8 * device
        )
        if near_wall:
            top.text(x + device, y, name, fontsize=6.5, fontweight="bold", color=_INK, ha="left", va="center")
        else:
            top.text(*label, name, fontsize=6.5, fontweight="bold", color=_INK, ha="center", va="center")
    for name, place in room.microphones.items():
        if place.at is None:
            continue
        top.add_patch(Circle(place.at, device / 2, facecolor=_MICROPHONE, edgecolor="none", zorder=3))
        top.text(place.at[0] + device, place.at[1], name, fontsize=6, color=_INK, va="center")
    bar, said = _SCALE[units]
    if room.size is not None:
        left, bottom = room.size[0] - 0.06 * span_x - bar, 0.05 * span_y
    else:
        left, bottom = across[0] + 0.04 * span_x, along[0] + 0.02 * span_y
    top.plot([left, left + bar], [bottom, bottom], color=_INK, linewidth=1.2, solid_capstyle="butt")
    top.text(left + bar / 2, bottom, said, fontsize=6.5, ha="center", va="bottom", color=_INK)

    # the side view: floor and walls, things, people as a head on a body, cameras with lines to whom they read
    side.plot(along, [0, 0], color=_WALL, linewidth=1.2)
    if room.size is not None:
        for at in (0, room.size[1]):
            side.plot([at, at], [0, up[1]], color=_WALL, linewidth=1.2)
    for name, place in room.things.items():
        if place.at is None or place.height is None:
            continue
        depth = (place.size or (thing, thing))[1]
        side.add_patch(Rectangle((place.at[1] - depth / 2, 0), depth, place.height, facecolor=_THING, alpha=0.8, edgecolor=_THING_EDGE, linewidth=0.8))
    heads = {}
    for name, place in room.people.items():
        if place.at is None or place.eyes is None:
            continue
        y = place.at[1]
        heads[name] = (y, place.eyes)
        side.plot([y, y], [0, place.eyes - head], color=_BODY, linewidth=2.6, solid_capstyle="butt", zorder=2)
        side.add_patch(Circle((y, place.eyes), head, facecolor="white", edgecolor=_INK, linewidth=1.1, zorder=3))
    for name, place in room.cameras.items():
        if place.at is None or place.height is None:
            continue
        y = place.at[1]
        for person in reads.get(name, []):
            if person in heads:
                side.plot([y, heads[person][0]], [place.height, heads[person][1]], color=_CAMERA, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
        side.add_patch(Rectangle((y - device / 2, place.height - device / 2), device, device, facecolor=_CAMERA, edgecolor="none", zorder=3))
        side.text(y, place.height + device, name, fontsize=6.5, fontweight="bold", color=_INK, ha="center", va="bottom")
    for name, place in room.microphones.items():
        if place.at is not None and place.height is not None:
            side.add_patch(Circle((place.at[1], place.height), device / 2, facecolor=_MICROPHONE, edgecolor="none", zorder=3))

    key.axis("off")
    handles = []
    if room.people:
        handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=_INK, markersize=7, label="person, arrow where they face"))
    if room.cameras:
        handles.append(Line2D([], [], marker="s", linestyle="", color=_CAMERA, markersize=6, label="camera, shaded what it sees"))
    if room.microphones:
        handles.append(Line2D([], [], marker="o", linestyle="", color=_MICROPHONE, markersize=5, label="microphone"))
    if room.things:
        handles.append(Line2D([], [], marker="s", linestyle="", markerfacecolor=_THING, markeredgecolor=_THING_EDGE, markersize=7, label="thing"))
    key.legend(handles=handles, loc="center", ncol=len(handles), frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.6)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    return path
