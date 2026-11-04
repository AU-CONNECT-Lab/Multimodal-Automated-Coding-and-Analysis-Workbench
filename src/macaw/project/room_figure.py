"""A drawing of the room for a paper: front and side elevations above and beside a plan, at 300 dpi, a length as long
in every view."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.config.schema import RoomSection
from macaw.project import room as geometry

DPI = 300
WIDTH_IN = 7.0
_PEOPLE = ("#1d4e89", "#e8590c", "#2b8a3e", "#862e9c", "#0b7285", "#c92a2a")
_CAMERA, _MICROPHONE, _THING, _INK, _GRID = "#343a40", "#2b8a3e", "#868e96", "#212529", "#dee2e6"
_BODY_CM, _HEAD_CM, _DEVICE_CM, _THING_CM, _MARGIN_CM = 22.0, 11.0, 9.0, 20.0, 40.0


def _ranges(room: RoomSection, units: str):
    xs, ys, zs = [], [], [0.0]
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
    margin = geometry.to_unit(_MARGIN_CM, units)
    return [(min(values) - margin, max(values) + margin) for values in (xs, ys)] + [(0.0, max(zs) + margin)]


def draw_room(room: RoomSection, roles: list[str], units: str, path: Path, *, title: str | None = None) -> Path:
    """Writes the drawing as a PNG; things left blank in the room are left out of it."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Polygon, Rectangle

    across, along, up = _ranges(room, units)
    span_x, span_y, span_z = (high - low for low, high in (across, along, up))
    figure = Figure(figsize=(WIDTH_IN, WIDTH_IN * (span_z + span_y) / (span_x + span_y) + 0.6), dpi=DPI)
    grid = figure.add_gridspec(2, 2, width_ratios=[span_x, span_y], height_ratios=[span_z, span_y], wspace=0.12, hspace=0.12)
    front, side, top = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, 0])
    key = figure.add_subplot(grid[1, 1])
    body, head, device, thing = (geometry.to_unit(value, units) for value in (_BODY_CM, _HEAD_CM, _DEVICE_CM, _THING_CM))
    reach = max(span_x, span_y)

    for axes, (horizontal, vertical), name in ((front, (across, up), "Front"), (side, (along, up), "Side"), (top, (across, along), "Top")):
        axes.set_xlim(*horizontal)
        axes.set_ylim(*vertical)
        axes.set_aspect("equal")
        axes.grid(color=_GRID, linewidth=0.5)
        axes.set_axisbelow(True)
        axes.tick_params(labelsize=6, length=2, color=_GRID)
        for spine in axes.spines.values():
            spine.set_color(_GRID)
        axes.set_title(name, fontsize=8, loc="left", color=_INK)
    front.set_ylabel(f"height ({units})", fontsize=7)
    top.set_xlabel(f"across ({units})", fontsize=7)
    top.set_ylabel(f"along ({units})", fontsize=7)
    side.set_xlabel(f"along ({units})", fontsize=7)
    for axes in (front, side):
        axes.axhline(0, color=_INK, linewidth=0.8)

    def label(axes, x, y, text, colour):
        axes.annotate(text, (x, y), textcoords="offset points", xytext=(0, 4), ha="center", va="bottom", fontsize=6.5, color=colour)

    for index, (name, place) in enumerate(room.people.items()):
        if place.at is None:
            continue
        colour = _PEOPLE[(roles.index(name) if name in roles else index) % len(_PEOPLE)]
        x, y = place.at
        top.add_patch(Circle((x, y), body, facecolor=colour, alpha=0.85, edgecolor="white", linewidth=0.6))
        if place.facing is not None:
            tip = np.array([x, y]) + 1.9 * body * geometry.heading(place.facing)[:2]
            top.annotate("", tip, (x, y), arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1, "mutation_scale": 7})
        label(top, x, y + body, name, colour)
        if place.eyes is not None:
            for axes, along_axis in ((front, x), (side, y)):
                axes.plot([along_axis, along_axis], [0, place.eyes - head], color=colour, linewidth=2.2, solid_capstyle="butt")
                axes.add_patch(Circle((along_axis, place.eyes), head, facecolor=colour, edgecolor="white", linewidth=0.6))
                label(axes, along_axis, place.eyes + head, name, colour)

    for name, place in room.cameras.items():
        if place.at is None:
            continue
        x, y = place.at
        if place.facing is not None and place.lens is not None:
            wedge = [np.array([x, y]) + reach * geometry.heading(angle)[:2] for angle in np.linspace(place.facing - place.lens / 2, place.facing + place.lens / 2, 24)]
            top.add_patch(Polygon([(x, y), *wedge], closed=True, facecolor=_CAMERA, alpha=0.07, edgecolor=_CAMERA, linewidth=0.4, linestyle=":"))
        top.add_patch(Rectangle((x - device, y - device), 2 * device, 2 * device, facecolor=_CAMERA, edgecolor="white", linewidth=0.5))
        label(top, x, y + device, name, _CAMERA)
        if place.height is not None:
            view = geometry.view_of(place)
            for axes, along_axis, which in ((front, x, 0), (side, y, 1)):
                axes.add_patch(Rectangle((along_axis - device, place.height - device), 2 * device, 2 * device, facecolor=_CAMERA, edgecolor="white", linewidth=0.5))
                label(axes, along_axis, place.height + device, name, _CAMERA)
                if view is not None:
                    tip = view.origin + 0.35 * reach * view.ahead
                    axes.plot([along_axis, tip[which]], [place.height, tip[2]], color=_CAMERA, linewidth=0.7, linestyle=":")

    for name, place in room.microphones.items():
        if place.at is None:
            continue
        x, y = place.at
        top.add_patch(Circle((x, y), device / 1.6, facecolor=_MICROPHONE, edgecolor="white", linewidth=0.5))
        label(top, x, y + device, name, _MICROPHONE)
        if place.height is not None:
            for axes, along_axis in ((front, x), (side, y)):
                axes.add_patch(Circle((along_axis, place.height), device / 1.6, facecolor=_MICROPHONE, edgecolor="white", linewidth=0.5))

    for name, place in room.things.items():
        if place.at is None:
            continue
        x, y = place.at
        width, depth = place.size or (thing, thing)
        top.add_patch(Rectangle((x - width / 2, y - depth / 2), width, depth, facecolor=_THING, alpha=0.35, edgecolor=_THING, linewidth=0.6))
        top.annotate(name, (x, y - depth / 2), textcoords="offset points", xytext=(0, -3), ha="center", va="top", fontsize=6.5, color=_INK)
        if place.height is not None:
            for axes, along_axis, extent in ((front, x, width), (side, y, depth)):
                axes.add_patch(Rectangle((along_axis - extent / 2, 0), extent, place.height, facecolor=_THING, alpha=0.35, edgecolor=_THING, linewidth=0.6))

    key.axis("off")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=_PEOPLE[0], label="person (arrow: facing)"),
        Line2D([], [], marker="s", linestyle="", color=_CAMERA, label="camera (shaded: what it sees)"),
        Line2D([], [], marker="o", linestyle="", color=_MICROPHONE, markersize=4, label="microphone"),
        Line2D([], [], marker="s", linestyle="", color=_THING, alpha=0.5, label="thing"),
    ]
    legend = key.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5, title=title, title_fontsize=7.5, alignment="left")
    legend.get_title().set_color(_INK)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    return path
