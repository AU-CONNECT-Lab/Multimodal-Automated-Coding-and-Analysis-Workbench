"""The Setup page's editor: the people, cameras, microphones and things of a setup in one list, the room drawn in 3D
and in three views where each is dragged, and the chosen one's settings beside them. Saved as setups/<name>.yaml;
positions may be left blank."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from pydantic import ValidationError

from macaw.config.loader import resolve_config
from macaw.config.schema import DEVICE_NAME, CameraDevice, MacawConfig, MicrophoneDevice, RoomSection, SetupSection
from macaw.exceptions import MacawError
from macaw.interface import project_file, ui
from macaw.project import room as geometry
from macaw.project.setup import write_setup

PICK, FIELD, USE, RENAME, ACT = "room-pick", "room-field", "room-use", "room-rename", "room-act"
KINDS = ("people", "cameras", "microphones", "things")
WALLS = "walls"
FIELDS = {
    WALLS: ("width", "length"),
    "people": ("x", "y", "eyes", "facing"),
    "cameras": ("x", "y", "height", "facing", "tilt", "lens"),
    "microphones": ("x", "y", "height"),
    "things": ("x", "y", "height", "width", "depth"),
}
USES = {"cameras": ("body", "face", "gaze", "hears"), "microphones": ("hears",)}
USE_LABELS = {"body": "Body", "face": "Face", "gaze": "Gaze", "hears": "Voice"}
HEADS = {
    "x": "Across", "y": "Along", "eyes": "Eye height", "height": "Height", "facing": "Facing (°)", "tilt": "Tilt down (°)",
    "lens": "Lens angle (°)", "width": "Width", "depth": "Depth", "length": "Length",
}
ANGLES = {"facing", "tilt", "lens"}
NOUN = {"people": "person", "cameras": "camera", "microphones": "microphone", "things": "thing"}
TITLES = {"people": "People", "cameras": "Cameras", "microphones": "Microphones", "things": "Things"}
# muted and apart, none read as standing for a sex
PEOPLE_COLOURS = ("#5f7f8f", "#c49a5a", "#7f9a6b", "#9a7b9a", "#8a877a", "#b07a63")
RESERVED = ("camera", "elsewhere", "off_screen", "unsure", "none", "other", "room")
# where a newly placed item goes, from the middle of the room, in centimetres apart
_SPREAD_CM = 40.0


def card(config):
    units = config.project.units
    room = config.setup.room
    started = any(place.at is not None for kind in KINDS for place in getattr(room, kind).values())
    return ui.card(
        "The setup",
        html.Div(id="room-message"),
        html.Div(
            [
                html.Div([html.Div(id="room-list"), add_box()], className="room-side room-left"),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.RadioItems(
                                    [{"label": "Plan", "value": "plan"}, {"label": "3D", "value": "free"}, {"label": "From a camera", "value": "camera"}],
                                    "free" if started else "plan", id="room-view-mode", inline=True, className="segmented",
                                ),
                                html.Div(
                                    [
                                        dcc.Dropdown([], None, id="room-view-camera", clearable=False, placeholder="camera", className="picker"),
                                        dcc.Dropdown([], None, id="room-view-recording", clearable=False, placeholder="recording", className="picker"),
                                        ui.field("At (s)", dcc.Input(id="room-view-at", type="number", min=0, step="any", value=0, debounce=True)),
                                    ],
                                    id="room-view-camera-tools", className="room-view-tools", style={"display": "none"},
                                ),
                            ],
                            className="room-view-bar",
                        ),
                        html.Div(id="room-main", className="room-canvas room-main"),
                        html.Div(id="room-plan", className="room-canvas room-main"),
                        html.Div(id="room-view-note", className="hint"),
                    ],
                    className="room-centre",
                ),
                html.Div(id="room-panel", className="room-side room-right"),
            ],
            className="room-editor",
        ),
        html.Div(
            [
                html.Span("Flat views drawn as", className="hint"),
                dcc.RadioItems(
                    [{"label": "Drawing", "value": "paper"}, {"label": "Model", "value": "model"}], "paper", id="room-flat-style",
                    inline=True, className="segmented", persistence=True,
                ),
            ],
            className="room-view-bar room-flat-bar",
        ),
        html.Div(
            [
                html.Div(id="room-top", className="room-canvas room-flat"),
                html.Div(id="room-front", className="room-canvas room-flat"),
                html.Div(id="room-side", className="room-canvas room-flat"),
            ],
            className="room-flats",
        ),
        dcc.Store(id="room-model", data=None),
        dcc.Store(id="room-saved", data=None),
        dcc.Store(id="room-selected", data=None),
        dcc.Store(id="room-moved", data=None),
        dcc.Store(id="room-frame", data=None),
        dcc.Store(id="room-sync", data=None),
        desc=f"Click someone or something to edit it, drag to move it. Lengths in {geometry.UNIT_NAMES[units]}.",
        actions=[
            html.Span(id="room-dirty"),
            ui.button("From the calibration", "room-calibrate", small=True),
            ui.button("Draw for a paper", "room-draw", icon_name="download", small=True),
            ui.button("Save the setup", "room-save", primary=True, small=True),
        ],
    )


def add_box():
    return html.Div(
        [
            ui.field("Add a", dcc.Dropdown(
                [{"label": NOUN[kind].capitalize(), "value": kind} for kind in KINDS], "cameras", id="room-add-kind",
                clearable=False, className="picker",
            )),
            ui.field("Named", dcc.Input(id="room-add-name", placeholder="e.g. side", debounce=False)),
            ui.button("Add", "room-add", small=True),
        ],
        className="room-add",
    )


def _colours(roles: list[str]) -> dict[str, str]:
    return {role: PEOPLE_COLOURS[index % len(PEOPLE_COLOURS)] for index, role in enumerate(roles)}


def _read(config, name: str) -> dict:
    path = Path(config.project.setups_dir) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    return data if isinstance(data, dict) else {}


def model_of(config, name: str) -> dict:
    """The saved setup as the editor holds it, read as written so a file that no longer checks can be put right here:
    kind -> name -> settings, with blank numbers as None."""
    data = _read(config, name)
    room = data.get("room") if isinstance(data.get("room"), dict) else {}
    roles = list(config.project.roles)
    size = room.get("size") or [None, None]
    model = {
        "setup": name, "units": config.project.units, "roles": roles, "colours": _colours(roles),
        WALLS: {"width": size[0], "length": size[1]},
    }
    names = {
        "people": roles,
        "cameras": list(data.get("cameras") or {}),
        "microphones": list(data.get("microphones") or {}),
        "things": list(room.get("things") or {}),
    }
    for kind in KINDS:
        model[kind] = {}
        for item in names[kind]:
            placed = (room.get(kind) or {}).get(item) or {}
            at = placed.get("at") or [None, None]
            entry = dict.fromkeys(FIELDS[kind])
            entry.update(x=at[0], y=at[1])
            for key in ("eyes", "height", "facing", "tilt", "lens"):
                if key in entry:
                    entry[key] = placed.get(key)
            if kind == "things":
                size = placed.get("size") or [None, None]
                entry.update(width=size[0], depth=size[1])
            if kind == "cameras":
                entry["calibration"] = placed.get("calibration")
            for use in USES.get(kind, ()):
                entry[use] = list(((data.get(kind) or {}).get(item) or {}).get(use) or [])
            model[kind][str(item)] = entry
    return model


def room_of(model: dict) -> dict:
    """The room section the model describes, leaving out what is blank."""
    room: dict = {}
    walls = model.get(WALLS) or {}
    if walls.get("width") is not None and walls.get("length") is not None:
        room["size"] = [float(walls["width"]), float(walls["length"])]
    for kind in KINDS:
        placed = {}
        for name, entry in model.get(kind, {}).items():
            spot = {}
            if entry.get("x") is not None and entry.get("y") is not None:
                spot["at"] = [float(entry["x"]), float(entry["y"])]
            for key in ("eyes", "height", "facing", "tilt", "lens"):
                if key in FIELDS[kind] and entry.get(key) is not None:
                    spot[key] = float(entry[key])
            if kind == "things" and entry.get("width") is not None and entry.get("depth") is not None:
                spot["size"] = [float(entry["width"]), float(entry["depth"])]
            if kind == "cameras" and entry.get("calibration"):
                spot["calibration"] = entry["calibration"]
            placed[name] = spot
        room[kind] = placed
    return room


def setup_data(model: dict, name: str) -> dict:
    return {
        "name": name,
        "cameras": {device: {use: list(entry.get(use) or []) for use in USES["cameras"]} for device, entry in model["cameras"].items()},
        "microphones": {device: {"hears": list(entry.get("hears") or [])} for device, entry in model["microphones"].items()},
        "room": room_of(model),
    }


def _plain_errors(error: ValidationError) -> str:
    return "; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())


def checked(config, name: str, model: dict) -> SetupSection:
    """The setup the editor describes, refused with a reason a person can act on."""
    if not re.fullmatch(DEVICE_NAME, name or ""):
        raise MacawError(f"{name!r}: a setup's name takes letters, digits, - and _ only")
    quiet = sorted(device for device, entry in model["microphones"].items() if not entry.get("hears"))
    if quiet:
        raise MacawError(f"microphone {', '.join(quiet)} carries nobody's voice: tick whose voices it carries, or remove it")
    try:
        return MacawConfig.model_validate({**config.model_dump(), "setup": setup_data(model, name)}).setup
    except ValidationError as error:
        raise MacawError(_plain_errors(error)) from None


def save(config_path: Path, name: str, model: dict) -> Path:
    config = resolve_config(config_path)
    return write_setup(Path(config.project.setups_dir) / f"{name}.yaml", checked(config, name, model))


def comparable(model: dict | None) -> dict | None:
    """The model without what the editor adds for drawing, to tell whether it differs from the saved one."""
    if model is None:
        return None
    return copy.deepcopy({key: value for key, value in model.items() if key not in ("setup", "units", "roles", "colours")})


def loose_setup(model: dict) -> tuple[SetupSection, RoomSection] | None:
    """The setup and room as far as they read, unchecked across devices, for notes on a setup not yet valid."""
    try:
        room = RoomSection.model_validate(room_of(model))
        cameras = {device: CameraDevice(**{use: entry.get(use) or [] for use in USES["cameras"]}) for device, entry in model["cameras"].items()}
        microphones = {device: MicrophoneDevice.model_construct(hears=entry.get("hears") or []) for device, entry in model["microphones"].items()}
    except ValidationError:
        return None
    return SetupSection.model_construct(cameras=cameras, microphones=microphones, room=room), room


def sight(setup: SetupSection, room: RoomSection, name: str) -> dict | None:
    """What a placed camera sees of each person it is read for: how far away they are, from how far round their face
    it looks (for face and gaze), whether they are in its picture; and, for two bodies or more, how much more one pixel
    spans at the farther one, by which a single scale misstates them. None until the camera is placed."""
    device, camera = setup.cameras.get(name), room.cameras.get(name)
    if device is None or camera is None or geometry.view_of(camera) is None:
        return None
    rows = []
    for person in device.people():
        place = room.people.get(person)
        if place is None or place.at is None or place.eyes is None:
            continue
        rows.append({
            "person": person, "distance": geometry.distance(camera, place),
            "angle": geometry.angle_to(camera, place) if person in device.face or person in device.gaze else None,
            "inside": geometry.in_picture(camera, [*place.at, place.eyes]),
        })
    body = sorted((row for row in rows if row["person"] in device.body), key=lambda row: row["distance"])
    spread = (body[-1]["person"], body[0]["person"], body[-1]["distance"] / body[0]["distance"]) if len(body) > 1 else None
    return {"rows": rows, "spread": spread}


def troubles(seen: dict | None) -> list[str]:
    """What a camera's view as laid out gets wrong: someone outside its picture, a face read from behind."""
    if not seen:
        return []
    found = [f"{row['person']} is outside the picture" for row in seen["rows"] if row["inside"] is False]
    found += [f"{row['person']}'s face is seen from behind" for row in seen["rows"] if row["angle"] is not None and row["angle"] > 90]
    return found


def _copy(model: dict) -> dict:
    return {key: ({name: dict(entry) for name, entry in value.items()} if key in KINDS else value) for key, value in model.items()}


def _number(value, field: str):
    if value in (None, ""):
        return None
    value = float(value)
    return value % 360 if field == "facing" else value


def moved(model: dict, kind: str, name: str, fields: dict) -> dict:
    """The model with the numbers a drag or a typed field gave one item."""
    if kind == WALLS:
        return {**_copy(model), WALLS: {**model[WALLS], **{field: _number(value, field) for field, value in fields.items() if field in FIELDS[WALLS]}}}
    if name not in model.get(kind, {}):
        return model
    model = _copy(model)
    entry = model[kind][name]
    given = {field: _number(value, field) for field, value in fields.items() if field in FIELDS[kind]}
    # numbers changed by hand are no longer the calibration's
    if kind == "cameras" and any(entry.get(field) != value for field, value in given.items()):
        entry["calibration"] = None
    entry.update(given)
    return model


def filled(model: dict, placing, calibration: str) -> dict:
    """The model with each camera's numbers from the calibration: its lens angle always, the rest once the room is tied."""
    model = _copy(model)
    for name, values in placing.values.items():
        entry = model["cameras"][name]
        entry.update({field: values[field] for field in FIELDS["cameras"] if field in values})
        if placing.tied:
            entry["calibration"] = calibration
    return model


def from_calibration(config_path: Path, model: dict):
    """(model, message) after taking the cameras' places from the project's calibration."""
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.extractors.video.pose.triangulate import calibration_path
    from macaw.project.placing import check_rows, floor_of, place_cameras, remarks, summary

    config = resolve_config(config_path)
    calibration = config.pose3d.calibration
    if not calibration:
        return no_update, ui.note("Choose a calibration under Cameras for three dimensions first.", "warn")
    source = calibration_path(config, calibration)
    if not source.is_file():
        return no_update, ui.note(f"The cameras of {calibration} are not found yet: press Find the cameras.", "warn")
    loose = loose_setup(model)
    if loose is None:
        return no_update, ui.note("A number in the room does not read; fix it first.", "bad")
    cameras = read_setup(source, config.pose3d.units)
    floor, problem = floor_of(config, calibration, cameras)
    placing = place_cameras(cameras, loose[1], model["units"], floor)
    said = [ui.note(summary(placing, calibration), "ok" if placing.tied else "warn")]
    said += [ui.note(text, "warn") for text in ([problem] if problem else []) + remarks(placing)]
    rows = check_rows(placing, model["units"])
    if rows:
        said.append(ui.table(["Camera", "What", "Given", "Placed", "Off by", "Agrees"], rows, numeric={"Given", "Placed", "Off by"}))
    return (filled(model, placing, calibration) if placing.values else no_update), html.Div(said)


def used(model: dict, kind: str, name: str, use: str, role: str, on: bool) -> dict:
    """The model with a device read, or not, for one person's body, face, gaze or voice."""
    if name not in model.get(kind, {}) or use not in USES.get(kind, ()):
        return model
    model = _copy(model)
    now = set(model[kind][name].get(use) or [])
    now = now | {role} if on else now - {role}
    model[kind][name][use] = [person for person in model["roles"] if person in now]
    return model


def _taken(model: dict) -> set[str]:
    return {item for kind in KINDS for item in model.get(kind, {})} | set(model["roles"])


def _valid_name(model: dict, kind: str, name: str) -> str:
    name = (name or "").strip()
    if not re.fullmatch(DEVICE_NAME, name):
        raise MacawError(f"a {NOUN[kind]}'s name takes letters, digits, - and _ only")
    if name in RESERVED:
        raise MacawError(f"{name} names something else in macaw; choose another name")
    if name in _taken(model):
        raise MacawError(f"{name} is already in this setup")
    return name


def added(model: dict, kind: str, name: str) -> tuple[dict, str]:
    """The model with a new item, unplaced; a person is a new role, which the caller writes into the project."""
    name = _valid_name(model, kind, name)
    model = _copy(model)
    entry = dict.fromkeys(FIELDS[kind])
    entry.update({use: [] for use in USES.get(kind, ())})
    if kind == "people":
        model["roles"] = [*model["roles"], name]
        model["colours"] = _colours(model["roles"])
    model[kind] = {**model[kind], name: entry}
    return model, name


def renamed(model: dict, kind: str, name: str, new: str) -> tuple[dict, str]:
    if kind == "people":
        raise MacawError("a person's role is renamed under Settings, since every step uses it")
    new = (new or "").strip()
    if new == name:
        return model, name
    new = _valid_name(model, kind, new)
    model = _copy(model)
    model[kind] = {new if key == name else key: entry for key, entry in model[kind].items()}
    return model, new


def removed(model: dict, kind: str, name: str) -> dict:
    if kind == "people":
        raise MacawError("a person is taken out of the project under Settings, since every step uses them; here they can be taken out of the room")
    model = _copy(model)
    model[kind] = {key: entry for key, entry in model[kind].items() if key != name}
    return model


def unplaced(model: dict, kind: str, name: str) -> dict:
    """The model with the item taken out of the room: its numbers blank, what it records kept."""
    return moved(model, kind, name, dict.fromkeys(FIELDS[kind]))


def placed(model: dict, kind: str, name: str) -> dict:
    """The model with the item put in the room: in the middle of the walls, or of what is placed, a little apart from
    what is there already. Heights and directions stay blank until given or dragged."""
    units = model["units"]
    walls = model[WALLS]
    spots = [(entry["x"], entry["y"]) for key in KINDS for entry in model[key].values() if entry.get("x") is not None and entry.get("y") is not None]
    if walls.get("width") is not None and walls.get("length") is not None:
        middle = (walls["width"] / 2, walls["length"] / 2)
    elif spots:
        middle = (sum(x for x, _ in spots) / len(spots), sum(y for _, y in spots) / len(spots))
    else:
        middle = (geometry.to_unit(200.0, units), geometry.to_unit(150.0, units))
    step = geometry.to_unit(_SPREAD_CM, units)
    places = geometry.DECIMALS[units]
    for index in range(50):
        spot = (round(middle[0] + step * (index % 5 - 2) * (index > 0), places), round(middle[1] + step * (index // 5), places))
        if all(abs(spot[0] - x) > step / 2 or abs(spot[1] - y) > step / 2 for x, y in spots):
            break
    return moved(model, kind, name, {"x": spot[0], "y": spot[1]})


def add_role(config_path: Path, role: str) -> None:
    """Writes one more person into project.yaml, among the roles every recording has."""
    config = resolve_config(config_path)
    roles = [*config.project.roles, role]
    text = project_file.with_values(config_path.read_text(encoding="utf-8"), {"project.roles": roles, "project.n_subjects": len(roles)})
    try:
        project_file.check(config_path, text)
    except Exception as error:
        raise MacawError(f"{role} cannot be added: {error}") from None
    project_file.save(config_path, text)


def own_colour(model: dict, kind: str, name: str) -> str | None:
    """A person's colour, or that of the one person a device records alone; None for a shared device or a thing."""
    if kind == "people":
        return model["colours"].get(name)
    entry = model[kind].get(name, {})
    people = {person for use in USES.get(kind, ()) if use != "hears" or kind == "microphones" for person in entry.get(use) or []}
    return model["colours"].get(next(iter(people))) if len(people) == 1 else None


def _is_placed(entry: dict) -> bool:
    return entry.get("x") is not None and entry.get("y") is not None


def room_size(model: dict):
    walls = model[WALLS]
    blank = walls.get("width") is None or walls.get("length") is None
    size = lambda field: dcc.Input(
        id={"type": FIELD, "kind": WALLS, "name": "room", "field": field}, type="number", value=walls.get(field), placeholder=field,
        debounce=True, min=0, step="any",
    )
    return html.Div(
        [
            html.Div("Room", className="room-group-title"),
            html.Div([size("width"), html.Span("×"), size("length"), html.Span(model["units"], className="hint")], className="room-size-inputs"),
            *([html.Div("Set the room's size first.", className="room-size-hint")] if blank else []),
        ],
        className="room-size blank" if blank else "room-size",
    )


def item_list(model: dict | None, selected: dict | None):
    if model is None:
        return html.P("No setup.", className="none")
    read = loose_setup(model)
    groups = [room_size(model)]
    for kind in KINDS:
        entries = []
        for name, entry in model[kind].items():
            chosen = bool(selected) and selected.get("kind") == kind and selected.get("name") == name
            colour = own_colour(model, kind, name)
            wrong = troubles(sight(*read, name)) if read is not None and kind == "cameras" else []
            entries.append(html.Button(
                [
                    html.Span(className=f"room-swatch {kind}", style={"background": colour} if colour else None),
                    html.Span(name, className="room-item-name"),
                    *([html.Span("!", className="room-alert", title="; ".join(wrong))] if wrong else []),
                    *([] if _is_placed(entry) else [html.Span("not placed", className="room-tag")]),
                ],
                id={"type": PICK, "kind": kind, "name": name}, className="room-item chosen" if chosen else "room-item", n_clicks=0,
                draggable="true", **{"data-kind": kind, "data-name": name},
            ))
        groups.append(html.Div(
            [html.Div(TITLES[kind], className="room-group-title"), *(entries or [html.Div(f"No {NOUN[kind]} yet.", className="room-empty")])],
            className="room-group",
        ))
    return groups


def _field(kind: str, name: str, field: str, value, units: str):
    unit = "°" if field in ANGLES else units
    label = HEADS[field].replace(" (°)", "")
    bounds = {"facing": (0, 360), "tilt": (-90, 90), "lens": (1, 179)}.get(field, (None, None))
    return ui.field(
        f"{label} ({unit})",
        dcc.Input(id={"type": FIELD, "kind": kind, "name": name, "field": field}, type="number", value=value, step="any", debounce=True,
                  min=bounds[0], max=bounds[1]),
    )


def panel(model: dict | None, selected: dict | None):
    if model is None:
        return None
    units = model["units"]
    kind, name = (selected or {}).get("kind"), (selected or {}).get("name")
    if kind not in KINDS or name not in model[kind]:
        return html.Div(
            [
                html.H3("Nothing chosen"),
                html.Ul([
                    html.Li("Ring: turn. Knob: tilt. Arrow: height."),
                    html.Li("R: turn 15°. Delete: out of the room."),
                ], className="room-help"),
            ]
        )
    entry = model[kind][name]
    head = [
        html.H3([
            name, " ", ui.pill(NOUN[kind]), *([" ", ui.pill("not placed", "warn")] if not _is_placed(entry) else []),
            *([" ", ui.pill(f"from the calibration {entry['calibration']}", "ok")] if entry.get("calibration") else []),
        ]),
    ]
    if kind != "people":
        head.append(ui.field(
            "Name" + (" (its folder in each recording)" if kind in USES else ""),
            dcc.Input(id={"type": RENAME, "kind": kind, "name": name}, value=name, debounce=True),
        ))
    numbers = [_field(kind, name, field, entry.get(field), units) for field in FIELDS[kind]]
    uses = []
    if kind in USES:
        rows = [
            [
                html.Span([html.Span(className="room-swatch", style={"background": model["colours"].get(role)}), role], className="room-who"),
                *[
                    dcc.Checklist(
                        [{"label": "", "value": role}], [role] if role in (entry.get(use) or []) else [],
                        id={"type": USE, "kind": kind, "name": name, "use": use, "role": role}, className="room-tick",
                    )
                    for use in USES[kind]
                ],
            ]
            for role in model["roles"]
        ]
        uses = [
            html.Div("Whom it records" if kind == "cameras" else "Whose voices it carries", className="room-group-title"),
            ui.table(["", *[USE_LABELS[use] for use in USES[kind]]], rows),
        ]
    if not _is_placed(entry):
        head.append(html.P("Click in the plan or a view to put it there.", className="room-size-hint"))
    act = lambda what: {"type": ACT, "kind": kind, "name": name, "act": what}
    buttons = [
        ui.button("Take out of the room", act("unplace"), small=True) if _is_placed(entry) else ui.button("Put in the middle", act("place"), small=True),
        *([ui.button(f"Remove this {NOUN[kind]}", act("remove"), small=True)] if kind != "people" else []),
    ]
    seen = seen_table(model, name) if kind == "cameras" else []
    return html.Div([*head, *uses, *seen, html.Div(numbers, className="room-numbers"), html.Div(buttons, className="buttons")])


def seen_table(model: dict, name: str) -> list:
    read = loose_setup(model)
    seen = sight(*read, name) if read is not None else None
    if not seen or not seen["rows"]:
        return []
    places = geometry.DECIMALS[model["units"]]
    flag = lambda text, bad: html.Span(text, className="room-bad") if bad else text
    rows = [
        [
            row["person"],
            f"{row['distance']:.{places}f}",
            "" if row["angle"] is None else flag(f"{row['angle']:.0f}°" + (" behind" if row["angle"] > 90 else ""), row["angle"] > 90),
            "" if row["inside"] is None else flag("yes" if row["inside"] else "no", not row["inside"]),
        ]
        for row in seen["rows"]
    ]
    parts = [
        html.Div("What it sees", className="room-group-title"),
        ui.table(["", f"Away ({model['units']})", "Face", "In picture"], rows, numeric={f"Away ({model['units']})"}),
    ]
    if seen["spread"] and seen["spread"][2] > 1.005:
        far, near, ratio = seen["spread"]
        parts.append(html.P(f"A pixel spans {ratio:.2f}× as much at {far} as at {near}.", className="room-bad" if ratio > 1.1 else "hint"))
    return parts


def draw_figure(config_path: Path, name: str):
    from macaw.interface import cli_bridge
    from macaw.interface.files import file_url

    argv = cli_bridge.build_command(["draw-room"], config_path=config_path, extra_flags=["--setup", name])
    code, text = cli_bridge.run_now(argv, cwd=config_path.parent)
    config = resolve_config(config_path)
    path = Path(config.project.setups_dir) / f"{name}.png"
    link = [html.A(f"Open {path.name}", href=file_url(path), target="_blank")] if code == 0 and path.is_file() else []
    return [ui.command(argv), ui.output(code, text), *link]


def changed_by(config_path: Path, model: dict, trigger, value, moved_by=None, add_kind=None, add_name=None):
    """(model, selection, message) after one change in the editor; no_update where nothing changes."""
    unchanged = lambda after: after if after != model else no_update
    if trigger == "room-moved":
        if not moved_by:
            return no_update, no_update, no_update
        if moved_by.get("op") == "unplace":
            return unchanged(unplaced(model, moved_by["kind"], moved_by["name"])), no_update, None
        return unchanged(moved(model, moved_by["kind"], moved_by["name"], moved_by.get("fields") or {})), no_update, None
    if trigger == "room-add":
        if not value:
            return no_update, no_update, no_update
        after, new = added(model, add_kind, add_name)
        said = None
        if add_kind == "people":
            add_role(config_path, new)
            said = ui.note(f"{new} is now one of the project's people, in every recording; say which devices record them and save the setup.", "ok")
        return after, {"kind": add_kind, "name": new}, said
    kind, name = trigger.get("kind"), trigger.get("name")
    if trigger["type"] == FIELD:
        return unchanged(moved(model, kind, name, {trigger["field"]: value})), no_update, None
    if trigger["type"] == USE:
        return unchanged(used(model, kind, name, trigger["use"], trigger["role"], bool(value))), no_update, None
    if trigger["type"] == RENAME:
        if value is None or value == name:
            return no_update, no_update, no_update
        after, new = renamed(model, kind, name, value)
        return after, {"kind": kind, "name": new}, None
    # buttons drawn afresh report themselves without a click
    if not value:
        return no_update, no_update, no_update
    if trigger["type"] == PICK:
        return no_update, {"kind": kind, "name": name}, None
    act = trigger["act"]
    if act == "place":
        return placed(model, kind, name), no_update, None
    if act == "unplace":
        return unplaced(model, kind, name), no_update, None
    return removed(model, kind, name), None, None


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("room-model", "data"), Output("room-saved", "data"), Output("room-selected", "data", allow_duplicate=True),
        Input("setup-pick", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def load(name):
        model = model_of(resolve_config(config_path), name)
        return model, comparable(model), None

    @app.callback(
        Output("room-model", "data", allow_duplicate=True), Output("room-selected", "data", allow_duplicate=True),
        Output("room-message", "children", allow_duplicate=True),
        Input("room-moved", "data"),
        Input({"type": FIELD, "kind": ALL, "name": ALL, "field": ALL}, "value"),
        Input({"type": USE, "kind": ALL, "name": ALL, "use": ALL, "role": ALL}, "value"),
        Input({"type": RENAME, "kind": ALL, "name": ALL}, "value"),
        Input({"type": PICK, "kind": ALL, "name": ALL}, "n_clicks"),
        Input({"type": ACT, "kind": ALL, "name": ALL, "act": ALL}, "n_clicks"),
        Input("room-add", "n_clicks"),
        State("room-model", "data"), State("room-add-kind", "value"), State("room-add-name", "value"),
        prevent_initial_call=True,
    )
    def edit(moved_by, _values, _uses, _names, _picks, _acts, _add, model, add_kind, add_name):
        if model is None or ctx.triggered_id is None:
            return no_update, no_update, no_update
        try:
            return changed_by(config_path, model, ctx.triggered_id, ctx.triggered[0]["value"], moved_by, add_kind, add_name)
        except MacawError as error:
            return no_update, no_update, ui.note(str(error), "bad")

    @app.callback(
        Output("room-list", "children"), Output("room-panel", "children"), Output("room-dirty", "children"),
        Output("room-view-camera", "options"),
        Input("room-model", "data"), Input("room-selected", "data"), Input("room-saved", "data"),
    )
    def render(model, selected, saved):
        dirty = ui.pill("unsaved changes", "warn") if model is not None and comparable(model) != saved else None
        cameras = list(model["cameras"]) if model else []
        return item_list(model, selected), panel(model, selected), dirty, cameras

    @app.callback(
        Output("room-message", "children", allow_duplicate=True), Output("room-saved", "data", allow_duplicate=True),
        Output("setup-files", "children", allow_duplicate=True),
        Input("room-save", "n_clicks"), State("room-model", "data"), State("setup-pick", "value"),
        prevent_initial_call=True,
    )
    def keep(clicks, model, name):
        from macaw.interface.pages.setup import holdings

        if not clicks or model is None:
            return no_update, no_update, no_update
        try:
            path = save(config_path, name, model)
        except MacawError as error:
            return ui.note(str(error), "bad"), no_update, no_update
        return ui.note(f"Saved {path.parent.name}/{path.name}.", "ok"), comparable(model), holdings(config_path)

    @app.callback(
        Output("room-model", "data", allow_duplicate=True), Output("room-message", "children", allow_duplicate=True),
        Input("room-calibrate", "n_clicks"), State("room-model", "data"),
        prevent_initial_call=True,
    )
    def calibrate(clicks, model):
        if not clicks or model is None:
            return no_update, no_update
        try:
            return from_calibration(config_path, model)
        except MacawError as error:
            return no_update, ui.note(str(error), "bad")

    @app.callback(
        Output("room-message", "children", allow_duplicate=True),
        Input("room-draw", "n_clicks"), State("room-model", "data"), State("room-saved", "data"), State("setup-pick", "value"),
        prevent_initial_call=True,
    )
    def draw(clicks, model, saved, name):
        if not clicks:
            return no_update
        if comparable(model) != saved:
            return ui.note("The drawing is made from the saved setup: save it first.", "warn")
        return draw_figure(config_path, name)

    @app.callback(
        Output("room-view-camera-tools", "style"), Output("room-view-camera", "value"), Output("room-view-recording", "options"),
        Output("room-view-recording", "value"),
        Input("room-view-mode", "value"), State("room-view-camera", "value"), State("room-view-camera", "options"), State("room-view-recording", "value"),
    )
    def view_tools(mode, camera, cameras, rid):
        from macaw.execution.runner import resolve_recordings

        if mode != "camera":
            return {"display": "none"}, no_update, no_update, no_update
        rids = resolve_recordings(resolve_config(config_path), None)
        names = [item if isinstance(item, str) else item.get("value") for item in (cameras or [])]
        return {}, camera if camera in names else (names[0] if names else None), rids, rid if rid in rids else (rids[0] if rids else None)

    @app.callback(
        Output("room-frame", "data"), Output("room-view-note", "children"), Output("room-selected", "data", allow_duplicate=True),
        Input("room-view-mode", "value"), Input("room-view-camera", "value"), Input("room-view-recording", "value"), Input("room-view-at", "value"),
        prevent_initial_call=True,
    )
    def frame(mode, camera, rid, at):
        from macaw.interface.pages.naming import frame_of

        if mode != "camera" or not camera:
            return None, None, no_update
        chosen = {"kind": "cameras", "name": camera}
        if not rid:
            return {"camera": camera}, "No recording to take a frame from.", chosen
        shown, problem = frame_of(config_path, rid, camera, float(at or 0.0))
        if problem:
            return {"camera": camera}, problem, chosen
        return shown, "Drag to aim, scroll for the lens angle, until the drawing lies on the picture.", chosen

    app.clientside_callback(
        """function (model, selected, frame, mode, style) {
            const state = {model: model, selected: selected, frame: frame, mode: mode, style: style};
            if (window.macawRoom) { window.macawRoom.update(state); } else { window.macawRoomPending = state; }
            return window.dash_clientside.no_update;
        }""",
        Output("room-sync", "data"),
        Input("room-model", "data"), Input("room-selected", "data"), Input("room-frame", "data"), Input("room-view-mode", "value"),
        Input("room-flat-style", "value"),
    )
