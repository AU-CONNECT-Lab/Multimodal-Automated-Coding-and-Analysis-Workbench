"""A change of the project's unit converts the lengths people already typed: the scale written by hand, the board's
square, the 3D gesture and measure thresholds, in project.yaml and in each recording's own file, the room of every setup with
what was measured in it, and the heights table. Measurements of an object keep the unit they were typed in, so they need nothing."""

from __future__ import annotations

import io
from pathlib import Path

import yaml

from macaw.interface import project_file
from macaw.project.room import CM_PER_UNIT

LENGTHS = (
    "scale.per_px", "pose3d.board.square", "pose3d.board.marker",
    "kinematics.gestures.still_speed_3d", "kinematics.gestures.peak_prominence_3d",
)
BY_ROLE = ("scale.per_px_by_role",)


def _number(value: float, factor: float):
    converted = round(float(value) * factor, 4)
    return int(converted) if converted.is_integer() else converted


def _get(data: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def settings_text(text: str, factor: float) -> str:
    """A settings file's text with its lengths converted, comments kept; unchanged when it holds none."""
    data = yaml.safe_load(text) or {}
    found = {key: _number(value, factor) for key in LENGTHS if isinstance(value := _get(data, key), (int, float))}
    found.update({key: {role: _number(each, factor) for role, each in value.items()} for key in BY_ROLE if isinstance(value := _get(data, key), dict) and value})
    text = project_file.with_values(text, found) if found else text
    return _measure_thresholds(text, factor)


def _measure_thresholds(text: str, factor: float) -> str:
    """below_3d of each built-in measure of a length or speed, which is typed in the project's unit."""
    from ruamel.yaml import YAML

    from macaw.extractors.video.pose.kinematics import BUILTIN_MEASURES

    def typed_in_units(item) -> bool:
        return (
            isinstance(item, dict) and isinstance(item.get("below_3d"), (int, float))
            and item.get("method", "distance") in BUILTIN_MEASURES and not item.get("per_body_size")
        )

    measures = _get(yaml.safe_load(text) or {}, "kinematics.measures")
    if not isinstance(measures, list) or not any(typed_in_units(item) for item in measures):
        return text
    loader = YAML()
    loader.preserve_quotes = True
    loader.width = 4096
    loader.indent(mapping=2, sequence=4, offset=2)
    data = loader.load(text)
    for item in data["kinematics"]["measures"]:
        if typed_in_units(item):
            item["below_3d"] = _number(item["below_3d"], factor)
    out = io.StringIO()
    loader.dump(data, out)
    return out.getvalue()


def _place(place: dict, factor: float) -> dict:
    place = dict(place or {})
    for key in ("at", "size"):
        if isinstance(place.get(key), list):
            place[key] = [_number(value, factor) for value in place[key]]
    for key in ("eyes", "height"):
        if isinstance(place.get(key), (int, float)):
            place[key] = _number(place[key], factor)
    # what a person measured for a camera placed by the calibration, which the next placing fits to
    if isinstance(place.get("measured"), dict):
        place["measured"] = _place(place["measured"], factor)
    return place


def room_data(room: dict, factor: float) -> dict:
    out = dict(room)
    if isinstance(room.get("size"), list):
        out["size"] = [_number(value, factor) for value in room["size"]]
    for kind in ("people", "cameras", "microphones", "things"):
        placed = {name: _place(place, factor) for name, place in (room.get(kind) or {}).items()}
        if placed:
            out[kind] = placed
    return out


def heights_table(path: Path, factor: float) -> bool:
    """The heights table's heights converted in place; False when there is none or it holds no number."""
    import pandas as pd

    if not path.is_file():
        return False
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    column = next((name for name in frame.columns if str(name).strip() == "height"), None)
    if column is None:
        return False
    numbers = pd.to_numeric(frame[column].str.strip(), errors="coerce")
    if numbers.isna().all():
        return False
    frame[column] = [str(_number(value, factor)) if pd.notna(value) else text for value, text in zip(numbers, frame[column])]
    frame.to_csv(path, index=False)
    return True


def convert_entered(config, project_yaml: Path, old: str, new: str) -> list[Path]:
    """Converts every length typed in old to new; config is the project's settings after the change. Returns the files changed."""
    factor = CM_PER_UNIT[old] / CM_PER_UNIT[new]
    changed = []
    recordings = Path(config.project.recording_config_dir)
    for path in [project_yaml, *(sorted(recordings.glob("*.yaml")) if recordings.is_dir() else [])]:
        text = path.read_text(encoding="utf-8")
        converted = settings_text(text, factor)
        if converted != text:
            project_file.save(path, converted)
            changed.append(path)
    setups = Path(config.project.setups_dir)
    for path in sorted(setups.glob("*.yaml")) if setups.is_dir() else []:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("room"), dict):
            continue
        data["room"] = room_data(data["room"], factor)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None), encoding="utf-8")
        changed.append(path)
    heights = Path(config.project.heights)
    if heights_table(heights, factor):
        changed.append(heights)
    return changed
