"""What people said about the things followed in one recording: where one starts, that one is not on a camera, that a start found by name is right."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from macaw.store._io import swap_in

STARTS, ABSENT, KEPT = "starts", "absent", "kept"


def read_decisions(path: Path) -> dict:
    if not path.is_file():
        return {STARTS: {}, ABSENT: {}, KEPT: {}}
    found = json.loads(path.read_text(encoding="utf-8"))
    return {key: found.get(key, {}) for key in (STARTS, ABSENT, KEPT)}


def write_decisions(path: Path, decisions: dict) -> Path:
    kept = {key: {camera: names for camera, names in decisions.get(key, {}).items() if names} for key in (STARTS, ABSENT, KEPT)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(kept, indent=2, ensure_ascii=False), encoding="utf-8")
    swap_in(temporary, path)
    return path


def stamp(by: str) -> str:
    return f"{by} {datetime.now().isoformat(timespec='seconds')}"


def tracking_key(decisions: dict) -> str:
    """What of the decisions changes the tracking: the starts people gave and the cameras they said a thing is not on."""
    return json.dumps({STARTS: decisions[STARTS], ABSENT: decisions[ABSENT]}, sort_keys=True)


def add_start(decisions: dict, camera: str, name: str, t: float, x: float, y: float, by: str) -> dict:
    """A further point on the thing at another moment; one at the same moment replaces it."""
    points = [point for point in decisions[STARTS].get(camera, {}).get(name, []) if abs(point["t"] - t) > 1e-6]
    points.append({"t": round(float(t), 3), "x": round(float(x), 1), "y": round(float(y), 1), "by": stamp(by)})
    decisions[STARTS].setdefault(camera, {})[name] = sorted(points, key=lambda point: point["t"])
    decisions[ABSENT].get(camera, {}).pop(name, None)
    decisions[KEPT].get(camera, {}).pop(name, None)
    return decisions


def set_absent(decisions: dict, camera: str, name: str, by: str) -> dict:
    decisions[ABSENT].setdefault(camera, {})[name] = stamp(by)
    decisions[STARTS].get(camera, {}).pop(name, None)
    decisions[KEPT].get(camera, {}).pop(name, None)
    return decisions


def keep(decisions: dict, camera: str, name: str, start: dict, by: str) -> dict:
    decisions[KEPT].setdefault(camera, {})[name] = {"start": start, "by": stamp(by)}
    return decisions


def clear(decisions: dict, camera: str, name: str) -> dict:
    for key in (STARTS, ABSENT, KEPT):
        decisions[key].get(camera, {}).pop(name, None)
    return decisions
