"""Who is who on each camera of a recording, as a person clicked it: a moment of the camera's file, counted from its
first picture, and a point on each person, as shares of the picture's width and height."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def read_names(path: Path) -> dict[str, list[dict]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_names(path: Path, names: dict[str, list[dict]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({camera: clicks for camera, clicks in names.items() if clicks}, indent=2), encoding="utf-8")
    return path


def add_clicks(path: Path, camera: str, at_s: float, points: dict[str, tuple[float, float]], by: str) -> dict[str, list[dict]]:
    """Clicks at one moment join the camera's earlier ones; clicking a person again at the same moment replaces them."""
    names = read_names(path)
    stamp = f"{by} {datetime.now().isoformat(timespec='seconds')}"
    kept = [click for click in names.get(camera, []) if not (click["person"] in points and click["at"] == at_s)]
    names[camera] = kept + [{"person": person, "at": at_s, "x": x, "y": y, "by": stamp} for person, (x, y) in points.items()]
    write_names(path, names)
    return names


def remove_clicks(path: Path, camera: str, at_s: float, person: str | None = None) -> int:
    """Takes back the clicks at one moment of a camera, or only that person's; returns how many were taken back."""
    names = read_names(path)
    clicks = names.get(camera, [])
    kept = [click for click in clicks if not (float(click["at"]) == at_s and person in (None, click["person"]))]
    names[camera] = kept
    write_names(path, names)
    return len(clicks) - len(kept)


def clear_clicks(path: Path, camera: str) -> dict[str, list[dict]]:
    names = read_names(path)
    names.pop(camera, None)
    write_names(path, names)
    return names
