"""Recording folders with one folder per device, and the setup file saying what each holds."""

from __future__ import annotations

from pathlib import Path

import yaml

EVERYONE = ("parent", "child")


def camera(people=EVERYONE, *, hears=()) -> dict:
    """A camera read for body, face and gaze of these people."""
    return {"body": list(people), "face": list(people), "gaze": list(people), "hears": list(hears)}


def mics(people=EVERYONE) -> dict:
    """One microphone per person, named <person>_mic."""
    return {f"{person}_mic": {"hears": [person]} for person in people}


def write_setup(project_yaml: Path, *, cameras: dict | None = None, microphones: dict | None = None, name: str = "main") -> Path:
    folder = Path(project_yaml).parent / "setups"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"cameras": cameras or {}, "microphones": microphones or {}}, sort_keys=False), encoding="utf-8")
    return path


def put(raw: Path, recording: str, device: str, file_name: str, data: bytes = b"") -> Path:
    folder = Path(raw) / recording / device
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    path.write_bytes(data)
    return path
