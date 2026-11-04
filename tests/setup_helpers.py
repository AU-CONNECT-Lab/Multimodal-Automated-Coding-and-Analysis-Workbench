"""Recording folders with one folder per device, and the setup file saying what each holds."""

from __future__ import annotations

from pathlib import Path

import yaml

EVERYONE = ("parent", "child")


def camera(people=EVERYONE, *, uses=("body", "face", "gaze"), hears=()) -> dict:
    """A camera read for these people: for body, face and gaze unless said otherwise."""
    return {**{use: list(people) for use in uses}, "hears": list(hears)}


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


def video_file(path, people=EVERYONE, *, uses=("body", "face", "gaze"), **fields):
    """A camera named like its file and read for these people."""
    from macaw.project.recording import MediaFile

    path, people = Path(path), tuple(people)
    return MediaFile(
        path=path, kind="video", device=path.name, uses={use: people for use in uses},
        subject_id=people[0] if len(people) == 1 else None, **fields,
    )


def recording_of(media, subjects=EVERYONE, *, recording_id="sub-003_ses-12mo", group_id="003", session_id="12mo", **fields):
    """A recording of these media; subjects are role names or Subject objects."""
    from macaw.project.recording import Recording, Subject

    people = [Subject(subject_id=subject, role=subject) if isinstance(subject, str) else subject for subject in subjects]
    return Recording(
        recording_id=recording_id, root=Path("."), subjects=people, media=list(media), group_id=group_id, session_id=session_id, **fields,
    )
