"""A person's confirmation of which file belongs to whom, recorded once per project."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from macaw.config.schema import MacawConfig
from macaw.exceptions import MacawError
from macaw.project.recording import Recording


def identity_basis(config: MacawConfig) -> dict:
    """What a confirmation covers: the naming rule and the people, so new recordings that follow it need no new one."""
    return {
        "media_pattern": config.project.media_pattern,
        "subjects": list(config.project.roles[: config.project.n_subjects]),
    }


def pairing(recording: Recording) -> dict:
    """File names per subject, plus the files that belong to nobody in particular."""
    owned = {
        subject.subject_id: [item.path.name for item in recording.media if item.subject_id == subject.subject_id]
        for subject in recording.subjects
    }
    shared = [item.path.name for item in recording.media if item.subject_id is None]
    return {"recording_id": recording.recording_id, "subjects": owned, "shared": shared}


def read_confirmation(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MacawError(f"{path} is not readable: {error}") from error


def write_confirmation(path: Path, basis: dict, pairings: list[dict], confirmed_by: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "basis": basis,
        "confirmed_by": confirmed_by,
        "confirmed_at": datetime.now().isoformat(timespec="seconds"),
        "pairings": pairings,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
