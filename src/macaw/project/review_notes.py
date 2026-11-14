"""What a person noted about a recording while reviewing it: tags from the project's list and free text, with who and when."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["recording_id", "tags", "text", "by", "at"]


def read_notes(path: Path) -> list[dict]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _write(path: Path, notes: list[dict]) -> None:
    path = Path(path)
    if not notes:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")


def add_note(path: Path, tags: list[str], text: str, by: str) -> dict:
    note = {"id": uuid.uuid4().hex[:8], "tags": list(tags), "text": text.strip(), "by": by, "at": datetime.now().isoformat(timespec="seconds")}
    _write(path, [*read_notes(path), note])
    return note


def remove_note(path: Path, note_id: str) -> bool:
    notes = read_notes(path)
    kept = [note for note in notes if note["id"] != note_id]
    _write(path, kept)
    return len(kept) < len(notes)


def note_tags(notes: list[dict]) -> list[str]:
    """Every tag the recording's notes carry, once, in the order first used."""
    return list(dict.fromkeys(tag for note in notes for tag in note["tags"]))


def notes_table(layout, recording_ids: list[str]) -> pd.DataFrame:
    rows = [
        {"recording_id": rid, "tags": "; ".join(note["tags"]), "text": note["text"], "by": note["by"], "at": note["at"]}
        for rid in recording_ids for note in read_notes(layout.review_notes_path(rid))
    ]
    return pd.DataFrame(rows, columns=COLUMNS)
