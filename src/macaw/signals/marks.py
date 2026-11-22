"""Spans a person marks on a recording's clock while looking at its signals, such as a child touching the cap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macaw import people_tables
from macaw.exceptions import ConfigError
from macaw.project.segments import seconds_cell, seconds_text

COLUMNS = ("recording", "mark", "start_s", "end_s", "note")
# the columns a table must have; note may be left out
NEEDED = COLUMNS[:4]


@dataclass(frozen=True)
class Mark:
    name: str
    start_s: float
    end_s: float
    note: str = ""


def marks_path(config) -> Path:
    return Path(config.project.marks)


def _read_frame(path: Path) -> pd.DataFrame:
    frame = people_tables.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in NEEDED if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(COLUMNS)}")
    if "note" not in frame.columns:
        frame["note"] = ""
    frame = frame.apply(lambda column: column.str.strip())
    return frame[(frame != "").any(axis=1)].reset_index(drop=True)


def read_marks(config) -> dict[str, list[Mark]]:
    """Each recording's marks in time order; no table means none. Many marks may share a name."""
    path = marks_path(config)
    if not path.is_file():
        return {}
    found: dict[str, list[Mark]] = {}
    for index, row in _read_frame(path).iterrows():
        line = index + 2   # the header is line 1
        if not row["recording"] or not row["mark"]:
            raise ConfigError(f"{path} line {line}: a mark needs both a recording and a name")
        start, end = seconds_cell(row["start_s"], path, line, "start_s"), seconds_cell(row["end_s"], path, line, "end_s")
        if start < 0 or end <= start:
            raise ConfigError(f"{path} line {line}: {row['mark']} has to end after it starts, at 0 s or later ({start:g} to {end:g} s)")
        found.setdefault(row["recording"], []).append(Mark(row["mark"], start, end, row["note"]))
    return {recording: sorted(marks, key=lambda m: (m.start_s, m.name)) for recording, marks in found.items()}


def marks_of(config, recording_id: str) -> list[Mark]:
    return read_marks(config).get(recording_id, [])


def _same(frame: pd.DataFrame, recording_id: str, mark: Mark) -> pd.Series:
    return (
        (frame["recording"] == recording_id) & (frame["mark"] == mark.name)
        & (pd.to_numeric(frame["start_s"], errors="coerce").sub(mark.start_s).abs() < 1e-6)
        & (pd.to_numeric(frame["end_s"], errors="coerce").sub(mark.end_s).abs() < 1e-6)
    )


def set_mark(config, recording_id: str, mark: Mark, old: Mark | None = None) -> Path:
    """Adds the mark, or changes the one that was old, keeping any other columns people added."""
    if not mark.name.strip():
        raise ConfigError("A mark needs a name")
    if mark.start_s < 0 or mark.end_s <= mark.start_s:
        raise ConfigError(f"{mark.name} has to end after it starts, at 0 s or later ({mark.start_s:g} to {mark.end_s:g} s)")
    path = marks_path(config)
    frame = _read_frame(path) if path.is_file() else pd.DataFrame(columns=list(COLUMNS))
    row = {"recording": recording_id, "mark": mark.name.strip(), "start_s": seconds_text(mark.start_s),
           "end_s": seconds_text(mark.end_s), "note": mark.note.strip()}
    same = _same(frame, recording_id, old) if old is not None else pd.Series(False, index=frame.index)
    if old is not None and not same.any():
        raise ConfigError(f"{path} has no mark {old.name} from {old.start_s:g} to {old.end_s:g} s in {recording_id}; it may have been changed elsewhere")
    if same.any():
        first = same[same].index[0]
        for column, value in row.items():
            frame.loc[first, column] = value
    else:
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True).fillna("")
    _write_frame(frame, path)
    return path


def remove_mark(config, recording_id: str, mark: Mark) -> Path:
    path = marks_path(config)
    frame = _read_frame(path) if path.is_file() else pd.DataFrame(columns=list(COLUMNS))
    same = _same(frame, recording_id, mark)
    if not same.any():
        raise ConfigError(f"{path} has no mark {mark.name} from {mark.start_s:g} to {mark.end_s:g} s in {recording_id}")
    _write_frame(frame.drop(index=same[same].index[0]), path)
    return path


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    order = frame.assign(_start=pd.to_numeric(frame["start_s"], errors="coerce")).sort_values(
        ["recording", "_start", "mark"], kind="stable"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    order.drop(columns="_start").to_csv(path, index=False, encoding="utf-8")
