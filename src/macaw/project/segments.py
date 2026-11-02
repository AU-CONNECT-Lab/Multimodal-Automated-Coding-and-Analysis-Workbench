"""Named parts of a recording, from the project's segments table, that each step's summaries are worked out again over."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError

COLUMNS = ("recording", "segment", "start_s", "end_s")
SUMMARY_STEP = "segments.summary"


@dataclass(frozen=True)
class Segment:
    name: str
    start_s: float
    end_s: float


def segments_path(config) -> Path:
    return Path(config.project.segments)


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(COLUMNS)}")
    frame = frame.apply(lambda column: column.str.strip())
    return frame[(frame != "").any(axis=1)].reset_index(drop=True)


def _seconds(text: str, path: Path, line: int, column: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ConfigError(f"{path} line {line}: {column} '{text}' is not a number of seconds") from None
    if not math.isfinite(value):
        raise ConfigError(f"{path} line {line}: {column} '{text}' is not a number of seconds")
    return value


def read_segments(config) -> dict[str, list[Segment]]:
    """Each recording's segments in time order; no table means none."""
    path = segments_path(config)
    if not path.is_file():
        return {}
    found: dict[str, list[Segment]] = {}
    for index, row in _read_frame(path).iterrows():
        line = index + 2   # the header is line 1
        recording, name = row["recording"], row["segment"]
        if not recording or not name:
            raise ConfigError(f"{path} line {line}: a segment needs both a recording and a name")
        start = _seconds(row["start_s"], path, line, "start_s")
        end = _seconds(row["end_s"], path, line, "end_s")
        if start < 0 or end <= start:
            raise ConfigError(f"{path} line {line}: {name} has to end after it starts, at 0 s or later ({start:g} to {end:g} s)")
        if any(segment.name == name for segment in found.get(recording, [])):
            raise ConfigError(f"{path} line {line}: {recording} has two segments named {name}")
        found.setdefault(recording, []).append(Segment(name, start, end))
    return {recording: sorted(segments, key=lambda s: (s.start_s, s.name)) for recording, segments in found.items()}


def segments_of(config, recording_id: str) -> list[Segment]:
    return read_segments(config).get(recording_id, [])


def check_recordings(config, recording_ids: list[str]) -> None:
    """A recording the table names that the project does not have is a typo, not something to skip."""
    unknown = sorted(set(read_segments(config)) - set(recording_ids) - set(config.project.excluded_recordings))
    if unknown:
        raise ConfigError(
            f"{segments_path(config)} names recordings this project does not have: {', '.join(unknown)}"
        )


def set_segment(config, recording_id: str, segment: Segment) -> Path:
    """Adds the segment, or moves the one of that name, keeping any other columns people added."""
    if not segment.name.strip():
        raise ConfigError("A segment needs a name")
    if segment.start_s < 0 or segment.end_s <= segment.start_s:
        raise ConfigError(f"{segment.name} has to end after it starts, at 0 s or later ({segment.start_s:g} to {segment.end_s:g} s)")
    path = segments_path(config)
    frame = _read_frame(path) if path.is_file() else pd.DataFrame(columns=list(COLUMNS))
    same = (frame["recording"] == recording_id) & (frame["segment"] == segment.name)
    row = {"recording": recording_id, "segment": segment.name, "start_s": f"{segment.start_s:g}", "end_s": f"{segment.end_s:g}"}
    if same.any():
        for column, value in row.items():
            frame.loc[same, column] = value
    else:
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True).fillna("")
    _write_frame(frame, path)
    return path


def remove_segment(config, recording_id: str, name: str) -> Path:
    path = segments_path(config)
    frame = _read_frame(path) if path.is_file() else pd.DataFrame(columns=list(COLUMNS))
    same = (frame["recording"] == recording_id) & (frame["segment"] == name)
    if not same.any():
        raise ConfigError(f"{path} has no segment {name} for {recording_id}")
    _write_frame(frame[~same], path)
    return path


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    order = frame.assign(_start=pd.to_numeric(frame["start_s"], errors="coerce")).sort_values(
        ["recording", "_start", "segment"], kind="stable"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    order.drop(columns="_start").to_csv(path, index=False, encoding="utf-8")


def clip_intervals(frame: pd.DataFrame, start_s: float, end_s: float, *, low: str = "t_start", high: str = "t_end") -> pd.DataFrame:
    """The rows that overlap the span, cut at its edges."""
    if frame.empty:
        return frame.copy()
    starts, ends = frame[low].to_numpy(dtype=float), frame[high].to_numpy(dtype=float)
    out = frame[(ends > start_s) & (starts < end_s)].copy()
    out[low] = np.maximum(out[low].to_numpy(dtype=float), start_s)
    out[high] = np.minimum(out[high].to_numpy(dtype=float), end_s)
    return out.reset_index(drop=True)


def clip_points(frame: pd.DataFrame, start_s: float, end_s: float, *, column: str = "t") -> pd.DataFrame:
    """The rows whose instant falls inside the span, its start included and its end not."""
    if frame.empty:
        return frame.copy()
    values = frame[column].to_numpy(dtype=float)
    return frame[(values >= start_s) & (values < end_s)].reset_index(drop=True)
