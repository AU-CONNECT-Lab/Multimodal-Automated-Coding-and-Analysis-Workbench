"""Equally spaced nominal sequences.

The long table cannot hold one: its value column is float64, and a state like AB is not a
number. Recurrence analysis and the categorical time-series tools around it eat exactly this
shape, so it is an export target of its own rather than another set of feature rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

EMPTY_STATE = "none"
SEQUENCE_COLUMNS = ("recording_id", "series", "t", "state")
JOINT_COLUMNS = ("recording_id", "t", "state")
DEFAULT_SERIES_COLUMNS = ("subject_id", "tier")


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([], dtype="object") for column in columns})


def to_state_sequence(
    intervals: pd.DataFrame,
    *,
    step_s: float,
    state_column: str = "label",
    series_columns: Sequence[str] = DEFAULT_SERIES_COLUMNS,
    t_start: float | None = None,
    t_end: float | None = None,
    empty_state: str = EMPTY_STATE,
) -> pd.DataFrame:
    """Labelled intervals resampled onto a fixed grid: recording_id | series | t | state.

    A cell takes the state covering most of it, ties going alphabetically, which is the rule the
    agreement grid already uses. Cells nobody covered get empty_state rather than being left out,
    because a sequence with holes is not a sequence.
    """
    if step_s <= 0:
        raise ValueError("step_s must be positive")
    if intervals is None or intervals.empty:
        return _empty(SEQUENCE_COLUMNS)

    frame = intervals.dropna(subset=[state_column]).copy()
    starts = pd.to_numeric(frame["t_start"], errors="coerce")
    ends = pd.to_numeric(frame["t_end"], errors="coerce")
    frame = frame[np.isfinite(starts) & np.isfinite(ends) & (ends > starts)]
    if frame.empty:
        return _empty(SEQUENCE_COLUMNS)

    frame["_start"] = pd.to_numeric(frame["t_start"], errors="coerce")
    frame["_end"] = pd.to_numeric(frame["t_end"], errors="coerce")
    origin = float(t_start) if t_start is not None else float(frame["_start"].min())
    stop = float(t_end) if t_end is not None else float(frame["_end"].max())
    n_cells = int(np.ceil((stop - origin) / step_s - 1e-9))
    if n_cells <= 0:
        return _empty(SEQUENCE_COLUMNS)

    if "recording_id" not in frame.columns:
        frame["recording_id"] = ""
    present = [column for column in series_columns if column in frame.columns]

    rows: list[tuple] = []
    for key, group in frame.groupby(["recording_id"] + present, dropna=False, sort=True):
        parts = key if isinstance(key, tuple) else (key,)
        recording_id = str(parts[0])
        series = "|".join(str(part) for part in parts[1:]) or "all"
        coverage = _coverage(group, origin, step_s, n_cells, state_column)
        for index in range(n_cells):
            rows.append((recording_id, series, origin + index * step_s, _winner(coverage[index], empty_state)))

    return pd.DataFrame(rows, columns=list(SEQUENCE_COLUMNS))


def _coverage(
    group: pd.DataFrame, origin: float, step_s: float, n_cells: int, state_column: str
) -> list[dict[str, float]]:
    cells: list[dict[str, float]] = [{} for _ in range(n_cells)]
    starts = group["_start"].to_numpy(dtype=float)
    ends = group["_end"].to_numpy(dtype=float)
    states = group[state_column].astype(str).to_numpy()

    for start, end, state in zip(starts, ends, states):
        first = max(0, int(np.floor((start - origin) / step_s)))
        last = min(n_cells, int(np.ceil((end - origin) / step_s)))
        for index in range(first, last):
            low = max(start, origin + index * step_s)
            high = min(end, origin + (index + 1) * step_s)
            if high <= low:
                continue
            bucket = cells[index]
            bucket[state] = bucket.get(state, 0.0) + (high - low)
    return cells


def _winner(bucket: dict[str, float], empty_state: str) -> str:
    if not bucket:
        return empty_state
    top = max(bucket.values())
    return sorted(state for state, seconds in bucket.items() if seconds == top)[0]


def to_joint_sequence(sequence: pd.DataFrame, *, separator: str = "") -> pd.DataFrame:
    """One state per time step for the whole recording: the per-series states concatenated in a
    fixed order. This is where AA / AB / BA / BB comes from."""
    if sequence is None or sequence.empty:
        return _empty(JOINT_COLUMNS)

    ordered = sequence.sort_values(["recording_id", "t", "series"], kind="stable")
    joined = (
        ordered.groupby(["recording_id", "t"], sort=True)["state"]
        .apply(lambda states: separator.join(states.astype(str)))
        .reset_index()
    )
    return joined.loc[:, list(JOINT_COLUMNS)]


def export_state_sequence(sequence: pd.DataFrame, path: Path) -> Path:
    """csv, because the toolboxes that want this shape live in R and MATLAB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sequence.to_csv(path, index=False, encoding="utf-8")
    return path
