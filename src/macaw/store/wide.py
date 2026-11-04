"""Export for downstream statistics users (R, MATLAB)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.store.schema import SEGMENT_TABLE_SCHEMA, Scale, coerce_frame

# entity plus time columns; enough to identify a row once features become columns
DEFAULT_INDEX = [
    "recording_id",
    "group_id",
    "session_id",
    "task",
    "run",
    "segment",
    "subject_id",
    "subject_id_b",
    "t_start",
    "t_end",
]


def to_wide(
    df: pd.DataFrame,
    *,
    index: list[str] | None = None,
    scale: Scale | None = None,
) -> pd.DataFrame:
    """Pivot long rows to one column per feature."""
    frame = with_segment(df)
    if scale is not None:
        wanted = scale.value if isinstance(scale, Scale) else scale
        frame = frame[frame["scale"] == wanted]

    keys = index or DEFAULT_INDEX
    # not pivot_table: it takes the cartesian product of the index levels, and t_start x
    # t_end alone would square the row count. groupby only ever sees observed combinations,
    # and dropna=False keeps rows whose subject_id_b or task is null, which is most of them.
    grouped = frame.groupby(keys + ["feature"], dropna=False, observed=True)["value"].mean()
    pivoted = grouped.unstack("feature")
    pivoted.columns.name = None
    return pivoted.reset_index()


def with_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Long-table rows, or per-segment ones, in one layout; a row of the whole recording has no segment."""
    return coerce_frame(df if "segment" in df.columns else df.assign(segment=pd.NA), SEGMENT_TABLE_SCHEMA, "export")


def export_csv(df: pd.DataFrame, path: Path, *, wide: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = to_wide(df) if wide else with_segment(df)
    out.to_csv(path, index=False, encoding="utf-8")
    return path
