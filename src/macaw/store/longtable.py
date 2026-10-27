from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from macaw.exceptions import SchemaViolationError
from macaw.store import _io
from macaw.store.schema import (
    ENTITY_COLUMNS,
    LONG_TABLE_IDENTITY,
    LONG_TABLE_KEY,
    LONG_TABLE_SCHEMA,
    Scale,
    coerce_frame,
    empty_frame,
)

NAME = "long table"


def write_long_table(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    """One file per source inside the directory; store/_io.py explains why."""
    return _io.write_partitioned(
        df, path, LONG_TABLE_SCHEMA, NAME, "source", append=append, key=LONG_TABLE_KEY
    )


def read_long_table(
    path: Path,
    *,
    recording_id: str | None = None,
    subject_id: str | None = None,
    source: str | None = None,
    scale: Scale | None = None,
    features: Sequence[str] | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Predicate pushdown at the parquet level; never load the whole table."""
    predicate = _io.all_of(
        _io.equals("recording_id", recording_id),
        _io.equals("subject_id", subject_id),
        _io.equals("source", source),
        _io.equals("scale", scale.value if isinstance(scale, Scale) else scale),
        _io.is_in("feature", features),
        _io.overlaps(t_range),
    )
    return _io.read_partitioned(path, LONG_TABLE_SCHEMA, NAME, predicate)


def concat_long_tables(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    collected = [frame for frame in frames if frame is not None and not frame.empty]
    if not collected:
        return empty_frame(LONG_TABLE_SCHEMA)
    return coerce_frame(pd.concat(collected, ignore_index=True), LONG_TABLE_SCHEMA, NAME)


def build_long_rows(
    *,
    recording_id: str,
    source: str,
    scale: Scale,
    values: pd.DataFrame,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    subject_id_b: str | None = None,
    execution_id: str | None = None,
    config_hash: str | None = None,
) -> pd.DataFrame:
    """Melt an extractor's wide (time x feature) output into long-table rows.

    values carries t_start, optionally t_end, and one column per feature. Pass
    Recording.entities() so the BIDS columns come along."""
    if "t_start" not in values.columns:
        raise SchemaViolationError(f"{source}: values must carry a t_start column")

    feature_columns = [c for c in values.columns if c not in {"t_start", "t_end"}]
    if not feature_columns:
        return empty_frame(LONG_TABLE_SCHEMA)

    melted = values.melt(
        id_vars=[c for c in ("t_start", "t_end") if c in values.columns],
        value_vars=feature_columns,
        var_name="feature",
        value_name="value",
    )
    if "t_end" not in melted.columns:
        melted["t_end"] = melted["t_start"]

    entities = entities or {}
    melted["recording_id"] = recording_id
    for column in ENTITY_COLUMNS:
        melted[column] = entities.get(column)
    melted["subject_id"] = subject_id
    melted["subject_id_b"] = subject_id_b
    melted["source"] = source
    melted["scale"] = scale.value if isinstance(scale, Scale) else scale
    melted["execution_id"] = execution_id
    melted["config_hash"] = config_hash
    return coerce_frame(melted, LONG_TABLE_SCHEMA, NAME)


def aggregate_to_window(
    df: pd.DataFrame,
    *,
    window_s: float,
    hop_s: float,
    how: str | dict[str, str] = "mean",
) -> pd.DataFrame:
    """Frame/event scale to window scale. Per-feature aggregation comes from the manifest."""
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must both be positive")

    frame = coerce_frame(df, LONG_TABLE_SCHEMA, NAME)
    if frame.empty:
        return frame

    # sliding windows overlap, so one row feeds every window that covers its t_start.
    # window k spans [k*hop, k*hop + window), right edge open, hence the +1
    starts = frame["t_start"].to_numpy(dtype=float)
    first = np.maximum(np.floor((starts - window_s) / hop_s) + 1.0, 0.0)
    last = np.floor(starts / hop_s)
    counts = np.maximum(last - first + 1.0, 0.0).astype(int)
    if counts.sum() == 0:
        return empty_frame(LONG_TABLE_SCHEMA)

    row_index = np.repeat(np.arange(len(frame)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)

    expanded = frame.iloc[row_index].copy()
    expanded["_window"] = np.repeat(first, counts) + offsets

    aggregated = _aggregate(expanded, list(LONG_TABLE_IDENTITY) + ["_window"], how)
    aggregated["t_start"] = aggregated.pop("_window") * hop_s
    aggregated["t_end"] = aggregated["t_start"] + window_s
    aggregated["scale"] = Scale.WINDOW.value
    return coerce_frame(aggregated, LONG_TABLE_SCHEMA, NAME)


def aggregate_to_recording(df: pd.DataFrame, how: str | dict[str, str] = "mean") -> pd.DataFrame:
    frame = coerce_frame(df, LONG_TABLE_SCHEMA, NAME)
    if frame.empty:
        return frame

    # the span comes out of the same groupby as the value: subject_id_b is null on most
    # rows, and re-aligning two results on grouping keys that contain NA is unreliable
    aggregated = _aggregate(
        frame,
        list(LONG_TABLE_IDENTITY),
        how,
        extras={"t_start": ("t_start", "min"), "t_end": ("t_end", "max")},
    )
    aggregated["scale"] = Scale.RECORDING.value
    return coerce_frame(aggregated, LONG_TABLE_SCHEMA, NAME)


def _aggregate(
    frame: pd.DataFrame,
    group_columns: list[str],
    how: str | dict[str, str],
    extras: dict[str, tuple[str, str]] | None = None,
):
    specification = {"value": ("value", None), **(extras or {})}

    def run(subset: pd.DataFrame, function: str) -> pd.DataFrame:
        spec = dict(specification)
        spec["value"] = ("value", function)
        return subset.groupby(group_columns, dropna=False, observed=True).agg(**spec).reset_index()

    if isinstance(how, str):
        return run(frame, how)

    # a dict names one function per feature; grouping the features by function keeps each
    # distinct function to a single vectorised pass
    functions = frame["feature"].map(how).fillna("mean")
    parts = [run(frame[functions == function], function) for function in pd.unique(functions)]
    return pd.concat(parts, ignore_index=True)
