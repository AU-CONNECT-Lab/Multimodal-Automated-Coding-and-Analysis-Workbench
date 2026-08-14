"""Discrete labelled intervals. Partitioned by layer: A, the LLM layers and the human
review pass are separate jobs, so they must not share a file."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from macaw.store import _io
from macaw.store.schema import ANNOTATION_SCHEMA, ENTITY_COLUMNS, coerce_frame, empty_frame

NAME = "annotations"


def write_annotations(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    return _io.write_partitioned(df, path, ANNOTATION_SCHEMA, NAME, "layer", append=append)


def read_annotations(
    path: Path,
    *,
    recording_id: str | None = None,
    subject_id: str | None = None,
    tier: str | None = None,
    layer: str | None = None,
    annotator: str | None = None,
    needs_review: bool | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    predicate = _io.all_of(
        _io.equals("recording_id", recording_id),
        _io.equals("subject_id", subject_id),
        _io.equals("tier", tier),
        _io.equals("layer", layer),
        _io.equals("annotator", annotator),
        _io.equals("needs_review", needs_review),
        _io.overlaps(t_range),
    )
    return _io.read_partitioned(path, ANNOTATION_SCHEMA, NAME, predicate)


def build_annotations(
    *,
    recording_id: str,
    tier: str,
    layer: str,
    annotator: str,
    intervals: pd.DataFrame,
    entities: Mapping[str, str | None] | None = None,
    subject_id: str | None = None,
    execution_id: str | None = None,
    config_hash: str | None = None,
) -> pd.DataFrame:
    """intervals carries t_start, t_end, label and optionally confidence, abstained and
    needs_review. Everything else is filled in from the arguments."""
    if intervals.empty:
        return empty_frame(ANNOTATION_SCHEMA)

    out = intervals.copy()
    entities = entities or {}
    out["recording_id"] = recording_id
    for column in ENTITY_COLUMNS:
        out[column] = entities.get(column)
    out["subject_id"] = subject_id
    out["tier"] = tier
    out["layer"] = layer
    out["annotator"] = annotator
    out["execution_id"] = execution_id
    out["config_hash"] = config_hash
    for column, default in (("confidence", float("nan")), ("abstained", False), ("needs_review", False)):
        if column not in out.columns:
            out[column] = default
    return coerce_frame(out, ANNOTATION_SCHEMA, NAME)


def tiers(df: pd.DataFrame) -> Sequence[str]:
    return sorted(df["tier"].dropna().unique().tolist())
