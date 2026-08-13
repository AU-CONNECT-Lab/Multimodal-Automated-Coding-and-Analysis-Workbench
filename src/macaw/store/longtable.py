from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from macaw.store.schema import Scale


def write_long_table(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    raise NotImplementedError


def read_long_table(
    path: Path,
    *,
    session_id: str | None = None,
    subject_id: str | None = None,
    source: str | None = None,
    scale: Scale | None = None,
    features: Sequence[str] | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Predicate pushdown at the parquet level; never load the whole table."""
    raise NotImplementedError


def concat_long_tables(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    raise NotImplementedError


def build_long_rows(
    *,
    session_id: str,
    source: str,
    scale: Scale,
    values: pd.DataFrame,
    subject_id: str | None = None,
    subject_id_b: str | None = None,
    run_id: str | None = None,
    config_hash: str | None = None,
) -> pd.DataFrame:
    """Melt an extractor's wide (time x feature) output into long-table rows."""
    raise NotImplementedError


def aggregate_to_window(
    df: pd.DataFrame,
    *,
    window_s: float,
    hop_s: float,
    how: str | dict[str, str] = "mean",
) -> pd.DataFrame:
    """Frame/event scale to window scale. Per-feature aggregation comes from the manifest."""
    raise NotImplementedError


def aggregate_to_session(df: pd.DataFrame, how: str | dict[str, str] = "mean") -> pd.DataFrame:
    raise NotImplementedError


def suggest_window_length(task_duration_s: float, tau_s: float, min_windows: int = 10) -> float:
    """Window length must follow task duration, not be hard-coded."""
    raise NotImplementedError
