"""Parquet mechanics shared by the three tables.

Every table is a directory of parquet files, one per writer, never a single file. Within
one recording several extractors run as separate cluster jobs at the same time, and a
read-concat-rewrite append into one file would corrupt it. One file per writer also makes
rerunning a single extractor idempotent: it replaces its own file and touches nothing else.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from macaw.exceptions import InputNotFoundError, SchemaViolationError
from macaw.store.schema import coerce_frame, empty_frame

COMPRESSION = "zstd"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def _slug(value) -> str:
    text = "unassigned" if value is None or pd.isna(value) else str(value)
    return _UNSAFE.sub("_", text)


def _write_atomic(df: pd.DataFrame, target: Path) -> None:
    # a half-written parquet that a later run mistakes for a cache hit is worse than a crash
    tmp = target.with_suffix(target.suffix + ".tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression=COMPRESSION)
    os.replace(tmp, target)


def read_file(path: Path, schema: dict[str, str], name: str) -> pd.DataFrame:
    return coerce_frame(pq.read_table(path).to_pandas(), schema, name)


def write_partitioned(
    df: pd.DataFrame,
    directory: Path,
    schema: dict[str, str],
    name: str,
    part_column: str,
    *,
    append: bool = False,
    key: Sequence[str] | None = None,
) -> Path:
    if part_column not in df.columns:
        raise SchemaViolationError(f"{name}: cannot partition, no {part_column} column")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame = coerce_frame(df, schema, name)

    for value, group in frame.groupby(part_column, dropna=False, sort=True):
        target = directory / f"{_slug(value)}.parquet"
        if append and target.is_file():
            group = pd.concat([read_file(target, schema, name), group], ignore_index=True)
            if key:
                group = group.drop_duplicates(subset=list(key), keep="last")
        _write_atomic(group.reset_index(drop=True), target)
    return directory


def read_partitioned(
    path: Path,
    schema: dict[str, str],
    name: str,
    predicate: ds.Expression | None = None,
) -> pd.DataFrame:
    """Accepts a directory or a single parquet file, so an archived export reads back too."""
    path = Path(path)
    if not path.exists():
        raise InputNotFoundError(f"{name} not found: {path}")

    if path.is_dir() and not any(path.glob("*.parquet")):
        return empty_frame(schema)

    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(filter=predicate)
    return coerce_frame(table.to_pandas(), schema, name)


def equals(column: str, value) -> ds.Expression | None:
    return None if value is None else ds.field(column) == value


def is_in(column: str, values: Sequence[str] | None) -> ds.Expression | None:
    return None if values is None else ds.field(column).isin(list(values))


def overlaps(t_range: tuple[float, float] | None) -> ds.Expression | None:
    """Intervals that touch the window, not only those starting inside it."""
    if t_range is None:
        return None
    low, high = t_range
    return (ds.field("t_end") > low) & (ds.field("t_start") < high)


def all_of(*predicates: ds.Expression | None) -> ds.Expression | None:
    active = [predicate for predicate in predicates if predicate is not None]
    if not active:
        return None
    combined = active[0]
    for predicate in active[1:]:
        combined = combined & predicate
    return combined
