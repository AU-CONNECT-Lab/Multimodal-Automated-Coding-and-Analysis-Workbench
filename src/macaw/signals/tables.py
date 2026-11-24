"""Signals another device exported as a table: a time column, a column per channel, and maybe a column of events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import UnreadableSignalError
from macaw.signals.snirf import MARKER_COLUMNS, Snirf

SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz")
# the column a table's own time is in, in seconds, whatever its case
TIME = "time"
# a column whose text, where there is any, names an event at that row's time
EVENT_COLUMNS = ("event", "marker")


def is_table(path: Path) -> bool:
    return path.name.lower().endswith(SUFFIXES)


def read_table(path: Path) -> tuple[Snirf, list[str]]:
    """The table as a SNIRF would read, and the columns left out for holding no numbers."""
    try:
        frame = pd.read_csv(path, sep="\t" if ".tsv" in path.name.lower() else ",", low_memory=False)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise UnreadableSignalError(f"{path.name}: cannot read the table: {error}") from None
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    if TIME not in lowered:
        raise UnreadableSignalError(f"{path.name} has no column named {TIME}; a table needs one, in seconds on its device's clock")
    times = pd.to_numeric(frame[lowered[TIME]], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).any():
        raise UnreadableSignalError(f"{path.name}: the {TIME} column holds no numbers")
    event = next((lowered[name] for name in EVENT_COLUMNS if name in lowered), None)
    markers = pd.DataFrame(columns=list(MARKER_COLUMNS))
    if event is not None:
        named = frame[event].astype(str).str.strip()
        rows = named.notna() & ~named.isin(["", "nan", "None"]) & np.isfinite(times)
        markers = pd.DataFrame({"name": named[rows].to_numpy(), "onset": times[rows.to_numpy()], "duration": 0.0})
    channels, skipped = [], []
    for column in frame.columns:
        if column in (lowered[TIME], event):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        (channels if values.notna().any() else skipped).append(column)
    keep = np.isfinite(times)
    data = frame.loc[keep, channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    found = Snirf(times=times[keep], data=data, channels=[str(column).strip() for column in channels],
                  kinds=["value"] * len(channels), markers=markers)
    return found, [str(column) for column in skipped]
