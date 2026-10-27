"""Plots stall past a few hundred thousand points. Min-max keeps the envelope shape."""

from __future__ import annotations

import numpy as np


def min_max_downsample(t: np.ndarray, y: np.ndarray, target_px: int) -> tuple[np.ndarray, np.ndarray]:
    """At most target_px points: the lowest and highest sample of each of target_px / 2 runs, in time order.
    NaN samples are gaps, kept where a whole run is NaN so the line still breaks there."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    buckets = max(target_px // 2, 1)
    if t.size <= 2 * buckets:
        return t, y
    keep = []
    for run in np.array_split(np.arange(t.size), buckets):
        values = y[run]
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size == 0:
            keep.append(run[0])
            continue
        low, high = run[finite[np.argmin(values[finite])]], run[finite[np.argmax(values[finite])]]
        keep.extend(sorted({low, high}))
    keep = np.asarray(keep)
    return t[keep], y[keep]
