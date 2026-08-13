"""Plots stall past a few hundred thousand points. Min-max keeps the envelope shape."""

from __future__ import annotations

import numpy as np


def min_max_downsample(t: np.ndarray, y: np.ndarray, target_px: int) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def lttb(t: np.ndarray, y: np.ndarray, target_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Largest-triangle-three-buckets, for curves where peak preservation matters less."""
    raise NotImplementedError
