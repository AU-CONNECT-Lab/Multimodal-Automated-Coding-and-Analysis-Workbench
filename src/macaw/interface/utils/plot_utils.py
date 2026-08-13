from __future__ import annotations

import numpy as np


def downsampled_line(t: np.ndarray, y: np.ndarray, *, target_px: int = 2000, name: str = "") -> object:
    raise NotImplementedError


def annotation_bands(intervals: list[tuple[float, float, str]]) -> list[dict]:
    raise NotImplementedError


def empty_figure(message: str) -> object:
    raise NotImplementedError
