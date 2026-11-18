"""A thing's outline in one frame: the polygons around its mask, in the video's pixels."""

from __future__ import annotations

import numpy as np

# parts of a mask smaller than this share of its largest part are specks, not the thing
_SPECK = 0.01
# px a polygon may stray from the mask's edge, which keeps the stored outline small
_SIMPLIFY_PX = 1.0


def mask_outline(mask: np.ndarray) -> list[list[int]]:
    """The outer edge of each part of a boolean mask, as flat [x0, y0, x1, y1, ...] lists."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    if not areas:
        return []
    largest = max(areas)
    kept = []
    for contour, area in zip(contours, areas):
        if area < _SPECK * largest:
            continue
        simple = cv2.approxPolyDP(contour, _SIMPLIFY_PX, True).reshape(-1, 2)
        if len(simple) >= 3:
            kept.append(simple.astype(int).ravel().tolist())
    return kept


def polygons(outline) -> list[np.ndarray]:
    """The stored outline as (n, 2) arrays of x and y."""
    if outline is None:
        return []
    return [np.asarray(part, dtype=float).reshape(-1, 2) for part in outline if len(part) >= 6]


def _inside_one(polygon: np.ndarray, x: float, y: float) -> bool:
    xs, ys = polygon[:, 0], polygon[:, 1]
    xn, yn = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (yn > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = xs + (y - ys) * (xn - xs) / (yn - ys)
    return bool(np.count_nonzero(crosses & (x < at)) % 2)


def _edge_distance(polygon: np.ndarray, x: float, y: float) -> float:
    start, end = polygon, np.roll(polygon, -1, axis=0)
    along = end - start
    length = np.einsum("ij,ij->i", along, along)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - start, along) / length, 0.0, 1.0)
    share = np.where(length > 0, share, 0.0)
    nearest = start + share[:, None] * along
    return float(np.min(np.hypot(nearest[:, 0] - x, nearest[:, 1] - y)))


def inside(outline, x: float, y: float) -> bool:
    return any(_inside_one(polygon, x, y) for polygon in polygons(outline))


def distance(outline, x: float, y: float) -> float:
    """px from the point to the nearest edge of the outline, 0 inside it; nan without an outline."""
    parts = polygons(outline)
    if not parts or not np.isfinite(x) or not np.isfinite(y):
        return float("nan")
    if any(_inside_one(polygon, x, y) for polygon in parts):
        return 0.0
    return min(_edge_distance(polygon, x, y) for polygon in parts)
