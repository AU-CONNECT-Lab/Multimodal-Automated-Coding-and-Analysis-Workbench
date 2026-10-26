"""Clock drift: cheap recorders run slightly fast or slow, so a first-frame offset walks off."""

from __future__ import annotations

import numpy as np

# at least this many segments before a slope means anything
MIN_SEGMENTS = 3


def segment_offsets(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: float,
    *,
    offset_s: float,
    segment_s: float,
    max_residual_s: float,
    min_confidence: float,
) -> list[tuple[float, float]]:
    """(segment midpoint on the reference clock, offset_s) for each segment that correlates.
    The target is shifted by the global offset first, so a segment only searches the small
    residual and a long offset cannot push the two pieces out of overlap."""
    from macaw.media.align import gcc_phat

    step = int(segment_s * sample_rate)
    shift = int(round(offset_s * sample_rate))
    if step <= 0:
        return []

    offsets: list[tuple[float, float]] = []
    # target sample j sits at reference sample j + shift
    for start in range(max(shift, 0), reference.size - step + 1, step):
        piece = start - shift
        if piece + step > target.size:
            break
        lag_s, confidence = gcc_phat(
            reference[start : start + step], target[piece : piece + step], sample_rate, max_residual_s
        )
        # silence or room noise correlates weakly and would only add scatter to the fit
        if confidence >= min_confidence:
            offsets.append(((start + step / 2) / sample_rate, offset_s + lag_s))
    return offsets


def estimate_drift(offsets: list[tuple[float, float]]) -> tuple[float, float]:
    """Linear fit; returns (drift_ppm, r_squared). High r2 with nonzero slope means real drift."""
    if len(offsets) < MIN_SEGMENTS:
        return 0.0, 0.0

    midpoints = np.array([point[0] for point in offsets], dtype=float)
    measured = np.array([point[1] for point in offsets], dtype=float)
    slope, intercept = np.polyfit(midpoints, measured, 1)

    predicted = slope * midpoints + intercept
    total = float(np.sum((measured - measured.mean()) ** 2))
    residual = float(np.sum((measured - predicted) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    # slope is seconds of offset per second elapsed, which is parts per million x 1e-6
    return float(slope * 1e6), float(r_squared)


def drift_is_real(drift_ppm: float | None, r_squared: float | None, max_drift_ppm: float) -> bool:
    """A large slope with a poor fit is noise, not a clock running fast, and re-timing on it
    would make the alignment worse."""
    if drift_ppm is None or r_squared is None:
        return False
    return abs(drift_ppm) > max_drift_ppm and r_squared >= 0.5
