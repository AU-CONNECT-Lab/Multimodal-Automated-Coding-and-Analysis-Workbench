"""Clock drift: cheap recorders run slightly fast or slow, so a first-frame offset walks off."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.config.schema import AlignSection

# at least this many segments before a slope means anything
MIN_SEGMENTS = 3


def segment_offsets(
    reference: Path,
    target: Path,
    *,
    segment_s: float,
    max_lag_s: float,
) -> list[tuple[float, float]]:
    """(segment_midpoint_s, offset_s) pairs."""
    from macaw.media.align import SAMPLE_RATE, load_normalised, gcc_phat

    # no length cap here: drift only shows up over the full recording
    reference_audio = load_normalised(reference, limit_s=None)
    target_audio = load_normalised(target, limit_s=None)

    step = int(segment_s * SAMPLE_RATE)
    usable = min(reference_audio.size, target_audio.size)
    if step <= 0 or usable < step:
        return []

    offsets: list[tuple[float, float]] = []
    for start in range(0, usable - step + 1, step):
        stop = start + step
        lag_s, confidence = gcc_phat(
            reference_audio[start:stop], target_audio[start:stop], SAMPLE_RATE, max_lag_s
        )
        if confidence > 0:
            offsets.append(((start + stop) / 2 / SAMPLE_RATE, lag_s))
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


def check_drift(reference: Path, target: Path, cfg: AlignSection) -> tuple[float, float, bool]:
    offsets = segment_offsets(
        reference, target, segment_s=cfg.drift_segment_s, max_lag_s=cfg.search_window_s
    )
    drift_ppm, r_squared = estimate_drift(offsets)
    # a large slope with a poor fit is noise, not a clock running fast, and re-timing on it
    # would make the alignment worse
    within_tolerance = abs(drift_ppm) <= cfg.max_drift_ppm or r_squared < 0.5
    return drift_ppm, r_squared, within_tolerance
