"""Clock drift: cheap recorders run slightly fast or slow, so a first-frame offset walks off."""

from __future__ import annotations

import numpy as np

# at least this many segments before a slope or an agreement count means anything
MIN_SEGMENTS = 3
# within this of the file's loud level a frame is sound: others' bleed on a close-talk mic, not mic noise
SOUNDING_DB = 30.0
MIN_SOUNDING_FRACTION = 0.05


def segment_offsets(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: float,
    *,
    offset_s: float,
    segment_s: float,
    max_residual_s: float,
) -> list[tuple[float, float, float]]:
    """(midpoint on the reference clock, offset_s, confidence) per segment where both files have sound."""
    from macaw.media.align import gcc_phat

    step = int(segment_s * sample_rate)
    shift = int(round(offset_s * sample_rate))
    if step <= 0:
        return []
    loud_reference, loud_target = loud_level(reference, sample_rate), loud_level(target, sample_rate)

    offsets: list[tuple[float, float, float]] = []
    # target sample j sits at reference sample j + shift, so a segment only searches the residual
    for start in range(max(shift, 0), reference.size - step + 1, step):
        piece = start - shift
        if piece + step > target.size:
            break
        reference_piece, target_piece = reference[start : start + step], target[piece : piece + step]
        # a silent stretch, such as a quiet participant's own mic, correlates at a random lag
        if not (has_sound(reference_piece, sample_rate, loud_reference) and has_sound(target_piece, sample_rate, loud_target)):
            continue
        lag_s, confidence = gcc_phat(reference_piece, target_piece, sample_rate, max_residual_s)
        offsets.append(((start + step / 2) / sample_rate, offset_s + lag_s, confidence))
    return offsets


def loud_level(signal: np.ndarray, sample_rate: float) -> float:
    from macaw.media.align import envelope

    frames = envelope(signal, sample_rate)
    return float(np.percentile(frames, 99)) if frames.size else 0.0


def has_sound(piece: np.ndarray, sample_rate: float, loud: float) -> bool:
    """At least MIN_SOUNDING_FRACTION of the frames within SOUNDING_DB of the file's loud level."""
    from macaw.media.align import envelope

    frames = envelope(piece, sample_rate)
    if loud <= 0 or frames.size == 0:
        return False
    return float((frames > loud * 10 ** (-SOUNDING_DB / 20)).mean()) >= MIN_SOUNDING_FRACTION


def estimate_drift(offsets: list[tuple[float, ...]]) -> tuple[float, float]:
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


def drift_is_real(
    drift_ppm: float | None, r_squared: float | None, overlap_s: float | None, tolerance_s: float
) -> bool:
    """A credible slope that walks the offset further than tolerance_s over the file; drift is not corrected."""
    if drift_ppm is None or r_squared is None or overlap_s is None:
        return False
    return r_squared >= 0.5 and drift_walk_s(drift_ppm, overlap_s) > tolerance_s


def drift_walk_s(drift_ppm: float, overlap_s: float) -> float:
    return abs(drift_ppm) * 1e-6 * overlap_s
