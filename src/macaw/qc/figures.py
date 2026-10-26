"""Static figures for review and for the report, one file each so a report links them rather
than embedding them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.exceptions import AlignmentFailedError, BackendNotAvailableError
from macaw.media.align import (
    ENVELOPE_EXCLUSION_S,
    ENVELOPE_STEP_S,
    FINE_WINDOW_S,
    MAX_CORRELATION_S,
    PHAT_EXCLUSION_S,
    AlignResult,
    envelope,
    envelope_correlation,
    fine_pieces,
    phat_correlation,
    top_two_peaks,
)

EXCERPT_S = 30.0


def require_matplotlib():
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise BackendNotAvailableError('Review figures need matplotlib: pip install "pymacaw[report]"') from error
    return Figure


def loudest_shared_excerpt(
    reference_env: np.ndarray, target_env: np.ndarray, shift: int, width: int
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """The stretch where both are loud, on the reference's frame grid: (first frame, reference
    piece, target piece). Target frame j sits at reference frame j + shift."""
    start = max(shift, 0)
    stop = min(reference_env.size, target_env.size + shift)
    if stop - start <= 0:
        return None
    reference_part = reference_env[start:stop]
    target_part = target_env[start - shift : stop - shift]
    width = min(width, reference_part.size)
    joint = np.convolve(np.minimum(reference_part, target_part), np.ones(width), mode="valid")
    at = int(np.argmax(joint))
    return start + at, reference_part[at : at + width], target_part[at : at + width]


def alignment_figure(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: float,
    result: AlignResult,
    search_window_s: float,
    path: Path,
) -> Path:
    """What a person or an agent needs to accept or reject one offset: each correlation peak
    against its nearest rival, both loudness envelopes on the recording clock, the drift fit."""
    Figure = require_matplotlib()
    figure = Figure(figsize=(10, 11), layout="constrained")
    coarse, fine, envelopes, drift = figure.subplots(4, 1)
    figure.suptitle(f"{result.target} against {result.reference}: offset {result.offset_s:+.3f} s")

    lags, window = envelope_correlation(reference, target, sample_rate, search_window_s)
    _peaks(coarse, lags, window, int(ENVELOPE_EXCLUSION_S / ENVELOPE_STEP_S), "s")
    coarse.set(
        xlabel="lag (s)", ylabel="envelope correlation",
        title=f"coarse, whole files: confidence {_format(result.coarse_confidence)}",
    )

    shift = int(round((result.coarse_offset_s or 0.0) * sample_rate))
    try:
        pieces = fine_pieces(reference, target, shift, int(MAX_CORRELATION_S * sample_rate))
    except AlignmentFailedError as error:
        _empty(fine, f"fine correlation not drawn: {error}")
    else:
        lags, window = phat_correlation(*pieces, sample_rate, FINE_WINDOW_S)
        _peaks(fine, (lags + shift / sample_rate) * 1000, window, int(PHAT_EXCLUSION_S * sample_rate), "ms")
        fine.set(
            xlabel="lag (ms)", ylabel="PHAT correlation",
            title=f"fine, first minutes of overlap: confidence {_format(result.confidence)}",
        )

    reference_env = envelope(reference, sample_rate)
    target_env = envelope(target, sample_rate)
    reference_env /= max(float(np.percentile(reference_env, 99)), 1e-12)
    target_env /= max(float(np.percentile(target_env, 99)), 1e-12)
    excerpt = loudest_shared_excerpt(
        reference_env, target_env, int(round(result.offset_s / ENVELOPE_STEP_S)), int(EXCERPT_S / ENVELOPE_STEP_S)
    )
    if excerpt is None:
        _empty(envelopes, "the two files do not overlap at this offset")
    else:
        first, reference_part, target_part = excerpt
        t = (first + np.arange(reference_part.size)) * ENVELOPE_STEP_S
        envelopes.fill_between(t, 0, reference_part, color="C0", lw=0, label=result.reference)
        envelopes.fill_between(t, 0, -target_part, color="C1", lw=0, label=f"{result.target}, shifted")
        envelopes.axhline(0, color="0.2", lw=0.5)
        envelopes.set(xlabel="recording time (s)", ylabel="loudness", title="aligned, the two should rise and fall together")
        envelopes.legend(loc="upper right", fontsize=8)

    if result.segment_offsets and len(result.segment_offsets) >= 2:
        t, offsets = np.array(result.segment_offsets, dtype=float).T
        drift.plot(t, offsets * 1000, "o", ms=3, color="C0")
        slope, intercept = np.polyfit(t, offsets, 1)
        drift.plot(t, (slope * t + intercept) * 1000, color="C3", lw=1)
        title = (
            "drift: too few segments for a slope" if result.drift_ppm is None
            else f"drift {result.drift_ppm:+.1f} ppm, r2 {result.drift_r2:.2f}"
        )
        drift.set(xlabel="recording time (s)", ylabel="offset (ms)", title=title)
    else:
        _empty(drift, "drift not measured")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=100)
    return path


def _peaks(axes, lags, window, exclusion: int, unit: str) -> None:
    best, rival = top_two_peaks(window, exclusion)
    axes.plot(lags, window, lw=0.6, color="0.45")
    axes.axvline(lags[best], color="C3", lw=1.2, label=f"chosen {lags[best]:+.3f} {unit}")
    axes.axvline(lags[rival], color="C0", lw=1.2, ls="--", label=f"runner-up {lags[rival]:+.3f} {unit}")
    axes.legend(loc="upper right", fontsize=8)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _empty(axes, message: str) -> None:
    axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
    axes.set_axis_off()
