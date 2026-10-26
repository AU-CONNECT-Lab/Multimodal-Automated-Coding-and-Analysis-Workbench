"""Multi-source alignment by generalised cross-correlation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from macaw.config.schema import AlignSection
from macaw.exceptions import AlignmentFailedError, MacawError
from macaw.media.drift import MIN_SEGMENTS, drift_is_real, drift_walk_s, estimate_drift, segment_offsets
from macaw.media.ffmpeg import decode_audio
from macaw.project.recording import Recording

SAMPLE_RATE = 16000
# an hour-long file would make the FFT the slowest thing in the pipeline; once the coarse lag
# has lined the files up, the opening minutes settle the fine offset
MAX_CORRELATION_S = 600.0
ENVELOPE_STEP_S = 0.05
# the coarse lag is good to a frame or two, so the fine search only needs a little room
FINE_WINDOW_S = 1.0
# early reflections land within this of the direct path and are the same event; the rivals
# worth flagging are another match altogether, farther away
PHAT_EXCLUSION_S = 0.05
# envelope correlations are smooth over about a syllable, so a rival has to be well clear
ENVELOPE_EXCLUSION_S = 2.0
# a segment within this of the offset found the same offset; about half a video frame
AGREEMENT_TOLERANCE_S = 0.02

GCC_PHAT = "gcc_phat"
MANUAL = "manual"
# nothing to correlate: taken to start with the reference until a person sets it
UNALIGNED = "unaligned"


@dataclass
class AlignResult:
    # file names, not paths, so a table written on one machine applies on another
    reference: str
    target: str
    offset_s: float
    confidence: float | None         # fine peak sharpness, 0-1; None when nothing was correlated
    coarse_offset_s: float | None = None
    coarse_confidence: float | None = None
    # share of sounding segments that found the same offset; None when too few to say
    agreement: float | None = None
    overlap_s: float | None = None   # how long the two files overlap at offset_s
    drift_ppm: float | None = None
    drift_r2: float | None = None
    # (midpoint_s, offset_s, confidence) per sounding segment
    segment_offsets: list[tuple[float, float, float]] | None = None
    method: str = GCC_PHAT
    metadata_offset_s: float | None = None
    note: str | None = None
    accepted: str | None = None      # who kept a flagged result, and when


def _correlate(a: np.ndarray, b: np.ndarray, rate: float, max_lag_s: float, *, phat: bool):
    """(lags_s, correlation) over +-max_lag_s; a positive lag means b started after a."""
    if a.size == 0 or b.size == 0:
        raise AlignmentFailedError("Cannot correlate an empty signal")

    size = int(2 ** np.ceil(np.log2(a.size + b.size)))
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    if phat:
        # keep only the phase. Magnitude carries the room and the mic, which differ between
        # the two recorders and would otherwise dominate the peak.
        spectrum /= np.maximum(np.abs(spectrum), 1e-12)
    correlation = np.fft.irfft(spectrum, size)

    max_lag = max(int(min(max_lag_s * rate, size // 2)), 1)
    window = np.concatenate((correlation[-max_lag:], correlation[: max_lag + 1]))
    return np.arange(-max_lag, max_lag + 1) / rate, window


def phat_correlation(a: np.ndarray, b: np.ndarray, sample_rate: float, max_lag_s: float):
    return _correlate(a, b, sample_rate, max_lag_s, phat=True)


def envelope(signal: np.ndarray, sample_rate: float, step_s: float = ENVELOPE_STEP_S) -> np.ndarray:
    step = max(int(step_s * sample_rate), 1)
    n = signal.size // step
    return np.sqrt((signal[: n * step].reshape(n, step) ** 2).mean(axis=1))


def envelope_correlation(reference: np.ndarray, target: np.ndarray, sample_rate: float, max_lag_s: float):
    """Loudness envelopes on a log scale, so a close-talk mic and a room mic still rise and fall
    together; good to a frame, blind to timbre, and cheap over a whole file."""
    def standardised(signal):
        env = envelope(signal, sample_rate)
        env = np.log(env / max(float(np.percentile(env, 99)), 1e-12) + 1e-3)
        return (env - env.mean()) / max(float(env.std()), 1e-12)

    a, b = standardised(reference), standardised(target)
    lags, window = _correlate(a, b, 1.0 / ENVELOPE_STEP_S, max_lag_s, phat=False)
    return lags, window / min(a.size, b.size)


def top_two_peaks(window: np.ndarray, exclusion: int) -> tuple[int, int]:
    """The highest peak and the best competitor more than exclusion samples away from it."""
    best = int(np.argmax(window))
    masked = window.copy()
    masked[max(best - exclusion, 0) : best + exclusion + 1] = -np.inf
    return best, int(np.argmax(masked))


def sharpness(window: np.ndarray, exclusion: int) -> tuple[int, float]:
    """(peak index, confidence). Measured against the best competing peak, not the noise
    floor: a periodic signal produces many tall peaks and that is exactly the case to flag."""
    best, rival = top_two_peaks(window, exclusion)
    peak = float(window[best])
    if peak <= 0:
        return best, 0.0
    competitor = max(float(window[rival]), 0.0) if np.isfinite(window[rival]) else 0.0
    return best, float(np.clip(1.0 - competitor / peak, 0.0, 1.0))


def gcc_phat(
    a: np.ndarray, b: np.ndarray, sample_rate: float, max_lag_s: float
) -> tuple[float, float]:
    """Returns (lag_s, confidence). A positive lag means b started after a."""
    lags, window = phat_correlation(a, b, sample_rate, max_lag_s)
    best, confidence = sharpness(window, max(int(PHAT_EXCLUSION_S * sample_rate), 1))
    return (float(lags[best]), confidence) if confidence > 0 else (0.0, 0.0)


def envelope_lag(reference: np.ndarray, target: np.ndarray, sample_rate: float, max_lag_s: float):
    lags, window = envelope_correlation(reference, target, sample_rate, max_lag_s)
    best, confidence = sharpness(window, int(ENVELOPE_EXCLUSION_S / ENVELOPE_STEP_S))
    return float(lags[best]), confidence


def fine_pieces(reference: np.ndarray, target: np.ndarray, shift: int, limit: int):
    """The overlapping stretch once target sample j is placed at reference sample j + shift,
    capped at limit samples."""
    start_reference, start_target = max(shift, 0), max(-shift, 0)
    length = min(reference.size - start_reference, target.size - start_target, limit)
    if length <= 0:
        raise AlignmentFailedError("The two files do not overlap at the coarse offset")
    return (
        reference[start_reference : start_reference + length],
        target[start_target : start_target + length],
    )


def load_normalised(path: Path, limit_s: float | None = MAX_CORRELATION_S) -> np.ndarray:
    samples = decode_audio(path, sample_rate=SAMPLE_RATE)
    if limit_s is not None:
        samples = samples[: int(limit_s * SAMPLE_RATE)]
    samples = samples - samples.mean()
    peak = np.max(np.abs(samples))
    return samples / peak if peak > 0 else samples


def align_signals(
    reference: np.ndarray, target: np.ndarray, cfg: AlignSection, *, names: tuple[str, str]
) -> AlignResult:
    """Both at SAMPLE_RATE. Coarse lag from envelopes over the whole files, then GCC-PHAT on
    the first minutes of overlap for the exact offset, then segment by segment over the whole
    overlap to cross-check it and measure drift."""
    coarse_s, coarse_confidence = envelope_lag(reference, target, SAMPLE_RATE, cfg.search_window_s)
    shift = int(round(coarse_s * SAMPLE_RATE))
    piece_reference, piece_target = fine_pieces(reference, target, shift, int(MAX_CORRELATION_S * SAMPLE_RATE))
    residual_s, confidence = gcc_phat(piece_reference, piece_target, SAMPLE_RATE, FINE_WINDOW_S)
    lag_s = shift / SAMPLE_RATE + residual_s
    result = AlignResult(
        reference=names[0], target=names[1], offset_s=lag_s, confidence=confidence,
        coarse_offset_s=coarse_s, coarse_confidence=coarse_confidence,
    )
    duration_s = min(reference.size, target.size) / SAMPLE_RATE
    placed = int(round(lag_s * SAMPLE_RATE))
    result.overlap_s = max(min(reference.size - max(placed, 0), target.size - max(-placed, 0)), 0) / SAMPLE_RATE
    segments = segment_offsets(
        reference,
        target,
        SAMPLE_RATE,
        offset_s=lag_s,
        segment_s=cfg.drift_segment_s,
        # room for twice the tolerated drift over the whole file, and never under a second
        max_residual_s=max(1.0, 2 * cfg.max_drift_ppm * 1e-6 * duration_s),
    )
    result.segment_offsets = segments
    result.agreement = agreement(segments, lag_s)
    if cfg.drift_check:
        # blunt segments only add scatter to the slope
        sharp = [segment for segment in segments if segment[2] >= cfg.min_confidence]
        if len(sharp) >= MIN_SEGMENTS:
            result.drift_ppm, result.drift_r2 = estimate_drift(sharp)
    return result


def agreement(segments: list[tuple[float, float, float]] | None, offset_s: float) -> float | None:
    if not segments or len(segments) < MIN_SEGMENTS:
        return None
    same = [confidence > 0 and abs(offset - offset_s) <= AGREEMENT_TOLERANCE_S for _, offset, confidence in segments]
    return sum(same) / len(same)


def drift_flagged(result: AlignResult) -> bool:
    return drift_is_real(result.drift_ppm, result.drift_r2, result.overlap_s, AGREEMENT_TOLERANCE_S)


def describe_drift(result: AlignResult) -> str:
    text = f"drift {result.drift_ppm:+.1f} ppm (r2 {result.drift_r2:.2f})"
    if result.overlap_s:
        text += f", {drift_walk_s(result.drift_ppm, result.overlap_s) * 1000:.0f} ms over {result.overlap_s:.0f} s"
    return text


def describe_agreement(result: AlignResult) -> str:
    if result.agreement is None:
        return "too few sounding segments to cross-check"
    total = len(result.segment_offsets or ())
    return f"same offset in {round(result.agreement * total)} of {total} sounding segments"


def low_confidence(result: AlignResult, cfg: AlignSection) -> bool:
    """A blunt peak is normal between a close-talk and a room mic; it is a doubt only when the segments disagree too."""
    if result.method != GCC_PHAT or result.confidence is None or result.confidence >= cfg.min_confidence:
        return False
    return result.agreement is None or result.agreement < cfg.min_agreement


def review_reason(result: AlignResult, cfg: AlignSection) -> str | None:
    """Why a person should look at this file before anything uses it; None for a hand-set or accepted one."""
    if result.method == MANUAL or result.accepted or result.target == result.reference:
        return None
    if result.method == UNALIGNED:
        return f"not aligned: {result.note}" if result.note else "not aligned"
    if low_confidence(result, cfg):
        return "low confidence"
    if drift_flagged(result):
        return "drift"
    return None


def metadata_offset(reference: str | None, target: str | None) -> float | None:
    """Start-time difference from container tags, same sign as offset_s. A hint for a person
    only: device clocks are set by hand and some devices write local time labelled as UTC."""
    try:
        return (datetime.fromisoformat(target) - datetime.fromisoformat(reference)).total_seconds()
    except (TypeError, ValueError):
        return None


def align_recording(
    recording: Recording,
    cfg: AlignSection,
    infos: dict,
    on_pair: Callable[[np.ndarray, np.ndarray, AlignResult], None] | None = None,
) -> list[AlignResult]:
    """Aligns every file to one reference that has audio. infos maps file name to its probe
    result; on_pair sees each correlated pair's signals, e.g. to draw it for review."""
    audible = [item for item in recording.media if infos[item.path.name].audio_streams()]
    if not audible:
        raise AlignmentFailedError(f"{recording.recording_id}: no file has an audio stream to align by")
    reference = recording.reference_media(among=audible)
    reference_name = reference.path.name
    reference_audio = load_normalised(reference.path, limit_s=None)

    results: list[AlignResult] = []
    for item in recording.media:
        name = item.path.name
        if item is reference:
            results.append(AlignResult(reference=name, target=name, offset_s=0.0, confidence=1.0))
            continue
        hint = metadata_offset(infos[reference_name].creation_time, infos[name].creation_time)
        unaligned = AlignResult(
            reference=reference_name, target=name, offset_s=0.0, confidence=None,
            method=UNALIGNED, metadata_offset_s=hint, note="no audio stream",
        )
        if item not in audible:
            results.append(unaligned)
            continue
        try:
            target_audio = load_normalised(item.path, limit_s=None)
            result = align_signals(reference_audio, target_audio, cfg, names=(reference_name, name))
        except MacawError as error:
            # one unreadable file should be reported, not stop the other files from aligning
            unaligned.note = str(error)
            results.append(unaligned)
            continue
        result.metadata_offset_s = hint
        if on_pair is not None:
            on_pair(reference_audio, target_audio, result)
        results.append(result)
    return results


def keep_manual(
    fresh: list[AlignResult], previous: list[AlignResult]
) -> tuple[list[AlignResult], list[str]]:
    """A re-run never overwrites an offset a person set, unless the reference it was set
    against is no longer the reference. Returns the merged table and the targets dropped."""
    reference = next((result.reference for result in fresh), None)
    manual = {result.target: result for result in previous if result.method == MANUAL}
    stale = [target for target, result in manual.items() if result.reference != reference]
    merged = [
        manual[result.target] if result.target in manual and result.target not in stale else result
        for result in fresh
    ]
    return merged, stale


def keep_accepted(
    fresh: list[AlignResult], previous: list[AlignResult]
) -> tuple[list[AlignResult], list[str]]:
    """A re-run keeps a person's acceptance while it still fits; also returns the targets it no longer fits."""
    accepted = {result.target: result for result in previous if result.accepted}
    dropped = []
    for result in fresh:
        old = accepted.get(result.target)
        if old is None or result.method == MANUAL:
            continue
        same = old.reference == result.reference and old.method == result.method and (
            result.method == UNALIGNED or abs(old.offset_s - result.offset_s) <= AGREEMENT_TOLERANCE_S
        )
        if same:
            result.accepted = old.accepted
        else:
            dropped.append(result.target)
    return fresh, dropped


def write_offset_table(results: list[AlignResult], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        entry = asdict(result)
        if result.segment_offsets is not None:
            entry["segment_offsets"] = [list(pair) for pair in result.segment_offsets]
        payload.append(entry)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_offset_table(path: Path) -> list[AlignResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for entry in payload:
        segments = entry.pop("segment_offsets", None)
        results.append(
            AlignResult(**entry, segment_offsets=[tuple(pair) for pair in segments] if segments else None)
        )
    return results


def apply_alignment(
    recording: Recording, results: list[AlignResult], *, recut: bool = False
) -> Recording:
    """Default is to record the offset and let readers add it; recut re-encodes instead.
    Files are matched by name, so a table made on another machine still applies."""
    by_name = {result.target: result for result in results}
    for item in recording.media:
        result = by_name.get(item.path.name)
        if result is None:
            continue
        item.offset_s = result.offset_s
        item.drift_ppm = result.drift_ppm
        item.offset_confidence = result.confidence

    if recut:
        from macaw.media.ffmpeg import apply_offset

        for item in recording.media:
            if item.offset_s:
                dest = item.path.parent / f"{item.path.stem}_aligned{item.path.suffix}"
                item.path = apply_offset(item.path, dest, offset_s=item.offset_s)
                item.offset_s = 0.0
    return recording
