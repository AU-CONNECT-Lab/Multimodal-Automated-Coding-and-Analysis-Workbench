"""Multi-source alignment by generalised cross-correlation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from macaw.config.schema import AlignSection
from macaw.exceptions import AlignmentFailedError, MacawError
from macaw.media.ffmpeg import decode_audio
from macaw.project.recording import Recording

SAMPLE_RATE = 16000
# an hour-long file would make the FFT the slowest thing in the pipeline; the global offset
# is already settled by the opening minutes
MAX_CORRELATION_S = 600.0


@dataclass
class AlignResult:
    reference: Path
    target: Path
    offset_s: float
    confidence: float                # peak sharpness, 0-1
    drift_ppm: float | None = None
    drift_r2: float | None = None
    segment_offsets: list[tuple[float, float]] | None = None


def gcc_phat(
    a: np.ndarray, b: np.ndarray, sample_rate: float, max_lag_s: float
) -> tuple[float, float]:
    """Returns (lag_s, confidence). PHAT weighting is what makes this robust to room response."""
    if a.size == 0 or b.size == 0:
        raise AlignmentFailedError("Cannot correlate an empty signal")

    size = int(2 ** np.ceil(np.log2(a.size + b.size)))
    spectrum = np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size))
    # PHAT: keep only the phase. Magnitude carries the room and the mic, which differ
    # between the two recorders and would otherwise dominate the peak.
    magnitude = np.abs(spectrum)
    spectrum /= np.maximum(magnitude, 1e-12)
    correlation = np.fft.irfft(spectrum, size)

    max_lag = int(min(max_lag_s * sample_rate, size // 2))
    window = np.concatenate((correlation[-max_lag:], correlation[: max_lag + 1]))
    peak_index = int(np.argmax(window))
    peak = float(window[peak_index])
    if peak <= 0:
        return 0.0, 0.0

    lag_s = (peak_index - max_lag) / sample_rate

    # sharpness against the best competing peak, not against the noise floor: a periodic
    # signal produces many tall peaks and that is exactly the case worth flagging
    exclusion = max(int(0.005 * sample_rate), 1)
    masked = window.copy()
    low = max(peak_index - exclusion, 0)
    masked[low : peak_index + exclusion + 1] = 0.0
    runner_up = float(np.max(masked)) if masked.size else 0.0
    confidence = float(np.clip(1.0 - max(runner_up, 0.0) / peak, 0.0, 1.0))
    return lag_s, confidence


def load_normalised(path: Path, limit_s: float | None = MAX_CORRELATION_S) -> np.ndarray:
    samples = decode_audio(path, sample_rate=SAMPLE_RATE)
    if limit_s is not None:
        samples = samples[: int(limit_s * SAMPLE_RATE)]
    samples = samples - samples.mean()
    peak = np.max(np.abs(samples))
    return samples / peak if peak > 0 else samples


def align_pair(reference: Path, target: Path, cfg: AlignSection) -> AlignResult:
    reference_audio = load_normalised(reference)
    target_audio = load_normalised(target)
    lag_s, confidence = gcc_phat(reference_audio, target_audio, SAMPLE_RATE, cfg.search_window_s)

    result = AlignResult(
        reference=Path(reference), target=Path(target), offset_s=lag_s, confidence=confidence
    )
    if cfg.drift_check:
        from macaw.media.drift import check_drift

        drift_ppm, r_squared, _ = check_drift(reference, target, cfg)
        result.drift_ppm = drift_ppm
        result.drift_r2 = r_squared
    return result


def align_recording(recording: Recording, cfg: AlignSection) -> list[AlignResult]:
    """Aligns every media file to recording.reference_media()."""
    reference = recording.reference_media()
    results: list[AlignResult] = []
    for item in recording.media:
        if item.path == reference.path:
            results.append(
                AlignResult(reference=reference.path, target=item.path, offset_s=0.0, confidence=1.0)
            )
            continue
        try:
            results.append(align_pair(reference.path, item.path, cfg))
        except MacawError as error:
            # a file with no usable audio cannot be aligned; record it at zero confidence
            # so QC reports it instead of the run stopping here
            results.append(
                AlignResult(
                    reference=reference.path,
                    target=item.path,
                    offset_s=0.0,
                    confidence=0.0,
                    segment_offsets=None,
                )
            )
            recording.metadata.setdefault("align_errors", {})[str(item.path)] = str(error)
    return results


def write_offset_table(results: list[AlignResult], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        entry = asdict(result)
        entry["reference"] = str(result.reference)
        entry["target"] = str(result.target)
        if result.segment_offsets is not None:
            entry["segment_offsets"] = [list(pair) for pair in result.segment_offsets]
        payload.append(entry)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_offset_table(path: Path) -> list[AlignResult]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = []
    for entry in payload:
        segments = entry.get("segment_offsets")
        results.append(
            AlignResult(
                reference=Path(entry["reference"]),
                target=Path(entry["target"]),
                offset_s=entry["offset_s"],
                confidence=entry["confidence"],
                drift_ppm=entry.get("drift_ppm"),
                drift_r2=entry.get("drift_r2"),
                segment_offsets=[tuple(pair) for pair in segments] if segments else None,
            )
        )
    return results


def apply_alignment(
    recording: Recording, results: list[AlignResult], *, recut: bool = False
) -> Recording:
    """Default is to record the offset and let readers add it; recut re-encodes instead."""
    offsets = {result.target: result for result in results}
    for item in recording.media:
        result = offsets.get(item.path)
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
