"""A file's clock found from its IMU: the cap's turning speed matched to the head's turning speed on the video."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, savgol_filter

from macaw.exceptions import ConfigError
from macaw.media.align import sharpness
from macaw.media.drift import MIN_SEGMENTS, estimate_drift

# the common grid both speeds are put on
GRID_HZ = 10.0
# head movement stays alike for a few seconds, so a rival peak nearer than this is the same match
EXCLUSION_S = 5.0
# the least time both speeds must share for a match to count, or half the video when that is shorter
MIN_OVERLAP_S = 30.0
# frames of the per-frame angles a median is taken over before differencing, against one-frame jumps
ANGLE_MEDIAN_FRAMES = 5
# s; the head's speed is the slope of its angles over this long, as a frame-to-frame slope is mostly the tracker's jitter
SLOPE_S = 0.4
# speeds above this percentile are clipped to it, so a tracking glitch or a knock does not decide the match
CLIP_PERCENTILE = 99.0
# s; how far a stretch's own match may lie from the whole file's and still be looked for
SEGMENT_SEARCH_S = 10.0
# s; a stretch whose match lies within this of the whole file's agrees with it
AGREEMENT_S = 0.3
AXIS_LETTERS = ("x", "y", "z")
FACE_ANGLES = ("face_yaw_deg", "face_pitch_deg", "face_roll_deg")
POSE_ANGLES = ("head_roll_deg",)


@dataclass
class Match:
    offset_s: float                      # recording time minus the file's own time
    confidence: float                    # peak sharpness against the best rival, 0-1
    correlation: float                   # Pearson r at the peak
    overlap_s: float
    agreement: float | None              # share of stretches finding the same offset; None with too few
    segments: list[tuple[float, float, float]] = field(default_factory=list)  # (midpoint on the recording clock, offset_s, confidence)
    drift_ppm: float | None = None
    drift_r2: float | None = None
    drift_intercept_s: float | None = None


def gyro_sensors(names: list[str]) -> dict[str, list[str]]:
    """Gyroscope axes grouped by sensor: a name less its x, y or z token names the sensor."""
    sensors: dict[str, list[str]] = {}
    for name in sorted(names):
        tokens = [token for token in re.split(r"[_\-\s]+", name.lower()) if token]
        kept = [token for token in tokens if token not in AXIS_LETTERS]
        if len(kept) == len(tokens) and len(tokens[-1]) > 1 and tokens[-1][-1] in AXIS_LETTERS:
            kept[-1] = kept[-1][:-1]   # GyroX: the axis letter written onto the name
        sensors.setdefault("_".join(kept), []).append(name)
    # axes told apart only by number (gyro_1, gyro_2, gyro_3) are one sensor
    if len(sensors) > 1 and all(len(axes) == 1 for axes in sensors.values()):
        return {"gyro": sorted(names)}
    return sensors


def sensor_pattern(axes: list[str]) -> str:
    """A signals.gyro value matching just these axes."""
    name = axes[0].lower()
    found = re.search(r"(?<![a-z])[xyz](?![a-z])", name) or re.search(r"[xyz]$", name)
    if found is None or len(axes) == 1:
        return "^(" + "|".join(_escaped(axis.lower()) for axis in axes) + ")$"
    return "^" + _escaped(name[: found.start()]) + "[xyz]" + _escaped(name[found.end():]) + "$"


def _escaped(text: str) -> str:
    return re.sub(r"([.^$*+?{}\[\]\\|()])", r"\\\1", text)


def gyro_speed(aux: pd.DataFrame, pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(own times, deg/s): the turning speed over one gyroscope's axes, each less its median; None without a gyroscope."""
    names = sorted(str(name) for name in aux["aux"].unique() if re.search(pattern, str(name), re.IGNORECASE))
    if not names:
        return None
    sensors = gyro_sensors(names)
    if len(sensors) > 1:
        choices = ", ".join(f"'{sensor_pattern(axes)}' ({', '.join(axes)})" for axes in sensors.values())
        raise ConfigError(f"signals.gyro {pattern!r} matches {len(sensors)} gyroscopes in the file; set signals.gyro to the one "
                          f"worn on the head, one of {choices}")
    first = aux[aux["aux"] == names[0]].sort_values("time")
    times = first["time"].to_numpy(dtype=float)
    squared = np.zeros(times.size)
    for name in names:
        axis = aux[aux["aux"] == name].sort_values("time")
        own, values = axis["time"].to_numpy(dtype=float), axis["value"].to_numpy(dtype=float)
        if own.size != times.size or not np.allclose(own, times):
            values = np.interp(times, own, values)
        squared += (values - np.nanmedian(values)) ** 2
    return times, np.sqrt(squared)


def head_speed(frames: pd.DataFrame, angles: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray] | None:
    """(recording times, deg/s): how fast the head turns on the video, from per-frame angles; NaN where they are missing."""
    frames = frames[frames["feature"].astype(str).isin(angles)]
    if frames.empty:
        return None
    wide = frames.pivot_table(index="t_start", columns="feature", values="value", observed=True).sort_index()
    times = wide.index.to_numpy(dtype=float)
    if times.size < 3:
        return None
    period = float(np.median(np.diff(times)))
    if period <= 0:
        return None
    grid = times[0] + period * np.arange(int(round((times[-1] - times[0]) / period)) + 1)
    right = np.clip(np.searchsorted(times, grid), 0, times.size - 1)
    left = np.clip(right - 1, 0, times.size - 1)
    # a grid point with no frame within half a period lies in a gap
    seen = np.minimum(np.abs(times[right] - grid), np.abs(times[left] - grid)) <= 0.5 * period
    length = min(max(int(round(SLOPE_S / period)) | 1, 5), grid.size - (grid.size % 2 == 0))
    if length < 3:
        return None
    squared = np.zeros(grid.size)
    for name in wide.columns:
        values = wide[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        seen &= np.interp(grid, times, finite.astype(float)) > 0.5
        angle = np.interp(grid, times[finite], values[finite])
        angle = pd.Series(angle).rolling(ANGLE_MEDIAN_FRAMES, center=True, min_periods=1).median().to_numpy()
        squared += savgol_filter(angle, length, min(2, length - 1), deriv=1, delta=period) ** 2
    speed = np.sqrt(squared)
    speed[~seen] = np.nan
    return grid, speed


def on_grid(times: np.ndarray, values: np.ndarray, start: float, count: int) -> np.ndarray:
    """The mean of the finite values in each 1/GRID_HZ bin from start, NaN for a bin with none."""
    values = np.asarray(values, dtype=float)
    index = np.floor((np.asarray(times, dtype=float) - start) * GRID_HZ).astype(np.int64)
    keep = np.isfinite(values) & (index >= 0) & (index < count)
    sums = np.bincount(index[keep], weights=values[keep], minlength=count)
    counts = np.bincount(index[keep], minlength=count)
    out = np.full(count, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def standardised(values: np.ndarray) -> np.ndarray:
    """Clipped at the top percentile, then centred and scaled; all NaN when there is nothing to scale."""
    if np.isfinite(values).sum() < 3:
        return np.full(values.shape, np.nan)
    values = np.minimum(values, np.nanpercentile(values, CLIP_PERCENTILE))
    spread = np.nanstd(values)
    return (values - np.nanmean(values)) / spread if spread > 0 else np.full(values.shape, np.nan)


def masked_pearson(video: np.ndarray, imu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lags, r, n): Pearson r of video[i + lag] with imu[i] over the bins both have, for every lag, and the bins it rests on."""
    mv, mi = np.isfinite(video).astype(float), np.isfinite(imu).astype(float)
    v, u = np.where(mv > 0, video, 0.0), np.where(mi > 0, imu, 0.0)

    def corr(a, b):
        return correlate(a, b, mode="full", method="fft")

    n = np.rint(corr(mv, mi))
    sxy, sx, sy, sxx, syy = corr(v, u), corr(v, mi), corr(mv, u), corr(v * v, mi), corr(mv, u * u)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (n * sxy - sx * sy) / np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))
    r[(n < 3) | ~np.isfinite(r)] = np.nan
    return correlation_lags(video.size, imu.size, mode="full"), r, n


def _vertex(scores: np.ndarray, best: int) -> float:
    """Where a parabola through the peak and its neighbours tops out, in bins from the peak."""
    if 0 < best < scores.size - 1 and np.all(np.isfinite(scores[best - 1 : best + 2])):
        left, middle, right = scores[best - 1 : best + 2]
        bend = left - 2 * middle + right
        if bend < 0:
            return float(np.clip(0.5 * (left - right) / bend, -0.5, 0.5))
    return 0.0


def _peak(lags, r, n, start_gap_s: float, allowed: np.ndarray) -> tuple[float, float, float, float] | None:
    """(offset_s, confidence, r, overlap_s) of the best allowed lag, or None when no lag is allowed."""
    scores = np.where(allowed & np.isfinite(r), r, -np.inf)
    if not np.isfinite(scores).any():
        return None
    best, confidence = sharpness(scores, int(EXCLUSION_S * GRID_HZ))
    offset = start_gap_s + (lags[best] + _vertex(scores, best)) / GRID_HZ
    return float(offset), float(confidence), float(r[best]), float(n[best] / GRID_HZ)


def _on_grids(video_times, video_speed, imu_times, imu_speed) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Both speeds standardised on GRID_HZ grids from their own first times, the video's start, and the gap between the starts."""
    video_start, imu_start = float(np.min(video_times)), float(np.min(imu_times))
    video = standardised(on_grid(video_times, video_speed, video_start, int((np.max(video_times) - video_start) * GRID_HZ) + 1))
    imu = standardised(on_grid(imu_times, imu_speed, imu_start, int((np.max(imu_times) - imu_start) * GRID_HZ) + 1))
    return video, imu, video_start, video_start - imu_start


def correlation_curve(video_times, video_speed, imu_times, imu_speed) -> tuple[np.ndarray, np.ndarray]:
    """(offset_s, r) at every offset that rests on enough overlap, as match() weighs them."""
    video, imu, _, gap = _on_grids(video_times, video_speed, imu_times, imu_speed)
    lags, r, n = masked_pearson(video, imu)
    allowed = n >= min(MIN_OVERLAP_S, 0.5 * video.size / GRID_HZ) * GRID_HZ
    return gap + lags[allowed] / GRID_HZ, r[allowed]


def match(
    video_times: np.ndarray,
    video_speed: np.ndarray,
    imu_times: np.ndarray,
    imu_speed: np.ndarray,
    *,
    hint_s: float | None = None,
    window_s: float | None = None,
    segment_s: float = 60.0,
    min_confidence: float = 0.3,
) -> Match | None:
    """The offset (recording time minus own time) at which the IMU's speed best follows the head's on the video."""
    video, imu, video_start, gap = _on_grids(video_times, video_speed, imu_times, imu_speed)
    if not np.isfinite(video).any() or not np.isfinite(imu).any():
        return None
    lags, r, n = masked_pearson(video, imu)
    offsets = gap + lags / GRID_HZ
    allowed = n >= min(MIN_OVERLAP_S, 0.5 * video.size / GRID_HZ) * GRID_HZ
    if hint_s is not None and window_s is not None:
        allowed &= np.abs(offsets - hint_s) <= window_s
    found = _peak(lags, r, n, gap, allowed)
    if found is None:
        return None
    offset, confidence, correlation, overlap = found
    segments = _segments(video, imu, video_start, gap, offset, segment_s)
    agreeing = [abs(at - offset) <= AGREEMENT_S and sure > 0 for _, at, sure in segments]
    result = Match(
        offset_s=offset, confidence=confidence, correlation=correlation, overlap_s=overlap,
        agreement=sum(agreeing) / len(agreeing) if len(segments) >= MIN_SEGMENTS else None, segments=segments,
    )
    sure = [point for point in segments if point[2] >= min_confidence]
    if len(sure) >= MIN_SEGMENTS:
        result.drift_ppm, result.drift_r2, result.drift_intercept_s = estimate_drift(sure)
    return result


def _segments(video, imu, video_start: float, gap: float, offset: float, segment_s: float) -> list[tuple[float, float, float]]:
    """(midpoint, offset_s, confidence) for each stretch of the video, each matched on its own near the whole file's offset."""
    size = max(int(round(segment_s * GRID_HZ)), 1)
    out = []
    for first in range(0, video.size - size // 2, size):
        piece = video[first : first + size]
        if np.isfinite(piece).sum() < size // 2:
            continue
        lags, r, n = masked_pearson(piece, imu)
        piece_gap = gap + first / GRID_HZ
        allowed = (np.abs(piece_gap + lags / GRID_HZ - offset) <= SEGMENT_SEARCH_S) & (n >= np.isfinite(piece).sum() // 2)
        found = _peak(lags, r, n, piece_gap, allowed)
        if found is not None:
            out.append((video_start + (first + piece.size / 2) / GRID_HZ, found[0], found[1]))
    return out
