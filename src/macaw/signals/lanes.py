"""What the Signals page draws under its player, on the played copy's own clock: carpets, lines and coded spans."""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np
import pandas as pd

# columns of a carpet image over the whole played copy; zooming in shows them as blocks
CARPET_COLUMNS = 3000
# rows of a carpet image at most, two pixels each at the lane's tallest
CARPET_ROWS = 192
# points of a line over the whole played copy
LINE_POINTS = 3000
# a channel's z-score beyond this is drawn as far as it goes, as nirspipe's carpets are
CARPET_Z = 3.0
COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c")


def png(grey: np.ndarray, alpha: np.ndarray) -> str:
    """A data URI of an 8-bit grey image with transparency, rows top to bottom."""
    height, width = grey.shape
    pixels = np.stack([grey, alpha], axis=-1).astype(np.uint8).reshape(height, width * 2)
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    body = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 4, 0, 0, 0))
    body += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(body).decode("ascii")


def _bins(played: np.ndarray, length: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.floor(np.asarray(played, dtype=float) / length * count).astype(np.int64)
    return index, (index >= 0) & (index < count)


def _binned_mean(index: np.ndarray, keep: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    use = keep & np.isfinite(values)
    sums = np.bincount(index[use], weights=values[use], minlength=count)
    counts = np.bincount(index[use], minlength=count)
    out = np.full(count, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def detrended_z(values: np.ndarray) -> np.ndarray:
    """A channel less its straight-line trend, in its own standard deviations; NaN where it has no number."""
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return out
    x = np.flatnonzero(finite).astype(float)
    slope, intercept = np.polyfit(x, values[finite], 1)
    rest = values[finite] - (slope * x + intercept)
    spread = rest.std()
    out[finite] = rest / spread if spread > 0 else 0.0
    return out


def channel_z(samples: pd.DataFrame, name: str, played: np.ndarray, length: float) -> np.ndarray:
    """A channel less its trend in its own standard deviations over what the copy shows, as its carpet row is."""
    values = samples[name].to_numpy(dtype=float).copy()
    values[(played < 0) | (played > length)] = np.nan
    return detrended_z(values)


def carpet(samples: pd.DataFrame, channels: list[str], played: np.ndarray, length: float) -> dict | None:
    """Each channel a row, less its trend and in its own standard deviations over what the copy shows, grey from -3 to 3."""
    shown = (played >= 0) & (played <= length)
    if not channels or not shown.any():
        return None
    count = int(min(CARPET_COLUMNS, max(shown.sum(), 1)))
    index, keep = _bins(played, length, count)
    rows = []
    for name in channels:
        rows.append(_binned_mean(index, keep, channel_z(samples, name, played, length), count))
    z, names = np.vstack(rows), list(channels)
    if len(names) > CARPET_ROWS:
        z, names = _pooled_rows(z, names)
    grey = np.where(np.isfinite(z), (np.clip(z, -CARPET_Z, CARPET_Z) + CARPET_Z) / (2 * CARPET_Z) * 255, 0)
    return {"image": png(grey, np.where(np.isfinite(z), 255, 0)), "rows": names}


def _pooled_rows(z: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Neighbouring channels as one row each, keeping per column the value farthest from 0, so a poor channel still stands out."""
    pooled, labels = [], []
    for group in np.array_split(np.arange(len(names)), CARPET_ROWS):
        block = z[group]
        farthest = np.argmax(np.where(np.isfinite(block), np.abs(block), -1.0), axis=0)
        pooled.append(block[farthest, np.arange(block.shape[1])])
        labels.append(names[group[0]] if group.size == 1 else f"{names[group[0]]} to {names[group[-1]]}")
    return np.vstack(pooled), labels


def lines(series: list[tuple[str, np.ndarray, np.ndarray]], length: float, *, scaled: bool = False,
          bounds: tuple[float, float] | None = None, unit: str = "") -> dict | None:
    """(name, played times, values) as one lane of lines on a shared scale, or each scaled to its own range; bounds fixes the scale."""
    count = LINE_POINTS
    drawn, lows, highs = [], [], []
    for position, (name, played, values) in enumerate(series):
        index, keep = _bins(played, length, count)
        mean = _binned_mean(index, keep, np.asarray(values, dtype=float), count)
        if not np.isfinite(mean).any():
            continue
        low, high = np.nanpercentile(mean, 1), np.nanpercentile(mean, 99)
        if scaled:
            mean = (mean - low) / (high - low) if high > low else mean * 0
            low, high = 0.0, 1.0
        lows.append(low)
        highs.append(high)
        drawn.append({"name": name, "colour": COLOURS[position % len(COLOURS)],
                      "values": [None if not np.isfinite(v) else round(float(v), 4) for v in mean]})
    if not drawn:
        return None
    low, high = bounds or (float(min(lows)), float(max(highs)))
    if high <= low:
        high = low + 1.0
    return {"step": length / count, "series": drawn, "low": low, "high": high, "scaled": scaled, "unit": unit}


def spans(rows: list[tuple[float, float, str]], played_of) -> list[dict]:
    """Coded spans on the played copy's clock, as the strip draws them."""
    return [{"a": round(float(played_of(a)), 3), "b": round(float(played_of(b)), 3), "label": label} for a, b, label in rows]
