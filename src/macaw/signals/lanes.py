"""What the Signals page draws under its player, on the played copy's own clock: stacked channels, lines and coded spans."""

from __future__ import annotations

import base64

import numpy as np

# points of a line over the whole played copy
LINE_POINTS = 3000
COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c")
# a stack's rows as change from each channel's mean, as % of it (raw light), or each scaled to its own range
DIFFERENCE, RELATIVE, OWN = "difference", "relative", "own"
# a stack's points go as 16-bit integers in steps of this share of the scale; the lowest integer is a gap
PACK_STEP = 0.002
PACK_GAP = -32768


def _bins(played: np.ndarray, length: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    index = np.floor(np.asarray(played, dtype=float) / length * count).astype(np.int64)
    return index, (index >= 0) & (index < count)


def point_count(played: list[np.ndarray], length: float) -> int:
    """Points over the played copy: no more than the densest series has samples on it, so a slow one draws a joined line."""
    within = [int(np.count_nonzero((np.asarray(times) >= 0) & (np.asarray(times) <= length))) for times in played]
    return int(min(LINE_POINTS, max(max(within, default=0), 2)))


def _binned_mean(index: np.ndarray, keep: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    use = keep & np.isfinite(values)
    sums = np.bincount(index[use], weights=values[use], minlength=count)
    counts = np.bincount(index[use], minlength=count)
    out = np.full(count, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def lines(series: list[tuple[str, np.ndarray, np.ndarray]], length: float, *, scaled: bool = False,
          bounds: tuple[float, float] | None = None, unit: str = "") -> dict | None:
    """(name, played times, values) as one lane of lines on a shared scale, or each scaled to its own range; bounds fixes the scale."""
    count = point_count([played for _, played, _ in series], length)
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


def stack(rows: list[tuple[str, list[tuple[str, str, np.ndarray, np.ndarray]]]], length: float, *, mode: str = DIFFERENCE,
          unit: str = "", events: list[float] | None = None) -> dict | None:
    """Rows of (name, its series as (name, colour, played times, values)) stacked as a montage on one scale, in its units, events as lines."""
    count = point_count([played for _, members in rows for _, _, played, _ in members], length)
    drawn, spreads = [], []
    for row, series in rows:
        out = []
        for name, colour, played, values in series:
            index, keep = _bins(played, length, count)
            mean = _binned_mean(index, keep, np.asarray(values, dtype=float), count)
            if not np.isfinite(mean).any():
                continue
            if mode == RELATIVE:
                level = np.nanmean(mean)
                mean = (mean / level - 1.0) * 100.0 if level else mean * np.nan
            if mode == OWN:
                low, high = np.nanpercentile(mean, 1), np.nanpercentile(mean, 99)
                mean = (2.0 * (mean - low) / (high - low) - 1.0) if high > low else mean * 0
            else:
                mean = mean - np.nanmedian(mean)
                spreads.append(float(np.nanpercentile(np.abs(mean), 98)))
            out.append({"name": name, "colour": colour, "values": mean})
        if out:
            drawn.append({"name": row, "series": out})
    if not drawn:
        return None
    scale = 1.0 if mode == OWN else float(np.median(spreads)) if spreads and np.median(spreads) > 0 else 1.0
    # sent in units of the scale, so values as small as molar concentrations keep their shape
    for row in drawn:
        for series in row["series"]:
            series["packed"] = pack(series.pop("values") / scale)
    return {"step": length / count, "rows": drawn, "scale": scale, "unit": "%" if mode == RELATIVE else unit, "own": mode == OWN,
            "events": [round(float(t), 3) for t in events or () if 0.0 <= t <= length]}


def pack(values: np.ndarray) -> str:
    """Values as base64 of little-endian 16-bit integers in steps of PACK_STEP, a gap where there is no number."""
    steps = np.clip(np.round(np.nan_to_num(values) / PACK_STEP), PACK_GAP + 1, -PACK_GAP - 1).astype("<i2")
    steps[~np.isfinite(values)] = PACK_GAP
    return base64.b64encode(steps.tobytes()).decode("ascii")


def unpack(text: str) -> np.ndarray:
    steps = np.frombuffer(base64.b64decode(text), dtype="<i2")
    return np.where(steps == PACK_GAP, np.nan, steps * PACK_STEP)
