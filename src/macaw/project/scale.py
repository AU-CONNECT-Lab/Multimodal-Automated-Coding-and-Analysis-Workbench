"""How many centimetres one pixel spans on a recording's video: a yardstick a person measured on a
frame, or a number written in the settings. Results stay in pixels; exports add centimetres."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HAND = "settings"
MEASURED = "yardstick"
CM_SUFFIX = "_cm"
# a person's apparent size is taken over stretches this long, long enough to sit up straight once
SIZE_WINDOW_S = 30.0
_SIZE_MIN_FRAMES = 10
_PIXEL_UNIT = re.compile(r"^px(?:\^(\d))?(/.*)?$")


@dataclass
class Measurement:
    video: str
    t: float                       # recording time of the frame it was measured on
    points: list[list[float]]      # the two ends, in pixels of the frames the video steps read
    length_cm: float
    object: str | None = None
    by: str | None = None
    at: str | None = None

    @property
    def length_px(self) -> float:
        (x1, y1), (x2, y2) = self.points
        return float(np.hypot(x2 - x1, y2 - y1))

    @property
    def cm_per_px(self) -> float:
        return self.length_cm / self.length_px


@dataclass
class RecordingScale:
    cm_per_px: float
    source: str                    # HAND or MEASURED
    video: str | None              # where the yardstick was measured; None when written by hand
    spread: float | None           # standard deviation over two or more measurements
    measurements: list[Measurement]

    def describe(self) -> str:
        if self.source == HAND:
            return f"{self.cm_per_px:.4g} cm per pixel, written in scale.cm_per_px"
        objects = sorted({m.object for m in self.measurements if m.object})
        what = f" of {', '.join(objects)}" if objects else ""
        spread = f" ± {self.spread:.2g}" if self.spread is not None else ""
        return f"{self.cm_per_px:.4g}{spread} cm per pixel from {len(self.measurements)} measurement(s){what} on {self.video}"


def read_measurements(path: Path) -> list[Measurement]:
    if not path.is_file():
        return []
    return [Measurement(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def write_measurements(measurements: list[Measurement], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in measurements], indent=2), encoding="utf-8")
    return path


def recording_scale(config, layout, recording_id: str) -> RecordingScale | None:
    """config is the one this recording runs, so a scale.cm_per_px in its own file counts."""
    measurements = read_measurements(layout.scale_path(recording_id))
    if config.scale.cm_per_px is not None:
        return RecordingScale(config.scale.cm_per_px, HAND, None, None, measurements)
    if not measurements:
        return None
    ratios = [item.cm_per_px for item in measurements]
    spread = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else None
    return RecordingScale(float(np.mean(ratios)), MEASURED, measurements[0].video, spread, measurements)


def pixel_power(unit: str | None) -> int | None:
    """How many pixel lengths a unit holds: 1 for px or px/s, 2 for px^2; None for any other unit."""
    match = _PIXEL_UNIT.match(unit or "")
    return int(match.group(1) or 1) if match else None


def in_centimetres(unit: str) -> str:
    return "cm" + unit[2:]


def centimetre_rows(rows: pd.DataFrame, units: dict[str, str | None], cm_per_px: float) -> pd.DataFrame:
    """A copy of every pixel-unit feature's rows in centimetres, each named <feature>_cm."""
    powers = {name: power for name, power in ((name, pixel_power(unit)) for name, unit in units.items()) if power}
    picked = rows[rows["feature"].isin(list(powers))].copy()
    factors = cm_per_px ** picked["feature"].map(powers).astype(float).to_numpy()
    picked["value"] = picked["value"].astype(float).to_numpy() * factors
    picked["feature"] = picked["feature"] + CM_SUFFIX
    return picked


def other_videos(keypoints: pd.DataFrame, video: str) -> list[str]:
    """Videos besides the yardstick's that people were measured on."""
    seen = keypoints[keypoints["subject_id"].notna() & keypoints["x"].notna()]
    return sorted(set(seen["video"].astype(str)) - {video})


def apparent_size(person: pd.DataFrame) -> pd.Series:
    """Per frame, the distance from the nose to the midpoint of the shoulders, indexed by time."""
    wide = person.pivot_table(index="t", columns="joint", values=["x", "y"], aggfunc="first")
    needed = ["nose", "left_shoulder", "right_shoulder"]
    if not all(("x", joint) in wide.columns for joint in needed):
        return pd.Series(dtype=float)
    mid_x = (wide[("x", "left_shoulder")] + wide[("x", "right_shoulder")]) / 2.0
    mid_y = (wide[("y", "left_shoulder")] + wide[("y", "right_shoulder")]) / 2.0
    return np.hypot(wide[("x", "nose")] - mid_x, wide[("y", "nose")] - mid_y).dropna()


def size_change(size: pd.Series, window_s: float = SIZE_WINDOW_S) -> tuple[float, float] | None:
    """(typical size, largest share by which a stretch departs from it). Each stretch counts its 90th
    percentile, since turning or bending only ever shortens a body part on the picture."""
    if size.empty:
        return None
    stretch = np.floor((size.index.to_numpy(dtype=float) - float(size.index.min())) / window_s)
    groups = size.groupby(stretch)
    tops = groups.quantile(0.9)[groups.size() >= _SIZE_MIN_FRAMES]
    if tops.empty:
        return None
    typical = float(tops.median())
    return typical, float((tops - typical).abs().max() / typical)
