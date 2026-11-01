"""How many centimetres one pixel spans: an object of known length a person measured on a frame of the
recording or of a calibration clip, or a number written in the settings, for everyone or per role. Results
stay in pixels; exports add centimetres. A scale holds only at the distance of the object it came from."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import MacawError
from macaw.project.recording import VIDEO_SUFFIXES

HAND = "settings"
MEASURED = "recording video"
CALIBRATION = "calibration clip"
CM_SUFFIX = "_cm"
DEPTH_NOTE = "right only for people as far from the camera as the object measured"
_SIZE_MIN_FRAMES = 10
_PIXEL_UNIT = re.compile(r"^px(?:\^(\d))?(/.*)?$")


@dataclass
class Measurement:
    video: str                     # the file it was measured on: one of the recording's videos, or a calibration clip
    t: float                       # time of the frame, on the recording's clock or the clip's own
    points: list[list[float]]      # the two ends, in pixels of the picture shown
    length_cm: float
    person: str | None = None      # role at whose place the object stood; None for everyone
    object: str | None = None
    by: str | None = None
    at: str | None = None
    frame_size: list[int] | None = None   # width and height of a calibration clip's picture

    @property
    def length_px(self) -> float:
        (x1, y1), (x2, y2) = self.points
        return float(np.hypot(x2 - x1, y2 - y1))

    @property
    def cm_per_px(self) -> float:
        return self.length_cm / self.length_px


@dataclass
class RecordingScale:
    by_role: dict[str | None, float]           # None: everyone without a number of their own
    spread: dict[str | None, float | None]     # standard deviation over two or more measurements
    source: str                                # HAND, MEASURED or CALIBRATION
    video: str | None                          # the recording's video or the clip it was measured on
    measurements: list[Measurement]

    def for_role(self, role: str | None) -> float | None:
        return self.by_role.get(role, self.by_role.get(None))

    def factor(self, role: str | None, role_b: str | None) -> float | None:
        """cm per pixel for a row: one person's own, a pair's when both share it, else everyone's."""
        if role_b is not None:
            first, second = self.for_role(role), self.for_role(role_b)
            return first if first is not None and first == second else None
        return self.for_role(role) if role is not None else self.by_role.get(None)

    def values(self) -> str:
        return "cm per pixel " + "; ".join(
            f"{value:.4g}{f' ± {self.spread[role]:.2g}' if self.spread.get(role) is not None else ''} for {role or 'everyone'}"
            for role, value in sorted(self.by_role.items(), key=lambda item: (item[0] is not None, item[0] or ""))
        )

    def describe(self) -> str:
        if self.source == HAND:
            return f"{self.values()}, written in the settings"
        objects = sorted({item.object for item in self.measurements if item.object})
        what = f" of {', '.join(objects)}" if objects else ""
        where = f"calibration clip {self.video}" if self.source == CALIBRATION else self.video
        return f"{self.values()}, from {len(self.measurements)} measurement(s){what} on {where}"


def read_measurements(path: Path) -> list[Measurement]:
    if not path.is_file():
        return []
    return [Measurement(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def write_measurements(measurements: list[Measurement], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(item) for item in measurements], indent=2), encoding="utf-8")
    return path


def recording_scale(config, layout, recording_id: str) -> RecordingScale | None:
    """config is the one this recording runs, so its own file counts. Numbers in the settings come first,
    then measurements on the recording's video, then the calibration clip it names."""
    cfg = config.scale
    hand = {**({None: cfg.cm_per_px} if cfg.cm_per_px is not None else {}), **cfg.cm_per_px_by_role}
    if hand:
        return RecordingScale(hand, {role: None for role in hand}, HAND, None, [])
    own = read_measurements(layout.scale_path(recording_id))
    if own:
        return from_measurements(own, MEASURED, own[0].video)
    if cfg.calibration:
        clip = read_measurements(layout.calibration_scale_path(cfg.calibration))
        if clip:
            return from_measurements(clip, CALIBRATION, cfg.calibration)
    return None


def from_measurements(measurements: list[Measurement], source: str, video: str) -> RecordingScale:
    by_role, spread = {}, {}
    for role in dict.fromkeys(item.person for item in measurements):
        ratios = [item.cm_per_px for item in measurements if item.person == role]
        by_role[role] = float(np.mean(ratios))
        spread[role] = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else None
    return RecordingScale(by_role, spread, source, video, measurements)


def calibration_clips(config) -> list[str]:
    folder = Path(config.project.calibration_dir)
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir() if path.suffix.lower() in VIDEO_SUFFIXES)


def clip_frame(config, clip: str, at_s: float, dest: Path) -> tuple[Path, tuple[int, int]]:
    """A calibration clip's frame, with square pixels where prep.video gives the recordings square pixels too."""
    from macaw.media.ffmpeg import extract_frame
    from macaw.media.probe import run_ffprobe

    path = Path(config.project.calibration_dir) / clip
    if not path.is_file():
        raise MacawError(f"no calibration clip {clip} in {config.project.calibration_dir}")
    extract_frame(path, at_s, dest, square_pixels=config.video.normalise)
    width, height = run_ffprobe(["-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(dest)]).strip().split(",")
    return dest, (int(width), int(height))


def picture_size(config, recording_id: str, video: str) -> tuple[int, int] | None:
    """Width and height of the frames the video steps read from a recording's video, once prep.video has run."""
    from macaw.extractors.prep import load_video_prep

    entry = load_video_prep(config, recording_id).get(video)
    size = (entry.get("prepared_size") or entry.get("size")) if entry else None
    return (int(size[0]), int(size[1])) if size else None


def measured_videos(keypoints: pd.DataFrame) -> set[str]:
    """Videos people were measured on."""
    seen = keypoints[keypoints["subject_id"].notna() & keypoints["x"].notna()]
    return set(seen["video"].astype(str))


def unusable(scale: RecordingScale, config, recording_id: str, keypoints: pd.DataFrame) -> str | None:
    """Why a measured scale cannot convert this recording's pixels; None when it can."""
    videos = sorted(measured_videos(keypoints))
    if scale.source == MEASURED:
        elsewhere = [video for video in videos if video != scale.video]
        if elsewhere:
            return f"people are measured on {', '.join(elsewhere)} too, not only on {scale.video} where the object was measured"
    if scale.source == CALIBRATION:
        if len(videos) > 1:
            return f"people are measured on {len(videos)} videos ({', '.join(videos)}), and a calibration clip stands for one camera"
        sizes = {tuple(item.frame_size) for item in scale.measurements if item.frame_size}
        picture = picture_size(config, recording_id, videos[0]) if videos else None
        if picture and sizes and sizes != {picture}:
            clip = " or ".join(f"{width}x{height}" for width, height in sorted(sizes))
            return (
                f"calibration clip {scale.video} is {clip} but {videos[0]} is {picture[0]}x{picture[1]}; "
                "a clip carries over only to video of the same size from the same camera"
            )
    return None


def pixel_power(unit: str | None) -> int | None:
    """How many pixel lengths a unit holds: 1 for px or px/s, 2 for px^2; None for any other unit."""
    match = _PIXEL_UNIT.match(unit or "")
    return int(match.group(1) or 1) if match else None


def in_centimetres(unit: str) -> str:
    return "cm" + unit[2:]


def centimetre_rows(rows: pd.DataFrame, units: dict[str, str | None], factor_of) -> pd.DataFrame:
    """A copy of every pixel-unit feature's rows in centimetres, each named <feature>_cm. factor_of gives the
    cm per pixel for a (subject_id, subject_id_b) pair, or None to leave those rows in pixels."""
    powers = {name: power for name, power in ((name, pixel_power(unit)) for name, unit in units.items()) if power}
    picked = rows[rows["feature"].isin(list(powers))]
    pieces = []
    for (first, second), group in picked.groupby(["subject_id", "subject_id_b"], dropna=False, observed=True):
        factor = factor_of(None if pd.isna(first) else str(first), None if pd.isna(second) else str(second))
        if factor is None:
            continue
        group = group.copy()
        group["value"] = group["value"].astype(float).to_numpy() * factor ** group["feature"].map(powers).astype(float).to_numpy()
        group["feature"] = group["feature"] + CM_SUFFIX
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else picked.iloc[0:0]


def apparent_size(person: pd.DataFrame) -> pd.Series:
    """Per frame, the distance from the nose to the midpoint of the shoulders, indexed by time."""
    wide = person.pivot_table(index="t", columns="joint", values=["x", "y"], aggfunc="first")
    needed = ["nose", "left_shoulder", "right_shoulder"]
    if not all(("x", joint) in wide.columns for joint in needed):
        return pd.Series(dtype=float)
    mid_x = (wide[("x", "left_shoulder")] + wide[("x", "right_shoulder")]) / 2.0
    mid_y = (wide[("y", "left_shoulder")] + wide[("y", "right_shoulder")]) / 2.0
    return np.hypot(wide[("x", "nose")] - mid_x, wide[("y", "nose")] - mid_y).dropna()


def size_change(size: pd.Series, window_s: float) -> tuple[float, float] | None:
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
