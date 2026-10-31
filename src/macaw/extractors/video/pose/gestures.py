"""Kinematics of each gesture a person coded: speed, submovements, holds, extent, height and gesture space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.pose.keypoints import PoseReviewGate
from macaw.extractors.video.pose.kinematics import (
    _frame_times,
    _keypoints_file,
    _length_unit,
    _midpoint,
    _points,
    _xy,
    keypoints_by_subject,
    velocity,
)
from macaw.extractors.video.pose.skeleton import JOINT_SET
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.annotations import read_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

SIDES = ("left", "right")
_BODY = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose", "left_eye", "right_eye")

# description, unit ({length} is px or m), aggregation, comparable across subjects
_SPECS: dict[str, tuple[str, str | None, str, bool]] = {
    "peak_speed": ("Highest speed of the {side} wrist during the gesture.", "{length}/s", "mean", False),
    "submovements": (
        "Peaks in the {side} wrist's speed that rise at least kinematics.gestures.peak_prominence above their "
        "surroundings: the strokes and returns a gesture is made of.",
        "count", "mean", False,
    ),
    "rhythm_sd_s": (
        "Standard deviation of the intervals between the {side} hand's submovement peaks; larger is less "
        "regular. Needs three peaks.",
        "s", "mean", False,
    ),
    "holds": (
        "Stretches between the first and last submovement in which the {side} wrist and elbow both move "
        "slower than kinematics.gestures.still_speed.",
        "count", "mean", False,
    ),
    "hold_s": ("Time spent in those holds.", "s", "mean", False),
    "extent": ("Area of the box around the {side} wrist's path.", "{length}^2", "mean", False),
    "max_height": (
        "Highest level the {side} wrist reaches: 0 below the body midline, 1 below mid-chest, 2 below the "
        "shoulders, 3 below the nose, 4 below the top of the head, 5 above it. Needs hips in view.",
        None, "max", True,
    ),
    "space_max": (
        "Outermost gesture space the {side} wrist enters: 1 centre-centre, 2 centre, 3 periphery, 4 extreme "
        "periphery, on a grid drawn from the hips, eyes and nose. Needs hips in view.",
        None, "max", True,
    ),
    "space_mode": ("Gesture space the {side} wrist is in most of the time, coded as space_max.", None, "median", True),
    "space_sectors": (
        "How many of the eight directions around the centre space (above, below, to either side and the four "
        "corners) the {side} wrist reaches in the periphery or beyond; 0 when it stays in the centre. Needs hips in view.",
        "count", "mean", True,
    ),
}
_HOLDS = ("holds", "hold_s")
# with kinematics.gestures.holds both_hands, the holds are measured once per gesture instead of per hand
_BOTH_HANDS_SPECS: dict[str, tuple[str, str | None, str, bool]] = {
    "holds": (
        "Stretches between the first and last submovement of either hand in which both wrists and both elbows move "
        "slower than kinematics.gestures.still_speed. Frames with either hand out of view are not still; empty when "
        "a hand is out of view throughout.",
        "count", "mean", False,
    ),
    "hold_s": ("Time spent in those holds.", "s", "mean", False),
}


@register_extractor
class GestureExtractor(Extractor):
    name = "pose.gestures"
    version = "0.2.0"
    scale = Scale.EVENT
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.smoothing_s", "kinematics.gestures")
    requires = (PoseReviewGate.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.kinematics.gestures.tier is not None

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        coded = layout_for(self.config, create=False).annotations_path(recording.recording_id)
        return [_keypoints_file(self.config, recording.recording_id), *sorted(coded.glob("*.parquet"))]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        segments, warnings = self._segments(recording)
        keypoints, by_subject = keypoints_by_subject(self.config, recording)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        frames = []
        for subject_id, own in segments.groupby("subject_id", observed=True):
            kp = by_subject.get(str(subject_id))
            if kp is None:
                warnings.append(self._warning(recording, str(subject_id), f"{len(own)} gesture(s) coded but no keypoints for this person"))
                continue
            values = self._measure(kp, own)
            unseen = int(values.filter(like="peak_speed").isna().all(axis=1).sum())
            if unseen:
                warnings.append(self._warning(recording, str(subject_id), f"{unseen} of {len(own)} gesture(s) fall where neither wrist was tracked"))
            frames.append(build_long_rows(values=values, scale=Scale.EVENT, subject_id=str(subject_id), **common))
        rows = concat_long_tables(frames)
        return ExtractorResult(rows=rows, specs=self._specs(_length_unit(keypoints)), warnings=warnings)

    def _segments(self, recording: Recording) -> tuple[pd.DataFrame, list[QCWarning]]:
        cfg = self.config.kinematics.gestures
        path = layout_for(self.config, create=False).annotations_path(recording.recording_id)
        coded = read_annotations(path, tier=cfg.tier) if path.exists() else pd.DataFrame(columns=["subject_id", "annotator", "t_start", "t_end"])
        if cfg.annotator is not None:
            coded = coded[coded["annotator"] == cfg.annotator]
        annotators = sorted(coded["annotator"].dropna().astype(str).unique())
        if len(annotators) > 1:
            # two people's segments of the same gesture would be counted twice
            message = f"tier {cfg.tier} was coded by {', '.join(annotators)}; set kinematics.gestures.annotator to measure one of them"
            return coded.iloc[0:0], [self._warning(recording, None, message)]
        nobody = coded["subject_id"].isna()
        warnings = [self._warning(recording, None, f"{int(nobody.sum())} gesture(s) on tier {cfg.tier} belong to no person and were left out")] if nobody.any() else []
        if coded.empty:
            warnings.append(self._warning(recording, None, f"no segments on tier {cfg.tier} to measure", severity="info"))
        return coded[~nobody], warnings

    def _measure(self, kp: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.kinematics.gestures
        smoothing_s = self.config.kinematics.smoothing_s
        times = _frame_times(kp)
        body = _points(kp, _BODY)
        sides = {}
        for side in SIDES:
            wrist = _points(kp, [f"{side}_wrist"]).reindex(times.index)
            sides[side] = (
                velocity(kp, f"{side}_wrist", smoothing_s=smoothing_s).reindex(times.index).to_numpy(),
                velocity(kp, f"{side}_elbow", smoothing_s=smoothing_s).reindex(times.index).to_numpy(),
                _xy(wrist, f"{side}_wrist"),
            )
        placed = body.reindex(times.index)
        heights = {side: height_levels(sides[side][2], placed) for side in SIDES}
        spaces = {side: gesture_spaces(sides[side][2], placed) for side in SIDES}
        sectors = {side: gesture_sectors(sides[side][2], placed) for side in SIDES}
        t = times.to_numpy(dtype=float)
        both_hands = cfg.holds == "both_hands"

        rows = []
        for segment in segments.sort_values("t_start").itertuples():
            inside = (t >= segment.t_start) & (t <= segment.t_end)
            row = {"t_start": float(segment.t_start), "t_end": float(segment.t_end)}
            for side in SIDES:
                wrist_speed, elbow_speed, xy = (values[inside] for values in sides[side])
                measured = gesture_kinematics(t[inside], wrist_speed, elbow_speed, xy, cfg.still_speed, cfg.peak_prominence)
                measured["max_height"] = _nanmax(heights[side][inside])
                measured["space_max"] = _nanmax(spaces[side][inside])
                measured["space_mode"] = _mode(spaces[side][inside])
                measured["space_sectors"] = _distinct_sectors(sectors[side][inside])
                if both_hands:
                    for name in _HOLDS:
                        del measured[name]
                row.update({f"gesture_{side}_{name}": value for name, value in measured.items()})
            if both_hands:
                wrists = [sides[side][0][inside] for side in SIDES]
                elbows = [sides[side][1][inside] for side in SIDES]
                measured = both_hands_holds(t[inside], wrists, elbows, cfg.still_speed, cfg.peak_prominence)
                row.update({f"gesture_{name}": value for name, value in measured.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def _specs(self, length_unit: str) -> list[FeatureSpec]:
        cfg = self.config.kinematics.gestures
        params = {"joint_set": JOINT_SET, "smoothing_s": self.config.kinematics.smoothing_s, **cfg.model_dump()}
        both_hands = cfg.holds == "both_hands"
        named = [
            (f"gesture_{side}_{name}", description.format(side=side), spec)
            for side in SIDES
            for name, (description, *spec) in _SPECS.items()
            if not (both_hands and name in _HOLDS)
        ]
        if both_hands:
            named += [(f"gesture_{name}", description, spec) for name, (description, *spec) in _BOTH_HANDS_SPECS.items()]
        return [
            FeatureSpec(
                name=name,
                description=description,
                unit=unit.format(length=length_unit) if unit else None,
                scale=Scale.EVENT,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation=aggregation,
                comparable_across_subjects=comparable,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for name, description, (unit, aggregation, comparable) in named
        ]

    def _warning(self, recording: Recording, subject_id, message: str, severity: str = "warn") -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message)


def gesture_kinematics(
    t: np.ndarray, wrist_speed: np.ndarray, elbow_speed: np.ndarray, xy: np.ndarray, still_speed: float, peak_prominence: float
) -> dict[str, float]:
    """One hand over one gesture; speeds per frame, NaN where the joint was not tracked."""
    if not np.isfinite(wrist_speed).any():
        return {name: np.nan for name in ("peak_speed", "submovements", "rhythm_sd_s", "holds", "hold_s", "extent")}
    peaks = submovement_peaks(t, wrist_speed, peak_prominence)
    stretches = holds(t, [wrist_speed, elbow_speed], still_speed, peaks)
    seen = xy[np.isfinite(xy).all(axis=1)]
    return {
        "peak_speed": float(np.nanmax(wrist_speed)),
        "submovements": float(len(peaks)),
        # how unevenly the submovements are spaced, the opposite of isochrony
        "rhythm_sd_s": float(np.std(np.diff(peaks))) if len(peaks) >= 3 else np.nan,
        "holds": float(len(stretches)),
        "hold_s": float(sum(end - start for start, end in stretches)),
        "extent": float(np.ptp(seen[:, 0]) * np.ptp(seen[:, 1])) if len(seen) >= 2 else np.nan,
    }


def both_hands_holds(
    t: np.ndarray, wrists: list[np.ndarray], elbows: list[np.ndarray], still_speed: float, peak_prominence: float
) -> dict[str, float]:
    """Holds with both hands still at once, between the first and last submovement of either hand."""
    if not all(np.isfinite(speed).any() for speed in wrists):
        return {"holds": np.nan, "hold_s": np.nan}
    peaks = np.sort(np.concatenate([submovement_peaks(t, speed, peak_prominence) for speed in wrists]))
    stretches = holds(t, [*wrists, *elbows], still_speed, peaks)
    return {"holds": float(len(stretches)), "hold_s": float(sum(end - start for start, end in stretches))}


def submovement_peaks(t: np.ndarray, speed: np.ndarray, prominence: float) -> np.ndarray:
    """Times of the speed peaks; each tracked stretch on its own, so a gap makes no peak."""
    found = []
    for run in _runs(np.isfinite(speed)):
        index, _ = find_peaks(speed[run], prominence=prominence)
        found.append(t[run][index])
    return np.concatenate(found) if found else np.empty(0)


def holds(t: np.ndarray, speeds: list[np.ndarray], still_speed: float, peaks: np.ndarray) -> list[tuple[float, float]]:
    """Stretches with every given joint still, between the first and last submovement peak."""
    if len(peaks) < 2:
        return []
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    with np.errstate(invalid="ignore"):
        still = np.logical_and.reduce([speed < still_speed for speed in speeds]) & (t > peaks[0]) & (t < peaks[-1])
    return [(float(t[run][0]), float(t[run][-1] + step)) for run in _runs(still)]


def height_levels(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, the vertical level the wrist is at, 0 to 5; image y grows downward."""
    up = -wrist[:, 1]
    shoulders = -_midpoint(body, "left_shoulder", "right_shoulder")[:, 1]
    hips = -_midpoint(body, "left_hip", "right_hip")[:, 1]
    nose = -body[("y", "nose")].to_numpy()
    eyes = -_midpoint(body, "left_eye", "right_eye")[:, 1]
    midline = (shoulders + hips) / 2.0
    chest = (midline + shoulders) / 2.0
    top = eyes + 2.0 * (eyes - nose)
    bounds = np.column_stack([midline, chest, shoulders, nose, top])
    known = np.isfinite(up) & np.isfinite(bounds).all(axis=1)
    with np.errstate(invalid="ignore"):
        levels = (up[:, None] > bounds).sum(axis=1).astype(float)
    return np.where(known, levels, np.nan)


def gesture_spaces(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, the gesture space the wrist is in, centre-centre 1 to extreme periphery 4."""
    x, up = wrist[:, 0], -wrist[:, 1]
    low, high, centre_up = _hips_box(body)
    eyes = -_midpoint(body, "left_eye", "right_eye")[:, 1]
    nose = -body[("y", "nose")].to_numpy()
    face = 2.0 * np.abs(body[("x", "left_eye")].to_numpy() - body[("x", "right_eye")].to_numpy())
    width = high - low
    top = eyes + (eyes - nose)
    with np.errstate(invalid="ignore"):
        centre_centre = (x > low) & (x < high) & (np.abs(up - centre_up) < width / 2.0)
        centre = (x > low - width / 2.0) & (x < high + width / 2.0) & (np.abs(up - centre_up) < width)
        periphery = (x > low - width / 2.0 - face) & (x < high + width / 2.0 + face) & (np.abs(up - centre_up) < top - centre_up)
    spaces = np.select([centre_centre, centre, periphery], [1.0, 2.0, 3.0], 4.0)
    known = np.isfinite(np.column_stack([x, up, low, high, centre_up, top, face])).all(axis=1)
    return np.where(known, spaces, np.nan)


def gesture_sectors(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, which of the eight cells around the centre space the wrist is in, 1 to 8, or 0 within it."""
    x, up = wrist[:, 0], -wrist[:, 1]
    low, high, centre_up = _hips_box(body)
    width = high - low
    with np.errstate(invalid="ignore"):
        column = np.where(x <= low - width / 2.0, 0, np.where(x >= high + width / 2.0, 2, 1))
        row = np.where(up >= centre_up + width, 0, np.where(up <= centre_up - width, 2, 1))
    # the edges of the centre space extended outward cut the periphery into a 3 x 3 grid less its middle
    cell = 3 * row + column
    sectors = np.where(cell < 4, cell + 1, cell).astype(float)
    spaces = gesture_spaces(wrist, body)
    return np.where(np.isfinite(spaces), np.where(spaces >= 3, sectors, 0.0), np.nan)


def _hips_box(body: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, the hips' outer x on each side and the height halfway between shoulders and hips, up positive."""
    hips_x = np.column_stack([body[("x", "left_hip")].to_numpy(), body[("x", "right_hip")].to_numpy()])
    centre_up = -(_midpoint(body, "left_shoulder", "right_shoulder")[:, 1] + _midpoint(body, "left_hip", "right_hip")[:, 1]) / 2.0
    return hips_x.min(axis=1), hips_x.max(axis=1), centre_up


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    """Index arrays of each unbroken stretch where mask holds."""
    index = np.flatnonzero(mask)
    if index.size == 0:
        return []
    return np.split(index, np.flatnonzero(np.diff(index) != 1) + 1)


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if np.isfinite(values).any() else np.nan


def _distinct_sectors(values: np.ndarray) -> float:
    known = values[np.isfinite(values)]
    return float(np.unique(known[known > 0]).size) if known.size else np.nan


def _mode(values: np.ndarray) -> float:
    known = values[np.isfinite(values)].astype(int)
    return float(np.bincount(known).argmax()) if known.size else np.nan
