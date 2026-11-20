"""Stretches in which a hand moves, frame by frame and as a tier a coder picks gestures from."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.pose.keypoints import PoseReviewGate
from macaw.extractors.video.pose.kinematics import (
    _frame_times,
    _keypoints_file,
    _length_unit,
    joined_annotations,
    keypoints_by_subject,
    velocity,
)
from macaw.extractors.video.pose.gestures import room_tracks
from macaw.extractors.video.pose.skeleton import JOINT_SET
from macaw.extractors.video.pose.triangulate import TriangulationExtractor, frame_stride, points_file, three_d_step
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.room import CM_PER_UNIT
from macaw.store import keypoints as keypoints_store
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

SIDES = ("left", "right")
TIER, TIER_3D = "moving", "moving_3d"
_SPAN_COLUMNS = ["t_start", "t_end"]


def moving_spans(times, speed, *, above: float, min_s: float, join_s: float) -> tuple[pd.DataFrame, int]:
    """(stretches with speed over above, joined when under join_s apart; how many were dropped as shorter than min_s)."""
    times, speed = np.asarray(times, dtype=float), np.asarray(speed, dtype=float)
    with np.errstate(invalid="ignore"):
        fast = speed > above
    if not fast.any():
        return pd.DataFrame(columns=_SPAN_COLUMNS, dtype=float), 0
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    # frames the person was not seen in are absent, so a jump in time ends a stretch too
    broken = np.r_[True, (np.diff(times) > 1.5 * period) | (fast[1:] != fast[:-1])]
    runs = pd.DataFrame({"t": times, "fast": fast, "run": np.cumsum(broken)})
    spans = runs[runs["fast"]].groupby("run").agg(t_start=("t", "first"), t_end=("t", "last"))
    joined: list[list[float]] = []
    for start, end in zip(spans["t_start"], spans["t_end"] + period):
        if joined and start - joined[-1][1] < join_s:
            joined[-1][1] = end
        else:
            joined.append([start, end])
    kept = [span for span in joined if span[1] - span[0] >= min_s]
    return pd.DataFrame(kept, columns=_SPAN_COLUMNS, dtype=float), len(joined) - len(kept)


def moving_frames(times, speed, spans: pd.DataFrame) -> np.ndarray:
    """1 in a kept stretch, 0 outside one where the speed is known, empty where it is not."""
    times, speed = np.asarray(times, dtype=float), np.asarray(speed, dtype=float)
    inside = np.zeros(len(times), dtype=bool)
    for start, end in zip(spans["t_start"], spans["t_end"]):
        inside |= (times >= start) & (times < end)
    return np.where(inside, 1.0, np.where(np.isfinite(speed), 0.0, np.nan))


def either_hand(by_side: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The stretches of the hands laid over each other: t_start, t_end, and as label the hand that moved or both."""
    each = sorted((start, end, side) for side, spans in by_side.items() for start, end in zip(spans["t_start"], spans["t_end"]))
    merged: list[list] = []
    for start, end, side in each:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].add(side)
        else:
            merged.append([start, end, {side}])
    labels = ["both" if len(sides) > 1 else next(iter(sides)) for _, _, sides in merged]
    return pd.DataFrame({"t_start": [m[0] for m in merged], "t_end": [m[1] for m in merged], "label": labels})


def hands_moving(times, speeds: dict[str, np.ndarray], *, above: float, min_s: float, join_s: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(each hand's 0/1 per frame, the stretches of either hand with the hand as label, how many were too short)."""
    values, by_side, dropped = pd.DataFrame({"t_start": np.asarray(times, dtype=float)}), {}, 0
    for side in SIDES:
        by_side[side], short = moving_spans(values["t_start"], speeds[side], above=above, min_s=min_s, join_s=join_s)
        values[f"{side}_hand_moving"] = moving_frames(values["t_start"], speeds[side], by_side[side])
        dropped += short
    return values, either_hand(by_side), dropped


@register_extractor
class MovingExtractor(Extractor):
    name = "pose.moving"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.smoothing_s", "kinematics.moving.speed", "kinematics.moving.min_s", "kinematics.moving.join_s", "setup.cameras")
    requires = (PoseReviewGate.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.kinematics.moving.speed is not None

    def check_available(self) -> None:
        if self.config.kinematics.moving.speed is None:
            raise ConfigError(
                "kinematics.moving.speed is not set: give the wrist speed above which a hand counts as moving, in pixels per "
                "second for these videos; the wrist speeds pose.kinematics writes help to choose it"
            )

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [_keypoints_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.kinematics.moving
        keypoints, by_subject = keypoints_by_subject(self.config, recording)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        warnings = [
            self._warning(recording, subject.subject_id, "warn", "no keypoints for this subject")
            for subject in recording.subjects
            if subject.subject_id not in by_subject
        ]
        frames, coded = [], []
        for subject_id, kp in by_subject.items():
            times = _frame_times(kp)
            speeds = {
                side: velocity(kp, f"{side}_wrist", smoothing_s=self.config.kinematics.smoothing_s).reindex(times.index).to_numpy()
                for side in SIDES
            }
            values, stretches, dropped = hands_moving(times, speeds, above=cfg.speed, min_s=cfg.min_s, join_s=cfg.join_s)
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, subject_id=subject_id, **common))
            coded.append(build_annotations(
                recording_id=recording.recording_id, tier=TIER, layer="pose", annotator=self.name, entities=common["entities"],
                intervals=stretches, subject_id=subject_id,
            ))
            warnings.append(self._warning(
                recording, subject_id, "info",
                f"{len(stretches)} stretch(es) of hand movement on the {TIER} tier to pick gestures from; "
                f"{dropped} shorter than {cfg.min_s:g} s left out",
            ))
        rows = concat_long_tables(frames)
        return ExtractorResult(
            rows=rows, specs=self._specs(set(rows["feature"]), _length_unit(keypoints)), annotations=joined_annotations(coded),
            warnings=warnings,
        )

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message)

    def _specs(self, emitted: set[str], length_unit: str) -> list[FeatureSpec]:
        cfg = self.config.kinematics.moving
        params = {"joint_set": JOINT_SET, "smoothing_s": self.config.kinematics.smoothing_s, "length_unit": length_unit, **cfg.model_dump(exclude={"speed_3d"})}
        return [
            FeatureSpec(
                name=f"{side}_hand_moving",
                description=(
                    f"1 while the {side} wrist moves faster than {cfg.speed:g} {length_unit}/s, in stretches at least {cfg.min_s:g} s "
                    f"long, with stretches less than {cfg.join_s:g} s apart counted as one; 0 otherwise. It says that the hand "
                    "moves, not that the movement is a gesture."
                ),
                unit=None,
                scale=Scale.FRAME,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                interpolation="none",
                aggregation="mean",
                missing=(
                    f"Empty in frames outside a stretch where the {side} wrist has no speed: it was not found on the camera this "
                    "person's body is measured from, or only in a run of frames shorter than the smoothing window "
                    "(kinematics.smoothing_s)."
                ),
                comparable_across_subjects=False,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for side in SIDES
            if f"{side}_hand_moving" in emitted
        ]


@register_extractor
class Moving3DExtractor(Extractor):
    name = "pose.moving3d"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = (
        "kinematics.smoothing_s", "kinematics.moving.speed_3d", "kinematics.moving.min_s", "kinematics.moving.join_s", "setup.cameras",
        "project.units", "pose3d.one_camera_every",
    )

    @classmethod
    def dependencies(cls, config) -> tuple[str, ...]:
        return (three_d_step(config) or TriangulationExtractor.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.kinematics.moving.speed_3d is not None and three_d_step(config) is not None

    def check_available(self) -> None:
        if self.config.kinematics.moving.speed_3d is None:
            raise ConfigError(
                "kinematics.moving.speed_3d is not set: give the wrist speed above which a hand counts as moving, in "
                f"{self.config.project.units} per second; the wrist speeds pose.kinematics3d writes help to choose it"
            )

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [points_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.kinematics.moving
        # the speed is typed in the project's unit, the points are in metres
        above = cfg.speed_3d * CM_PER_UNIT[self.config.project.units] / 100.0
        points = keypoints_store.read_keypoints(points_file(self.config, recording.recording_id), recording_id=recording.recording_id)
        people = {str(subject_id): own for subject_id, own in points.groupby("subject_id", observed=True)}
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        warnings = [
            _warning(self.name, recording, subject.subject_id, "warn", "this person was never placed in 3D")
            for subject in recording.subjects
            if str(subject.subject_id) not in people
        ]
        frames, coded = [], []
        for subject_id, placed in people.items():
            times, sides, _ = room_tracks(placed, self.config.kinematics.smoothing_s, frame_stride(self.config))
            values, stretches, dropped = hands_moving(
                times, {side: sides[side][0] for side in SIDES}, above=above, min_s=cfg.min_s, join_s=cfg.join_s,
            )
            values.columns = [column if column == "t_start" else f"{column}_3d" for column in values.columns]
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, subject_id=subject_id, **common))
            coded.append(build_annotations(
                recording_id=recording.recording_id, tier=TIER_3D, layer="pose", annotator=self.name, entities=common["entities"],
                intervals=stretches, subject_id=subject_id,
            ))
            warnings.append(_warning(
                self.name, recording, subject_id, "info",
                f"{len(stretches)} stretch(es) of hand movement in 3D on the {TIER_3D} tier to pick gestures from; "
                f"{dropped} shorter than {cfg.min_s:g} s left out",
            ))
        rows = concat_long_tables(frames)
        return ExtractorResult(rows=rows, specs=self._specs(set(rows["feature"])), annotations=joined_annotations(coded), warnings=warnings)

    def _specs(self, emitted: set[str]) -> list[FeatureSpec]:
        cfg, units = self.config.kinematics.moving, self.config.project.units
        params = {"joint_set": JOINT_SET, "smoothing_s": self.config.kinematics.smoothing_s, "units": units, **cfg.model_dump(exclude={"speed"})}
        return [
            FeatureSpec(
                name=f"{side}_hand_moving_3d",
                description=(
                    f"1 while the {side} wrist, placed in 3D, moves faster than {cfg.speed_3d:g} {units}/s, in stretches at least "
                    f"{cfg.min_s:g} s long, with stretches less than {cfg.join_s:g} s apart counted as one; 0 otherwise. It says "
                    "that the hand moves, not that the movement is a gesture."
                ),
                unit=None,
                scale=Scale.FRAME,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                interpolation="none",
                aggregation="mean",
                missing=(
                    f"Empty in frames outside a stretch where the {side} wrist has no speed in 3D: it was not placed in 3D, or only "
                    "in a run of frames shorter than the smoothing window (kinematics.smoothing_s)."
                ),
                comparable_across_subjects=True,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for side in SIDES
            if f"{side}_hand_moving_3d" in emitted
        ]


def _warning(check: str, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
    return QCWarning(check=check, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message)
