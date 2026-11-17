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
from macaw.extractors.video.pose.skeleton import JOINT_SET
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

SIDES = ("left", "right")
TIER = "moving"
_SPAN_COLUMNS = ["t_start", "t_end"]


def moving_spans(times, speed, *, above: float, min_s: float, join_s: float) -> tuple[pd.DataFrame, int]:
    """(each kept stretch with speed over above, how many were dropped as shorter than min_s); stretches less than
    join_s apart count as one, and a frame without a speed is not moving."""
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


@register_extractor
class MovingExtractor(Extractor):
    name = "pose.moving"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.smoothing_s", "kinematics.moving", "setup.cameras")
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
            values, by_side, dropped = pd.DataFrame({"t_start": times.to_numpy(dtype=float)}), {}, 0
            for side in SIDES:
                speed = velocity(kp, f"{side}_wrist", smoothing_s=self.config.kinematics.smoothing_s).reindex(times.index).to_numpy()
                by_side[side], short = moving_spans(values["t_start"], speed, above=cfg.speed, min_s=cfg.min_s, join_s=cfg.join_s)
                values[f"{side}_hand_moving"] = moving_frames(values["t_start"], speed, by_side[side])
                dropped += short
            stretches = either_hand(by_side)
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
        params = {"joint_set": JOINT_SET, "smoothing_s": self.config.kinematics.smoothing_s, "length_unit": length_unit, **cfg.model_dump()}
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
