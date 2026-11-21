"""How far each hand is from each thing followed, frame by frame, and the stretches in which a hand is on one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.objects.outline import distance
from macaw.extractors.video.objects.review import ObjectReviewGate
from macaw.extractors.video.objects.track import load_objects, objects_file
from macaw.extractors.video.pose.keypoints import PoseReviewGate
from macaw.extractors.video.pose.kinematics import _frame_times, _keypoints_file, joined_annotations, keypoints_by_subject
from macaw.extractors.video.pose.moving import SIDES, either_hand
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from macaw.config.schema import thing_slug

TIER = "on_{thing}"
_SPAN_COLUMNS = ["t_start", "t_end"]


slug = thing_slug


def wrist_distances(wrists: pd.DataFrame, things: pd.DataFrame) -> pd.Series:
    """px from each wrist row (frame, x, y) to the thing's outline in the same frame, 0 inside it, empty where it is not in view."""
    outlines = things.drop_duplicates("frame").set_index("frame")["outline"]
    return pd.Series(
        [distance(outlines[frame], x, y) if frame in outlines.index else np.nan for frame, x, y in zip(wrists["frame"], wrists["x"], wrists["y"])],
        index=wrists.index, dtype=float,
    )


def on_spans(times, on) -> pd.DataFrame:
    """The stretches of frames marked 1, each to the end of its last frame; a jump in time ends one."""
    times, on = np.asarray(times, dtype=float), np.asarray(on, dtype=float) == 1
    if not on.any():
        return pd.DataFrame(columns=_SPAN_COLUMNS, dtype=float)
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    broken = np.r_[True, (np.diff(times) > 1.5 * period) | (on[1:] != on[:-1])]
    runs = pd.DataFrame({"t": times, "on": on, "run": np.cumsum(broken)})
    spans = runs[runs["on"]].groupby("run").agg(t_start=("t", "first"), t_end=("t", "last"))
    return pd.DataFrame({"t_start": spans["t_start"].to_numpy(), "t_end": spans["t_end"].to_numpy() + period})


@register_extractor
class ObjectTouchExtractor(Extractor):
    name = "objects.touch"
    version = "0.3.0"
    scale = Scale.FRAME
    group = "objects"
    phase = Phase.DERIVE
    config_sections = ("objects.names", "objects.touch_px", "setup.cameras")
    requires = (ObjectReviewGate.name, PoseReviewGate.name)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.objects.names) and config.objects.touch_px is not None

    def check_available(self) -> None:
        if self.config.objects.touch_px is None:
            raise ConfigError(
                "objects.touch_px is not set: give how near a wrist has to come to a thing's outline to count as a hand on it, "
                "in pixels of these videos, about a hand's length"
            )

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [_keypoints_file(self.config, recording.recording_id), objects_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.objects
        keypoints, by_subject = keypoints_by_subject(self.config, recording)
        things = load_objects(self.config, recording.recording_id)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        warnings = [
            self._warning(recording, subject.subject_id, "warn", "no keypoints for this subject")
            for subject in recording.subjects
            if subject.subject_id not in by_subject
        ]
        frames, coded = [], []
        for subject_id, kp in by_subject.items():
            video = str(kp["video"].iloc[0])
            times = _frame_times(kp)
            values = pd.DataFrame({"t_start": times.to_numpy(dtype=float)}, index=times.index)
            mine = things[things["video"] == video]
            counted = []
            for name in cfg.names:
                shown = mine[mine["label"] == name]
                if shown.empty:
                    continue
                by_side = {}
                for side in SIDES:
                    wrist = kp[kp["joint"] == f"{side}_wrist"].set_index("frame").reindex(times.index)[["x", "y"]].reset_index()
                    gap = wrist_distances(wrist, shown).to_numpy()
                    values[f"{side}_wrist_to_{slug(name)}"] = gap
                    on = np.where(np.isfinite(gap), (gap <= cfg.touch_px).astype(float), np.nan)
                    values[f"{side}_hand_on_{slug(name)}"] = on
                    by_side[side] = on_spans(values["t_start"], on)
                stretches = either_hand(by_side)
                coded.append(build_annotations(
                    recording_id=recording.recording_id, tier=TIER.format(thing=slug(name)), layer="objects", annotator=self.name,
                    entities=common["entities"], intervals=stretches, subject_id=subject_id,
                ))
                counted.append(f"{name} {len(stretches)}")
            if not counted:
                warnings.append(self._warning(recording, subject_id, "warn", f"no thing is followed on {video}, the camera this person's body is measured on"))
                continue
            frames.append(build_long_rows(values=values.reset_index(drop=True), scale=Scale.FRAME, subject_id=subject_id, camera=video, **common))
            warnings.append(self._warning(recording, subject_id, "info", f"stretches with a hand on a thing: {', '.join(counted)} (read from the picture: over the thing as the camera sees it)"))
        rows = concat_long_tables(frames)
        return ExtractorResult(
            rows=rows, specs=self._specs(set(rows["feature"]) if not rows.empty else set()), annotations=joined_annotations(coded),
            warnings=warnings,
        )

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message)

    def _specs(self, emitted: set[str]) -> list[FeatureSpec]:
        cfg = self.config.objects
        params = {"names": list(cfg.names), "touch_px": cfg.touch_px}
        holds = (
            "It is read from the picture: a hand in front of or behind the thing counts as on it, so it marks a hand over the "
            "thing as the camera sees it, not contact."
        )
        missing = "Empty in frames where this wrist was not found, or the thing was out of view."
        specs = []
        for name in cfg.names:
            for side in SIDES:
                far, on = f"{side}_wrist_to_{slug(name)}", f"{side}_hand_on_{slug(name)}"
                if far in emitted:
                    specs.append(FeatureSpec(
                        name=far, description=f"Pixels from the {side} wrist to the outline of {name} in the picture, 0 inside it. {holds}",
                        unit="px", scale=Scale.FRAME, group=self.group, extractor=self.name, extractor_version=self.version, params=params,
                        interpolation="none", aggregation="mean", missing=missing, comparable_across_subjects=False,
                        comparable_across_backends=False, validation="synthetic",
                    ))
                if on in emitted:
                    specs.append(FeatureSpec(
                        name=on, description=f"1 while the {side} wrist is within {cfg.touch_px:g} px of the outline of {name}, 0 otherwise. {holds}",
                        unit=None, scale=Scale.FRAME, group=self.group, extractor=self.name, extractor_version=self.version, params=params,
                        interpolation="none", aggregation="mean", missing=missing, comparable_across_subjects=False,
                        comparable_across_backends=False, validation="synthetic",
                    ))
        return specs
