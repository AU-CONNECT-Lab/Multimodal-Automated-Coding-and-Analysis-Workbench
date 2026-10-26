"""Derived quantities. These go to the long table; raw keypoints do not."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from macaw.exceptions import SchemaViolationError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate, own_views, primary_views
from macaw.extractors.video.pose.skeleton import JOINT_SET, STANDARD_JOINTS
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import keypoints as keypoints_store
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import UNITS_METRE, Scale

DEFAULT_SMOOTHING_S = 0.3

# cubic, so the third derivative (jerk) is still a fitted quantity rather than identically zero
POLYORDER = 3

# a horizontal length only ever shrinks under rotation, so its upper envelope is the person's size
BODY_SIZE_QUANTILE = 0.95

_ARM_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
_WRISTS = ("left_wrist", "right_wrist")
_DERIVATIVES = (("speed", 1), ("acceleration", 2), ("jerk", 3))

# description, unit ({length} is px or m), comparable across subjects; a length in pixels
# depends on how far each person sits from the camera
_SUBJECT_SPECS: dict[str, tuple[str, str | None, bool]] = {
    "hand_distance": ("Distance between the two wrists.", "{length}", False),
    **{
        f"{wrist}_{name}": (
            f"Magnitude of the {name} of the {wrist.replace('_', ' ')}, from a Savitzky-Golay "
            "fit within each unbroken run of frames.",
            "{length}/s" if order == 1 else f"{{length}}/s^{order}",
            False,
        )
        for wrist in _WRISTS
        for name, order in _DERIVATIVES
    },
    "head_yaw_index": (
        "Nose position between the eyes, (d_right - d_left) / (d_right + d_left): 0 facing the "
        "camera, positive when turned toward the person's own left. Not an angle.",
        None,
        True,
    ),
    "head_roll_deg": (
        "Image-plane tilt of the line through the eyes, positive clockwise.",
        "deg",
        True,
    ),
    "postural_openness": (
        "Horizontal reach of shoulders, elbows and wrists in units of this person's shoulder "
        "width: about 1 with the arms inside the shoulders, larger as they spread.",
        None,
        True,
    ),
}

_PAIR_SPECS: dict[str, tuple[str, str | None, bool]] = {
    "interpersonal_distance": (
        "Distance between the two shoulder midpoints in the image. It changes when people turn "
        "or move in depth, not only when they approach each other.",
        "{length}",
        False,
    ),
    "envelope_overlap": (
        "Intersection of the two keypoint bounding boxes over the smaller box. Overlap in the "
        "image is not proximity in the room.",
        None,
        True,
    ),
}


@register_extractor
class KinematicsExtractor(Extractor):
    name = "pose.kinematics"
    version = "0.3.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics",)
    requires = (PoseReviewGate.name,)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [_keypoints_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        smoothing_s = self.config.kinematics.smoothing_s
        keypoints = keypoints_store.read_keypoints(
            _keypoints_file(self.config, recording.recording_id),
            recording_id=recording.recording_id,
        )
        keypoints = keypoints[keypoints["subject_id"].notna()]
        subjects = [subject.subject_id for subject in recording.subjects]
        own = own_views(recording)
        views = primary_views(keypoints, recording)
        # each person is measured on one view: their own camera, else the shared one that saw them most
        by_subject = {
            subject_id: keypoints[(keypoints["subject_id"] == subject_id) & (keypoints["video"] == views[subject_id])]
            for subject_id in subjects
            if subject_id in views
        }
        shared = keypoints[~keypoints["video"].isin(set(own.values()))]
        warnings = [
            self._warning(recording, subject_id, "warn", "no keypoints for this subject")
            for subject_id in subjects
            if subject_id not in by_subject
        ]

        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        frames = [
            build_long_rows(
                values=_subject_values(kp, smoothing_s),
                scale=Scale.FRAME,
                subject_id=subject_id,
                **common,
            )
            for subject_id, kp in by_subject.items()
        ]

        # people filmed by separate cameras do not share an image plane, so no pixel distance
        # between them means anything
        pairs = {pair: _pair_view(shared, *pair) for pair in combinations([s for s in subjects if s in by_subject], 2)}
        pairs = {pair: view for pair, view in pairs.items() if view is not None}
        if own and not pairs and len(by_subject) > 1:
            warnings.append(
                self._warning(
                    recording, None, "info",
                    "each person has their own camera, so pairwise quantities were not computed "
                    "(pose.read_shared measures them on a shared camera)",
                )
            )
        else:
            for (a, b), view in pairs.items():
                frames.append(
                    build_long_rows(
                        values=_pair_values(view[view["subject_id"] == a], view[view["subject_id"] == b]),
                        scale=Scale.FRAME,
                        subject_id=a,
                        subject_id_b=b,
                        **common,
                    )
                )

        rows = concat_long_tables(frames)
        length_unit = _length_unit(keypoints)
        return ExtractorResult(
            rows=rows,
            specs=self._specs(set(rows["feature"]), length_unit, smoothing_s),
            warnings=warnings,
        )

    def _specs(self, emitted: set[str], length_unit: str, smoothing_s: float) -> list[FeatureSpec]:
        params = {
            "joint_set": JOINT_SET,
            "smoothing_s": smoothing_s,
            "polyorder": POLYORDER,
            "body_size_quantile": BODY_SIZE_QUANTILE,
        }
        specs = []
        for name, (description, unit, comparable) in {**_SUBJECT_SPECS, **_PAIR_SPECS}.items():
            if name not in emitted:
                continue
            specs.append(
                FeatureSpec(
                    name=name,
                    description=description,
                    unit=unit.format(length=length_unit) if unit else None,
                    scale=Scale.FRAME,
                    group=self.group,
                    extractor=self.name,
                    extractor_version=self.version,
                    params=params,
                    interpolation="none",
                    aggregation="mean",
                    comparable_across_subjects=comparable,
                    comparable_across_backends=False,
                    validation="synthetic",
                )
            )
        return specs

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(
            check=self.name,
            recording_id=recording.recording_id,
            subject_id=subject_id,
            severity=severity,
            message=message,
        )


def _pair_view(shared: pd.DataFrame, a: str, b: str) -> pd.DataFrame | None:
    """The rows of the shared camera on which the two were seen together most; none if never."""
    seen = shared[shared["subject_id"].isin([a, b]) & shared["x"].notna()]
    together = seen.groupby(["video", "frame"], observed=True)["subject_id"].nunique()
    together = together[together == 2].groupby(level="video").size()
    if together.empty:
        return None
    return shared[shared["video"] == together.idxmax()]


def _keypoints_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).keypoints_path(recording_id) / f"{PoseExtractor.name}.parquet"


def _length_unit(keypoints: pd.DataFrame) -> str:
    return "m" if (keypoints["units"] == UNITS_METRE).any() else "px"


def _subject_values(kp: pd.DataFrame, smoothing_s: float) -> pd.DataFrame:
    values = pd.DataFrame({"t_start": _frame_times(kp), "hand_distance": hand_distance(kp)})
    for wrist in _WRISTS:
        values[f"{wrist}_speed"] = velocity(kp, wrist, smoothing_s=smoothing_s)
        values[f"{wrist}_acceleration"] = acceleration(kp, wrist, smoothing_s=smoothing_s)
        values[f"{wrist}_jerk"] = jerk(kp, wrist, smoothing_s=smoothing_s)
    head = head_orientation(kp)
    values["head_yaw_index"] = head["yaw_index"]
    values["head_roll_deg"] = head["roll_deg"]
    values["postural_openness"] = postural_openness(kp)
    return values.reset_index(drop=True)


def _pair_values(kp_a: pd.DataFrame, kp_b: pd.DataFrame) -> pd.DataFrame:
    distance = interpersonal_distance(kp_a, kp_b)
    values = pd.DataFrame(
        {
            "t_start": _frame_times(kp_a).reindex(distance.index),
            "interpersonal_distance": distance,
            "envelope_overlap": envelope_overlap(kp_a, kp_b),
        }
    )
    return values.reset_index(drop=True)


def _frame_times(kp: pd.DataFrame) -> pd.Series:
    return kp.groupby("frame")["t"].first().sort_index()


def _points(kp: pd.DataFrame, joints: Sequence[str]) -> pd.DataFrame:
    """One row per frame the person was seen in, with an (x|y, joint) column per joint."""
    rows = kp[kp["joint"].isin(joints)]
    if rows.duplicated(["frame", "joint"]).any():
        raise SchemaViolationError(
            "keypoints: one person has two rows for the same frame and joint; kinematics reads "
            "one camera per person"
        )
    wide = rows.pivot(index="frame", columns="joint", values=["x", "y"])
    columns = pd.MultiIndex.from_product([["x", "y"], list(joints)])
    return wide.reindex(index=np.sort(kp["frame"].unique()), columns=columns).astype(float)


def _xy(points: pd.DataFrame, joint: str) -> np.ndarray:
    return np.column_stack([points[("x", joint)].to_numpy(), points[("y", joint)].to_numpy()])


def _midpoint(points: pd.DataFrame, a: str, b: str) -> np.ndarray:
    return (_xy(points, a) + _xy(points, b)) / 2.0


def _seconds_per_frame(kp: pd.DataFrame) -> float:
    times = _frame_times(kp)
    if len(times) < 2:
        return float("nan")
    steps = np.diff(times.to_numpy()) / np.diff(times.index.to_numpy())
    return float(np.median(steps))


def _window_frames(smoothing_s: float, seconds_per_frame: float) -> int:
    frames = max(int(round(smoothing_s / seconds_per_frame)), POLYORDER + 2)
    return frames if frames % 2 else frames + 1


def hand_distance(kp: pd.DataFrame) -> pd.Series:
    points = _points(kp, _WRISTS)
    gap = _xy(points, "left_wrist") - _xy(points, "right_wrist")
    return pd.Series(np.hypot(gap[:, 0], gap[:, 1]), index=points.index)


def _derivative(kp: pd.DataFrame, joint: str, order: int, smoothing_s: float) -> pd.Series:
    """No fit spans a missing frame, so a gap never turns into an apparent jump."""
    points = _points(kp, [joint])
    frames = points.index.to_numpy()
    xy = _xy(points, joint)
    out = np.full(len(frames), np.nan)
    step = _seconds_per_frame(kp)
    if not step > 0:
        return pd.Series(out, index=points.index)

    window = _window_frames(smoothing_s, step)
    valid = ~np.isnan(xy).any(axis=1)
    starts = np.ones(len(frames), dtype=bool)
    starts[1:] = (np.diff(frames) != 1) | (valid[1:] != valid[:-1])
    run_id = np.cumsum(starts)
    for run in np.unique(run_id[valid]):
        index = np.flatnonzero(run_id == run)
        if len(index) < window:
            continue
        dx = savgol_filter(xy[index, 0], window, POLYORDER, deriv=order, delta=step)
        dy = savgol_filter(xy[index, 1], window, POLYORDER, deriv=order, delta=step)
        out[index] = np.hypot(dx, dy)
    return pd.Series(out, index=points.index)


def velocity(kp: pd.DataFrame, joint: str, *, smoothing_s: float = DEFAULT_SMOOTHING_S) -> pd.Series:
    return _derivative(kp, joint, 1, smoothing_s)


def acceleration(
    kp: pd.DataFrame, joint: str, *, smoothing_s: float = DEFAULT_SMOOTHING_S
) -> pd.Series:
    return _derivative(kp, joint, 2, smoothing_s)


def jerk(kp: pd.DataFrame, joint: str, *, smoothing_s: float = DEFAULT_SMOOTHING_S) -> pd.Series:
    return _derivative(kp, joint, 3, smoothing_s)


def head_orientation(kp: pd.DataFrame) -> pd.DataFrame:
    """Image-plane proxies from nose and eyes. Pitch is left out: 2D body keypoints carry no
    stable cue for it."""
    points = _points(kp, ("nose", "left_eye", "right_eye"))
    nose, left, right = (_xy(points, joint) for joint in ("nose", "left_eye", "right_eye"))
    to_left = np.hypot(*(nose - left).T)
    to_right = np.hypot(*(nose - right).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        yaw_index = (to_right - to_left) / (to_right + to_left)

    # an eye line has no direction, so the angle folds into [-90, 90) whichever way the person faces
    line = left - right
    angle = np.degrees(np.arctan2(line[:, 1], line[:, 0]))
    roll = (angle + 90.0) % 180.0 - 90.0
    return pd.DataFrame({"yaw_index": yaw_index, "roll_deg": roll}, index=points.index)


def nod_events(head: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError(
        "Nods are pitch movements and 2D body keypoints give no stable pitch; this belongs with "
        "the face backend's head pose"
    )


def interpersonal_distance(kp_a: pd.DataFrame, kp_b: pd.DataFrame) -> pd.Series:
    """Metres with depth, pixels without; the unit goes into the manifest either way."""
    a = _points(kp_a, ("left_shoulder", "right_shoulder"))
    b = _points(kp_b, ("left_shoulder", "right_shoulder"))
    shared = a.index.intersection(b.index)
    a, b = a.loc[shared], b.loc[shared]
    gap = _midpoint(a, "left_shoulder", "right_shoulder") - _midpoint(b, "left_shoulder", "right_shoulder")
    return pd.Series(np.hypot(gap[:, 0], gap[:, 1]), index=shared)


def postural_openness(kp: pd.DataFrame) -> pd.Series:
    points = _points(kp, _ARM_JOINTS)
    shoulders = _xy(points, "left_shoulder") - _xy(points, "right_shoulder")
    width = np.hypot(shoulders[:, 0], shoulders[:, 1])
    size = np.nanquantile(width, BODY_SIZE_QUANTILE) if np.isfinite(width).any() else float("nan")
    if not size > 0:
        return pd.Series(np.nan, index=points.index)

    # an area would reward arms hanging straight down; reach is what spreading them changes
    x = np.column_stack([points[("x", joint)].to_numpy() for joint in _ARM_JOINTS])
    with np.errstate(invalid="ignore"):
        reach = x.max(axis=1) - x.min(axis=1)
    return pd.Series(reach / size, index=points.index)


def _boxes(kp: pd.DataFrame) -> pd.DataFrame:
    rows = kp[kp["joint"].isin(STANDARD_JOINTS) & kp["x"].notna() & kp["y"].notna()]
    return rows.groupby("frame").agg(
        x0=("x", "min"), x1=("x", "max"), y0=("y", "min"), y1=("y", "max")
    )


def envelope_overlap(kp_a: pd.DataFrame, kp_b: pd.DataFrame) -> pd.Series:
    a, b = _boxes(kp_a), _boxes(kp_b)
    frames = np.sort(kp_a["frame"].unique())
    frames = frames[np.isin(frames, kp_b["frame"].unique())]
    a, b = a.reindex(frames), b.reindex(frames)
    width = (np.minimum(a.x1, b.x1) - np.maximum(a.x0, b.x0)).clip(lower=0)
    height = (np.minimum(a.y1, b.y1) - np.maximum(a.y0, b.y0)).clip(lower=0)
    smaller = np.minimum((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    overlap = (width * height) / smaller.where(smaller > 0)
    return pd.Series(overlap.to_numpy(dtype=float), index=pd.Index(frames, name="frame"))
