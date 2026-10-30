"""Derived quantities. These go to the long table; raw keypoints do not."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from macaw.exceptions import ConfigError, SchemaViolationError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate, own_views, primary_views
from macaw.extractors.video.pose.skeleton import JOINT_SET, STANDARD_JOINTS
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project_scripts import script_function, script_path
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

# points a measure may name besides the skeleton's joints
_CENTRES = {"shoulder_center": ("left_shoulder", "right_shoulder"), "hip_center": ("left_hip", "right_hip")}
SELF = "self"
BUILTIN_MEASURES = ("distance", "approach_speed", "offset_toward")

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
    version = "0.4.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics",)
    requires = (PoseReviewGate.name,)

    def check_available(self) -> None:
        roles = set(self.config.project.roles)
        for measure in self.config.kinematics.measures:
            where = f"kinematics.measures {measure.name}"
            if measure.name in {**_SUBJECT_SPECS, **_PAIR_SPECS}:
                raise ConfigError(f"{where}: pose.kinematics already has a feature of that name; choose another")
            if measure.person not in roles or (measure.to_person not in roles and measure.to_person != SELF):
                raise ConfigError(
                    f"{where}: person and to_person name a role in project.roles ({', '.join(sorted(roles))}); "
                    f"to_person may also be {SELF}"
                )
            if measure.method in BUILTIN_MEASURES:
                _check_joints(where, measure)
            elif "." in measure.method:
                try:
                    script_function(self.config.project.scripts_dir, measure.method)
                except ConfigError as error:
                    raise ConfigError(f"{where}: {error}") from None
            else:
                raise ConfigError(
                    f"{where}: method is {', '.join(BUILTIN_MEASURES)}, or file.function for a function "
                    f"in {self.config.project.scripts_dir}"
                )

    def inputs_for(self, recording: Recording) -> list[Path]:
        scripts = [
            script_path(self.config.project.scripts_dir, measure.method)
            for measure in self.config.kinematics.measures
            if measure.method not in BUILTIN_MEASURES
        ]
        return [_keypoints_file(self.config, recording.recording_id), *scripts]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        smoothing_s = self.config.kinematics.smoothing_s
        keypoints, by_subject = keypoints_by_subject(self.config, recording)
        subjects = [subject.subject_id for subject in recording.subjects]
        own = own_views(recording)
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

        for measure in self.config.kinematics.measures:
            pieces, notes = self._measure_rows(measure, recording, by_subject, shared, common)
            frames += pieces
            warnings += notes

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
        catalog = {name: (*spec, params) for name, spec in {**_SUBJECT_SPECS, **_PAIR_SPECS}.items()}
        for measure in self.config.kinematics.measures:
            catalog[measure.name] = (*measure_spec(measure), {**params, **measure.model_dump()})
        specs = []
        for name, (description, unit, comparable, spec_params) in catalog.items():
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
                    params=spec_params,
                    interpolation="none",
                    aggregation="mean",
                    comparable_across_subjects=comparable,
                    comparable_across_backends=False,
                    validation="synthetic",
                )
            )
        return specs

    def _measure_rows(self, measure, recording: Recording, by_subject, shared, common):
        function = None
        if measure.method not in BUILTIN_MEASURES:
            function = script_function(self.config.project.scripts_dir, measure.method)
        seen = [subject for subject in recording.subjects if subject.subject_id in by_subject]
        rows, notes = [], []
        for a in [subject.subject_id for subject in seen if subject.role == measure.person]:
            others = [a] if measure.to_person == SELF else [
                subject.subject_id for subject in seen if subject.role == measure.to_person and subject.subject_id != a
            ]
            for b in others:
                if b == a:
                    kp_a = kp_b = by_subject[a]
                else:
                    # one image plane for both, as for every pairwise quantity
                    view = _pair_view(shared, a, b)
                    if view is None:
                        notes.append(self._warning(
                            recording, a, "info", f"{measure.name}: {a} and {b} are never on one camera together"
                        ))
                        continue
                    kp_a, kp_b = view[view["subject_id"] == a], view[view["subject_id"] == b]
                values = measure_values(
                    measure, kp_a, kp_b, smoothing_s=self.config.kinematics.smoothing_s, function=function
                )
                table = pd.DataFrame(
                    {"t_start": _frame_times(kp_a).reindex(values.index).to_numpy(), measure.name: values.to_numpy()}
                )
                rows.append(build_long_rows(
                    values=table, scale=Scale.FRAME, subject_id=a, subject_id_b=None if b == a else b, **common
                ))
        return rows, notes

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(
            check=self.name,
            recording_id=recording.recording_id,
            subject_id=subject_id,
            severity=severity,
            message=message,
        )


def keypoints_by_subject(config, recording: Recording) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Everyone's keypoints, and each person's on the one view they are measured on: their own
    camera, else the shared one that saw them most."""
    keypoints = keypoints_store.read_keypoints(_keypoints_file(config, recording.recording_id), recording_id=recording.recording_id)
    keypoints = keypoints[keypoints["subject_id"].notna()]
    views = primary_views(keypoints, recording)
    by_subject = {
        subject.subject_id: keypoints[(keypoints["subject_id"] == subject.subject_id) & (keypoints["video"] == views[subject.subject_id])]
        for subject in recording.subjects
        if subject.subject_id in views
    }
    return keypoints, by_subject


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
    points = _points(kp, [joint])
    fitted = _smoothed_derivative(_xy(points, joint), points.index.to_numpy(), _seconds_per_frame(kp), smoothing_s, order)
    return pd.Series(np.hypot(fitted[:, 0], fitted[:, 1]), index=points.index)


def _smoothed_derivative(values: np.ndarray, frames: np.ndarray, step: float, smoothing_s: float, order: int) -> np.ndarray:
    """Per column; no fit spans a missing frame, so a gap never turns into an apparent jump."""
    out = np.full(values.shape, np.nan)
    if not step > 0:
        return out
    window = _window_frames(smoothing_s, step)
    valid = ~np.isnan(values).any(axis=1)
    starts = np.ones(len(frames), dtype=bool)
    starts[1:] = (np.diff(frames) != 1) | (valid[1:] != valid[:-1])
    run_id = np.cumsum(starts)
    for run in np.unique(run_id[valid]):
        index = np.flatnonzero(run_id == run)
        if len(index) < window:
            continue
        out[index] = savgol_filter(values[index], window, POLYORDER, deriv=order, delta=step, axis=0)
    return out


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
    size = _body_size(kp)
    if not size > 0:
        return pd.Series(np.nan, index=points.index)

    # an area would reward arms hanging straight down; reach is what spreading them changes
    x = np.column_stack([points[("x", joint)].to_numpy() for joint in _ARM_JOINTS])
    with np.errstate(invalid="ignore"):
        reach = x.max(axis=1) - x.min(axis=1)
    return pd.Series(reach / size, index=points.index)


def _body_size(kp: pd.DataFrame) -> float:
    points = _points(kp, ("left_shoulder", "right_shoulder"))
    shoulders = _xy(points, "left_shoulder") - _xy(points, "right_shoulder")
    width = np.hypot(shoulders[:, 0], shoulders[:, 1])
    size = np.nanquantile(width, BODY_SIZE_QUANTILE) if np.isfinite(width).any() else float("nan")
    return float(size) if size > 0 else float("nan")


def measure_values(
    measure, kp_a: pd.DataFrame, kp_b: pd.DataFrame, *, smoothing_s: float = DEFAULT_SMOOTHING_S, function=None
) -> pd.Series:
    """One value per frame both people were seen in; function stands in for a script method."""
    frames = np.intersect1d(kp_a["frame"].unique(), kp_b["frame"].unique())
    if function is not None:
        values = _script_values(measure.method, function, kp_a, kp_b, frames)
    elif measure.method == "offset_toward":
        values = _offset_toward(kp_a, measure.joints[0], kp_b, measure.to_joints[0], frames)
    else:
        values = _nearest_distance(kp_a, measure.joints, kp_b, measure.to_joints, frames)
        if measure.method == "approach_speed":
            # a shrinking distance is an approach, so it reads positive
            values = -_smoothed_derivative(values[:, None], frames, _seconds_per_frame(kp_a), smoothing_s, 1)[:, 0]
    if measure.per_body_size:
        values = values / _body_size(kp_a)
    return pd.Series(values, index=pd.Index(frames, name="frame"))


def _joint_xy(kp: pd.DataFrame, joints: Sequence[str], frames: np.ndarray) -> np.ndarray:
    """(frame, joint, xy); a centre is the midpoint of its two joints."""
    real = list(dict.fromkeys(part for joint in joints for part in _CENTRES.get(joint, (joint,))))
    points = _points(kp, real).reindex(frames)
    return np.stack(
        [_midpoint(points, *_CENTRES[joint]) if joint in _CENTRES else _xy(points, joint) for joint in joints], axis=1
    )


def _nearest_distance(kp_a, joints, kp_b, to_joints, frames) -> np.ndarray:
    a = _joint_xy(kp_a, joints, frames)[:, :, None, :]
    b = _joint_xy(kp_b, to_joints, frames)[:, None, :, :]
    gaps = np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]).reshape(len(frames), -1)
    return np.fmin.reduce(gaps, axis=1)


def _offset_toward(kp_a, joint, kp_b, to_joint, frames) -> np.ndarray:
    own = _joint_xy(kp_a, [joint, "shoulder_center"], frames)
    centre = own[:, 1, 0]
    target = _joint_xy(kp_b, [to_joint], frames)[:, 0, 0]
    return (own[:, 0, 0] - centre) * np.sign(target - centre)


def _script_values(reference: str, function, kp_a, kp_b, frames) -> np.ndarray:
    values = np.asarray(function(_skeleton_table(kp_a, frames), _skeleton_table(kp_b, frames)), dtype=float).reshape(-1)
    if len(values) != len(frames):
        raise SchemaViolationError(f"{reference} returned {len(values)} values for {len(frames)} rows; give one per row")
    return values


def _skeleton_table(kp: pd.DataFrame, frames: np.ndarray) -> pd.DataFrame:
    """What a script measure gets per person: a row per frame with t and each joint's x and y."""
    points = _points(kp, STANDARD_JOINTS).reindex(frames)
    columns = {"t": _frame_times(kp).reindex(frames).to_numpy()}
    for joint in STANDARD_JOINTS:
        columns[f"{joint}_x"] = points[("x", joint)].to_numpy()
        columns[f"{joint}_y"] = points[("y", joint)].to_numpy()
    return pd.DataFrame(columns, index=pd.Index(frames, name="frame"))


def _check_joints(where: str, measure) -> None:
    if not measure.joints or not measure.to_joints:
        raise ConfigError(f"{where}: {measure.method} needs joints and to_joints")
    unknown = sorted(set(measure.joints + measure.to_joints) - set(STANDARD_JOINTS) - set(_CENTRES))
    if unknown:
        raise ConfigError(
            f"{where}: no joint called {', '.join(unknown)}; use {', '.join(STANDARD_JOINTS)}, {', '.join(_CENTRES)}"
        )
    if measure.method == "offset_toward" and (len(measure.joints) != 1 or len(measure.to_joints) != 1):
        raise ConfigError(f"{where}: offset_toward takes one joint and one to_joint")


def measure_spec(measure) -> tuple[str, str | None, bool]:
    who = f"{measure.person}'s {', '.join(measure.joints)}"
    owner = "their own" if measure.to_person == SELF else f"{measure.to_person}'s"
    target = f"{owner} {', '.join(measure.to_joints)}"
    if measure.method == "distance":
        text, unit = f"Distance from {who} to {target}, the nearest pair where several are listed.", "{length}"
    elif measure.method == "approach_speed":
        text, unit = f"Rate at which the distance from {who} to {target} shrinks; positive while closing in.", "{length}/s"
    elif measure.method == "offset_toward":
        text, unit = f"Horizontal offset of {who} from their own shoulder midpoint, positive toward {target}.", "{length}"
    else:
        return f"Computed by the project script {measure.method}.", None, False
    if measure.per_body_size:
        text += f" In units of {measure.person}'s shoulder width."
        unit = "1/s" if unit.endswith("/s") else None
    return text, unit, measure.per_body_size


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
