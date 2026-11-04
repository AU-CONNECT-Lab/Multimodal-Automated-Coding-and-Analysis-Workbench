"""3D from two or more calibrated cameras: each person's points triangulated from every camera that saw them,
and measures in metres beside the 2D ones."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError, MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.video.pose.cameras import UNITS_TO_METRES, Camera, read_setup, triangulate
from macaw.extractors.video.pose.keypoints import PoseReviewGate
from macaw.extractors.video.pose.kinematics import (
    _CENTRES,
    BUILTIN_MEASURES,
    POLYORDER,
    SELF,
    _keypoints_file,
    _smoothed_derivative,
    below_outputs,
    below_specs,
    call_script,
    check_measures,
    joined_annotations,
    measure_ends,
)
from macaw.extractors.video.pose.skeleton import JOINT_SET, STANDARD_JOINTS
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.scale import picture_size
from macaw.project_scripts import script_function, script_path
from macaw.store import keypoints as keypoints_store
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, UNITS_METRE, Scale, coerce_frame, empty_frame

VIDEO_3D = "3d"
QUALITY_NAME = "triangulation.csv"
# bones whose length should hold still; the hips are often hidden at a table
BONES = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
)
# a bone whose length spreads more than this share of its median has misplaced ends
_BONE_SPREAD = 0.15
# cameras that disagree on a person by more than this share of pose3d.max_reprojection_px on median see someone else
_DISAGREE_SHARE = 1 / 3
_WRISTS = ("left_wrist", "right_wrist")
_DERIVATIVES = (("speed", 1), ("acceleration", 2), ("jerk", 3))


def calibration_path(config, name: str | None = None) -> Path:
    """The file a calibration (pose3d.calibration unless named) stands for: a .toml in project.calibration_dir, or
    the one macaw calibrate-cameras made from a folder of board clips there."""
    name = name or config.pose3d.calibration
    source = Path(config.project.calibration_dir) / name
    return layout_for(config, create=False).camera_setup_path(name) if source.is_dir() else source


def points_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).keypoints_path(recording_id) / f"{TriangulationExtractor.name}.parquet"


@register_extractor
class TriangulationExtractor(Extractor):
    name = "pose.triangulate"
    version = "0.2.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("pose3d",)
    requires = (PoseReviewGate.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.pose3d.calibration is not None

    def check_available(self) -> None:
        name = self.config.pose3d.calibration
        if name is None:
            raise ConfigError("pose.triangulate needs pose3d.calibration: a folder of board clips or a calibration file")
        path = calibration_path(self.config)
        if not path.is_file():
            folder = Path(self.config.project.calibration_dir) / name
            raise ConfigError(
                f"run macaw calibrate-cameras {name} first, to find the cameras from its board clips" if folder.is_dir()
                else f"pose3d.calibration {name}: no such file or folder in {self.config.project.calibration_dir}"
            )

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [_keypoints_file(self.config, recording.recording_id), calibration_path(self.config)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.pose3d
        rid = recording.recording_id
        setup = read_setup(calibration_path(self.config), cfg.units)
        keypoints = keypoints_store.read_keypoints(_keypoints_file(self.config, rid), recording_id=rid)
        keypoints = keypoints[keypoints["subject_id"].notna() & keypoints["joint"].isin(STANDARD_JOINTS)]
        views, warnings = self._views(recording, setup.cameras, keypoints)
        if len(views) < 2:
            raise MacawError(
                f"{rid}: {len(views)} of the calibration's cameras ({', '.join(camera.name for camera in setup.cameras)}) "
                "match a camera people were measured on, and 3D needs two; a calibrated camera is matched to the "
                "recording's camera of the same name"
            )
        points, quality = triangulate_people(keypoints, views, max_error_px=cfg.max_reprojection_px)
        points[["x", "y", "z"]] *= UNITS_TO_METRES[setup.units]
        warnings += self._quality_warnings(rid, quality, points)

        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        quality.to_csv(directory / QUALITY_NAME, index=False)
        rows = coerce_frame(
            points.assign(
                recording_id=rid, source=self.name, video=VIDEO_3D, track_id=pd.NA, units=UNITS_METRE, metric_scale_valid=True,
            ),
            KEYPOINT_SCHEMA, "keypoints",
        )
        return ExtractorResult(
            rows=empty_frame(LONG_TABLE_SCHEMA), keypoints=rows, artifacts={QUALITY_NAME: directory / QUALITY_NAME}, warnings=warnings,
        )

    def _views(self, recording: Recording, cameras: list[Camera], keypoints: pd.DataFrame) -> tuple[list, list[QCWarning]]:
        """(camera, video) for each calibrated camera that filmed people in this recording, matched by name."""
        seen = set(keypoints["video"].astype(str))
        by_label = {item.name: item.name for item in recording.media if item.kind == "video"}
        views, warnings = [], []
        for camera in cameras:
            video = by_label.get(camera.name)
            if video is None or video not in seen:
                warnings.append(self._warning(recording.recording_id, None, "info", f"camera {camera.name} has no video with people in it here"))
                continue
            picture = picture_size(self.config, recording.recording_id, video)
            if picture is not None and tuple(picture) != tuple(camera.size):
                warnings.append(self._warning(
                    recording.recording_id, None, "warn",
                    f"{video} is {picture[0]}x{picture[1]} but camera {camera.name} was calibrated at {camera.size[0]}x{camera.size[1]}; left out",
                ))
                continue
            views.append((camera, video))
        unmatched = sorted(seen - {video for _, video in views})
        if unmatched:
            warnings.append(self._warning(recording.recording_id, None, "info", f"no calibrated camera for {', '.join(unmatched)}, so 3D does not use it"))
        return views, warnings

    def _quality_warnings(self, rid: str, quality: pd.DataFrame, points: pd.DataFrame) -> list[QCWarning]:
        cfg = self.config.pose3d
        warnings = []
        for subject_id, own in quality.groupby("subject_id"):
            share = own["triangulated"].sum() / max(own["seen"].sum(), 1)
            error = float(np.nanmedian(own["error_px"])) if own["error_px"].notna().any() else float("nan")
            warnings.append(self._warning(
                rid, subject_id, "info",
                f"in 3D in {share:.0%} of the frames a camera saw them; the cameras' views of them miss each other by {error:.1f} px on median",
            ))
            if error > cfg.max_reprojection_px * _DISAGREE_SHARE:
                warnings.append(self._warning(
                    rid, subject_id, "warn",
                    f"the cameras disagree on this person by {error:.1f} px on median: check that they carry the same name on "
                    "every camera, and that the cameras did not move since the calibration",
                ))
        for subject_id, own in points.groupby("subject_id"):
            for (first, second), spread in bone_spread(own).items():
                if spread > _BONE_SPREAD:
                    bone = f"{first.replace('_', ' ')} to {second.replace('_', ' ')}"
                    warnings.append(self._warning(
                        rid, subject_id, "warn", f"the {bone} changes length by {spread:.0%}, so its ends are misplaced in some frames",
                    ))
        return warnings

    def _warning(self, rid: str, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=rid, subject_id=subject_id, severity=severity, message=message)


def triangulate_people(keypoints: pd.DataFrame, views: list[tuple[Camera, str]], *, max_error_px: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each person's joints in the room, on the frames of the camera with the most frames; every other
    camera's nearest frame counts as the same moment. Returns the points, in the calibration's units,
    and per person and joint how often they were seen, placed, and how far the cameras missed each other."""
    clocks = {video: _clock(keypoints[keypoints["video"] == video]) for _, video in views}
    step, start, first, last = clocks[max(clocks, key=lambda video: clocks[video][3] - clocks[video][2])]
    grid = np.arange(first, last + 1)
    t = start + step * grid
    nearest = [np.round((t - clocks[video][1]) / clocks[video][0]).astype(int) for _, video in views]
    cameras = [camera for camera, _ in views]
    joints = list(STANDARD_JOINTS)

    points, quality = [], []
    for subject_id, own in keypoints.groupby("subject_id", observed=True):
        pixels = np.full((len(grid), len(joints), len(views), 2), np.nan)
        weights = np.zeros((len(grid), len(joints), len(views)))
        for index, (_, video) in enumerate(views):
            wide = own[own["video"] == video].pivot_table(index="frame", columns="joint", values=["x", "y", "confidence"], aggfunc="first")
            for j, joint in enumerate(joints):
                if ("x", joint) in wide.columns and ("y", joint) in wide.columns:
                    pixels[:, j, index, 0] = wide[("x", joint)].reindex(nearest[index]).to_numpy()
                    pixels[:, j, index, 1] = wide[("y", joint)].reindex(nearest[index]).to_numpy()
                    weights[:, j, index] = np.nan_to_num(wide[("confidence", joint)].reindex(nearest[index]).to_numpy() if ("confidence", joint) in wide.columns else 1.0)
        weights[~np.isfinite(pixels).all(axis=-1)] = 0.0
        flat, used, error = triangulate(cameras, pixels.reshape(-1, len(views), 2), weights.reshape(-1, len(views)), max_error_px=max_error_px)
        found = np.isfinite(flat).all(axis=1)
        frame, joint = np.divmod(np.arange(len(flat)), len(joints))
        confidence = np.where(used, weights.reshape(-1, len(views)), 0.0).sum(axis=1) / np.maximum(used.sum(axis=1), 1)
        points.append(pd.DataFrame({
            "subject_id": str(subject_id), "frame": grid[frame[found]], "t": t[frame[found]], "joint": np.asarray(joints)[joint[found]],
            "x": flat[found, 0], "y": flat[found, 1], "z": flat[found, 2], "confidence": confidence[found],
        }))
        seen = (weights > 0).any(axis=-1)
        errors = error.reshape(len(grid), len(joints))
        quality.append(pd.DataFrame({
            "subject_id": str(subject_id), "joint": joints, "seen": seen.sum(axis=0),
            "triangulated": found.reshape(len(grid), len(joints)).sum(axis=0),
            "error_px": [float(np.nanmedian(column)) if np.isfinite(column).any() else np.nan for column in errors.T],
            "views": [float(np.median(column[column > 0])) if (column > 0).any() else np.nan for column in used.sum(axis=1).reshape(len(grid), len(joints)).T],
        }))
    columns = ["subject_id", "frame", "t", "joint", "x", "y", "z", "confidence"]
    return (
        pd.concat(points, ignore_index=True) if points else pd.DataFrame(columns=columns),
        pd.concat(quality, ignore_index=True) if quality else pd.DataFrame(columns=["subject_id", "joint", "seen", "triangulated", "error_px", "views"]),
    )


def _clock(rows: pd.DataFrame) -> tuple[float, float, int, int]:
    """(seconds per frame, time of frame 0, first frame, last frame) of one video's keypoints."""
    stamps = rows.groupby("frame")["t"].first()
    frames, times = stamps.index.to_numpy(dtype=float), stamps.to_numpy(dtype=float)
    if len(frames) < 2:
        raise MacawError("a camera saw people on fewer than two frames, too few to place its frames in time")
    step, start = np.polyfit(frames, times, 1)
    return float(step), float(start), int(frames.min()), int(frames.max())


def bone_spread(points: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Per bone, how much its length varies: 1.4826 times the median absolute deviation over the median."""
    wide = points.pivot_table(index="frame", columns="joint", values=["x", "y", "z"], aggfunc="first")
    spread = {}
    for first, second in BONES:
        if ("x", first) not in wide.columns or ("x", second) not in wide.columns:
            continue
        ends = [np.column_stack([wide[(axis, joint)] for axis in "xyz"]) for joint in (first, second)]
        length = np.linalg.norm(ends[0] - ends[1], axis=1)
        length = length[np.isfinite(length)]
        if len(length) >= 10 and np.median(length) > 0:
            spread[(first, second)] = float(1.4826 * np.median(np.abs(length - np.median(length))) / np.median(length))
    return spread


# description, unit
_SUBJECT_SPECS: dict[str, tuple[str, str]] = {
    "hand_distance_3d": ("Distance between the two wrists in the room.", "m"),
    **{
        f"{wrist}_{name}_3d": (
            f"Magnitude of the {name} of the {wrist.replace('_', ' ')} in the room, from a Savitzky-Golay fit within each unbroken run of frames.",
            "m/s" if order == 1 else f"m/s^{order}",
        )
        for wrist in _WRISTS
        for name, order in _DERIVATIVES
    },
}
_PAIR_SPECS: dict[str, tuple[str, str]] = {
    "interpersonal_distance_3d": ("Distance between the two shoulder midpoints in the room.", "m"),
    "head_distance_3d": ("Distance between the midpoints of the two people's ears.", "m"),
    "head_toward_deg_3d": (
        "Angle between where this person's head points (from the midpoint of the ears to the nose) and the "
        "direction to the other person's head; 0 when it points straight at them. Not gaze.",
        "deg",
    ),
    "body_toward_deg_3d": (
        "Angle between where this person's chest faces (square to the shoulders and to the spine from the hip midpoint, on "
        "the nose's side) and the direction to the other person's shoulder midpoint; 0 facing them, 180 with the back to "
        "them. Missing while the hips or the nose are not placed.",
        "deg",
    ),
}
# compared with an independent 3D reconstruction of real recordings; the rest only with simulated cameras
_AGAINST_REFERENCE = frozenset(
    {"interpersonal_distance_3d", "head_distance_3d", "head_toward_deg_3d", "left_wrist_speed_3d", "right_wrist_speed_3d"}
)


def measures_3d(config) -> list:
    """The project's own measures that also run in 3D: every built-in one, and scripts marked in_3d."""
    return [measure for measure in config.kinematics.measures if measure.method in BUILTIN_MEASURES or measure.in_3d]


@register_extractor
class Kinematics3DExtractor(Extractor):
    name = "pose.kinematics3d"
    version = "0.4.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.smoothing_s", "kinematics.measures")
    requires = (TriangulationExtractor.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.pose3d.calibration is not None

    def check_available(self) -> None:
        taken = {*_SUBJECT_SPECS, *_PAIR_SPECS} | {measure.name for measure in self.config.kinematics.measures}
        for measure in measures_3d(self.config):
            if f"{measure.name}_3d" in taken:
                raise ConfigError(
                    f"kinematics.measures {measure.name}: its 3D version would be called {measure.name}_3d, which another "
                    "feature already is; choose another name"
                )
        check_measures(self.config)

    def inputs_for(self, recording: Recording) -> list[Path]:
        scripts = [
            script_path(self.config.project.scripts_dir, measure.method)
            for measure in measures_3d(self.config)
            if measure.method not in BUILTIN_MEASURES
        ]
        return [points_file(self.config, recording.recording_id), *scripts]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        smoothing_s = self.config.kinematics.smoothing_s
        points = keypoints_store.read_keypoints(points_file(self.config, recording.recording_id), recording_id=recording.recording_id)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        people = {str(subject_id): own for subject_id, own in points.groupby("subject_id", observed=True)}
        frames = [
            build_long_rows(values=subject_values_3d(own, smoothing_s), scale=Scale.FRAME, subject_id=subject_id, **common)
            for subject_id, own in people.items()
        ]
        for a, b in permutations(sorted(people), 2):
            values = pair_values_3d(people[a], people[b], symmetric=a < b)
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, subject_id=a, subject_id_b=b, **common))
        coded = []
        for measure in measures_3d(self.config):
            pieces, stretches = self._measure_rows(measure, recording, people, common)
            frames += pieces
            coded += stretches
        rows = concat_long_tables(frames)
        return ExtractorResult(
            rows=rows, specs=self._specs(set(rows["feature"]) if not rows.empty else set(), smoothing_s), annotations=joined_annotations(coded),
        )

    def _measure_rows(self, measure, recording: Recording, people: dict[str, pd.DataFrame], common: dict) -> tuple[list, list]:
        function = None if measure.method in BUILTIN_MEASURES else script_function(self.config.project.scripts_dir, measure.method)
        # unlike in the picture, two people placed in the room can always be measured against each other
        placed = [subject for subject in recording.subjects if subject.subject_id in people]
        rows, stretches = [], []
        for a in [subject.subject_id for subject in placed if subject.role == measure.person]:
            others = [a] if measure.to_person == SELF else [
                subject.subject_id for subject in placed if subject.role == measure.to_person and subject.subject_id != a
            ]
            for b in others:
                values = measure_values_3d(measure, people[a], people[b], smoothing_s=self.config.kinematics.smoothing_s, function=function)
                rows.append(build_long_rows(values=values, scale=Scale.FRAME, subject_id=a, subject_id_b=None if b == a else b, **common))
                if measure.below_3d is not None:
                    feature = f"{measure.name}_3d"
                    below = measure.below_3d * below_3d_to_metres(measure, self.config.project.units)
                    summary, coded = below_outputs(feature, below, a, b, values["t_start"], values[feature], common)
                    rows += summary
                    stretches.append(coded)
        return rows, stretches

    def _specs(self, emitted: set[str], smoothing_s: float) -> list[FeatureSpec]:
        params = {"joint_set": JOINT_SET, "smoothing_s": smoothing_s, "polyorder": POLYORDER}
        catalog = {name: (description, unit, True, params) for name, (description, unit) in {**_SUBJECT_SPECS, **_PAIR_SPECS}.items()}
        below = []
        for measure in measures_3d(self.config):
            catalog[f"{measure.name}_3d"] = (*measure_spec_3d(measure), {**params, **measure.model_dump()})
            if measure.below_3d is not None:
                typed = below_3d_unit(measure, self.config.project.units)
                below += below_specs(f"{measure.name}_3d", f"{measure.below_3d:g}{f' {typed}' if typed else ''}", {**params, **measure.model_dump()}, emitted, self)
        return below + [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.FRAME, group=self.group, extractor=self.name,
                extractor_version=self.version, params=spec_params, interpolation="none", aggregation="mean",
                comparable_across_subjects=comparable, comparable_across_backends=False,
                validation="against_reference" if name in _AGAINST_REFERENCE else "synthetic",
            )
            for name, (description, unit, comparable, spec_params) in catalog.items()
            if name in emitted
        ]


def _joints(points: pd.DataFrame) -> tuple[np.ndarray, pd.Series, dict[str, np.ndarray]]:
    """(frames, time per frame, joint -> (n, 3)) with NaN where a joint was not placed."""
    wide = points.pivot_table(index="frame", columns="joint", values=["x", "y", "z", "t"], aggfunc="first")
    frames = wide.index.to_numpy()
    times = wide["t"].bfill(axis=1).iloc[:, 0]
    found = {
        joint: np.column_stack([wide[(axis, joint)].to_numpy() for axis in "xyz"])
        for joint in STANDARD_JOINTS
        if ("x", joint) in wide.columns
    }
    return frames, times, found


def _part(found: dict[str, np.ndarray], *joints: str) -> np.ndarray | None:
    """Midpoint of the named joints, or None when one is never placed."""
    if not all(joint in found for joint in joints):
        return None
    return np.mean([found[joint] for joint in joints], axis=0)


def _step(frames: np.ndarray, times: np.ndarray) -> float:
    """Seconds per frame; 0 with a single frame, which leaves every derivative empty."""
    return float(np.median(np.diff(times) / np.diff(frames))) if len(frames) > 1 else 0.0


def subject_values_3d(points: pd.DataFrame, smoothing_s: float) -> pd.DataFrame:
    frames, times, found = _joints(points)
    values = pd.DataFrame({"t_start": times.to_numpy(dtype=float)})
    if "left_wrist" in found and "right_wrist" in found:
        values["hand_distance_3d"] = np.linalg.norm(found["left_wrist"] - found["right_wrist"], axis=1)
    step = _step(frames, times.to_numpy(dtype=float))
    for wrist in _WRISTS:
        if wrist not in found:
            continue
        for name, order in _DERIVATIVES:
            fitted = _smoothed_derivative(found[wrist], frames, step, smoothing_s, order)
            values[f"{wrist}_{name}_3d"] = np.linalg.norm(fitted, axis=1)
    return values


def pair_values_3d(points_a: pd.DataFrame, points_b: pd.DataFrame, *, symmetric: bool) -> pd.DataFrame:
    """On the frames both were placed; distances once per pair, where a person's head points for each of them."""
    frames_a, times_a, found_a = _joints(points_a)
    frames_b, _, found_b = _joints(points_b)
    shared, in_a, in_b = np.intersect1d(frames_a, frames_b, return_indices=True)
    values = pd.DataFrame({"t_start": times_a.to_numpy(dtype=float)[in_a]})
    pick_a = {joint: xyz[in_a] for joint, xyz in found_a.items()}
    pick_b = {joint: xyz[in_b] for joint, xyz in found_b.items()}
    head_a, head_b = _part(pick_a, "left_ear", "right_ear"), _part(pick_b, "left_ear", "right_ear")
    if symmetric:
        shoulders_a, shoulders_b = _part(pick_a, "left_shoulder", "right_shoulder"), _part(pick_b, "left_shoulder", "right_shoulder")
        if shoulders_a is not None and shoulders_b is not None:
            values["interpersonal_distance_3d"] = np.linalg.norm(shoulders_a - shoulders_b, axis=1)
        if head_a is not None and head_b is not None:
            values["head_distance_3d"] = np.linalg.norm(head_a - head_b, axis=1)
    if head_a is not None and head_b is not None and "nose" in pick_a:
        values["head_toward_deg_3d"] = _angle(pick_a["nose"] - head_a, head_b - head_a)
    chest, shoulders_a = chest_facing(pick_a), _part(pick_a, "left_shoulder", "right_shoulder")
    shoulders_b = _part(pick_b, "left_shoulder", "right_shoulder")
    if chest is not None and shoulders_b is not None:
        values["body_toward_deg_3d"] = _angle(chest, shoulders_b - shoulders_a)
    return values


def chest_facing(found: dict[str, np.ndarray]) -> np.ndarray | None:
    """Square to the shoulder line and the spine; the nose picks the front, so swapped left and right do not turn it round."""
    shoulders, hips = _part(found, "left_shoulder", "right_shoulder"), _part(found, "left_hip", "right_hip")
    if shoulders is None or hips is None or "nose" not in found:
        return None
    normal = np.cross(found["left_shoulder"] - found["right_shoulder"], shoulders - hips)
    with np.errstate(invalid="ignore"):
        front = np.sign(np.sum(normal * (found["nose"] - shoulders), axis=1))
    return normal * front[:, None]


def _angle(one: np.ndarray, other: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(one * other, axis=1) / (np.linalg.norm(one, axis=1) * np.linalg.norm(other, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def measure_values_3d(measure, points_a: pd.DataFrame, points_b: pd.DataFrame, *, smoothing_s: float, function=None) -> pd.DataFrame:
    """A project measure on the frames both people were placed on; function stands in for a script method."""
    frames_a, times_a, found_a = _joints(points_a)
    frames_b, _, found_b = _joints(points_b)
    frames, in_a, in_b = np.intersect1d(frames_a, frames_b, return_indices=True)
    times = times_a.to_numpy(dtype=float)[in_a]
    a = {joint: xyz[in_a] for joint, xyz in found_a.items()}
    b = {joint: xyz[in_b] for joint, xyz in found_b.items()}
    if function is not None:
        values = call_script(measure.method, function, _skeleton_table_3d(a, frames, times), _skeleton_table_3d(b, frames, times))
    elif measure.method == "offset_toward":
        own, centre, target = _at(a, measure.joints[0], len(frames)), _at(a, "shoulder_center", len(frames)), _at(b, measure.to_joints[0], len(frames))
        toward = target - centre
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.sum((own - centre) * toward, axis=1) / np.linalg.norm(toward, axis=1)
    else:
        ends = np.stack([_at(a, joint, len(frames)) for joint in measure.joints], axis=1)[:, :, None, :]
        targets = np.stack([_at(b, joint, len(frames)) for joint in measure.to_joints], axis=1)[:, None, :, :]
        values = np.fmin.reduce(np.linalg.norm(ends - targets, axis=-1).reshape(len(frames), -1), axis=1)
        if measure.method == "approach_speed":
            values = -_smoothed_derivative(values[:, None], frames, _step(frames, times), smoothing_s, 1)[:, 0]
    if measure.per_body_size:
        values = values / _shoulder_width(found_a)
    return pd.DataFrame({"t_start": times, f"{measure.name}_3d": values})


def _at(found: dict[str, np.ndarray], joint: str, n: int) -> np.ndarray:
    """A joint or a centre the measures may name, NaN throughout when it was never placed."""
    point = _part(found, *_CENTRES.get(joint, (joint,)))
    return np.full((n, 3), np.nan) if point is None else point


def _shoulder_width(found: dict[str, np.ndarray]) -> float:
    # in the room a shoulder width does not shrink as the person turns, so its median is their size
    if "left_shoulder" not in found or "right_shoulder" not in found:
        return float("nan")
    width = np.linalg.norm(found["left_shoulder"] - found["right_shoulder"], axis=1)
    return float(np.nanmedian(width)) if np.isfinite(width).any() else float("nan")


def _skeleton_table_3d(found: dict[str, np.ndarray], frames: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """What a script measure gets per person in 3D: t and each joint's x, y and z in metres."""
    columns = {"t": times}
    for joint in STANDARD_JOINTS:
        xyz = found.get(joint, np.full((len(frames), 3), np.nan))
        columns.update({f"{joint}_{axis}": xyz[:, index] for index, axis in enumerate("xyz")})
    return pd.DataFrame(columns, index=pd.Index(frames, name="frame"))


def below_3d_to_metres(measure, units: str) -> float:
    """below_3d is typed in the project's unit for a length or speed in the room, else in the measure's own unit."""
    from macaw.project.room import CM_PER_UNIT

    if measure.method not in BUILTIN_MEASURES or measure.per_body_size:
        return 1.0
    return CM_PER_UNIT[units] / 100.0


def below_3d_unit(measure, units: str) -> str | None:
    if measure.method not in BUILTIN_MEASURES or measure.per_body_size:
        return measure_spec_3d(measure)[1]
    return f"{units}/s" if measure.method == "approach_speed" else units


def measure_spec_3d(measure) -> tuple[str, str | None, bool]:
    """(description, unit, comparable across subjects) of a project measure's 3D version."""
    if measure.method not in BUILTIN_MEASURES:
        return f"Computed by the project script {measure.method} on the points in the room.", None, False
    who, target = measure_ends(measure)
    if measure.method == "distance":
        text, unit = f"Distance in the room from {who} to {target}, the nearest pair where several are listed.", "m"
    elif measure.method == "approach_speed":
        text, unit = f"Rate at which the distance in the room from {who} to {target} shrinks; positive while closing in.", "m/s"
    else:
        text, unit = f"How far {who} sits from their own shoulder midpoint in the direction of {target}; negative when away from it.", "m"
    if measure.per_body_size:
        text += f" In units of {measure.person}'s shoulder width."
        unit = "1/s" if unit == "m/s" else None
    return text, unit, True
