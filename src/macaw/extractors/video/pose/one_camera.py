"""3D from each person's own camera: a body model fitted to every person in the picture (pose.mesh, run in the
environment the model is installed in), then each named person placed from the body that matches their keypoints,
set to their stated height where the heights table gives one (pose.one_camera)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from macaw.exceptions import BackendNotAvailableError, ConfigError, ExternalToolError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, load_video_prep
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.project.heights import read_heights
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.project.room import CM_PER_UNIT
from macaw.store import keypoints as keypoints_store
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, Scale, coerce_frame, empty_frame

BUILT_IN = "multihmr"
PYTHON_ENV_VAR = "MACAW_MULTIHMR_PYTHON"
CODE_ENV_VAR = "MACAW_MULTIHMR"
RUNNER = Path(__file__).with_name("one_camera_runner.py")
QUALITY_NAME = "one_camera.csv"
_UNITS_METRE = "metre"
# a fitted body is someone's when its keypoints fall within this share of that person's size in the picture
_SAME_PERSON = 0.3
_BONE_SPREAD = 0.15
_FEW_PLACED = 0.5


def mesh_file(camera: str) -> str:
    return f"{camera}.mesh.npz"


def body_cameras(recording: Recording) -> dict[str, object]:
    """camera -> its media file, for each camera someone's body is measured on."""
    return {item.device: item for item in recording.views("body").values()}


def _python() -> str:
    path = os.environ.get(PYTHON_ENV_VAR)
    if not path or not Path(path).is_file():
        raise BackendNotAvailableError(
            f"pose.mesh runs the body model in its own environment: set {PYTHON_ENV_VAR} to that environment's python"
        )
    return path


def runner_path(config) -> Path:
    chosen = config.pose3d.one_camera
    if chosen == BUILT_IN:
        return RUNNER
    path = Path(config.project.scripts_dir) / chosen
    if not path.is_file():
        raise ConfigError(f"pose3d.one_camera {chosen}: neither {BUILT_IN} nor a script in {config.project.scripts_dir}")
    return path


@register_extractor
class MeshExtractor(Extractor):
    """Every person a body model finds in each frame of the cameras bodies are measured on, before anyone is named."""

    name = "pose.mesh"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose3d.one_camera", "pose3d.one_camera_every", "setup.cameras", "video")
    requires = (MediaPrepExtractor.name, VideoPrepExtractor.name)
    resource_profile = "gpu_standard"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return False                    # pose.one_camera pulls it in

    def check_available(self) -> None:
        if self.config.pose3d.one_camera is None:
            raise ConfigError("pose.mesh needs pose3d.one_camera")
        _python()
        if self.config.pose3d.one_camera == BUILT_IN:
            code = os.environ.get(CODE_ENV_VAR)
            if not code or not Path(code).is_dir():
                raise BackendNotAvailableError(f"pose3d.one_camera {BUILT_IN}: set {CODE_ENV_VAR} to the folder of the Multi-HMR code")
        runner_path(self.config)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return sorted(item.path for item in body_cameras(recording).values()) + [runner_path(self.config)]

    def extra_key(self, recording: Recording) -> str:
        # the picture's angle, where the room gives it, sets the lens the model assumes
        return json.dumps({camera: self._fov(camera) for camera in sorted(body_cameras(recording))})

    def _fov(self, camera: str) -> float | None:
        placed = self.config.setup.room.cameras.get(camera)
        return None if placed is None or placed.lens is None else float(placed.lens)

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        videos = load_video_prep(self.config, recording.recording_id)
        workspace.mkdir(parents=True, exist_ok=True)
        artifacts, warnings = {}, []
        for camera, item in sorted(body_cameras(recording).items()):
            # the frames pose.keypoints numbered are the prepared copy's
            frames = (videos.get(camera) or {}).get("path") or item.path
            out = workspace / mesh_file(camera)
            job = workspace / f"{camera}.job.json"
            job.write_text(json.dumps({
                "video": str(frames), "out": str(out), "code": os.environ.get(CODE_ENV_VAR, ""), "checkpoint": None,
                "fov_deg": self._fov(camera), "every": self.config.pose3d.one_camera_every,
            }), encoding="utf-8")
            run_runner(_python(), runner_path(self.config), job)
            artifacts[out.name] = out
            if self._fov(camera) is None:
                warnings.append(QCWarning(
                    check=self.name, recording_id=recording.recording_id, subject_id=None, severity="info",
                    message=f"{camera}: the model estimated the picture's angle itself; giving the camera's lens angle in the setup's room fixes it",
                ))
        return ExtractorResult(rows=empty_frame(LONG_TABLE_SCHEMA), artifacts=artifacts, warnings=warnings)


def run_runner(python: str, runner: Path, job: Path) -> None:
    done = subprocess.run([python, str(runner), str(job)], capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-15:])
        raise ExternalToolError(f"the body model stopped ({runner.name}):\n{tail}")


def read_mesh(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in ("frame", "points", "pixels", "height", "focal")}


def match(people: dict[str, np.ndarray], found: np.ndarray) -> dict[str, int]:
    """person -> which fitted body is theirs in one frame, by how close its keypoints fall to theirs in the picture.
    people: person -> (17, 2) pixels with NaN where unseen; found: (n, 17, 2)."""
    names = list(people)
    if not names or not len(found):
        return {}
    cost = np.full((len(names), len(found)), np.inf)
    for row, name in enumerate(names):
        mine = people[name]
        seen = np.isfinite(mine).all(axis=1)
        if seen.sum() < 2:
            continue
        size = float(np.linalg.norm(np.nanmax(mine[seen], axis=0) - np.nanmin(mine[seen], axis=0)))
        if size <= 0:
            continue
        for col in range(len(found)):
            cost[row, col] = float(np.median(np.linalg.norm(found[col][seen] - mine[seen], axis=1))) / size
    rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
    return {names[row]: int(col) for row, col in zip(rows, cols) if cost[row, col] <= _SAME_PERSON}


@register_extractor
class OneCameraExtractor(Extractor):
    """Each named person's body points in 3D from the body model fitted to them on the camera their body is measured on."""

    name = "pose.one_camera"
    version = "0.1.1"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("pose3d.one_camera", "pose3d.one_camera_every", "project.units", "setup.cameras")
    requires = ("pose.review", MeshExtractor.name)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.pose3d.one_camera is not None

    def check_available(self) -> None:
        from macaw.extractors.video.pose.triangulate import three_d_step

        three_d_step(self.config)

    def inputs_for(self, recording: Recording) -> list[Path]:
        from macaw.extractors.video.pose.kinematics import _keypoints_file

        return [_keypoints_file(self.config, recording.recording_id)]

    def extra_key(self, recording: Recording) -> str:
        heights = read_heights(self.config)
        return json.dumps({person: heights.get((recording.recording_id, person)) for person in sorted(self.config.project.roles)})

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        from macaw.extractors.video.pose.kinematics import _keypoints_file
        from macaw.extractors.video.pose.triangulate import bone_spread

        rid = recording.recording_id
        keypoints = keypoints_store.read_keypoints(_keypoints_file(self.config, rid), recording_id=rid)
        keypoints = keypoints[keypoints["subject_id"].notna() & keypoints["joint"].isin(STANDARD_JOINTS)]
        heights = read_heights(self.config)
        folder = layout_for(self.config, create=False).artifacts_dir(rid, MeshExtractor.name)
        views = body_cameras(recording)
        pieces, quality, warnings = [], [], []
        joint_index = {joint: index for index, joint in enumerate(STANDARD_JOINTS)}
        for camera in sorted(views):
            path = folder / mesh_file(camera)
            if not path.is_file():
                continue
            mesh = read_mesh(path)
            people = sorted(person for person, item in recording.views("body").items() if item.device == camera)
            own = keypoints[(keypoints["video"] == camera) & keypoints["subject_id"].isin(people)]
            placed: dict[str, list] = {person: [] for person in people}
            for frame, group in own.groupby("frame"):
                pixels = {}
                for person, mine in group.groupby("subject_id"):
                    grid = np.full((len(STANDARD_JOINTS), 2), np.nan)
                    for row in mine.itertuples(index=False):
                        if np.isfinite(row.x) and np.isfinite(row.y):
                            grid[joint_index[row.joint]] = (row.x, row.y)
                    pixels[str(person)] = grid
                here = np.flatnonzero(mesh["frame"] == frame)
                for person, which in match(pixels, mesh["pixels"][here]).items():
                    body = here[which]
                    seen = np.isfinite(pixels[person]).all(axis=1)
                    placed[person].append((frame, mesh["points"][body], seen, float(mesh["height"][body])))
            # the model reads every nth frame, so only those count toward how often someone was placed
            read = own[own["frame"] % self.config.pose3d.one_camera_every == 0]
            for person in people:
                seen_frames = int(read.loc[read["subject_id"] == person, "frame"].nunique())
                rows, warning = self._person(rid, person, placed[person], seen_frames, heights.get((rid, person)), own)
                warnings += warning
                if rows is not None:
                    # points from one camera are in that camera's frame, which the video column names
                    pieces.append(rows.assign(video=camera))
                quality.append({"subject_id": person, "camera": camera, "frames_seen": seen_frames, "frames_placed": len(placed[person]),
                                "model_height_m": float(np.median([entry[3] for entry in placed[person]])) if placed[person] else np.nan,
                                "stated_height": heights.get((rid, person))})
        points = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["subject_id", "video", "frame", "t", "joint", "confidence", "x", "y", "z"])
        for person, own_points in points.groupby("subject_id"):
            for (first, second), spread in bone_spread(own_points).items():
                if spread > _BONE_SPREAD:
                    bone = f"{first.replace('_', ' ')} to {second.replace('_', ' ')}"
                    warnings.append(self._warning(rid, person, "warn", f"the {bone} changes length by {spread:.0%}, so the fitted body is misplaced in some frames"))
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(quality).to_csv(directory / QUALITY_NAME, index=False)
        stated = {person for person in points["subject_id"].unique() if heights.get((rid, person)) is not None}
        rows = coerce_frame(
            points.assign(
                recording_id=rid, source=self.name, track_id=pd.NA, units=_UNITS_METRE,
                metric_scale_valid=points["subject_id"].isin(stated),
            ),
            KEYPOINT_SCHEMA, "keypoints",
        )
        return ExtractorResult(rows=empty_frame(LONG_TABLE_SCHEMA), keypoints=rows, artifacts={QUALITY_NAME: directory / QUALITY_NAME}, warnings=warnings)

    def _person(self, rid, person, placed, seen_frames, stated, own) -> tuple[pd.DataFrame | None, list[QCWarning]]:
        if not placed:
            return None, [self._warning(rid, person, "warn", "no fitted body matched their keypoints, so they are not placed in 3D")]
        share = len(placed) / max(seen_frames, 1)
        model_m = float(np.median([entry[3] for entry in placed]))
        warnings = []
        if stated is None:
            scale = 1.0
            warnings.append(self._warning(
                rid, person, "warn",
                f"no height in the heights table, so placed at the size of the body the model fitted ({model_m:.2f} m tall); their distance from the "
                "camera and from others is only as right as that size, which for a small child is too large",
            ))
        else:
            stated_m = stated * CM_PER_UNIT[self.config.project.units] / 100.0
            scale = stated_m / model_m
            warnings.append(self._warning(
                rid, person, "info", f"set to the stated height, {stated_m:.2f} m, from the fitted body's {model_m:.2f} m (scaled by {scale:.2f})",
            ))
        every = self.config.pose3d.one_camera_every
        read = f", counting the one frame in {every} the body model read" if every > 1 else ""
        warnings.append(self._warning(
            rid, person, "warn" if share < _FEW_PLACED else "info", f"placed in 3D in {share:.0%} of the frames they were seen in{read}",
        ))
        times = own.groupby("frame")["t"].first()
        confidence = own.set_index(["subject_id", "frame", "joint"])["confidence"]
        rows = []
        for frame, points, seen, _ in placed:
            for index, joint in enumerate(STANDARD_JOINTS):
                if seen[index]:
                    x, y, z = points[index] * scale
                    rows.append({"subject_id": person, "frame": int(frame), "t": float(times[frame]), "joint": joint,
                                 "confidence": float(confidence.get((person, frame, joint), np.nan)), "x": x, "y": y, "z": z})
        return pd.DataFrame(rows), warnings

    def _warning(self, rid: str, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=rid, subject_id=subject_id, severity=severity, message=message)
