from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macaw.extractors.base import Backend
from macaw.extractors.video.pose.skeleton import JOINT_SET, STANDARD_JOINTS
from macaw.store.schema import BODY_MODEL_SCHEMA, KEYPOINT_SCHEMA

# What a backend fills in, and nothing more. recording_id, subject_id, source and t are the
# extractor's job: a backend sees one file and its own tracks, so letting it name a subject
# would scatter identity logic across every backend instead of keeping it in one testable place.
BACKEND_KEYPOINT_COLUMNS: tuple[str, ...] = (
    "frame",
    "joint",
    "x",
    "y",
    "z",
    "confidence",
    "track_id",
    "units",
    "metric_scale_valid",
)
BACKEND_KEYPOINT_SCHEMA: dict[str, str] = {
    column: KEYPOINT_SCHEMA[column] for column in BACKEND_KEYPOINT_COLUMNS
}

# Same subset rule for the parameter table, plus track_id, which the stored table has no
# column for: it is only the join key that carries parameters from a track to a subject, and
# the extractor drops it once assign_subjects has filled subject_id in.
_EXTRACTOR_OWNED = ("recording_id", "subject_id", "source", "t")
BACKEND_BODY_MODEL_SCHEMA: dict[str, str] = {
    "track_id": KEYPOINT_SCHEMA["track_id"],
    **{
        column: spec
        for column, spec in BODY_MODEL_SCHEMA.items()
        if column not in _EXTRACTOR_OWNED
    },
}

# Declared capability, so feature code branches on what a backend says it returns instead of
# sniffing for an all-NaN column and guessing.
PROVIDES_KEYPOINTS_2D = "keypoints_2d"
PROVIDES_KEYPOINTS_3D = "keypoints_3d"
PROVIDES_SHAPE = "shape"
PROVIDES_POSE = "pose"
PROVIDES_CAMERA = "camera"
PROVIDES_BODY_MODEL = "body_model"

PROVIDES_2D_ONLY = frozenset({PROVIDES_KEYPOINTS_2D})
PROVIDES_FULL_3D = frozenset(
    {
        PROVIDES_KEYPOINTS_2D,
        PROVIDES_KEYPOINTS_3D,
        PROVIDES_SHAPE,
        PROVIDES_POSE,
        PROVIDES_CAMERA,
        PROVIDES_BODY_MODEL,
    }
)


@dataclass
class PoseFrames:
    """One backend's reading of one video file."""

    keypoints: pd.DataFrame                     # BACKEND_KEYPOINT_SCHEMA
    body_model: pd.DataFrame | None = None      # BACKEND_BODY_MODEL_SCHEMA, 3D backends only
    provides: frozenset[str] = PROVIDES_2D_ONLY


class PoseBackend(Backend):
    joint_set: str = JOINT_SET
    joint_names: tuple[str, ...] = STANDARD_JOINTS
    provides: frozenset[str] = PROVIDES_2D_ONLY

    def infer(self, video_path: Path, *, device: str, min_confidence: float) -> PoseFrames:
        """One file in, that file's own tracks out. Four rules hold for every backend:

        joint is a name from STANDARD_JOINTS, and any extra point the backend has goes
        through skeleton.extra_joint() so it can never be read as a standard one;

        track_id is the backend's own numbering before repair, and is null when the backend
        does no tracking at all, never a 0 or -1 standing in for one;

        subject_id is not filled in here, and neither is t: an offset is applied once, by the
        extractor, the same way prep.media applies it once for audio;

        z is metres or NaN. A pseudo-depth in units of torso width is neither, and feeding it
        to interpersonal_distance would produce numbers that look right and are not.
        """
        raise NotImplementedError


class RTMPoseBackend(PoseBackend):
    """Optional, and deliberately not implemented: mmengine / mmcv / mmpose usually has to be
    compiled against a specific torch and CUDA. check_available is what users meet instead."""

    name = "rtmpose"

    def check_available(self) -> None:
        raise NotImplementedError


class MediaPipeBackend(PoseBackend):
    """2D only. It does report a z, but that z is relative to the hip midpoint in units of
    torso width, so it is dropped rather than written to a column that means metres."""

    name = "mediapipe"

    def check_available(self) -> None:
        raise NotImplementedError


class YOLOPoseBackend(PoseBackend):
    """AGPL licensed, so it is invoked as a subprocess and never imported."""

    name = "yolo"

    def check_available(self) -> None:
        raise NotImplementedError

    def _run_subprocess(self, video_path: Path, out_path: Path) -> Path:
        raise NotImplementedError


class DWPoseBackend(PoseBackend):
    name = "dwpose"

    def check_available(self) -> None:
        raise NotImplementedError
