from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend
from macaw.extractors.video.pose.skeleton import (
    JOINT_SET,
    STANDARD_JOINTS,
    extra_joint,
    is_standard,
)
from macaw.store.schema import (
    BODY_MODEL_SCHEMA,
    KEYPOINT_SCHEMA,
    UNITS_PIXEL,
    coerce_frame,
    empty_frame,
)

BACKEND_GROUP = "pose"

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

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        min_confidence: float,
        model: str | Path | None = None,
        max_people: int = 1,
    ) -> PoseFrames:
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


MEDIAPIPE_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

# 17 of those 33 spell the standard skeleton exactly, so mapping is a membership test; the
# other 16 keep their native name behind the backend prefix, which leaves the hands and feet
# available without a feature ever reading a point only one backend has

_POSE_MODEL_ENV_VAR = "MACAW_POSE_MODEL"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def landmark_rows(
    poses,
    *,
    frame_index: int,
    width: int,
    height: int,
    min_confidence: float,
) -> list[dict]:
    """One frame of detections to BACKEND_KEYPOINT_SCHEMA rows. Pure, so it can be tested
    without a model file or a video: it only reads x, y and visibility off each landmark."""
    rows: list[dict] = []
    for slot, landmarks in enumerate(poses):
        for index, landmark in enumerate(landmarks):
            if index >= len(MEDIAPIPE_LANDMARKS):
                break                  # a newer model with extra points stays readable
            native = MEDIAPIPE_LANDMARKS[index]
            confidence = float(getattr(landmark, "visibility", float("nan")))
            below = not (confidence >= min_confidence)
            rows.append(
                {
                    "frame": frame_index,
                    "joint": native if is_standard(native) else extra_joint("mediapipe", native),
                    # normalised coordinates are meaningless once a video is resized, so they
                    # are stored as pixels and units says so
                    "x": float("nan") if below else float(landmark.x) * width,
                    "y": float("nan") if below else float(landmark.y) * height,
                    # world landmarks exist but are metres from that person's own hip, which
                    # cannot answer how far apart two people are; see the z rule above
                    "z": float("nan"),
                    "confidence": confidence,
                    "track_id": slot,
                    "units": UNITS_PIXEL,
                    "metric_scale_valid": None,
                }
            )
    return rows


@register_backend(BACKEND_GROUP)
class MediaPipeBackend(PoseBackend):
    """2D only. It does report a z, but that z is relative to the hip midpoint in units of
    torso width, so it is dropped rather than written to a column that means metres.

    track_id is the detection slot within a frame, not a track: nothing keeps slot 0 on the
    same person from one frame to the next, which is exactly the input track_repair exists
    to fix. Running this backend without repair is a bug, not a shortcut.
    """

    name = "mediapipe"

    @property
    def version(self) -> str:
        import mediapipe

        return getattr(mediapipe, "__version__", "unknown")

    def check_available(self) -> None:
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "The 'mediapipe' backend needs mediapipe and opencv: "
                'pip install "pymacaw[vision]"'
            ) from error

    def _model_path(self, model: str | Path | None) -> Path:
        """Task files are not shipped with the wheel and are not downloaded on demand, so the
        lookup order matches ffmpeg's: configured value, environment, then a clear failure."""
        import os

        candidate = model or os.environ.get(_POSE_MODEL_ENV_VAR)
        if not candidate:
            raise ExternalToolError(
                "The 'mediapipe' backend needs a pose landmarker task file. Set pose.model, "
                f"or {_POSE_MODEL_ENV_VAR}, to a copy of {_POSE_MODEL_URL}"
            )
        path = Path(candidate)
        if not path.is_file():
            raise ExternalToolError(f"pose.model does not point at a file: {path}")
        return path

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        min_confidence: float,
        model: str | Path | None = None,
        max_people: int = 1,
    ) -> PoseFrames:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        model_path = self._model_path(model)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0:
                raise ExternalToolError(f"video reports no frame rate: {video_path}")

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=(
                        BaseOptions.Delegate.GPU
                        if str(device).lower() == "cuda"
                        else BaseOptions.Delegate.CPU
                    ),
                ),
                running_mode=RunningMode.VIDEO,
                num_poses=max(int(max_people), 1),
                # detection stays permissive and the threshold is applied per keypoint below,
                # so a low-confidence frame is still counted as a miss rather than vanishing
                min_pose_detection_confidence=0.5,
            )

            rows: list[dict] = []
            with PoseLandmarker.create_from_options(options) as landmarker:
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    )
                    # VIDEO mode requires strictly increasing timestamps, and the frame index
                    # is the only clock the backend can be sure of; recording time is the
                    # extractor's job
                    result = landmarker.detect_for_video(
                        image, int(round(frame_index * 1000.0 / fps))
                    )
                    rows.extend(
                        landmark_rows(
                            result.pose_landmarks,
                            frame_index=frame_index,
                            width=width,
                            height=height,
                            min_confidence=min_confidence,
                        )
                    )
                    frame_index += 1
        finally:
            capture.release()

        if not rows:
            return PoseFrames(keypoints=empty_frame(BACKEND_KEYPOINT_SCHEMA))
        frame = coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
        return PoseFrames(keypoints=frame, provides=self.provides)


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


def resolve_backend(name: str) -> PoseBackend:
    """Exactly the named backend, no fallback chain. Prosody has one because it has a pure
    python last resort; here a silent downgrade would change the joint set under the user."""
    from macaw.extractors import get_backend

    backend = get_backend(BACKEND_GROUP, name)()
    backend.check_available()
    return backend
