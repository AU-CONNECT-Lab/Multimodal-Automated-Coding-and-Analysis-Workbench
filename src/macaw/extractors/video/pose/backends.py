from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend
from macaw.media.ffmpeg import run_ffmpeg
from macaw.media.probe import probe
from macaw.telemetry import quiet_environment
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
# the extractor drops it once subject_id has been filled in.
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
    frame_size: tuple[int, int] | None = None   # (width, height) px; points outside it are dropped


class PoseBackend(Backend):
    joint_set: str = JOINT_SET
    joint_names: tuple[str, ...] = STANDARD_JOINTS
    provides: frozenset[str] = PROVIDES_2D_ONLY
    # what the backend sends out while it runs that no setting here can stop; shown before a run
    reporting: str | None = None

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        min_confidence: float,
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
    ) -> PoseFrames:
        """One file in, that file's own tracks out. frames is a half-open range of the file's
        own frame numbers, None for all of it; frame numbers stay file-absolute either way.
        input_size is the side frames are scaled to for a backend that takes one.
        Four rules hold for every backend:

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
    without a model file or a video: it only reads x, y, visibility and presence off each landmark."""
    rows: list[dict] = []
    for slot, landmarks in enumerate(poses):
        for index, landmark in enumerate(landmarks):
            if index >= len(MEDIAPIPE_LANDMARKS):
                break                  # a newer model with extra points stays readable
            native = MEDIAPIPE_LANDMARKS[index]
            confidence = _score(landmark, "visibility")
            # visibility alone passes a point extrapolated past the image edge; presence is the
            # model's own estimate that the point is inside the frame
            presence = _score(landmark, "presence")
            if presence < confidence:
                confidence = presence
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


def _score(landmark, name: str) -> float:
    value = getattr(landmark, name, None)
    return float("nan") if value is None else float(value)


@register_backend(BACKEND_GROUP)
class MediaPipeBackend(PoseBackend):
    """2D only. It does report a z, but that z is relative to the hip midpoint in units of
    torso width, so it is dropped rather than written to a column that means metres.

    track_id is the detection slot within a frame, not a track: nothing keeps slot 0 on the
    same person from one frame to the next, which is exactly the input track_repair exists
    to fix. Running this backend without repair is a bug, not a shortcut.
    """

    name = "mediapipe"
    reporting = (
        "pose.backend mediapipe: MediaPipe sends usage and performance metrics to Google while it "
        "runs (not the video itself), and has no switch to turn this off. "
        "pose.backend: yolo runs offline."
    )

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
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
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

            first, stop = frames if frames is not None else (0, None)
            frame_index = _seek(capture, cv2, max(int(first), 0))

            rows: list[dict] = []
            with PoseLandmarker.create_from_options(options) as landmarker:
                while stop is None or frame_index < stop:
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
        return PoseFrames(keypoints=frame, provides=self.provides, frame_size=(width, height))


def _seek(capture, cv2, first: int) -> int:
    """Jump to a frame, or read up to it when the container will not say where it landed."""
    if first == 0:
        return 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first:
        return first
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    index = 0
    while index < first and capture.grab():
        index += 1
    return index


YOLO_ENV_VAR = "MACAW_YOLO"
# the order the COCO pose models write their 17 points in
COCO_KEYPOINT_ORDER: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# boxes below this never reach the label files; keypoints are thresholded on their own by min_confidence
YOLO_DETECTION_CONFIDENCE = 0.25
YOLO_TRACKER = "bytetrack.yaml"
_LABEL_FRAME = re.compile(r"_(\d+)\.txt$")


def yolo_label_rows(
    lines: list[str], *, frame_index: int, width: int, height: int, min_confidence: float
) -> list[dict]:
    """One frame's label lines to BACKEND_KEYPOINT_SCHEMA rows. A line is class, box (4), then x y
    confidence per point, all normalised, then the box confidence and, when tracking, the track id."""
    n_values = 3 * len(COCO_KEYPOINT_ORDER)
    rows: list[dict] = []
    for line in lines:
        numbers = [float(value) for value in line.split()]
        if not numbers:
            continue
        tail = len(numbers) - 5 - n_values
        if tail not in (1, 2):
            raise ExternalToolError(
                f"a yolo label line has {len(numbers)} values; expected a 17-point pose model with "
                "keypoint confidences"
            )
        track_id = int(numbers[-1]) if tail == 2 else None
        points = numbers[5 : 5 + n_values]
        for index, joint in enumerate(COCO_KEYPOINT_ORDER):
            x, y, confidence = points[3 * index : 3 * index + 3]
            # some versions write a point they do not trust at the image corner
            below = not (confidence >= min_confidence) or (x == 0.0 and y == 0.0)
            rows.append(
                {
                    "frame": frame_index,
                    "joint": joint,
                    "x": float("nan") if below else x * width,
                    "y": float("nan") if below else y * height,
                    "z": float("nan"),
                    "confidence": confidence,
                    "track_id": track_id,
                    "units": UNITS_PIXEL,
                    "metric_scale_valid": None,
                }
            )
    return rows


def yolo_device(device: str) -> str | None:
    """The config's device in the form the yolo command takes; None lets it choose."""
    name = str(device).lower()
    if name == "auto":
        return None
    if name.startswith("cuda"):
        return name.partition(":")[2] or "0"
    return name


@register_backend(BACKEND_GROUP)
class YOLOPoseBackend(PoseBackend):
    """Runs the ultralytics `yolo` command from its own environment and reads the label files it
    writes. The package never imports it, so its AGPL licence stays with the user's own install."""

    name = "yolo"

    @property
    def version(self) -> str:
        result = subprocess.run(
            [self._executable(), "version"], capture_output=True, text=True, env=quiet_environment()
        )
        lines = result.stdout.strip().splitlines()
        return lines[-1].strip() if result.returncode == 0 and lines else "unknown"

    def _executable(self) -> str:
        configured = os.environ.get(YOLO_ENV_VAR)
        found = configured if configured and Path(configured).is_file() else shutil.which("yolo")
        if not found:
            raise BackendNotAvailableError(
                "The 'yolo' backend runs the `yolo` command from ultralytics, which pymacaw does not "
                "install. Create a separate environment, `pip install ultralytics lap` there, and point "
                f"${YOLO_ENV_VAR} at its yolo executable, or put it on PATH. pymacaw runs it offline: it "
                "reports no usage and installs nothing by itself"
            )
        return found

    def check_available(self) -> None:
        self._executable()

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        min_confidence: float,
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
    ) -> PoseFrames:
        """Everyone in the frame comes back with the tracker's ids; picking max_people is left to assignment."""
        if model is None or not Path(model).is_file():
            raise ExternalToolError(
                "The 'yolo' backend needs pose.model to point at a pose weights file you have downloaded, "
                f"such as yolo11x-pose.pt; nothing is fetched while it runs (got {model})"
            )
        info = probe(video_path)
        stream = next(iter(info.video_streams()), None)
        if stream is None or not stream.width or not stream.height:
            raise ExternalToolError(f"no video stream with a frame size in {video_path}")

        first, stop = frames if frames is not None else (0, None)
        first = max(int(first), 0)
        n_frames = round((stream.duration_s or info.duration_s) * (stream.fps or 0.0))
        whole = first == 0 and (stop is None or stop >= n_frames > 0)
        rows: list[dict] = []
        with tempfile.TemporaryDirectory(prefix="macaw_yolo_") as scratch:
            scratch = Path(scratch)
            source = video_path if whole else _window_clip(video_path, first, stop, scratch)
            labels = self._run_subprocess(
                source, scratch / "out", model=model, input_size=input_size, device=device
            )
            for path in sorted(labels.glob("*.txt")):
                match = _LABEL_FRAME.search(path.name)
                if match is None:
                    continue
                # label files count the frames of the clip they came from, from 1
                frame_index = first + int(match.group(1)) - 1
                if stop is not None and frame_index >= stop:
                    continue
                rows.extend(
                    yolo_label_rows(
                        path.read_text(encoding="utf-8").splitlines(),
                        frame_index=frame_index,
                        width=stream.width,
                        height=stream.height,
                        min_confidence=min_confidence,
                    )
                )

        if not rows:
            return PoseFrames(keypoints=empty_frame(BACKEND_KEYPOINT_SCHEMA))
        frame = coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
        return PoseFrames(keypoints=frame, provides=self.provides, frame_size=(stream.width, stream.height))

    def _run_subprocess(
        self, video_path: Path, out_path: Path, *, model, input_size: int | None, device: str
    ) -> Path:
        """Returns the directory of per-frame label files."""
        command = [
            self._executable(), "pose", "track",
            f"model={model}", f"source={video_path}",
            f"imgsz={input_size or 640}", f"conf={YOLO_DETECTION_CONFIDENCE}", f"tracker={YOLO_TRACKER}",
            "save=False", "save_txt=True", "save_conf=True",
            f"project={out_path}", "name=run", "exist_ok=True", "verbose=False",
        ]
        chosen = yolo_device(device)
        if chosen is not None:
            command.append(f"device={chosen}")
        result = subprocess.run(command, capture_output=True, text=True, env=quiet_environment())
        if result.returncode != 0:
            tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
            raise ExternalToolError(f"yolo failed on {Path(video_path).name}:\n{tail}")
        return out_path / "run" / "labels"


def _window_clip(video_path: Path, first: int, stop: int | None, scratch: Path) -> Path:
    """Frames [first, stop) as a clip of their own, chosen by frame number so none is gained or lost."""
    chosen = f"gte(n\\,{first})" if stop is None else f"between(n\\,{first}\\,{stop - 1})"
    clip = scratch / "window.mp4"
    # an encoder every ffmpeg build has, at a quality the detector cannot tell from the source
    run_ffmpeg([
        "-i", str(video_path), "-an",
        "-vf", f"select={chosen},setpts=PTS-STARTPTS", "-fps_mode", "passthrough",
        "-c:v", "mpeg4", "-q:v", "2", str(clip),
    ])
    return clip


class DWPoseBackend(PoseBackend):
    name = "dwpose"

    def check_available(self) -> None:
        raise NotImplementedError


NO_BACKEND_MESSAGE = (
    "pose.backend is not set, and there is no default: each choice has a cost to weigh. "
    'mediapipe: pip install "pymacaw[vision]"; it sends usage metrics to Google and cannot be told '
    "not to. yolo: runs offline; it needs the ultralytics `yolo` command in an environment of its "
    "own, under the AGPL-3.0 licence."
)


def resolve_backend(name: str | None) -> PoseBackend:
    """Exactly the named backend, no fallback chain. Prosody has one because it has a pure
    python last resort; here a silent downgrade would change the joint set under the user."""
    from macaw.extractors import get_backend

    if name is None:
        raise BackendNotAvailableError(NO_BACKEND_MESSAGE)
    backend = get_backend(BACKEND_GROUP, name)()
    backend.check_available()
    return backend
