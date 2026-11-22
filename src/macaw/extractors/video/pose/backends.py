from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend, distribution_version
from macaw.media.ffmpeg import SEEK_MARGIN_S, run_ffmpeg
from macaw.media.probe import frame_rate, jumps_exactly, probe
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

if TYPE_CHECKING:
    from macaw.extractors.prep import SourceVideo

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
    **{column: KEYPOINT_SCHEMA[column] for column in BACKEND_KEYPOINT_COLUMNS},
    # how sure the detector is that the person is there, NaN for a backend that gives none; kept with the detections only
    "box_score": "float64",
}

# Same subset rule for the parameter table, plus track_id, which the stored table has no
# column for: it is only the join key that carries parameters from a track to a subject, and
# the extractor drops it once subject_id has been filled in.
_EXTRACTOR_OWNED = ("recording_id", "subject_id", "source", "video", "t")
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
    # False when track_id only numbers the people within each frame, so who is who needs pose.track_repair
    tracks: bool = True
    # whether pose.input_size reaches the model
    takes_input_size: bool = False

    def check_model(self, model: str | Path | None) -> None:
        """Stops before a run when the weights it would read are not there."""

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
        source: SourceVideo | None = None,
    ) -> PoseFrames:
        """One file in, that file's own tracks out. frames is a half-open range of the file's
        own frame numbers, None for all of it; frame numbers stay file-absolute either way.
        input_size is the side frames are scaled to for a backend that takes one. source is the
        original file when video_path is prep.video's copy of it, for a backend that reads
        something made from the original; frames and pixels returned are video_path's. Every point
        comes back with its confidence and where the model put it, however unsure: the cut at
        pose.min_confidence and outside the picture is made when the detections are read.
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
            rows.append(
                {
                    "frame": frame_index,
                    "joint": native if is_standard(native) else extra_joint("mediapipe", native),
                    # normalised coordinates are meaningless once a video is resized, so they
                    # are stored as pixels and units says so
                    "x": float(landmark.x) * width,
                    "y": float(landmark.y) * height,
                    # world landmarks exist but are metres from that person's own hip, which
                    # cannot answer how far apart two people are; see the z rule above
                    "z": float("nan"),
                    "confidence": confidence,
                    "track_id": slot,
                    "units": UNITS_PIXEL,
                    "metric_scale_valid": None,
                    "box_score": float("nan"),
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
    tracks = False
    reporting = (
        "pose.backend mediapipe: MediaPipe sends usage and performance metrics to Google while it "
        "runs (not the video itself), and has no switch to turn this off. "
        "pose.backend: yolo runs offline."
    )

    @property
    def version(self) -> str:
        return distribution_version("mediapipe")

    def check_available(self) -> None:
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "The 'mediapipe' backend needs mediapipe and opencv: "
                'pip install "pymacaw[vision]"'
            ) from error

    def check_model(self, model: str | Path | None) -> None:
        self._model_path(model)

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
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
        source: SourceVideo | None = None,
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
            frame_index = _seek(capture, cv2, max(int(first), 0), Path(video_path))

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
                        )
                    )
                    frame_index += 1
        finally:
            capture.release()

        if not rows:
            return PoseFrames(keypoints=empty_frame(BACKEND_KEYPOINT_SCHEMA))
        frame = coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
        return PoseFrames(keypoints=frame, provides=self.provides, frame_size=(width, height))


def _seek(capture, cv2, first: int, path: Path) -> int:
    """Jump to a frame, or read up to it in a transport stream or when the container will not say where it landed."""
    if first == 0:
        return 0
    if jumps_exactly(probe(path)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first:
            return first
        # setting the position back to 0 does not always rewind
        capture.open(str(path))
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
# boxes below this never reach the label files; keypoints are cut at pose.min_confidence when the detections are read
YOLO_DETECTION_CONFIDENCE = 0.25
YOLO_TRACKER = "bytetrack.yaml"
_LABEL_FRAME = re.compile(r"_(\d+)\.txt$")


def yolo_label_rows(lines: list[str], *, frame_index: int, width: int, height: int) -> list[dict]:
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
        box_score = numbers[5 + n_values]
        points = numbers[5 : 5 + n_values]
        for index, joint in enumerate(COCO_KEYPOINT_ORDER):
            x, y, confidence = points[3 * index : 3 * index + 3]
            # some versions write a point they do not trust at the image corner
            below = x == 0.0 and y == 0.0
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
                    "box_score": box_score,
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


@functools.lru_cache(maxsize=None)
def _yolo_version(executable: str) -> str:
    """Asked once per process: each ask starts the command, which imports its whole library."""
    result = subprocess.run([executable, "version"], capture_output=True, text=True, env=quiet_environment())
    lines = result.stdout.strip().splitlines()
    return lines[-1].strip() if result.returncode == 0 and lines else "unknown"


@register_backend(BACKEND_GROUP)
class YOLOPoseBackend(PoseBackend):
    """Runs the ultralytics `yolo` command from its own environment and reads the label files it
    writes. The package never imports it, so its AGPL licence stays with the user's own install."""

    name = "yolo"
    takes_input_size = True

    @property
    def version(self) -> str:
        return _yolo_version(self._executable())

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

    def check_model(self, model: str | Path | None) -> None:
        if model is None or not Path(model).is_file():
            raise BackendNotAvailableError(
                "The 'yolo' backend needs pose.model to point at a pose weights file you have downloaded, "
                f"such as yolo11x-pose.pt; nothing is fetched while it runs (got {model})"
            )

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
        source: SourceVideo | None = None,
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


VITPOSE_DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"
# people the detector is less sure of than this are not looked at for points
VITPOSE_PERSON_SCORE = 0.3
# frames looked at together
_VITPOSE_BATCH = 8


def vitpose_rows(people: list[dict], scores: list[float], *, frame_index: int) -> list[dict]:
    """One frame's people, as ViTPose's processor gives them, to BACKEND_KEYPOINT_SCHEMA rows."""
    rows: list[dict] = []
    for slot, (person, box_score) in enumerate(zip(people, scores)):
        points, confidences = person["keypoints"], person["scores"]
        for index, joint in enumerate(COCO_KEYPOINT_ORDER):
            rows.append({
                "frame": frame_index, "joint": joint, "x": float(points[index][0]), "y": float(points[index][1]), "z": float("nan"),
                "confidence": float(confidences[index]), "track_id": slot, "units": UNITS_PIXEL, "metric_scale_valid": None,
                "box_score": float(box_score),
            })
    return rows


@register_backend(BACKEND_GROUP)
class ViTPoseBackend(PoseBackend):
    """ViTPose+ through transformers on the people RT-DETR finds; Apache-2.0 code and weights, the 17 COCO points, offline.

    track_id is the person's place among those found in the frame, not a track, so pose.track_repair says who is who.
    """

    name = "vitpose"
    tracks = False
    default_model = "usyd-community/vitpose-plus-base"

    def __init__(self) -> None:
        self._parts: dict = {}

    @property
    def version(self) -> str:
        return distribution_version("transformers")

    def check_available(self) -> None:
        try:
            import cv2  # noqa: F401
            import torch  # noqa: F401
            import transformers
        except ImportError as error:
            raise BackendNotAvailableError(
                "The 'vitpose' backend needs PyTorch, transformers and opencv: install PyTorch for your GPU first, then "
                'pip install "pymacaw[vision,asr]"'
            ) from error
        if not all(hasattr(transformers, name) for name in ("VitPoseForPoseEstimation", "RTDetrForObjectDetection")):
            raise BackendNotAvailableError(f"transformers {transformers.__version__} has no ViTPose or RT-DETR: pip install -U transformers")

    def check_model(self, model: str | Path | None) -> None:
        from macaw.extractors.hub import check_hub_model

        check_hub_model(str(model or self.default_model), "pose.model")
        check_hub_model(VITPOSE_DETECTOR, "the RT-DETR person detector of the vitpose backend")

    def _load(self, model: str | Path | None, device: str):
        key = (str(model or self.default_model), device)
        if key not in self._parts:
            from transformers import AutoImageProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

            detector = RTDetrForObjectDetection.from_pretrained(VITPOSE_DETECTOR).to(device).eval()
            person = next(index for index, label in detector.config.id2label.items() if label.lower() == "person")
            # the torchvision processors, about twice as fast as the PIL ones
            self._parts[key] = (
                AutoImageProcessor.from_pretrained(VITPOSE_DETECTOR, use_fast=True), detector, person,
                AutoImageProcessor.from_pretrained(key[0], use_fast=True), VitPoseForPoseEstimation.from_pretrained(key[0]).to(device).eval(),
            )
        return self._parts[key]

    def infer(
        self,
        video_path: Path,
        *,
        device: str,
        model: str | Path | None = None,
        max_people: int = 1,
        frames: tuple[int, int] | None = None,
        input_size: int | None = None,
        source: SourceVideo | None = None,
    ) -> PoseFrames:
        """Everyone the detector finds comes back; picking max_people is left to assignment."""
        import cv2

        from macaw.extractors.audio.content.asr import resolve_device

        device = resolve_device(str(device or "auto"))
        parts = self._load(model, device)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")
        rows: list[dict] = []
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            first, stop = frames if frames is not None else (0, None)
            frame_index = _seek(capture, cv2, max(int(first), 0), Path(video_path))
            batch: list[tuple[int, object]] = []
            while stop is None or frame_index < stop:
                ok, frame = capture.read()
                if not ok:
                    break
                batch.append((frame_index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                frame_index += 1
                if len(batch) == _VITPOSE_BATCH:
                    rows += self._look(batch, parts, device)
                    batch = []
            if batch:
                rows += self._look(batch, parts, device)
        finally:
            capture.release()
        if not rows:
            return PoseFrames(keypoints=empty_frame(BACKEND_KEYPOINT_SCHEMA))
        frame = coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
        return PoseFrames(keypoints=frame, provides=self.provides, frame_size=(width, height))

    def _look(self, batch: list[tuple[int, object]], parts, device: str) -> list[dict]:
        import torch

        detect_processor, detector, person, pose_processor, pose = parts
        images = [image for _, image in batch]
        # resized on the card for the detector; the crops for the points are quicker cut on the processor
        on_card = [torch.from_numpy(image).permute(2, 0, 1).to(device) for image in images]
        inputs = detect_processor(images=on_card, return_tensors="pt", device=device)
        with torch.inference_mode():
            found = detect_processor.post_process_object_detection(
                detector(**inputs), target_sizes=torch.tensor([image.shape[:2] for image in images]), threshold=VITPOSE_PERSON_SCORE
            )
        seen, boxes, scores = [], [], []
        for (index, image), result in zip(batch, found):
            keep = (result["labels"] == person).cpu().numpy()
            if keep.any():
                xyxy = result["boxes"].cpu().numpy()[keep]
                seen.append((index, image))
                boxes.append([[a, b, c - a, d - b] for a, b, c, d in xyxy.tolist()])
                scores.append(result["scores"].cpu().numpy()[keep].tolist())
        if not seen:
            return []
        inputs = pose_processor([image for _, image in seen], boxes=boxes, return_tensors="pt").to(device)
        with torch.inference_mode():
            heatmaps = pose(**inputs, dataset_index=torch.zeros(sum(map(len, boxes)), dtype=torch.long, device=device))
        people = pose_processor.post_process_pose_estimation(heatmaps, boxes=boxes)
        rows: list[dict] = []
        for (index, _), frame_people, frame_scores in zip(seen, people, scores):
            rows += vitpose_rows(
                [{"keypoints": p["keypoints"].cpu().numpy(), "scores": p["scores"].cpu().numpy()} for p in frame_people],
                frame_scores, frame_index=index,
            )
        return rows


def _window_clip(video_path: Path, first: int, stop: int | None, scratch: Path) -> Path:
    """Frames [first, stop) as a clip of their own, chosen by time on the file's own clock after a jump close to them."""
    info = probe(video_path)
    stream = next(iter(info.video_streams()), None)
    rate = frame_rate(stream) if stream is not None else None
    if not rate:
        raise ExternalToolError(f"no frame rate in {video_path} to find frame {first} by")
    origin = stream.start_s if stream.start_s is not None else info.start_s
    # frame k is shown k intervals after the first; half an interval either side absorbs rounded timestamps
    chosen = f"gte(t\\,{origin + float((first - Fraction(1, 2)) / rate):.6f})"
    if stop is not None:
        chosen += f"*lt(t\\,{origin + float((stop - Fraction(1, 2)) / rate):.6f})"
    clip = scratch / "window.mp4"
    jump = ["-noaccurate_seek", "-ss", f"{max(float(first / rate) - SEEK_MARGIN_S, 0.0):.6f}"] if jumps_exactly(info) else []
    # an encoder every ffmpeg build has, at a quality the detector cannot tell from the source
    run_ffmpeg([
        "-copyts", *jump, "-i", str(video_path), "-an",
        "-vf", f"select={chosen},setpts=PTS-STARTPTS", "-fps_mode", "passthrough",
        "-c:v", "mpeg4", "-q:v", "2", str(clip),
    ])
    return clip


NO_BACKEND_MESSAGE = (
    "pose.backend is not set, and there is no default: each choice has a cost to weigh. "
    'mediapipe: pip install "pymacaw[vision]"; it sends usage metrics to Google and cannot be told '
    "not to. yolo: runs offline; it needs the ultralytics `yolo` command in an environment of its "
    "own, under the AGPL-3.0 licence."
)


def resolve_backend(name: str | None) -> PoseBackend:
    """Exactly the named backend, no fallback chain. Prosody has one because it has a pure
    python last resort; here a silent downgrade would change the joint set under the user."""
    backend = named_backend(name)
    backend.check_available()
    return backend


def named_backend(name: str | None) -> PoseBackend:
    """The named backend, made without checking it can run."""
    from macaw.extractors import get_backend

    if name is None:
        raise BackendNotAvailableError(NO_BACKEND_MESSAGE)
    return get_backend(BACKEND_GROUP, name)()
