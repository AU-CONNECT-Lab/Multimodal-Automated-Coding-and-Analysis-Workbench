from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend

BACKEND_GROUP = "face"
# what every face backend returns per face and frame, besides its action units and landmarks; angles in degrees
FACE_COLUMNS = ("frame", "face_x", "face_y", "face_width", "face_height", "face_score", "pitch", "roll", "yaw")
# when gaze is asked for, in degrees; pitch up and yaw to the person's own right are positive
GAZE_COLUMNS = ("gaze_pitch", "gaze_yaw")
# frames decoded and handed over at a time; the model batches within it
CHUNK_FRAMES = 256


class FaceBackend(Backend):
    provides_gaze: bool = False
    provides_action_units: bool = True
    # a note the run plan prints when this backend sends anything off the machine
    reporting: str | None = None

    def infer(
        self,
        video_path: Path,
        *,
        frames: tuple[int, int] | None,
        device: str,
        model: str | None,
        batch_size: int,
        landmarks: bool,
        gaze: bool = False,
    ) -> pd.DataFrame:
        """One row per face and frame (the file's own frame numbers): FACE_COLUMNS, AU columns
        (AU01, AU02, ...), and x_0, y_0, ... landmark and GAZE_COLUMNS when asked for."""
        raise NotImplementedError

    def prefetch(self, *, model: str | None, gaze: bool) -> None:
        """Fetch the weights infer would download; a backend whose models the user installs has none."""
        return None


@register_backend(BACKEND_GROUP)
class PyFeatBackend(FaceBackend):
    """MIT code; most of its model weights are for non-commercial research. The user installs it."""

    name = "pyfeat"
    provides_gaze = True
    models = ("v1", "v2")

    @property
    def version(self) -> str:
        import feat

        return str(getattr(feat, "__version__", "unknown"))

    def check_available(self) -> None:
        try:
            import feat  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "face.backend pyfeat needs py-feat, which pymacaw does not install: its code is MIT, but most "
                "of its model weights are for non-commercial research only. If that fits your use, install "
                "PyTorch for your GPU first and then: pip install py-feat"
            ) from error

    def prefetch(self, *, model: str | None, gaze: bool) -> None:
        self.detector(model, "cpu", gaze)

    def detector(self, model: str | None, device: str, gaze: bool = False):
        from feat import Detectorv1, Detectorv2

        model = model or "v1"
        if model not in self.models:
            raise BackendNotAvailableError(f"face.model for pyfeat is one of {', '.join(self.models)}, not {model}")
        # identity embeddings are never needed: who is who comes from pose; v2 has gaze built in
        if model == "v2":
            return Detectorv2(device=device, identity_model=None)
        return Detectorv1(device=device, identity_model=None, gaze_model="l2cs" if gaze else None, emotion_model=None)

    def infer(self, video_path, *, frames, device, model, batch_size, landmarks, gaze=False):
        import cv2
        import torch

        from macaw.extractors.video.pose.backends import _seek

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = self.detector(model, device, gaze)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")
        pieces = []
        try:
            first, stop = frames if frames is not None else (0, None)
            index = _seek(capture, cv2, max(int(first), 0))
            while stop is None or index < stop:
                chunk = []
                while len(chunk) < CHUNK_FRAMES and (stop is None or index + len(chunk) < stop):
                    ok, image = capture.read()
                    if not ok:
                        break
                    chunk.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not chunk:
                    break
                tensor = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2).contiguous()
                found = detector.detect(tensor, data_type="tensor", batch_size=batch_size, progress_bar=False)
                pieces.append(_rows(pd.DataFrame(found), index, landmarks, gaze))
                index += len(chunk)
        finally:
            capture.release()
        pieces = [piece for piece in pieces if not piece.empty]
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(FACE_COLUMNS))


OPENFACE_ENV_VAR = "MACAW_OPENFACE"
# the multi-face tool, so a shared camera's faces are all read; it writes the same columns as FeatureExtraction
OPENFACE_TOOL = "FaceLandmarkVidMulti"
# OpenFace scores action unit intensity 0 to 5; these share a number with the units face.au reports
OPENFACE_INTENSITY_MAX = 5.0


def openface_rows(found: pd.DataFrame, offset: int, landmarks: bool, gaze: bool, action_units: set[str]) -> pd.DataFrame:
    """OpenFace's csv in the backend contract. Its angles are radians in camera axes; pitch, roll and both gaze
    angles come out signed against py-feat's, so they are flipped to match it."""
    found = found[found["success"] == 1].reset_index(drop=True)
    xs = found[[f"x_{index}" for index in range(68)]].to_numpy(dtype=float)
    ys = found[[f"y_{index}" for index in range(68)]].to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            # OpenFace counts frames from 1
            "frame": found["frame"].astype(int).to_numpy() - 1 + offset,
            "face_x": xs.min(axis=1), "face_y": ys.min(axis=1),
            "face_width": xs.max(axis=1) - xs.min(axis=1), "face_height": ys.max(axis=1) - ys.min(axis=1),
            "face_score": found["confidence"].to_numpy(dtype=float),
            "pitch": -np.degrees(found["pose_Rx"].to_numpy(dtype=float)),
            "roll": -np.degrees(found["pose_Rz"].to_numpy(dtype=float)),
            "yaw": np.degrees(found["pose_Ry"].to_numpy(dtype=float)),
        }
    )
    if gaze:
        out["gaze_pitch"] = -np.degrees(found["gaze_angle_y"].to_numpy(dtype=float))
        out["gaze_yaw"] = -np.degrees(found["gaze_angle_x"].to_numpy(dtype=float))
    for unit in sorted(action_units):
        if f"{unit}_r" in found:
            out[unit] = found[f"{unit}_r"].to_numpy(dtype=float) / OPENFACE_INTENSITY_MAX
    if landmarks:
        points = pd.DataFrame(np.hstack([xs, ys]), columns=[*(f"x_{i}" for i in range(68)), *(f"y_{i}" for i in range(68))])
        out = pd.concat([out, points], axis=1)
    return out


@register_backend(BACKEND_GROUP)
class OpenFaceBackend(FaceBackend):
    """OpenFace 2, run as a program the user installs: its licence is for academic, non-commercial use."""

    name = "openface"
    provides_gaze = True

    @property
    def version(self) -> str:
        return Path(self._executable()).parent.name

    def _executable(self) -> str:
        import os
        import shutil

        configured = os.environ.get(OPENFACE_ENV_VAR)
        candidates = []
        if configured:
            path = Path(configured)
            candidates += [path] if path.is_file() else [path / f"{OPENFACE_TOOL}.exe", path / OPENFACE_TOOL]
        found = next((str(path) for path in candidates if path.is_file()), None) or shutil.which(OPENFACE_TOOL)
        if not found:
            raise BackendNotAvailableError(
                f"face.backend openface runs OpenFace 2's {OPENFACE_TOOL}, which pymacaw does not install: its licence "
                "is for academic, non-commercial use. Install it (with its model files) and point "
                f"${OPENFACE_ENV_VAR} at its folder, or put {OPENFACE_TOOL} on PATH"
            )
        return found

    def check_available(self) -> None:
        self._executable()

    def infer(self, video_path, *, frames, device, model, batch_size, landmarks, gaze=False):
        import subprocess
        import tempfile

        from macaw.extractors.video.face.au import ACTION_UNITS
        from macaw.extractors.video.pose.backends import _window_clip

        first, stop = frames if frames is not None else (0, None)
        with tempfile.TemporaryDirectory(prefix="macaw_openface_") as scratch:
            scratch = Path(scratch)
            source = _window_clip(Path(video_path), first, stop, scratch) if first or stop is not None else Path(video_path)
            command = [self._executable(), "-f", str(source), "-out_dir", str(scratch / "out"), "-2Dfp", "-pose", "-aus", "-q"]
            done = subprocess.run(command + (["-gaze"] if gaze else []), capture_output=True, text=True, errors="replace")
            written = scratch / "out" / f"{source.stem}.csv"
            if done.returncode != 0 or not written.is_file():
                tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-5:])
                raise ExternalToolError(f"OpenFace failed on {video_path} (exit {done.returncode}): {tail}")
            found = pd.read_csv(written, skipinitialspace=True)
        if found.empty:
            return pd.DataFrame(columns=list(FACE_COLUMNS))
        return openface_rows(found, first, landmarks, gaze, set(ACTION_UNITS))


FACE_MODEL_ENV_VAR = "MACAW_FACE_MODEL"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
# the 68 points of the iBUG layout that face.movement reads, picked from MediaPipe's 478-point mesh:
# jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, outer lips 48-59, inner lips 60-67
MEDIAPIPE_TO_68 = (
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
)
# faces looked for in one frame; a shared camera rarely shows more people than this
MEDIAPIPE_MAX_FACES = 6


def head_angles(matrix: np.ndarray) -> tuple[float, float, float]:
    """(pitch, roll, yaw) in degrees from MediaPipe's face transformation matrix, signed as py-feat signs them."""
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    pitch = -np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = np.degrees(np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return float(pitch), float(roll), float(yaw)


def mediapipe_face_rows(landmarks, matrices, *, frame_index: int, width: int, height: int, keep_landmarks: bool) -> list[dict]:
    rows = []
    for points, matrix in zip(landmarks, matrices):
        xs = np.array([point.x for point in points]) * width
        ys = np.array([point.y for point in points]) * height
        pitch, roll, yaw = head_angles(matrix)
        row = {
            "frame": frame_index,
            "face_x": float(xs.min()), "face_y": float(ys.min()),
            "face_width": float(xs.max() - xs.min()), "face_height": float(ys.max() - ys.min()),
            # the landmarker keeps a face only above its presence threshold and reports no score of its own
            "face_score": float("nan"),
            "pitch": pitch, "roll": roll, "yaw": yaw,
        }
        if keep_landmarks:
            for index, mesh in enumerate(MEDIAPIPE_TO_68):
                row[f"x_{index}"] = float(xs[mesh])
                row[f"y_{index}"] = float(ys[mesh])
        rows.append(row)
    return rows


@register_backend(BACKEND_GROUP)
class MediaPipeFaceBackend(FaceBackend):
    """Landmarks and head angles; no action units and no gaze. Apache-licensed weights, so no research-only limit."""

    name = "mediapipe"
    provides_gaze = False
    provides_action_units = False
    reporting = (
        "face.backend mediapipe: MediaPipe sends usage and performance metrics to Google while it runs "
        "(not the video itself), and has no switch to turn this off."
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
                'face.backend mediapipe needs mediapipe and opencv: pip install "pymacaw[vision]"'
            ) from error

    def _model_path(self, model: str | None) -> Path:
        import os

        candidate = model or os.environ.get(FACE_MODEL_ENV_VAR)
        if not candidate:
            raise ExternalToolError(
                f"face.backend mediapipe needs a face landmarker task file. Set face.model, or {FACE_MODEL_ENV_VAR}, "
                f"to a copy of {_FACE_MODEL_URL}"
            )
        path = Path(candidate)
        if not path.is_file():
            raise ExternalToolError(f"face.model does not point at a file: {path}")
        return path

    def infer(self, video_path, *, frames, device, model, batch_size, landmarks, gaze=False):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        from macaw.extractors.video.pose.backends import _seek

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path(model))),
            running_mode=RunningMode.VIDEO,
            num_faces=MEDIAPIPE_MAX_FACES,
            output_facial_transformation_matrixes=True,
        )
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")
        rows: list[dict] = []
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0:
                raise ExternalToolError(f"video reports no frame rate: {video_path}")
            first, stop = frames if frames is not None else (0, None)
            index = _seek(capture, cv2, max(int(first), 0))
            with FaceLandmarker.create_from_options(options) as landmarker:
                while stop is None or index < stop:
                    ok, image = capture.read()
                    if not ok:
                        break
                    found = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        int(round(index * 1000.0 / fps)),
                    )
                    rows += mediapipe_face_rows(
                        found.face_landmarks, found.facial_transformation_matrixes,
                        frame_index=index, width=width, height=height, keep_landmarks=landmarks,
                    )
                    index += 1
        finally:
            capture.release()
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(FACE_COLUMNS))


def _rows(found: pd.DataFrame, offset: int, landmarks: bool, gaze: bool = False) -> pd.DataFrame:
    found = found[found["FaceScore"].notna()] if "FaceScore" in found else found.iloc[:0]
    out = pd.DataFrame(
        {
            "frame": found["frame"].astype(int).to_numpy() + offset,
            "face_x": found["FaceRectX"].to_numpy(dtype=float),
            "face_y": found["FaceRectY"].to_numpy(dtype=float),
            "face_width": found["FaceRectWidth"].to_numpy(dtype=float),
            "face_height": found["FaceRectHeight"].to_numpy(dtype=float),
            "face_score": found["FaceScore"].to_numpy(dtype=float),
            # py-feat gives head angles in radians
            "pitch": np.degrees(found["Pitch"].to_numpy(dtype=float)),
            "roll": np.degrees(found["Roll"].to_numpy(dtype=float)),
            "yaw": np.degrees(found["Yaw"].to_numpy(dtype=float)),
        }
    )
    if gaze:
        for column in GAZE_COLUMNS:
            out[column] = np.degrees(found[column].to_numpy(dtype=float))
    extra = [column for column in found.columns if column.startswith("AU")]
    if landmarks:
        extra += [column for column in found.columns if column.startswith(("x_", "y_")) and column[2:].isdigit()]
    return pd.concat([out, found[extra].reset_index(drop=True).astype(float)], axis=1)


def resolve_backend(name: str) -> FaceBackend:
    from macaw.extractors import get_backend

    backend = get_backend(BACKEND_GROUP, name)()
    backend.check_available()
    return backend
