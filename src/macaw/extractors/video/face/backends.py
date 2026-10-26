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


class OpenFaceBackend(FaceBackend):
    """Academic-only licence, needs compiling."""

    name = "openface"
    provides_gaze = True

    def check_available(self) -> None:
        raise NotImplementedError

    def _run_subprocess(self, video_path: Path, out_dir: Path) -> Path:
        raise NotImplementedError


class MediaPipeFaceBackend(FaceBackend):
    name = "mediapipe"
    provides_gaze = False

    def check_available(self) -> None:
        raise NotImplementedError

    def approximate_gaze(self, landmarks: pd.DataFrame) -> pd.DataFrame:
        """Iris landmarks give a coarse gaze direction; the degraded fallback for mutual gaze."""
        raise NotImplementedError


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
