from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Backend


class FaceBackend(Backend):
    provides_gaze: bool = False

    def infer(self, video_path: Path, *, action_units: bool, landmarks: bool) -> pd.DataFrame:
        raise NotImplementedError


class PyFeatBackend(FaceBackend):
    """Default: MIT, pip-installable. Installation experience decides adoption here."""

    name = "pyfeat"
    provides_gaze = False

    def check_available(self) -> None:
        raise NotImplementedError


class OpenFaceBackend(FaceBackend):
    """Academic-only licence, needs compiling. The only backend with real gaze vectors."""

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
