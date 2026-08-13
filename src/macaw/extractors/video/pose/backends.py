from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Backend


class PoseBackend(Backend):
    joint_names: tuple[str, ...] = ()

    def infer(self, video_path: Path, *, device: str, min_confidence: float) -> pd.DataFrame:
        """KEYPOINT_SCHEMA rows. z is NaN for every 2D backend."""
        raise NotImplementedError


class RTMPoseBackend(PoseBackend):
    name = "rtmpose"

    def check_available(self) -> None:
        raise NotImplementedError


class MediaPipeBackend(PoseBackend):
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
