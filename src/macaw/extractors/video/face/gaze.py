from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class GazeExtractor(Extractor):
    name = "face.gaze"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "face"
    config_sections = ("face",)
    requires = ("face.au",)

    def check_available(self) -> None:
        """Full gaze needs the openface backend; mediapipe gives the degraded version."""
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def mutual_gaze(gaze_a: pd.DataFrame, gaze_b: pd.DataFrame, *, tolerance_deg: float) -> pd.Series:
    raise NotImplementedError


def gaze_following(gaze_a: pd.DataFrame, gaze_b: pd.DataFrame, *, lag_s: float) -> pd.Series:
    raise NotImplementedError


def gaze_shift_events(gaze: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def looking_at_partner_fraction(gaze: pd.DataFrame, *, tolerance_deg: float) -> float:
    raise NotImplementedError
