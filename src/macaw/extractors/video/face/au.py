from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class FaceExtractor(Extractor):
    name = "face.au"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "face"
    config_sections = ("face",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Emits a warning on child data: AU models are trained on adults."""
        raise NotImplementedError


def blink_rate(landmarks: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def mouth_opening(landmarks: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def cross_backend_agreement(au_a: pd.DataFrame, au_b: pd.DataFrame) -> pd.DataFrame:
    """QC only. AU values are never merged across backends; different intensity calibration."""
    raise NotImplementedError
