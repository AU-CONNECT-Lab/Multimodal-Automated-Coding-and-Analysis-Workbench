from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class FrameProsodyExtractor(Extractor):
    name = "prosody.frame"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "prosody"
    config_sections = ("prosody", "segment")

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def f0_contour(audio_path: Path, cfg, *, f0_min: float, f0_max: float) -> pd.DataFrame:
    raise NotImplementedError


def intensity_contour(audio_path: Path, cfg) -> pd.DataFrame:
    """Intensity is not comparable across speakers -- the manifest must say so."""
    raise NotImplementedError


def formant_contours(audio_path: Path, cfg, n_formants: int = 4) -> pd.DataFrame:
    raise NotImplementedError
