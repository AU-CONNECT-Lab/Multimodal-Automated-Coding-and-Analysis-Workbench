"""Voice report per turn: jitter, shimmer, HNR, CPPS, spectral measures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class VoiceReportExtractor(Extractor):
    name = "prosody.voice_report"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "prosody"
    config_sections = ("prosody", "segment")
    requires = ("turns.structure",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def voice_report(audio_path: Path, t_start: float, t_end: float, *, f0_min: float, f0_max: float) -> dict:
    raise NotImplementedError


def f0_slope(f0: pd.Series, t: pd.Series) -> float:
    """Rising slope reads as a question, falling as a statement. Cheap and useful."""
    raise NotImplementedError


def voiced_fraction(f0: pd.Series) -> float:
    raise NotImplementedError


def words_per_second(n_words: int, duration_s: float) -> float:
    raise NotImplementedError
