"""Layer A: physical facts defined by rules. Near-perfect, fully automatic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.config.schema import CodingSection
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class LayerACodingExtractor(Extractor):
    name = "coding.layer_a"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    config_sections = ("coding", "turns")
    requires = ("turns.transitions",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def code_backchannels(turns: pd.DataFrame, cfg: CodingSection) -> pd.DataFrame:
    """Lexicon match, short duration, does not take the floor. Three rules is enough."""
    raise NotImplementedError


def code_pause_types(turns: pd.DataFrame, cfg) -> pd.DataFrame:
    """turn-internal vs switching pause; well established and highly reliable."""
    raise NotImplementedError


def code_interruptions(transitions: pd.DataFrame, cfg: CodingSection) -> pd.DataFrame:
    raise NotImplementedError


def code_mutual_gaze(gaze: pd.DataFrame, cfg) -> pd.DataFrame:
    raise NotImplementedError


def code_gesture_speech_alignment(strokes: pd.DataFrame, prominence: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
