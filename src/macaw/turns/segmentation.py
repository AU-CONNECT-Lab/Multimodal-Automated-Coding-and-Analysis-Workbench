from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.config.schema import TurnsSection
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class TurnStructureExtractor(Extractor):
    name = "turns.structure"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "turns"
    config_sections = ("turns", "text")
    requires = ("text.asr",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def utterances_to_turns(transcript: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Merge consecutive same-speaker utterances separated by less than max_gap_s."""
    raise NotImplementedError


def turn_durations(turns: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def dominance(turns: pd.DataFrame) -> dict[str, float]:
    raise NotImplementedError


def silences(turns: pd.DataFrame, *, min_s: float) -> pd.DataFrame:
    """Joint silence: gaps after merging all speakers' talk into one timeline."""
    raise NotImplementedError
