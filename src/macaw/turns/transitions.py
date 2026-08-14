from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.config.schema import TurnsSection
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class TransitionExtractor(Extractor):
    name = "turns.transitions"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "turns"
    config_sections = ("turns",)
    requires = ("turns.structure",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def detect_transitions(turns: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Gap is positive, overlap negative. Stored per speaker pair, never as A/B."""
    raise NotImplementedError


def classify_overlaps(transitions: pd.DataFrame, *, interruption_s: float) -> pd.DataFrame:
    """Overlap beyond interruption_s takes the floor; shorter is a backchannel."""
    raise NotImplementedError


def floor_transfer_efficiency(transitions: pd.DataFrame, cfg: TurnsSection) -> dict[str, float]:
    raise NotImplementedError


def response_latency(transitions: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def transition_matrix(turns: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """N x N, not the hard-coded AA/AB/BA/BB four-state version."""
    raise NotImplementedError
