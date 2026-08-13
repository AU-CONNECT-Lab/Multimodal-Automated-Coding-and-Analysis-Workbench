"""Transfer, sample and multiscale entropy are excluded: all are biased on short series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class EntropyExtractor(Extractor):
    name = "turns.entropy"
    version = "0.1.0"
    scale = Scale.WINDOW
    group = "turns"
    config_sections = ("turns",)
    requires = ("turns.transitions",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        """Warns when the task duration yields too few windows to interpret."""
        raise NotImplementedError


def transition_entropy(matrix: pd.DataFrame) -> float:
    """Shannon entropy over the N x N state distribution; max is 2 bits when N=2."""
    raise NotImplementedError


def permutation_entropy(series: np.ndarray, *, order: int = 3, normalise: bool = True) -> float:
    raise NotImplementedError


def speaker_entropy(talk_time: dict[str, float]) -> float:
    raise NotImplementedError


def conditional_entropy(x: np.ndarray, y: np.ndarray) -> float:
    """H(partner response | my utterance); quantifies direction of influence."""
    raise NotImplementedError
