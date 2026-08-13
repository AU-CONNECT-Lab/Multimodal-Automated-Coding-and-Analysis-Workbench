"""Layer B: speech acts. The evidence is in the utterance itself, which is why this works."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class LayerBCodingExtractor(Extractor):
    name = "coding.layer_b"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    config_sections = ("coding", "turns")
    requires = ("turns.structure",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def annotate_turns(turns: pd.DataFrame, codebook: Codebook, backend, cfg) -> pd.DataFrame:
    raise NotImplementedError


def clean_transcript(segments: pd.DataFrame, backend, cfg) -> pd.DataFrame:
    raise NotImplementedError


def refine_codebook(codebook: Codebook, examples: pd.DataFrame, backend, cfg) -> Codebook:
    """Pure assistance; output never enters the data tables."""
    raise NotImplementedError
