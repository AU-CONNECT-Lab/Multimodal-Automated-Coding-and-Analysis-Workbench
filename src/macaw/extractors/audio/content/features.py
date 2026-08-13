from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class TextFeatureExtractor(Extractor):
    name = "text.features"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "text"
    config_sections = ("text",)
    requires = ("text.asr",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def mlu(utterances: pd.DataFrame) -> float:
    raise NotImplementedError


def type_token_ratio(tokens: list[str]) -> float:
    raise NotImplementedError


def unigram_entropy(tokens: list[str]) -> float:
    """Lexical diversity without TTR's length bias."""
    raise NotImplementedError


def first_use_share(transcript: pd.DataFrame) -> dict[str, float]:
    """Who introduces new words, i.e. who drives the topic."""
    raise NotImplementedError


def sentence_embeddings(texts: list[str], model_name: str) -> pd.DataFrame:
    raise NotImplementedError
