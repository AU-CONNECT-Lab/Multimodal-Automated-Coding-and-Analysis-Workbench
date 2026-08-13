"""eGeMAPSv02 and ComParE_2016 via openSMILE -- citeable, reviewers accept them, near-free."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class StandardSetExtractor(Extractor):
    name = "prosody.standard_sets"
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


def egemaps(audio_path: Path, segments: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def compare_2016(audio_path: Path, segments: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
