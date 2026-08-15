"""The judgement half of layer B: the codebook applied to what the observation pass wrote down.

This is the half that has no single truth, so its output keeps confidence and abstentions and
goes through the review queue. It reads the whole coding section, codebook included, which is
what makes a codebook edit recompute this pass and only this pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class JudgementExtractor(Extractor):
    name = "coding.judge"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    config_sections = ("coding", "turns")
    requires = ("coding.observe",)
    resource_profile = "gpu_standard"

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def judge_observations(
    observations: pd.DataFrame, codebook: Codebook, backend, cfg
) -> pd.DataFrame:
    """Every unit gets a label, an abstention, or a low confidence; none of them get dropped."""
    raise NotImplementedError
