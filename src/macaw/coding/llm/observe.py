"""The observation half of layer B: what was said and done, described in the model's own words,
with no codebook in sight.

Split from the judgement half so that changing a code definition does not force the expensive
pass to run again. Its config_sections deliberately stop at coding.llm, which is why editing the
codebook leaves its cache entries valid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale

OBSERVATION_TIER = "observation"


class ObservationExtractor(Extractor):
    name = "coding.observe"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    config_sections = ("coding.llm", "turns")
    requires = ("turns.structure", "text.asr")
    resource_profile = "gpu_standard"

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def observe_turns(turns: pd.DataFrame, backend, cfg) -> pd.DataFrame:
    """One free-text observation per unit. No labels, so nothing here depends on a codebook."""
    raise NotImplementedError


def load_observations(config, recording_id: str) -> pd.DataFrame:
    raise NotImplementedError
