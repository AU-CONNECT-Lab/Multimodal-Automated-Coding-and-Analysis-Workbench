"""Model assistance that never reaches the data tables. The coding passes themselves live in
observe.py and judge.py."""

from __future__ import annotations

import pandas as pd

from macaw.coding.codebook import Codebook


def clean_transcript(segments: pd.DataFrame, backend, cfg) -> pd.DataFrame:
    raise NotImplementedError


def refine_codebook(codebook: Codebook, examples: pd.DataFrame, backend, cfg) -> Codebook:
    """Pure assistance; output never enters the data tables."""
    raise NotImplementedError
