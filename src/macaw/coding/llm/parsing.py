from __future__ import annotations

import pandas as pd

from macaw.coding.codebook import Codebook


def parse_annotation_response(text: str, codebook: Codebook, expected_ids: list[str]) -> pd.DataFrame:
    """Missing or unparseable items become abstentions, never silent drops."""
    raise NotImplementedError


def parse_cleanup_response(text: str, expected_ids: list[str]) -> pd.DataFrame:
    raise NotImplementedError


def parse_refinement_response(text: str) -> Codebook:
    raise NotImplementedError


def repair_json(text: str) -> str:
    """Small local models often wrap JSON in prose or truncate it."""
    raise NotImplementedError
