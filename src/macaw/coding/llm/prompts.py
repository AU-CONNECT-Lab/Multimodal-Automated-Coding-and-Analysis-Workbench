"""One prompt set for every backend; switching provider must not change coding logic."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.coding.codebook import Codebook

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_template(name: str) -> str:
    raise NotImplementedError


def build_annotation_prompt(codebook: Codebook, utterances: pd.DataFrame, context: pd.DataFrame | None = None) -> tuple[str, str]:
    """Returns (system, user)."""
    raise NotImplementedError


def build_observation_prompt(utterances: pd.DataFrame, context: pd.DataFrame | None = None) -> tuple[str, str]:
    """No codebook argument on purpose: the observation pass must not know the categories."""
    raise NotImplementedError


def build_judgement_prompt(codebook: Codebook, observations: pd.DataFrame) -> tuple[str, str]:
    raise NotImplementedError


def build_transcript_cleanup_prompt(segments: pd.DataFrame) -> tuple[str, str]:
    raise NotImplementedError


def build_codebook_refinement_prompt(codebook: Codebook, examples: pd.DataFrame) -> tuple[str, str]:
    raise NotImplementedError
