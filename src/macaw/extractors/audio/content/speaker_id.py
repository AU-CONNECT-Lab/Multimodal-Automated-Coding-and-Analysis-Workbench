"""Who is who: three-source vote, then a human confirms. Never trust an Adult/Child label."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from macaw.config.schema import TextSection
from macaw.project.session import Session


@dataclass
class IdentityVote:
    subject_id: str
    speaker_label: str
    lexical_score: float
    f0_score: float
    anchor_score: float
    combined: float
    evidence: dict = field(default_factory=dict)


def vote_by_lexical_form(transcript: pd.DataFrame, cfg: TextSection) -> dict[str, float]:
    raise NotImplementedError


def vote_by_f0(prosody: pd.DataFrame, cfg) -> dict[str, float]:
    raise NotImplementedError


def vote_by_kinship_anchor(transcript: pd.DataFrame, cfg: TextSection) -> dict[str, float]:
    """"Mommy, look" identifies the speaker as the child and the addressee as the parent."""
    raise NotImplementedError


def combine_votes(session: Session, transcript: pd.DataFrame, prosody: pd.DataFrame, cfg: TextSection) -> list[IdentityVote]:
    raise NotImplementedError


def apply_identity(transcript: pd.DataFrame, votes: list[IdentityVote]) -> pd.DataFrame:
    raise NotImplementedError


def needs_confirmation(votes: list[IdentityVote], *, margin: float = 0.2) -> bool:
    raise NotImplementedError
