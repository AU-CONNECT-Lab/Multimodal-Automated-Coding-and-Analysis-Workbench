"""Who is who: three-source vote, then a human confirms. Never trust an Adult/Child label."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macaw.config.schema import TextSection
from macaw.project.recording import Recording


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


def combine_votes(recording: Recording, transcript: pd.DataFrame, prosody: pd.DataFrame, cfg: TextSection) -> list[IdentityVote]:
    raise NotImplementedError


def apply_identity(transcript: pd.DataFrame, votes: list[IdentityVote]) -> pd.DataFrame:
    raise NotImplementedError


def needs_confirmation(votes: list[IdentityVote], *, margin: float = 0.2) -> bool:
    raise NotImplementedError


def suggest_voices(config, recording: Recording) -> dict | None:
    """Which voice on the shared microphone each person's mouth moves with; None without faces or voices."""
    from macaw.exceptions import InputNotFoundError
    from macaw.extractors.audio.content.diarization import read_diarization
    from macaw.project.layout import layout_for
    from macaw.qc.checks import MOUTH_SOURCE, RecordingResults

    rid = recording.recording_id
    mouth = RecordingResults(layout_for(config, create=False), rid).long_table(MOUTH_SOURCE, features=["mouth_movement"])
    try:
        turns = read_diarization(config, rid)
    except InputNotFoundError:
        return None
    if mouth.empty or turns.empty:
        return None
    return match_voices(turns[turns["exclusive"].astype(bool)], mouth)


def match_voices(turns: pd.DataFrame, mouth: pd.DataFrame) -> dict | None:
    """One voice per person, the assignment with the highest summed rank correlation of mouth movement with that voice talking."""
    from scipy.optimize import linear_sum_assignment

    from macaw.extractors.audio.content.diarization import NO_ONE
    from macaw.qc.checks import _MOUTH_BIN_S, _rank_contrast, _within

    labels = sorted(str(label) for label in turns["label"].unique())
    scores: dict[str, dict[str, tuple[float, float]]] = {}
    for subject_id, rows in mouth.groupby("subject_id", dropna=True, observed=True):
        bins = (rows["t_start"].to_numpy(dtype=float) // _MOUTH_BIN_S).astype(int)
        moving = pd.Series(rows["value"].to_numpy(dtype=float)).groupby(bins).mean().dropna()
        if moving.size < 10:
            continue
        centres = (moving.index.to_numpy() + 0.5) * _MOUTH_BIN_S
        level = moving.rank().to_numpy()
        scores[str(subject_id)] = {}
        for label in labels:
            own = turns[turns["label"] == label]
            mask = _within(centres, own["t_start"].to_numpy(dtype=float), own["t_end"].to_numpy(dtype=float))
            scores[str(subject_id)][label] = _rank_contrast(level, mask) if mask.any() else (float("nan"), float("nan"))
    if not scores:
        return None
    subjects = sorted(scores)
    rho = np.array([[scores[subject][label][0] for label in labels] for subject in subjects])
    rows, cols = linear_sum_assignment(np.nan_to_num(rho, nan=-1.0), maximize=True)
    people = {label: NO_ONE for label in labels}
    evidence = {}
    for row, col in zip(rows, cols):
        subject, label = subjects[row], labels[col]
        people[label] = subject
        others = {other: scores[subject][other][0] for other in labels if other != label and np.isfinite(scores[subject][other][0])}
        rival = max(others, key=others.get) if others else None
        evidence[subject] = {
            "label": label, "rho": scores[subject][label][0], "z": scores[subject][label][1],
            **({"rival": rival, "rival_rho": others[rival]} if rival else {}),
        }
    return {"people": people, "evidence": evidence}


def describe_suggestion(suggestion: dict) -> str:
    shown = [
        f"{label} is {who}" + (
            f" (rho {suggestion['evidence'][who]['rho']:.2f}, z {suggestion['evidence'][who]['z']:.1f})" if who in suggestion["evidence"] else ""
        )
        for label, who in sorted(suggestion["people"].items())
    ]
    return "; ".join(shown)
