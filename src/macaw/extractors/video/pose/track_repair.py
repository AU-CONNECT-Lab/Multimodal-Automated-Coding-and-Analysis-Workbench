"""body_id swapping mid-recording. No existing tool tells you this happened."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SwapEvent:
    t: float
    frame: int
    track_a: int
    track_b: int
    evidence: str
    score: float


def detect_swaps(keypoints: pd.DataFrame, *, max_gap_s: float) -> list[SwapEvent]:
    """Large simultaneous jumps in opposite directions are the signature of a swap."""
    raise NotImplementedError


def repair_tracks(keypoints: pd.DataFrame, swaps: list[SwapEvent]) -> pd.DataFrame:
    raise NotImplementedError


def assign_subjects(
    keypoints: pd.DataFrame,
    subject_ids: list[str],
    *,
    identity_anchors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Map repaired tracks onto stable subject_ids, anchored by initial screen position."""
    raise NotImplementedError


def track_quality(keypoints: pd.DataFrame) -> pd.DataFrame:
    """Continuity, mean confidence, missing fraction. Bad tracks warn instead of emitting NaN."""
    raise NotImplementedError
