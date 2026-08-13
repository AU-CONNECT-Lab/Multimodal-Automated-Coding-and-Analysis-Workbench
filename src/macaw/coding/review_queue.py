"""Partial automation plus human review. Full automation is a fantasy."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_queue(annotations: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Low confidence and every abstention go to the queue."""
    raise NotImplementedError


def write_queue(queue: pd.DataFrame, path: Path) -> Path:
    raise NotImplementedError


def read_queue(path: Path) -> pd.DataFrame:
    raise NotImplementedError


def apply_decisions(annotations: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Human decisions overwrite model labels and clear needs_review."""
    raise NotImplementedError


def queue_summary(queue: pd.DataFrame) -> dict:
    raise NotImplementedError
