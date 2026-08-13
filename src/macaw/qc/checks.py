"""The five quality checks. This is the part no existing tool does."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from macaw.config.schema import MacawConfig
from macaw.extractors.base import QCWarning
from macaw.project.session import Session

_CHECKS: dict[str, Callable] = {}


def register_check(name: str) -> Callable:
    raise NotImplementedError


def check_alignment_confidence(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def check_clock_drift(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def check_transcript_match(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    """Energy-ratio test: does this transcript belong to this recording at all."""
    raise NotImplementedError


def check_track_quality(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def check_octave_error(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    """Doubled or halved f0 relative to the speaker's own distribution."""
    raise NotImplementedError


def check_voiced_fraction(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def check_au_child_reliability(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    """AU models degrade badly on children -- always warn, never silently pass."""
    raise NotImplementedError


def check_window_count(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def run_checks(session: Session, config: MacawConfig, results) -> list[QCWarning]:
    raise NotImplementedError


def warnings_to_frame(warnings: list[QCWarning]) -> pd.DataFrame:
    raise NotImplementedError
