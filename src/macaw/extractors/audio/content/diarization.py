from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Backend
from macaw.project.session import Session


class DiarizationBackend(Backend):
    def diarize(self, audio_path: Path, *, n_speakers: int | None = None) -> pd.DataFrame:
        raise NotImplementedError


class PyannoteBackend(DiarizationBackend):
    name = "pyannote"

    def check_available(self) -> None:
        """Model weights need HuggingFace terms accepted; say so in the error."""
        raise NotImplementedError


class PerTrackBackend(DiarizationBackend):
    """One mic per person: assign by track. More accurate than any diarizer."""

    name = "per_track"

    def check_available(self) -> None:
        raise NotImplementedError


def assign_speakers(segments: pd.DataFrame, diarization: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def resolve_diarization(session: Session, mode: str) -> DiarizationBackend:
    raise NotImplementedError
