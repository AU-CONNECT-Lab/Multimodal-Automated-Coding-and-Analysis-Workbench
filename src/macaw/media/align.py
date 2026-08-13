"""Multi-source alignment by generalised cross-correlation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from macaw.config.schema import AlignSection
from macaw.project.session import Session


@dataclass
class AlignResult:
    reference: Path
    target: Path
    offset_s: float
    confidence: float                # peak sharpness, 0-1
    drift_ppm: float | None = None
    drift_r2: float | None = None
    segment_offsets: list[tuple[float, float]] | None = None


def gcc_phat(a: np.ndarray, b: np.ndarray, sample_rate: float, max_lag_s: float) -> tuple[float, float]:
    """Returns (lag_s, confidence). PHAT weighting is what makes this robust to room response."""
    raise NotImplementedError


def align_pair(
    reference: Path,
    target: Path,
    cfg: AlignSection,
) -> AlignResult:
    raise NotImplementedError


def align_session(session: Session, cfg: AlignSection) -> list[AlignResult]:
    """Aligns every media file to session.reference_media()."""
    raise NotImplementedError


def write_offset_table(results: list[AlignResult], path: Path) -> Path:
    raise NotImplementedError


def read_offset_table(path: Path) -> list[AlignResult]:
    raise NotImplementedError


def apply_alignment(session: Session, results: list[AlignResult], *, recut: bool = False) -> Session:
    raise NotImplementedError
