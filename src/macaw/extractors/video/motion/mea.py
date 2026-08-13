"""Motion Energy Analysis: frame differencing per ROI. Cheap, stable, independent of pose."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class MEAExtractor(Extractor):
    name = "motion.mea"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "motion"
    config_sections = ("motion",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def frame_difference_energy(
    video_path: Path,
    rois: dict[str, tuple[int, int, int, int]],
    *,
    threshold: int,
    smoothing_s: float,
) -> pd.DataFrame:
    raise NotImplementedError


def suggest_rois(video_path: Path, n_subjects: int) -> dict[str, tuple[int, int, int, int]]:
    """Initial guess for the GUI to draw; the user always adjusts them."""
    raise NotImplementedError
