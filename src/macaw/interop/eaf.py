"""Two-way .eaf interchange, annotation layer only. Signals are not annotations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.project.recording import Recording


def export_eaf(
    annotations: pd.DataFrame,
    recording: Recording,
    out_path: Path,
    *,
    tiers: list[str] | None = None,
    controlled_vocabulary: dict[str, list[str]] | None = None,
) -> Path:
    raise NotImplementedError


def import_eaf(path: Path, *, recording_id: str) -> pd.DataFrame:
    raise NotImplementedError


def roundtrip_check(annotations: pd.DataFrame, recording: Recording, tmp_dir: Path) -> list[str]:
    """Export then re-import must reproduce every annotation exactly."""
    raise NotImplementedError


def tier_for_subject(subject_id: str, kind: str) -> str:
    raise NotImplementedError
