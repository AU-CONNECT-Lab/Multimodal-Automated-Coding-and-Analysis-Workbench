"""Self-contained single-file HTML: mailable, archivable, usable as supplementary material."""

from __future__ import annotations

from pathlib import Path

from macaw.config.schema import MacawConfig
from macaw.extractors.base import QCWarning
from macaw.project.recording import Recording


def build_recording_report(
    recording: Recording,
    config: MacawConfig,
    warnings: list[QCWarning],
    out_path: Path,
) -> Path:
    raise NotImplementedError


def build_project_report(
    recording_ids: list[str],
    config: MacawConfig,
    warnings: list[QCWarning],
    out_path: Path,
) -> Path:
    """The only cross-recording step; on a cluster it is one job after the whole array."""
    raise NotImplementedError


def alignment_panel(recording: Recording) -> str:
    raise NotImplementedError


def prosody_panel(recording: Recording) -> str:
    raise NotImplementedError


def track_panel(recording: Recording) -> str:
    raise NotImplementedError


def warning_table(warnings: list[QCWarning]) -> str:
    raise NotImplementedError
