from __future__ import annotations

from pathlib import Path


def extract_audio(
    src: Path,
    dest: Path,
    *,
    stream_index: int | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    raise NotImplementedError


def cut(src: Path, dest: Path, *, start_s: float, end_s: float | None = None, recode: bool = False) -> Path:
    raise NotImplementedError


def apply_offset(src: Path, dest: Path, *, offset_s: float) -> Path:
    raise NotImplementedError


def resample_for_drift(src: Path, dest: Path, *, drift_ppm: float) -> Path:
    raise NotImplementedError


def extract_thumbnails(src: Path, dest_dir: Path, *, every_s: float = 10.0) -> list[Path]:
    raise NotImplementedError


def run_ffmpeg(args: list[str]) -> str:
    """Raises ExternalToolError with ffmpeg's stderr attached on non-zero exit."""
    raise NotImplementedError
