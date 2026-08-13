from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str
    sample_rate: float | None = None
    channels: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None


@dataclass
class MediaInfo:
    path: Path
    duration_s: float
    streams: list[StreamInfo]
    variable_frame_rate: bool = False


def probe(path: Path) -> MediaInfo:
    raise NotImplementedError


def frame_timestamps(path: Path, stream_index: int = 0) -> list[float]:
    """Exact presentation timestamps -- currentTime += 1/fps drifts on VFR video."""
    raise NotImplementedError


def require_ffmpeg() -> str:
    raise NotImplementedError
