from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from macaw.config.schema import MacawConfig

MediaKind = Literal["audio", "video", "depth"]


@dataclass
class MediaFile:
    path: Path
    kind: MediaKind
    stream_index: int | None = None
    subject_id: str | None = None      # set when this is a per-person mic or camera
    sample_rate: float | None = None
    fps: float | None = None
    duration_s: float | None = None
    offset_s: float = 0.0              # seconds to add to this file's clock to reach session time
    drift_ppm: float | None = None
    offset_confidence: float | None = None


@dataclass
class Subject:
    subject_id: str
    role: str | None = None            # free-form label, never a binary A/B assumption
    age_months: int | None = None
    sex: str | None = None
    audio_track: str | None = None
    roi: tuple[int, int, int, int] | None = None


@dataclass
class Session:
    session_id: str
    root: Path
    subjects: list[Subject] = field(default_factory=list)
    media: list[MediaFile] = field(default_factory=list)
    t_task_start: float | None = None
    t_task_end: float | None = None
    metadata: dict = field(default_factory=dict)

    def subject(self, subject_id: str) -> Subject:
        raise NotImplementedError

    def audio_for(self, subject_id: str) -> MediaFile | None:
        raise NotImplementedError

    def reference_media(self) -> MediaFile:
        """The clock everything else is aligned to."""
        raise NotImplementedError

    def pairs(self) -> list[tuple[str, str]]:
        """Ordered subject pairs; pairwise features are stored per pair, dyad is just N=2."""
        raise NotImplementedError

    def duration_s(self) -> float:
        raise NotImplementedError


def discover_sessions(config: MacawConfig) -> list[Session]:
    """Scan raw_dir and build Session objects. Excluded ids are dropped here, once."""
    raise NotImplementedError


def load_session(path: Path) -> Session:
    raise NotImplementedError


def save_session(session: Session, path: Path) -> Path:
    raise NotImplementedError
