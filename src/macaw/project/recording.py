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
    offset_s: float = 0.0              # seconds to add to this file's clock to reach recording time
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
class Recording:
    """One stretch of media sharing a single clock. The parallel unit of the whole package:
    one array index, one output directory, one alignment."""

    recording_id: str
    root: Path
    # BIDS entities parsed out of the path; all optional, a flat dataset leaves them None
    group_id: str | None = None        # the people recorded together, BIDS sub-
    session_id: str | None = None      # visit or wave, BIDS ses-
    task: str | None = None
    run: str | None = None
    subjects: list[Subject] = field(default_factory=list)
    media: list[MediaFile] = field(default_factory=list)
    t_task_start: float | None = None
    t_task_end: float | None = None
    metadata: dict = field(default_factory=dict)

    def entities(self) -> dict[str, str | None]:
        """The identity columns every long-table row carries."""
        raise NotImplementedError

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


def detect_recordings(raw_dir: Path, pattern: str, prefix: str = "") -> list[str]:
    """Walk raw_dir with the entity pattern and return recording ids, sorted. The prefix is
    stripped from the returned ids."""
    raise NotImplementedError


def parse_entities(relative_path: Path, pattern: str) -> dict[str, str | None]:
    """Pull group_id / session_id / task / run out of a path using the project's pattern.
    Entities the pattern does not mention come back as None."""
    raise NotImplementedError


def make_recording_id(entities: dict[str, str | None]) -> str:
    """Join the present entities into one id that is unique and stays readable in a filename."""
    raise NotImplementedError


def parse_recordings_input(value: str) -> list[str]:
    """A comma-separated list, or a path to a file holding one id per line."""
    raise NotImplementedError


def build_recording(recording_id: str, config: MacawConfig) -> Recording:
    """Resolve one recording on demand from directory convention plus its optional override file."""
    raise NotImplementedError


def recording_override_path(recording_id: str, config: MacawConfig) -> Path | None:
    raise NotImplementedError
