"""Who said each utterance. With one microphone per person the answer is the track it came
off, but only after the partner's bleed has been taken back out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.base import Backend
from macaw.project.recording import Recording

BACKEND_GROUP = "diarization"

_SILENCE_FLOOR_DB = -120.0
# a mic's own loud level: its owner's speech, whatever the gain it was recorded at
_GAIN_FRAME_S = 0.02
_GAIN_PERCENTILE = 99.0
# a mic further below the loudest than this is missing its owner, not turned down
_MAX_GAIN_GAP_DB = 30.0


class DiarizationBackend(Backend):
    version = "0.1.0"

    def diarize(self, audio_path: Path, *, n_speakers: int | None = None) -> pd.DataFrame:
        """Speaker turns found in one mixed recording: t_start, t_end, speaker."""
        raise NotImplementedError

    def attribute(
        self, segments: pd.DataFrame, tracks: dict[str, Path], *, margin_db: float
    ) -> pd.DataFrame:
        """Add is_crosstalk to per-track ASR segments that already carry a subject_id."""
        raise NotImplementedError


@register_backend(BACKEND_GROUP)
class PyannoteBackend(DiarizationBackend):
    name = "pyannote"

    def check_available(self) -> None:
        """Model weights need HuggingFace terms accepted; say so in the error."""
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "pyannote.audio is not installed. Install it with "
                '`pip install "pymacaw[asr]"`, accept the terms for '
                "pyannote/speaker-diarization-3.1 on HuggingFace, and log in with "
                "`huggingface-cli login`."
            ) from None
        raise BackendNotAvailableError(
            "The pyannote diarizer is not wired up yet. Use `text.diarization: per_track` "
            "when every person has their own microphone."
        )


@register_backend(BACKEND_GROUP)
class PerTrackBackend(DiarizationBackend):
    """One mic per person: assign by track. More accurate than any diarizer."""

    name = "per_track"

    def check_available(self) -> None:
        return None

    def attribute(
        self, segments: pd.DataFrame, tracks: dict[str, Path], *, margin_db: float
    ) -> pd.DataFrame:
        return suppress_crosstalk(segments, tracks, margin_db=margin_db)


@register_backend(BACKEND_GROUP)
class NoDiarizationBackend(DiarizationBackend):
    """Take the track attribution at face value, bleed and all."""

    name = "none"

    def check_available(self) -> None:
        return None

    def attribute(
        self, segments: pd.DataFrame, tracks: dict[str, Path], *, margin_db: float
    ) -> pd.DataFrame:
        out = segments.copy()
        out["is_crosstalk"] = False
        return out


def _level_db(audio: np.ndarray, sr: float, t_start: float, t_end: float) -> float:
    first = max(int(round(t_start * sr)), 0)
    last = min(int(round(t_end * sr)), audio.size)
    if last <= first:
        return _SILENCE_FLOOR_DB
    rms = float(np.sqrt(np.mean(np.square(audio[first:last]))))
    return max(20.0 * np.log10(rms), _SILENCE_FLOOR_DB) if rms > 0 else _SILENCE_FLOOR_DB


def _gain_db(audio: np.ndarray, sr: float) -> float:
    hop = max(int(_GAIN_FRAME_S * sr), 1)
    frames = audio[: audio.size // hop * hop].reshape(-1, hop)
    if frames.size == 0:
        return 0.0
    power = np.mean(np.square(frames, dtype=float), axis=1)
    return float(np.percentile(10.0 * np.log10(np.maximum(power, 10 ** (_SILENCE_FLOOR_DB / 10))), _GAIN_PERCENTILE))


def suppress_crosstalk(
    segments: pd.DataFrame, tracks: dict[str, Path], *, margin_db: float
) -> pd.DataFrame:
    """Flag utterances that are the neighbour's voice leaking into this person's microphone.

    Two people in one room means each voice reaches both microphones, and an ASR run per
    track transcribes the leaked copy as well, so the same sentence is attributed to both.
    A leaked copy is always the quieter one, so an utterance is bleed when some other track
    is louder than its own over the same interval by more than the margin. Each track is
    measured against its own loud level first, so a mic recorded at a lower gain does not
    lose its owner's speech to a louder neighbour. Genuine simultaneous speech survives:
    each person is still loudest on their own microphone.

    Times must be on the prepared-track clock, where sample 0 is the same instant on every
    track."""
    out = segments.copy()
    if out.empty:
        out["is_crosstalk"] = pd.Series([], dtype=bool)
        return out
    if len(tracks) < 2:
        out["is_crosstalk"] = False
        return out

    audio = {subject_id: load_audio(Path(path)) for subject_id, path in tracks.items()}
    gain = {subject_id: _gain_db(samples, sr) for subject_id, (samples, sr) in audio.items()}
    loudest = max(gain.values())
    gain = {subject_id: max(value, loudest - _MAX_GAIN_GAP_DB) for subject_id, value in gain.items()}
    flags = []
    for row in out.itertuples(index=False):
        own = audio.get(row.subject_id)
        if own is None:
            flags.append(False)
            continue
        own_db = _level_db(own[0], own[1], row.t_start, row.t_end) - gain[row.subject_id]
        others = [
            _level_db(samples, sr, row.t_start, row.t_end) - gain[subject_id]
            for subject_id, (samples, sr) in audio.items()
            if subject_id != row.subject_id
        ]
        flags.append(bool(others) and max(others) - own_db > margin_db)
    out["is_crosstalk"] = flags
    return out


def assign_speakers(segments: pd.DataFrame, diarization: pd.DataFrame) -> pd.DataFrame:
    """Give each segment the diarized speaker it overlaps most, for the shared-mic path."""
    out = segments.copy()
    if out.empty:
        return out
    if diarization.empty:
        out["subject_id"] = None
        return out

    starts = diarization["t_start"].to_numpy(dtype=float)
    ends = diarization["t_end"].to_numpy(dtype=float)
    speakers = diarization["speaker"].to_numpy()

    picked = []
    for row in out.itertuples(index=False):
        overlap = np.minimum(ends, row.t_end) - np.maximum(starts, row.t_start)
        best = int(np.argmax(overlap))
        picked.append(speakers[best] if overlap[best] > 0 else None)
    out["subject_id"] = picked
    return out


def resolve_diarization(recording: Recording, mode: str) -> DiarizationBackend:
    from macaw.extractors import get_backend

    backend = get_backend(BACKEND_GROUP, mode)()
    backend.check_available()
    return backend
