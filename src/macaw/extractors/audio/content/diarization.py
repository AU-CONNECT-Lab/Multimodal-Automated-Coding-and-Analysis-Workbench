"""Who said each utterance. With one microphone per person the answer is the track it came
off, but only after the partner's bleed has been taken back out."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, InputNotFoundError
from macaw.extractors import register_backend, register_extractor
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.base import Backend, Extractor, ExtractorResult, Gate, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.longtable import build_long_rows
from macaw.store.schema import Scale

BACKEND_GROUP = "diarization"
# the one mode that reads voices off a shared microphone; the others attribute by track
SHARED_MODE = "pyannote"

DIARIZATION_NAME = "diarization.parquet"
VOICES_NAME = "voices.json"
NO_ONE = "none"

DIARIZATION_SCHEMA: dict[str, str] = {
    "track": "string",
    "label": "string",
    "t_start": "float64",
    "t_end": "float64",
    "exclusive": "bool",
}

# a voice with less speech than this is more likely a cough, a door or a cut-off word than a person
_MIN_VOICE_S = 5.0
_CLIPS_PER_VOICE = 3
_CLIP_S = 5.0

_SILENCE_FLOOR_DB = -120.0
# a mic's own loud level: its owner's speech, whatever the gain it was recorded at
_GAIN_FRAME_S = 0.02
_GAIN_PERCENTILE = 99.0
# a mic further below the loudest than this is missing its owner, not turned down
_MAX_GAIN_GAP_DB = 30.0


class DiarizationBackend(Backend):
    version = "0.1.0"

    def diarize(self, audio_path: Path, *, n_speakers: int | None = None) -> pd.DataFrame:
        """Speaker turns in one mixed recording: t_start, t_end, label, and exclusive, true on the
        copy where overlapping speech is given to one speaker only."""
        raise NotImplementedError

    def attribute(
        self, segments: pd.DataFrame, tracks: dict[str, Path], *, margin_db: float
    ) -> pd.DataFrame:
        """Add is_crosstalk to per-track ASR segments that already carry a subject_id."""
        raise NotImplementedError


@register_backend(BACKEND_GROUP)
class PyannoteBackend(DiarizationBackend):
    name = SHARED_MODE

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1", *, device: str = "auto"):
        self.model = model
        self.device = device
        self._pipeline = None

    @property
    def version(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("pyannote.audio")
        except PackageNotFoundError:
            return "unknown"

    def check_available(self) -> None:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "pyannote.audio is not installed. Install PyTorch for your CUDA version first, then "
                f'`pip install "pymacaw[asr]"`; accept the terms for {self.model} on HuggingFace and log in '
                "with `hf auth login`."
            ) from None

    def _load(self):
        if self._pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            from macaw.extractors.audio.content.asr import resolve_device

            pipeline = Pipeline.from_pretrained(self.model)
            if pipeline is None:
                raise BackendNotAvailableError(
                    f"{self.model} could not be loaded: accept its terms on HuggingFace and log in with `hf auth login`"
                )
            self._pipeline = pipeline.to(torch.device(resolve_device(self.device)))
        return self._pipeline

    def diarize(self, audio_path: Path, *, n_speakers: int | None = None) -> pd.DataFrame:
        import torch

        samples, sr = load_audio(Path(audio_path))
        # handing over the samples skips pyannote's own decoder, which needs ffmpeg libraries on the path
        waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32)).unsqueeze(0)
        output = self._load()({"waveform": waveform, "sample_rate": int(sr)}, num_speakers=n_speakers)
        rows = []
        for exclusive, annotation in (
            (False, output.speaker_diarization), (True, output.exclusive_speaker_diarization)
        ):
            for turn, label in annotation:
                rows.append({"t_start": float(turn.start), "t_end": float(turn.end), "label": str(label), "exclusive": exclusive})
        return pd.DataFrame(rows, columns=["t_start", "t_end", "label", "exclusive"])


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


def resolve_diarizer(config) -> DiarizationBackend:
    from macaw.extractors import get_backend

    cfg = config.text
    cls = get_backend(BACKEND_GROUP, cfg.diarization)
    backend = cls(cfg.diarization_model, device=cfg.device) if cfg.diarization == SHARED_MODE else cls()
    backend.check_available()
    return backend


def shared_track(prep, recording_id: str) -> str:
    """The one shared audio file the voices are read from."""
    if len(prep.shared_tracks) != 1:
        found = ", ".join(sorted(prep.shared_tracks)) or "none"
        raise InputNotFoundError(
            f"{recording_id}: text.diarization {SHARED_MODE} reads exactly one shared audio file, found: {found}"
        )
    return next(iter(prep.shared_tracks))


@register_extractor
class DiarizationExtractor(Extractor):
    """Who speaks when on a shared microphone, as anonymous voices; text.review says who they are."""

    name = "text.diarize"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = ("text.diarization", "text.diarization_model", "text.device", "segment")
    requires = (MediaPrepExtractor.name,)
    resource_profile = "gpu_standard"

    def check_available(self) -> None:
        if self.config.text.diarization == SHARED_MODE:
            resolve_diarizer(self.config)

    def backend_info(self) -> tuple[str | None, str | None]:
        if self.config.text.diarization != SHARED_MODE:
            return (None, None)
        return (SHARED_MODE, resolve_diarizer(self.config).version)

    def inputs_for(self, recording: Recording) -> list[Path]:
        prep = load_prep(self.config, recording.recording_id)
        return [prep.shared_track_path(name) for name in sorted(prep.shared_tracks)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        prep = load_prep(self.config, recording.recording_id)
        values = {"n_shared_tracks_diarized": 0.0}
        if self.config.text.diarization != SHARED_MODE:
            return ExtractorResult(rows=self._rows(recording, prep, values), specs=self._specs(values))

        track = shared_track(prep, recording.recording_id)
        path = prep.shared_track_path(track)
        turns = resolve_diarizer(self.config).diarize(path, n_speakers=len(recording.subjects))
        turns["t_start"] = turns["t_start"] + prep.t_origin_s
        turns["t_end"] = turns["t_end"] + prep.t_origin_s
        turns["track"] = track
        voices = describe_voices(turns, path, prep.t_origin_s, workspace)
        values = {
            "n_shared_tracks_diarized": 1.0,
            "n_voices_found": float(len(voices)),
            "overlapped_speech_fraction": overlapped_fraction(turns),
        }
        artifacts = {
            DIARIZATION_NAME: _io.write_file(turns, workspace / DIARIZATION_NAME, DIARIZATION_SCHEMA, "diarization"),
            VOICES_NAME: _write_json(workspace / VOICES_NAME, voices),
        }
        artifacts.update({clip["file"]: workspace / clip["file"] for voice in voices.values() for clip in voice["clips"]})
        return ExtractorResult(
            rows=self._rows(recording, prep, values),
            specs=self._specs(values),
            artifacts=artifacts,
            warnings=self._warnings(recording, voices, track),
        )

    def _rows(self, recording: Recording, prep, values: dict[str, float]) -> pd.DataFrame:
        frame = pd.DataFrame(
            {"t_start": [prep.t_origin_s], "t_end": [prep.t_origin_s + prep.duration_s], **{name: [value] for name, value in values.items()}}
        )
        return build_long_rows(
            values=frame, scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name,
            entities=recording.entities(), subject_id=None,
        )

    def _specs(self, values: dict[str, float]) -> list[FeatureSpec]:
        params = {"model": self.config.text.diarization_model} if self.config.text.diarization == SHARED_MODE else {}
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                extractor=self.name, extractor_version=self.version, params=params, validation="synthetic",
            )
            for name, (description, unit) in _DIARIZATION_SPECS.items()
            if name in values
        ]

    def _warnings(self, recording: Recording, voices: dict, track: str) -> list[QCWarning]:
        rid = recording.recording_id
        warnings = []
        faint = [label for label, voice in voices.items() if voice["speech_s"] < _MIN_VOICE_S]
        if faint:
            warnings.append(QCWarning(
                check="diarization_faint_voice", recording_id=rid, subject_id=None, severity="info",
                message=f"on {track}, {', '.join(faint)} speak under {_MIN_VOICE_S:g} s in all; likely not a person",
            ))
        if len(voices) < len(recording.subjects):
            warnings.append(QCWarning(
                check="diarization_voice_count", recording_id=rid, subject_id=None, severity="warn",
                message=f"{len(voices)} voices found on {track} for {len(recording.subjects)} people",
            ))
        return warnings


_DIARIZATION_SPECS = {
    "n_shared_tracks_diarized": ("Shared audio files read for voices; 0 when every person has their own track.", "count"),
    "n_voices_found": ("Distinct voices the diarizer found on the shared track.", "count"),
    "overlapped_speech_fraction": ("Share of the time anyone speaks during which two or more voices do.", None),
}


def overlapped_fraction(turns: pd.DataFrame) -> float:
    spans = turns[~turns["exclusive"].astype(bool)]
    if spans.empty:
        return float("nan")
    edges = np.concatenate([spans["t_start"].to_numpy(dtype=float), spans["t_end"].to_numpy(dtype=float)])
    steps = np.concatenate([np.ones(len(spans)), -np.ones(len(spans))])
    order = np.lexsort((steps, edges))
    edges, active = edges[order], np.cumsum(steps[order])
    widths = np.diff(edges)
    speaking = float(widths[active[:-1] >= 1].sum())
    return float(widths[active[:-1] >= 2].sum()) / speaking if speaking > 0 else float("nan")


def describe_voices(turns: pd.DataFrame, track_path: Path, t_origin_s: float, workspace: Path) -> dict:
    """Per voice its total speech and a few clips to listen to, cut from its longest turns alone."""
    samples, sr = load_audio(Path(track_path))
    exclusive = turns[turns["exclusive"].astype(bool)]
    voices = {}
    for label, own in exclusive.groupby("label", observed=True):
        longest = own.assign(length=own["t_end"] - own["t_start"]).nlargest(_CLIPS_PER_VOICE, "length").sort_values("t_start")
        clips = []
        for index, turn in enumerate(longest.itertuples(index=False), start=1):
            middle = (turn.t_start + turn.t_end) / 2
            start = max(turn.t_start, middle - _CLIP_S / 2)
            end = min(turn.t_end, middle + _CLIP_S / 2)
            name = f"clip_{label}_{index}.wav"
            _write_wav(workspace / name, samples, sr, start - t_origin_s, end - t_origin_s)
            clips.append({"file": name, "t_start": float(start), "t_end": float(end)})
        voices[str(label)] = {
            "speech_s": float((own["t_end"] - own["t_start"]).sum()),
            "first_s": float(own["t_start"].min()),
            "clips": clips,
        }
    return voices


def _write_wav(path: Path, samples: np.ndarray, sr: float, t_start: float, t_end: float) -> None:
    first, last = max(int(t_start * sr), 0), min(int(t_end * sr), samples.size)
    pcm = (np.clip(samples[first:last], -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sr))
        out.writeframes(pcm.tobytes())


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_voices(config, recording_id: str) -> dict:
    path = layout_for(config, create=False).artifacts_dir(recording_id, DiarizationExtractor.name) / VOICES_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_diarization(config, recording_id: str) -> pd.DataFrame:
    path = layout_for(config, create=False).artifacts_dir(recording_id, DiarizationExtractor.name) / DIARIZATION_NAME
    if not path.is_file():
        raise InputNotFoundError(f"{recording_id}: {path} is missing, run {DiarizationExtractor.name} first")
    return pd.read_parquet(path)


def read_voice_owners(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def voice_names(config, recording_id: str) -> dict[str, str | None]:
    """Each voice's person as a person named it, None for a voice left out; empty until someone has."""
    named = read_voice_owners(layout_for(config, create=False).voice_owners_path(recording_id))
    return {label: (None if who == NO_ONE else who) for label, who in named.get("people", {}).items()}


@register_extractor
class SpeakerReviewGate(Gate):
    """Held on a shared microphone until a person has said which voice is whom."""

    name = "text.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = ("text.diarization",)
    requires = (DiarizationExtractor.name,)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return []

    def pending(self, recording: Recording) -> str | None:
        if self.config.text.diarization != SHARED_MODE:
            return None
        rid = recording.recording_id
        voices = read_voices(self.config, rid)
        if set(voice_names(self.config, rid)) == set(voices):
            return None
        from macaw.extractors.audio.content.speaker_id import describe_suggestion, suggest_voices

        clips = layout_for(self.config, create=False).artifacts_dir(rid, DiarizationExtractor.name)
        suggestion = suggest_voices(self.config, recording)
        hint = (
            f"; from mouth movement {describe_suggestion(suggestion)}, kept with `macaw accept-voices -r {rid}`"
            if suggestion else "; with face.movement run on each person's camera a suggestion appears here"
        )
        return (
            f"nobody has said which voice is whom ({', '.join(sorted(voices)) or 'none found'}): listen to the clips "
            f"in {clips}{hint}, or name them with `macaw accept-voices -r {rid} --rename LABEL=PERSON,...` "
            f"({NO_ONE} leaves a voice out)"
        )
