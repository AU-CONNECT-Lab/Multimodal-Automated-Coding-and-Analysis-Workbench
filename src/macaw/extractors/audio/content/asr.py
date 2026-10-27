from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend, register_extractor
from macaw.extractors.audio.content.diarization import (
    BACKEND_GROUP as DIARIZATION_GROUP,
    SHARED_MODE,
    SpeakerReviewGate,
    read_diarization,
    resolve_diarization,
    shared_track,
    voice_inputs,
    voice_names,
)
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale, empty_frame

BACKEND_GROUP = "asr"

TRANSCRIPT_NAME = "transcript.parquet"
WORDS_NAME = "words.parquet"
DROPPED_NAME = "dropped.parquet"

# text cannot live in the long table, whose value column is a float, so the transcript is an
# artifact and only the countable quantities derived from it become features
TRANSCRIPT_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "utterance_index": "int64",
    "t_start": "float64",
    "t_end": "float64",
    "text": "string",
    "n_words": "int64",
    "confidence": "float64",
    "avg_logprob": "float64",     # faster-whisper only
    "no_speech_prob": "float64",  # faster-whisper only
    "language": "string",
}

WORD_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "utterance_index": "int64",
    "t_start": "float64",
    "t_end": "float64",
    "word": "string",
    "score": "float64",
}

# what the recogniser heard but the transcript leaves out, kept for a person to check
DROPPED_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "t_start": "float64",
    "t_end": "float64",
    "text": "string",
    "reason": "string",
    "detail": "string",
}

# whisperx transcribes in batches; 8 fits comfortably next to a large-v3 model on 24 GB
WHISPERX_BATCH_SIZE = 8
WHISPER_SAMPLE_RATE = 16000

_CROSSTALK_WARN_FRACTION = 0.5
# a word just outside every diarized turn still belongs to the nearest one within this
_WORD_REACH_S = 0.5
# an own-mic utterance at least this much inside the voice of someone without a mic can be their bleed
_ROOM_BLEED_SHARE = 0.5
# and only while the owner's own voice covers less than this of it, since overlapping talk is both people's
_OWNER_ABSENT_SHARE = 0.2


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    # ctranslate2 has no float16 kernels on CPU and refuses to load rather than falling back
    if device == "cpu" and compute_type in {"float16", "half"}:
        return "int8"
    return compute_type


class ASRBackend(Backend):
    # mean confidence below which a transcript is unlikely to be usable, on this backend's own
    # scale; None where nobody has measured one, since the scales are not comparable
    low_confidence: float | None = None

    def __init__(self, model: str, *, device: str = "auto", compute_type: str = "float16"):
        self.model = model
        self.device = resolve_device(device)
        self.compute_type = resolve_compute_type(compute_type, self.device)
        self._model = None

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (utterances, words); the words frame is empty unless asked for."""
        raise NotImplementedError


def _empty_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    utterances = empty_frame(TRANSCRIPT_SCHEMA).drop(columns=["recording_id", "subject_id"])
    words = empty_frame(WORD_SCHEMA).drop(columns=["recording_id", "subject_id"])
    return utterances, words


@register_backend(BACKEND_GROUP)
class WhisperXBackend(ASRBackend):
    name = "whisperx"

    @property
    def version(self) -> str:
        return _package_version("whisperx")

    def check_available(self) -> None:
        try:
            import whisperx  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "whisperx is not installed. Install PyTorch for your CUDA version first, "
                'then `pip install "pymacaw[asr]"`.'
            ) from None

    def _load(self):
        if self._model is None:
            import whisperx

            self._model = whisperx.load_model(
                self.model, self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import whisperx

        # whisperx's own loader shells out to an ffmpeg on PATH; prepared tracks are 16 kHz wav already
        samples, _ = load_audio(Path(audio_path), sample_rate=WHISPER_SAMPLE_RATE)
        audio = samples.astype(np.float32)
        result = self._load().transcribe(
            audio, batch_size=WHISPERX_BATCH_SIZE, language=language
        )
        detected = result.get("language") or language

        segments = result.get("segments") or []
        if word_timestamps and segments:
            model, metadata = whisperx.load_align_model(
                language_code=detected, device=self.device
            )
            segments = whisperx.align(
                segments, model, metadata, audio, self.device, return_char_alignments=False
            )["segments"]

        rows = []
        words = []
        for index, segment in enumerate(segments):
            scores = [
                float(word["score"])
                for word in segment.get("words") or []
                if word.get("score") is not None
            ]
            rows.append(
                {
                    "utterance_index": index,
                    "t_start": float(segment["start"]),
                    "t_end": float(segment["end"]),
                    "text": str(segment.get("text", "")).strip(),
                    "confidence": float(np.mean(scores)) if scores else np.nan,
                    "avg_logprob": np.nan,
                    "no_speech_prob": np.nan,
                    "language": detected,
                }
            )
            for word in segment.get("words") or []:
                # a numeral or a symbol comes back from the aligner without a time span
                if word.get("start") is None or word.get("end") is None:
                    continue
                words.append(
                    {
                        "utterance_index": index,
                        "t_start": float(word["start"]),
                        "t_end": float(word["end"]),
                        "word": str(word.get("word", "")).strip(),
                        "score": float(word["score"]) if word.get("score") is not None else np.nan,
                    }
                )

        if not rows:
            return _empty_parts()
        return pd.DataFrame(rows), (pd.DataFrame(words) if words else _empty_parts()[1])


@register_backend(BACKEND_GROUP)
class FasterWhisperBackend(ASRBackend):
    name = "faster_whisper"
    low_confidence = 0.5

    @property
    def version(self) -> str:
        return _package_version("faster_whisper")

    def check_available(self) -> None:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "faster-whisper is not installed. Install PyTorch for your CUDA version "
                'first, then `pip install "pymacaw[asr]"`.'
            ) from None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # prepared tracks are padded to a common length, and Whisper invents text on long
        # silences, so the built-in VAD gate stays on
        segments, info = self._load().transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,
        )
        detected = getattr(info, "language", None) or language

        rows = []
        words = []
        for index, segment in enumerate(segments):
            logprob = float(getattr(segment, "avg_logprob", np.nan))
            rows.append(
                {
                    "utterance_index": index,
                    "t_start": float(segment.start),
                    "t_end": float(segment.end),
                    "text": str(segment.text).strip(),
                    "confidence": math.exp(logprob) if math.isfinite(logprob) else np.nan,
                    "avg_logprob": logprob,
                    "no_speech_prob": float(getattr(segment, "no_speech_prob", np.nan)),
                    "language": detected,
                }
            )
            for word in getattr(segment, "words", None) or []:
                words.append(
                    {
                        "utterance_index": index,
                        "t_start": float(word.start),
                        "t_end": float(word.end),
                        "word": str(word.word).strip(),
                        "score": float(getattr(word, "probability", np.nan)),
                    }
                )

        if not rows:
            return _empty_parts()
        return pd.DataFrame(rows), (pd.DataFrame(words) if words else _empty_parts()[1])


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name.replace("_", "-"))
    except PackageNotFoundError:
        return "unknown"


_EVENT_SPECS = {
    "utterance_duration_s": ("Length of one transcribed utterance.", "s"),
    "n_words": ("Words in the utterance, counted on whitespace.", "count"),
    "speech_rate_wps": ("Words per second within the utterance, an articulation rate.", "word/s"),
    "asr_confidence": ("Recogniser confidence, 0-1. Defined differently by each backend.", None),
}

_SUMMARY_SPECS = {
    "n_utterances": ("Utterances attributed to this speaker.", "count"),
    "total_speech_s": ("Seconds this speaker held the floor.", "s"),
    "speech_fraction": ("Share of the task window this speaker was talking.", None),
    "total_words": ("Words this speaker produced.", "count"),
    "words_per_minute": ("Words per minute of their own speech, not of the session.", "word/min"),
    "mean_utterance_s": ("Mean utterance length.", "s"),
}


@register_extractor
class ASRExtractor(Extractor):
    name = "text.asr"
    version = "0.7.0"
    scale = Scale.EVENT
    group = "text"
    config_sections = (
        "text.asr_backend", "text.asr_model", "text.language", "text.device", "text.compute_type",
        "text.diarization", "text.crosstalk_margin_db", "text.word_timestamps", "segment",
    )
    requires = (MediaPrepExtractor.name,)
    resource_profile = "gpu_standard"

    @classmethod
    def dependencies(cls, config) -> tuple[str, ...]:
        # on a shared microphone nothing is anyone's until a person has named the voices
        return cls.requires + ((SpeakerReviewGate.name,) if config.text.diarization == SHARED_MODE else ())

    def __init__(self, config):
        super().__init__(config)
        self._backend: ASRBackend | None = None

    def backend(self) -> ASRBackend:
        if self._backend is None:
            from macaw.extractors import get_backend

            cfg = self.config.text
            # no fallback chain: there is no offline stand-in for a speech recogniser, and a
            # silent switch would change word timings without changing the numbers' name
            backend = get_backend(BACKEND_GROUP, cfg.asr_backend)(
                cfg.asr_model, device=cfg.device, compute_type=cfg.compute_type
            )
            backend.check_available()
            self._backend = backend
        return self._backend

    def check_available(self) -> None:
        from macaw.extractors import get_backend

        self.backend()
        if self.config.text.diarization != SHARED_MODE:
            get_backend(DIARIZATION_GROUP, self.config.text.diarization)().check_available()

    def _shared(self) -> bool:
        return self.config.text.diarization == SHARED_MODE

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, str(backend.version))

    def inputs_for(self, recording: Recording) -> list[Path]:
        corrected = layout_for(self.config, create=False).corrected_transcript_path(recording.recording_id)
        return [*voice_inputs(self.config, recording, load_prep(self.config, recording.recording_id)), corrected]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        prep = load_prep(self.config, recording.recording_id)
        backend = self.backend()
        warnings: list[QCWarning] = []
        self._dropped: list[pd.DataFrame] = []
        if layout_for(self.config, create=False).corrected_transcript_path(recording.recording_id).is_file():
            utterances, words = self._from_correction(recording, prep, warnings)
        elif not self._shared():
            utterances, words = self._from_own_tracks(recording, prep, backend, warnings)
        else:
            # whoever has a microphone of their own is transcribed from it; the shared one covers the rest
            utterances, words = self._from_shared_track(recording, prep, backend, warnings)
            if prep.tracks:
                own_utterances, own_words = self._from_own_tracks(recording, prep, backend, warnings)
                own_utterances = self._drop_room_bleed(recording, prep, own_utterances, warnings)
                own_words = _keep_utterances(own_words, own_utterances)
                utterances = _concat([own_utterances, utterances], TRANSCRIPT_SCHEMA, drop={"n_words"})
                words = _concat([own_words, words], WORD_SCHEMA)

        utterances = utterances.reset_index(drop=True)
        utterances["n_words"] = (
            utterances["text"].fillna("").astype(str).str.split().str.len().astype("int64")
        )
        dropped = _concat(self._dropped, DROPPED_SCHEMA)
        dropped["recording_id"] = recording.recording_id
        # ASR times are on the prepared track, whose sample 0 sits at t_origin_s
        for frame in (utterances, words, dropped):
            frame["t_start"] = frame["t_start"] + prep.t_origin_s
            frame["t_end"] = frame["t_end"] + prep.t_origin_s

        artifacts = {
            TRANSCRIPT_NAME: write_transcript(utterances, workspace / TRANSCRIPT_NAME),
            WORDS_NAME: _io.write_file(words, workspace / WORDS_NAME, WORD_SCHEMA, "words"),
            DROPPED_NAME: _io.write_file(dropped, workspace / DROPPED_NAME, DROPPED_SCHEMA, "dropped utterances"),
        }

        warnings += self._quality_warnings(recording, utterances, prep)
        return ExtractorResult(
            rows=self._rows(recording, utterances, prep),
            specs=self._specs(backend),
            artifacts=artifacts,
            warnings=warnings,
        )

    def _from_correction(self, recording: Recording, prep, warnings: list[QCWarning]):
        """The transcript a person corrected, on the prepared-track clock like a recogniser's; no words, nothing dropped."""
        from macaw.interop.transcript_eaf import read_transcript_eaf

        layout = layout_for(self.config, create=False)
        rid = recording.recording_id
        corrected = read_transcript_eaf(
            layout.corrected_transcript_path(rid), [subject.subject_id for subject in recording.subjects]
        )
        utterances = corrected.assign(
            recording_id=rid,
            utterance_index=corrected.groupby("subject_id").cumcount().astype("int64"),
            t_start=corrected["t_start"] - prep.t_origin_s,
            t_end=corrected["t_end"] - prep.t_origin_s,
            confidence=np.nan,
            language=self.config.text.language,
        )
        stamp_path = layout.corrected_transcript_stamp_path(rid)
        stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.is_file() else {}
        warnings.append(
            QCWarning(
                check="asr_corrected", recording_id=rid, subject_id=None, severity="info",
                message=f"the transcript is the one {stamp.get('by', 'a person')} corrected; speech was not recognised again",
            )
        )
        return _concat([utterances], TRANSCRIPT_SCHEMA, drop={"n_words"}), _concat([], WORD_SCHEMA)

    def _keep_dropped(self, frame: pd.DataFrame) -> None:
        self.__dict__.setdefault("_dropped", []).append(frame)

    def _from_shared_track(self, recording: Recording, prep, backend, warnings: list[QCWarning]):
        """One transcript of the shared microphone, each word given to the person whose voice it falls in."""
        cfg = self.config.text
        rid = recording.recording_id
        utterances, words = backend.transcribe(
            prep.shared_track_path(shared_track(prep, rid)), language=cfg.language, word_timestamps=True
        )
        utterances, words, outside = within_track(utterances, words, prep.duration_s)
        if outside:
            warnings.append(
                QCWarning(
                    check="asr_outside_track", recording_id=rid, subject_id=None, severity="info",
                    message=f"{outside} utterance(s) the backend returned lie outside the prepared track and were dropped.",
                )
            )
        turns = _track_clock_turns(self.config, rid, prep)
        utterances, words, unplaced = split_by_voice(
            utterances, words, turns, voice_names(self.config, rid), elsewhere=frozenset(prep.tracks)
        )
        if unplaced:
            warnings.append(
                QCWarning(
                    check="asr_unplaced_words", recording_id=rid, subject_id=None, severity="info",
                    message=f"{unplaced} word(s) fell outside every named voice and were left out.",
                )
            )
        for frame in (utterances, words):
            frame["recording_id"] = rid
        return _concat([utterances], TRANSCRIPT_SCHEMA, drop={"n_words"}), _concat([words], WORD_SCHEMA)

    def _drop_room_bleed(self, recording: Recording, prep, utterances: pd.DataFrame, warnings: list[QCWarning]) -> pd.DataFrame:
        """A person heard only on the shared microphone still leaks into the others' own microphones.
        An utterance is their bleed when it lies mostly in their voice and the owner's is all but absent."""
        names = voice_names(self.config, recording.recording_id)
        turns = _track_clock_turns(self.config, recording.recording_id, prep, exclusive=False)
        turns = turns.assign(subject_id=turns["label"].map(lambda label: names.get(label)))
        roomers = set(turns["subject_id"].dropna()) - set(prep.tracks)
        flags = np.zeros(len(utterances), dtype=bool)
        whose = [""] * len(utterances)
        for index, row in enumerate(utterances.itertuples(index=False)):
            length = row.t_end - row.t_start
            if length <= 0:
                continue
            overlap = (np.minimum(turns["t_end"], row.t_end) - np.maximum(turns["t_start"], row.t_start)).clip(lower=0)
            own = float(overlap[turns["subject_id"] == row.subject_id].sum())
            in_room = overlap[turns["subject_id"].isin(roomers)].groupby(turns["subject_id"]).sum()
            other = float(in_room.sum())
            flags[index] = other >= _ROOM_BLEED_SHARE * length and own < _OWNER_ABSENT_SHARE * length
            whose[index] = f"in the voice of {in_room.idxmax()}" if other > 0 else ""
        self._keep_dropped(
            utterances[flags].assign(
                reason="room bleed", detail=[whose[index] for index in np.flatnonzero(flags)]
            )
        )
        for subject_id in utterances.loc[flags, "subject_id"].unique():
            mine = utterances["subject_id"] == subject_id
            warnings.append(
                QCWarning(
                    check="asr_room_bleed", recording_id=recording.recording_id, subject_id=str(subject_id), severity="info",
                    message=(
                        f"{int((flags & mine).sum())} of {int(mine.sum())} utterances on their own microphone were in the voice of "
                        "someone heard only on the shared microphone and were dropped."
                    ),
                )
            )
        return utterances[~flags]

    def _from_own_tracks(self, recording: Recording, prep, backend, warnings: list[QCWarning]):
        cfg = self.config.text
        tracks = {subject_id: prep.track_path(subject_id) for subject_id in prep.tracks}
        utterance_parts: list[pd.DataFrame] = []
        word_parts: list[pd.DataFrame] = []

        for subject in recording.subjects:
            if subject.subject_id not in prep.tracks:
                if self._shared():
                    continue
                warnings.append(
                    QCWarning(
                        check="asr_missing_track",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message=(
                            f"No prepared track for '{subject.subject_id}', so nothing they "
                            "said is transcribed."
                        ),
                    )
                )
                continue

            utterances, words = backend.transcribe(
                tracks[subject.subject_id],
                language=cfg.language,
                word_timestamps=cfg.word_timestamps,
            )
            utterances, words, outside = within_track(utterances, words, prep.duration_s)
            if outside:
                warnings.append(
                    QCWarning(
                        check="asr_outside_track",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="info",
                        message=f"{outside} utterance(s) the backend returned lie outside the prepared track and were dropped.",
                    )
                )
            for frame in (utterances, words):
                frame["recording_id"] = recording.recording_id
                frame["subject_id"] = subject.subject_id
            utterance_parts.append(utterances)
            word_parts.append(words)

        utterances = _concat(utterance_parts, TRANSCRIPT_SCHEMA, drop={"n_words"})
        words = _concat(word_parts, WORD_SCHEMA)

        # beside a shared microphone the own tracks still bleed into each other
        diarizer = resolve_diarization(recording, "per_track" if self._shared() else cfg.diarization)
        flagged = diarizer.attribute(utterances, tracks, margin_db=cfg.crosstalk_margin_db)
        warnings += self._crosstalk_warnings(recording, flagged)

        bleed = flagged[flagged["is_crosstalk"].astype(bool)]
        self._keep_dropped(
            bleed.assign(
                reason="crosstalk",
                detail=[
                    f"{by:.1f} dB louder on {on}" if isinstance(on, str) else ""
                    for on, by in zip(bleed.get("louder_on", [None] * len(bleed)), bleed.get("louder_by_db", [np.nan] * len(bleed)))
                ],
            )
        )
        extra = [column for column in ("is_crosstalk", "louder_on", "louder_by_db") if column in flagged]
        utterances = flagged[~flagged["is_crosstalk"].astype(bool)].drop(columns=extra)
        return utterances, _keep_utterances(words, utterances)

    def _crosstalk_warnings(self, recording: Recording, flagged: pd.DataFrame) -> list[QCWarning]:
        warnings = []
        for subject_id, group in flagged.groupby("subject_id", dropna=False, observed=True):
            dropped = int(group["is_crosstalk"].astype(bool).sum())
            if not dropped:
                continue
            fraction = dropped / len(group)
            warnings.append(
                QCWarning(
                    check="asr_crosstalk",
                    recording_id=recording.recording_id,
                    subject_id=str(subject_id),
                    severity="warn" if fraction > _CROSSTALK_WARN_FRACTION else "info",
                    message=(
                        f"{dropped} of {len(group)} utterances ({fraction:.0%}) were louder on "
                        "another track and dropped as microphone bleed."
                    ),
                    details={"crosstalk_margin_db": self.config.text.crosstalk_margin_db},
                )
            )
        return warnings

    def _quality_warnings(self, recording: Recording, utterances, prep) -> list[QCWarning]:
        warnings = []
        span = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)

        for subject in recording.subjects:
            if subject.subject_id not in prep.tracks and not self._shared():
                continue
            mine = utterances[utterances["subject_id"] == subject.subject_id]
            if mine.empty:
                warnings.append(
                    QCWarning(
                        check="asr_no_speech",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message="No speech was transcribed for them.",
                        t_start=span[0],
                        t_end=span[1],
                    )
                )
                continue
            confidence = float(mine["confidence"].mean())
            threshold = self.backend().low_confidence
            if threshold is not None and math.isfinite(confidence) and confidence < threshold:
                warnings.append(
                    QCWarning(
                        check="asr_confidence",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message=(
                            f"Mean recogniser confidence is {confidence:.2f}. The transcript "
                            "is unlikely to be usable as text, though the timings may still be."
                        ),
                        t_start=span[0],
                        t_end=span[1],
                    )
                )

        languages = {str(value) for value in utterances["language"].dropna().unique()}
        if len(languages) > 1:
            warnings.append(
                QCWarning(
                    check="asr_language_mismatch",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"Tracks were autodetected as different languages ({', '.join(sorted(languages))}). "
                        "Set text.language to pin one."
                    ),
                )
            )
        return warnings

    def _rows(self, recording: Recording, utterances: pd.DataFrame, prep) -> pd.DataFrame:
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        frames = []

        for subject_id, group in utterances.groupby("subject_id", dropna=False, observed=True):
            duration = (group["t_end"] - group["t_start"]).to_numpy(dtype=float)
            counts = group["n_words"].to_numpy(dtype=float)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": group["t_start"].to_numpy(dtype=float),
                            "t_end": group["t_end"].to_numpy(dtype=float),
                            "utterance_duration_s": duration,
                            "n_words": counts,
                            "speech_rate_wps": np.divide(
                                counts, duration, out=np.full_like(counts, np.nan), where=duration > 0
                            ),
                            "asr_confidence": group["confidence"].to_numpy(dtype=float),
                        }
                    ),
                    scale=Scale.EVENT,
                    subject_id=str(subject_id),
                    **common,
                )
            )

            total_speech = float(duration.sum())
            total_words = float(counts.sum())
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [prep.t_origin_s],
                            "t_end": [prep.t_origin_s + prep.duration_s],
                            "n_utterances": [float(len(group))],
                            "total_speech_s": [total_speech],
                            "speech_fraction": [
                                total_speech / prep.duration_s if prep.duration_s > 0 else np.nan
                            ],
                            "total_words": [total_words],
                            "words_per_minute": [
                                total_words / total_speech * 60.0 if total_speech > 0 else np.nan
                            ],
                            "mean_utterance_s": [float(duration.mean()) if duration.size else np.nan],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=str(subject_id),
                    **common,
                )
            )
        return concat_long_tables(frames)

    def _specs(self, backend: ASRBackend) -> list[FeatureSpec]:
        params = {"asr_model": self.config.text.asr_model, "language": self.config.text.language}
        specs = [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.EVENT,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=backend.name,
                backend_version=str(backend.version),
                params=params,
                aggregation="mean",
                comparable_across_backends=feature != "asr_confidence",
                validation="none",
            )
            for feature, (description, unit) in _EVENT_SPECS.items()
        ]
        specs += [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=backend.name,
                backend_version=str(backend.version),
                params=params,
                aggregation="mean",
                validation="none",
            )
            for feature, (description, unit) in _SUMMARY_SPECS.items()
        ]
        return specs


def _concat(parts: list[pd.DataFrame], schema: dict[str, str], drop: set[str] | None = None):
    columns = [name for name in schema if name not in (drop or set())]
    if not parts:
        return empty_frame(schema).loc[:, columns]
    # a column only some recognisers give (avg_logprob, no_speech_prob) is empty for the others
    return pd.concat(parts, ignore_index=True).reindex(columns=columns)


def within_track(
    utterances: pd.DataFrame, words: pd.DataFrame, duration_s: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Clips to [0, duration_s]; a backend that reads an existing transcript can return the whole session."""
    keep = (utterances["t_start"] < duration_s) & (utterances["t_end"] > 0)
    outside = int((~keep).sum())
    utterances = utterances[keep].copy()
    utterances["t_start"] = utterances["t_start"].clip(lower=0.0)
    utterances["t_end"] = utterances["t_end"].clip(upper=duration_s)
    if not words.empty:
        words = words[
            words["utterance_index"].isin(utterances["utterance_index"])
            & (words["t_start"] < duration_s) & (words["t_end"] > 0)
        ].copy()
        words["t_start"] = words["t_start"].clip(lower=0.0)
        words["t_end"] = words["t_end"].clip(upper=duration_s)
    return utterances, words, outside


def _keep_utterances(words: pd.DataFrame, utterances: pd.DataFrame) -> pd.DataFrame:
    """Words follow their utterance; indices keep their original values so the two files join."""
    if words.empty:
        return words.reset_index(drop=True)
    kept = pd.MultiIndex.from_arrays([utterances["subject_id"], utterances["utterance_index"]])
    mask = pd.MultiIndex.from_arrays([words["subject_id"], words["utterance_index"]]).isin(kept)
    return words[mask].reset_index(drop=True)


def write_transcript(df: pd.DataFrame, path: Path) -> Path:
    return _io.write_file(df, Path(path), TRANSCRIPT_SCHEMA, "transcript")


def read_transcript(path: Path) -> pd.DataFrame:
    return _io.read_file(Path(path), TRANSCRIPT_SCHEMA, "transcript")


def read_words(path: Path) -> pd.DataFrame:
    return _io.read_file(Path(path), WORD_SCHEMA, "words")


def load_transcript(config, recording_id: str) -> pd.DataFrame:
    """Where every downstream text extractor starts, the way audio ones start at load_prep."""
    from macaw.project.layout import layout_for

    root = layout_for(config, create=False).artifacts_dir(
        recording_id, ASRExtractor.name
    )
    return read_transcript(root / TRANSCRIPT_NAME)


def _track_clock_turns(config, recording_id: str, prep, *, exclusive: bool = True) -> pd.DataFrame:
    turns = read_diarization(config, recording_id)
    return turns[turns["exclusive"].astype(bool) == exclusive].assign(
        t_start=lambda frame: frame["t_start"] - prep.t_origin_s, t_end=lambda frame: frame["t_end"] - prep.t_origin_s
    )


def split_by_voice(
    utterances: pd.DataFrame, words: pd.DataFrame, turns: pd.DataFrame, names: dict[str, str | None],
    *, elsewhere: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Each word goes to the voice turn it overlaps most, or the nearest within reach; a run of
    one person's words inside a recogniser segment is one utterance. Words of people in elsewhere
    are left to their own microphone. Returns the count of words no named voice took."""
    columns = [column for column in utterances.columns if column not in {"utterance_index", "t_start", "t_end", "text", "confidence"}]
    if words.empty:
        # a backend without word times can only be split at its segments
        placed = assign_voices(utterances, turns)
        placed["subject_id"] = placed.pop("label").map(lambda label: names.get(label))
        unplaced = int(placed["subject_id"].isna().sum())
        kept = placed[placed["subject_id"].notna() & ~placed["subject_id"].isin(elsewhere)].copy()
        kept["utterance_index"] = kept.sort_values("t_start").groupby("subject_id").cumcount()
        return kept, words.assign(subject_id=pd.Series(dtype="string")), unplaced

    placed = assign_voices(words, turns)
    placed["subject_id"] = placed.pop("label").map(lambda label: names.get(label))
    unplaced = int(placed["subject_id"].isna().sum())
    placed = placed[placed["subject_id"].notna() & ~placed["subject_id"].isin(elsewhere)]
    placed = placed.sort_values(["t_start", "t_end"]).reset_index(drop=True)
    # a new utterance starts wherever the speaker or the recogniser segment changes
    run = (placed["subject_id"] != placed["subject_id"].shift()) | (placed["utterance_index"] != placed["utterance_index"].shift())
    placed["run"] = run.cumsum()
    parent = utterances.set_index("utterance_index")
    rows = []
    for _, part in placed.groupby("run", sort=True):
        source = parent.loc[int(part["utterance_index"].iloc[0])]
        rows.append({
            "subject_id": part["subject_id"].iloc[0],
            "run": int(part["run"].iloc[0]),
            "t_start": float(part["t_start"].min()),
            "t_end": float(part["t_end"].max()),
            "text": " ".join(part["word"].astype(str)),
            "confidence": float(part["score"].mean()) if part["score"].notna().any() else np.nan,
            **{column: source[column] for column in columns},
        })
    split = pd.DataFrame(rows)
    split["utterance_index"] = split.groupby("subject_id").cumcount()
    index = split.set_index("run")["utterance_index"]
    placed["utterance_index"] = placed["run"].map(index)
    return split.drop(columns=["run"]), placed.drop(columns=["run"]), unplaced


def assign_voices(spans: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Add label: the turn each span overlaps most, else the nearest turn within reach, else None."""
    out = spans.copy()
    if turns.empty:
        out["label"] = None
        return out
    starts = turns["t_start"].to_numpy(dtype=float)
    ends = turns["t_end"].to_numpy(dtype=float)
    labels = turns["label"].astype(str).to_numpy()
    picked = []
    for t_start, t_end in zip(out["t_start"].to_numpy(dtype=float), out["t_end"].to_numpy(dtype=float)):
        overlap = np.minimum(ends, t_end) - np.maximum(starts, t_start)
        best = int(np.argmax(overlap))
        if overlap[best] > 0:
            picked.append(labels[best])
            continue
        # negative overlap is the gap to that turn
        picked.append(labels[best] if -overlap[best] <= _WORD_REACH_S else None)
    out["label"] = picked
    return out
