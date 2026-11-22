from __future__ import annotations

import functools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend, register_extractor
from macaw.extractors.audio.content.diarization import (
    BACKEND_GROUP as DIARIZATION_GROUP,
    DIARIZATION_NAME,
    LINES_GATE,
    OTHER,
    SHARED_MODE,
    DiarizationExtractor,
    SpeakerReviewGate,
    inside,
    read_diarization,
    refuse_older,
    resolve_diarization,
    shared_track,
    speaker_names,
    turn_takers,
    voice_inputs,
)
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning, distribution_version
from macaw.extractors.hub import check_hub_model
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale, empty_frame

BACKEND_GROUP = "asr"
SHARED_ASR = "text.shared_asr"

TRANSCRIPT_NAME = "transcript.parquet"
# the warning that says a run used a person's corrected transcript; the Methods text counts it
CORRECTED_CHECK = "asr_corrected"
SHARED_UTTERANCES_NAME = "shared_utterances.parquet"
SHARED_WORDS_NAME = "shared_words.parquet"
# what the recogniser gave outside the shared track, for text.asr's dropped utterances
SHARED_OUTSIDE_NAME = "shared_outside.parquet"
# what each voice on a shared microphone said and how high it is, before anyone names it
VOICE_LINES_NAME = "voice_lines.parquet"
VOICE_PITCH_NAME = "voice_pitch.json"
WORDS_NAME = "words.parquet"
DROPPED_NAME = "dropped.parquet"
BLEED_NAME = "bleed.parquet"

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
    "louder_on": "string",
    "louder_by_db": "float64",
    "gain_db": "float64",
    "gain_raised": "bool",
    "room_voice": "string",
    "room_voice_share": "float64",
    "own_voice_share": "float64",
}

# every utterance of a person's own microphone the bleed checks measured, kept or dropped, so a margin can be judged
BLEED_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "utterance_index": "int64",
    "t_start": "float64",
    "t_end": "float64",
    "louder_on": "string",
    "louder_by_db": "float64",
    "gain_db": "float64",
    "gain_raised": "bool",
    "room_voice": "string",
    "room_voice_share": "float64",
    "own_voice_share": "float64",
    "dropped_as": "string",
}
_CROSSTALK_COLUMNS = ["louder_on", "louder_by_db", "gain_db", "gain_raised"]
_ROOM_COLUMNS = ["room_voice", "room_voice_share", "own_voice_share"]

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
    # the model an empty text.asr_model stands for
    default_model = ""

    def __init__(self, model: str | None, *, device: str = "auto", compute_type: str = "float16"):
        self.model = model or self.default_model
        self._device, self._compute_type = device, compute_type
        self._model = None
        self._aligners: dict = {}

    # resolved when first used, so making a backend to key its step loads no library
    @functools.cached_property
    def device(self) -> str:
        return resolve_device(self._device)

    @functools.cached_property
    def compute_type(self) -> str:
        return resolve_compute_type(self._compute_type, self.device)

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (utterances, words); the words frame is empty unless asked for."""
        raise NotImplementedError

    def prefetch(self, *, language: str, align: bool) -> None:
        """Fetch the weights transcribe would download; a backend that reads existing transcripts has none."""
        return None


def _empty_parts() -> tuple[pd.DataFrame, pd.DataFrame]:
    utterances = empty_frame(TRANSCRIPT_SCHEMA).drop(columns=["recording_id", "subject_id"])
    words = empty_frame(WORD_SCHEMA).drop(columns=["recording_id", "subject_id"])
    return utterances, words


def whisper_repo(model: str) -> str:
    """The hub repository a whisper model name stands for, as the recogniser looks it up."""
    try:
        from faster_whisper.utils import _MODELS
    except ImportError:
        return model
    return _MODELS.get(model, model)


@register_backend(BACKEND_GROUP)
class WhisperXBackend(ASRBackend):
    name = "whisperx"
    default_model = "large-v3"

    @property
    def version(self) -> str:
        return _package_version("whisperx")

    def check_available(self) -> None:
        refuse_older("whisperx", (3, 8), "the first release built on pyannote.audio 4")
        try:
            import whisperx  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "whisperx is not installed. It needs an environment of its own: install the PyTorch release it requires for "
                'your CUDA version first, then `pip install "pymacaw[whisperx]"`.'
            ) from None
        check_hub_model(whisper_repo(self.model), "text.asr_model")

    def _load(self):
        if self._model is None:
            import whisperx

            self._model = whisperx.load_model(
                self.model, self.device, compute_type=self.compute_type
            )
        return self._model

    def _aligner(self, language: str):
        """(model, metadata) for word timing in this language, loaded once for all the tracks of a task."""
        if language not in self._aligners:
            import whisperx

            try:
                self._aligners[language] = whisperx.load_align_model(language_code=language, device=self.device)
            except (ValueError, OSError) as error:
                raise BackendNotAvailableError(
                    f"no word-timing model could be loaded for the language '{language}' ({error}); where the language "
                    "was detected, set text.language to the one spoken, and fetch its model with `macaw fetch` before an offline run"
                ) from error
        return self._aligners[language]

    def prefetch(self, *, language: str, align: bool) -> None:
        self._load()
        if align:
            self._aligner(language)

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
            model, metadata = self._aligner(detected)
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
            last = float(segment["start"])
            for word in segment.get("words") or []:
                # a numeral or a symbol comes back from the aligner without a time span; it is kept at the moment the
                # word before it ends, so it goes where that word goes
                start = float(word["start"]) if word.get("start") is not None else last
                end = float(word["end"]) if word.get("end") is not None else start
                last = end
                words.append(
                    {
                        "utterance_index": index,
                        "t_start": start,
                        "t_end": end,
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
    default_model = "large-v3"
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

            if self.device == "cuda":
                # on Windows ctranslate2 finds cuBLAS only among the CUDA libraries torch has loaded
                import torch  # noqa: F401
            self._model = WhisperModel(
                self.model, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def prefetch(self, *, language: str, align: bool) -> None:
        self._load()

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # faster-whisper's own decoder breaks on PyAV releases newer than it was written for
        samples, _ = load_audio(Path(audio_path), sample_rate=WHISPER_SAMPLE_RATE)
        # prepared tracks are padded to a common length, and Whisper invents text on long
        # silences, so the built-in VAD gate stays on
        segments, info = self._load().transcribe(
            samples.astype(np.float32),
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
    "speech_fraction": ("Share of the analysed part, or the segment, this speaker was talking.", None),
    "total_words": ("Words this speaker produced.", "count"),
    "words_per_minute": ("Words per minute of their own speech, not of the session.", "word/min"),
    "mean_utterance_s": ("Mean utterance length.", "s"),
}

_UNTRANSCRIBED = (
    "nothing of this person was transcribed in the span: no microphone of their own and no voice on a shared one named as "
    "theirs, no speech of theirs recognised or in the corrected transcript, all of it dropped as another person's bleed, or a "
    "segment in which they did not speak"
)
_MISSING = {
    "utterance_duration_s": "Never empty on an utterance row. A person of whom nothing was transcribed has no utterance rows.",
    "n_words": (
        "Never empty on an utterance row. A person of whom nothing was transcribed has no utterance rows. 0 means the "
        "recogniser returned an utterance without text."
    ),
    "speech_rate_wps": "Empty for an utterance of no length. 0 means an utterance without text.",
    "asr_confidence": (
        "Empty where the recogniser gave no confidence for the utterance: always for a transcript a person corrected, with "
        "whisperx on a person's own microphone when text.word_timestamps is off, and where none of its words was scored."
    ),
    "total_words": f"Empty, not 0, where {_UNTRANSCRIBED}. 0 means their utterances held no text.",
    "words_per_minute": f"Empty where their utterances have no length, and where {_UNTRANSCRIBED}.",
    "summary": f"Empty, not 0, where {_UNTRANSCRIBED}.",
}


# Qwen3-ASR reads speech in pieces of at most this length, as Whisper does
_PIECE_S = 30.0
_WHISPER_NAMES = frozenset({"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo", "turbo",
                            "tiny.en", "base.en", "small.en", "medium.en", "distil-large-v2", "distil-large-v3"})
# pieces recognised and aligned at once, fixed because the batch nudges the words recognised
_QWEN_BATCH = 16
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def speech_pieces(samples: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> list[tuple[float, float]]:
    """Silero's speech stretches, as faster-whisper finds them, joined into pieces of the recording."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    found = get_speech_timestamps(samples, VadOptions(max_speech_duration_s=_PIECE_S, min_silence_duration_ms=160), sampling_rate=sample_rate)
    return joined_stretches([(stretch["start"] / sample_rate, stretch["end"] / sample_rate) for stretch in found])


def joined_stretches(stretches: list[tuple[float, float]], longest_s: float = _PIECE_S) -> list[tuple[float, float]]:
    """Speech stretches joined, silences between them included, into unbroken pieces no longer than longest_s."""
    pieces: list[list[float]] = []
    for start, end in stretches:
        if pieces and end - pieces[-1][0] <= longest_s:
            pieces[-1][1] = end
        else:
            pieces.append([start, end])
    return [(start, end) for start, end in pieces]


def sentences_of(text: str, words: list[dict], count) -> list[tuple[str, list[dict]]]:
    """A piece's text cut after each full stop, question or exclamation mark, each sentence with its aligned words;
    the whole piece as one where the words cannot be shared out."""
    sentences = [part for part in _SENTENCE_END.split(text.strip()) if part]
    sizes = [count(sentence) for sentence in sentences]
    if len(sentences) < 2 or sum(sizes) != len(words):
        return [(text.strip(), words)]
    edges = np.cumsum([0, *sizes])
    return [(sentence, words[a:b]) for sentence, a, b in zip(sentences, edges[:-1], edges[1:]) if b > a]


def track_language(pieces: list[tuple[float, float]], heard: list[tuple[str, str | None]]) -> str | None:
    """The language heard over the most speech, counting pieces with any text."""
    spoken: dict[str | None, float] = {}
    for (start, end), (text, said) in zip(pieces, heard):
        if text.strip():
            spoken[said] = spoken.get(said, 0.0) + end - start
    return max(spoken, key=spoken.get) if spoken else None


@register_backend(BACKEND_GROUP)
class Qwen3ASRBackend(ASRBackend):
    """Qwen3-ASR through transformers, with word times from Qwen3-ForcedAligner; Apache-2.0 code and weights."""

    name = "qwen3"
    default_model = "Qwen/Qwen3-ASR-1.7B-hf"
    aligner_model = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

    def __init__(self, model: str | None, *, device: str = "auto", compute_type: str = "float16"):
        super().__init__(model, device=device, compute_type=compute_type)
        self._aligner = None

    @property
    def version(self) -> str:
        return distribution_version("transformers")

    def check_available(self) -> None:
        if self.model in _WHISPER_NAMES:
            raise BackendNotAvailableError(
                f"text.asr_model {self.model} is a Whisper model; with text.asr_backend qwen3 leave text.asr_model empty, or name a "
                "Qwen3-ASR model such as Qwen/Qwen3-ASR-0.6B-hf"
            )
        try:
            import faster_whisper  # noqa: F401
            import torch  # noqa: F401
            import transformers
        except ImportError:
            raise BackendNotAvailableError(
                "text.asr_backend qwen3 needs PyTorch, transformers and faster-whisper (for its speech detector): install "
                'PyTorch for your GPU first, then `pip install "pymacaw[asr]"`.'
            ) from None
        if not hasattr(transformers, "Qwen3ASRForTokenClassification"):
            raise BackendNotAvailableError(
                f"transformers {transformers.__version__} has no Qwen3-ASR: pip install -U transformers, in an environment without whisperx."
            )
        check_hub_model(self.model, "text.asr_model")
        check_hub_model(self.aligner_model, "the Qwen3-ForcedAligner word timing")

    def _parts(self, name: str):
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, Qwen3ASRForTokenClassification

        kind = Qwen3ASRForConditionalGeneration if name == self.model else Qwen3ASRForTokenClassification
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        return AutoProcessor.from_pretrained(name), kind.from_pretrained(name, dtype=dtype).to(self.device).eval()

    def _load(self):
        if self._model is None:
            self._model = self._parts(self.model)
        return self._model

    def _load_aligner(self):
        if self._aligner is None:
            self._aligner = self._parts(self.aligner_model)
        return self._aligner

    def prefetch(self, *, language: str, align: bool) -> None:
        self._load()
        if align:
            self._load_aligner()

    def transcribe(
        self, audio_path: Path, *, language: str | None, word_timestamps: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        samples, _ = load_audio(Path(audio_path), sample_rate=WHISPER_SAMPLE_RATE)
        samples = samples.astype(np.float32)
        pieces = speech_pieces(samples)
        if not pieces:
            return _empty_parts()
        heard = self._recognise(samples, pieces, language)
        aligned = self._align(samples, pieces, heard, Path(audio_path).name) if word_timestamps else [[] for _ in pieces]
        rows, words = [], []
        processor = self._load_aligner()[0] if word_timestamps else None
        for (start, end), (text, spoken), timed in zip(pieces, heard, aligned):
            if not text.strip():
                continue
            count = (lambda part: len(processor.split_words_for_alignment(part, spoken))) if processor else (lambda part: 0)
            for sentence, own in sentences_of(text, timed, count) if timed else [(text.strip(), [])]:
                index = len(rows)
                rows.append({
                    "utterance_index": index,
                    "t_start": own[0]["t_start"] if own else start,
                    "t_end": own[-1]["t_end"] if own else end,
                    "text": sentence, "confidence": np.nan, "avg_logprob": np.nan, "no_speech_prob": np.nan, "language": spoken,
                })
                words += [{"utterance_index": index, **word, "score": np.nan} for word in own]
        empty_rows, empty_words = _empty_parts()
        return (pd.DataFrame(rows, columns=empty_rows.columns) if rows else empty_rows,
                pd.DataFrame(words, columns=empty_words.columns) if words else empty_words)

    def _recognise(self, samples: np.ndarray, pieces: list[tuple[float, float]], language: str | None) -> list[tuple[str, str | None]]:
        """Each piece's text and language; with none given, the track's language is the one heard over most of its speech,
        as Whisper takes one per file, and pieces heard as another are recognised again in it."""
        heard = self._heard(samples, pieces, list(range(len(pieces))), language)
        if language is not None:
            return heard
        spoken = track_language(pieces, heard)
        # a language the model cannot be told is left for the aligner check to name
        again = [i for i, (text, said) in enumerate(heard) if said != spoken and text.strip()] if spoken in self._known_languages() else []
        heard = [(text, spoken) for text, _ in heard]
        for i, said in zip(again, self._heard(samples, pieces, again, spoken) if again else []):
            heard[i] = said
        return heard

    @staticmethod
    def _known_languages() -> set[str]:
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        return set(LANGUAGE_CODE_TO_NAME)

    def _heard(self, samples: np.ndarray, pieces, indices: list[int], language: str | None) -> list[tuple[str, str | None]]:
        """The text and language, as an ISO code where the model names one it knows, of the pieces at indices, in that order."""
        import torch
        from transformers.models.qwen3_asr.processing_qwen3_asr import LANGUAGE_CODE_TO_NAME

        codes = {name: code for code, name in LANGUAGE_CODE_TO_NAME.items()}
        processor, model = self._load()
        heard: dict[int, tuple[str, str | None]] = {}
        # pieces of like length go together, so a batch is padded little
        order = sorted(indices, key=lambda i: pieces[i][1] - pieces[i][0])
        for first in range(0, len(order), _QWEN_BATCH):
            batch = order[first:first + _QWEN_BATCH]
            audio = [samples[int(pieces[i][0] * WHISPER_SAMPLE_RATE):int(pieces[i][1] * WHISPER_SAMPLE_RATE)] for i in batch]
            inputs = processor.apply_transcription_request(audio, language=language).to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=448, do_sample=False)
            parsed = processor.decode(generated[:, inputs["input_ids"].shape[1]:], return_format="parsed")
            for i, said in zip(batch, [parsed] if isinstance(parsed, dict) else parsed):
                named = said.get("language")
                heard[i] = (said.get("transcription") or "", language or codes.get(named, named))
        return [heard[i] for i in indices]

    def _align(self, samples: np.ndarray, pieces, heard, track: str) -> list[list[dict]]:
        import torch
        from transformers.models.qwen3_asr.processing_qwen3_asr import FORCED_ALIGNER_LANGUAGES, LANGUAGE_CODE_TO_NAME

        processor, model = self._load_aligner()
        aligned: list[list[dict]] = [[] for _ in pieces]
        todo = [i for i, (text, _) in enumerate(heard) if text.strip()]
        for i in todo:
            spoken = heard[i][1]
            if LANGUAGE_CODE_TO_NAME.get(spoken, spoken) not in FORCED_ALIGNER_LANGUAGES:
                raise BackendNotAvailableError(
                    f"{track}: Qwen3-ForcedAligner gives word times in {', '.join(sorted(FORCED_ALIGNER_LANGUAGES))}, and "
                    f"{spoken} was heard; set text.asr_backend: faster_whisper, or text.word_timestamps: false"
                )
        todo.sort(key=lambda i: pieces[i][1] - pieces[i][0])
        for first in range(0, len(todo), _QWEN_BATCH):
            batch = todo[first:first + _QWEN_BATCH]
            audio = [samples[int(pieces[i][0] * WHISPER_SAMPLE_RATE):int(pieces[i][1] * WHISPER_SAMPLE_RATE)] for i in batch]
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio, [heard[i][0] for i in batch], language=[heard[i][1] for i in batch]
            )
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.inference_mode():
                logits = model(**inputs).logits
            found = processor.decode_forced_alignment(logits, inputs["input_ids"], word_lists, model.config.timestamp_token_id)
            for i, items in zip(batch, found):
                start = pieces[i][0]
                aligned[i] = [{"t_start": start + w["start_time"], "t_end": start + w["end_time"], "word": w["text"]} for w in items]
        return aligned


@register_extractor
class ASRExtractor(Extractor):
    name = "text.asr"
    version = "0.13.0"
    scale = Scale.EVENT
    group = "text"
    config_sections = (
        "text.asr_backend", "text.asr_model", "text.language", "text.device", "text.compute_type",
        "text.diarization", "text.extra_voices", "text.crosstalk_margin_db", "text.word_timestamps", "segment",
    )
    requires = (MediaPrepExtractor.name,)
    resource_profile = "gpu_standard"

    @classmethod
    def dependencies(cls, config) -> tuple[str, ...]:
        # on a shared microphone nothing is anyone's until a person has named the voices
        return cls.requires + ((SHARED_ASR, SpeakerReviewGate.name, LINES_GATE) if config.text.diarization == SHARED_MODE else ())

    def __init__(self, config):
        super().__init__(config)
        self._backend: ASRBackend | None = None

    def backend(self) -> ASRBackend:
        if self._backend is None:
            # no fallback chain: there is no offline stand-in for a speech recogniser, and a
            # silent switch would change word timings without changing the numbers' name
            backend = self._named_backend()
            backend.check_available()
            self._backend = backend
        return self._backend

    def _named_backend(self) -> ASRBackend:
        from macaw.extractors import get_backend

        cfg = self.config.text
        return get_backend(BACKEND_GROUP, cfg.asr_backend)(cfg.asr_model, device=cfg.device, compute_type=cfg.compute_type)

    def backends(self) -> list[Backend]:
        from macaw.extractors import get_backend

        # each track's crosstalk is judged by the diarization backend when every voice has its own
        shared = self._shared()
        return [self._named_backend(), *([] if shared else [get_backend(DIARIZATION_GROUP, self.config.text.diarization)()])]

    def check_available(self) -> None:
        from macaw.extractors import get_backend

        self.backend()
        if self.config.text.diarization != SHARED_MODE:
            get_backend(DIARIZATION_GROUP, self.config.text.diarization)().check_available()

    def prefetch(self) -> None:
        cfg = self.config.text
        # the alignment model follows the language; English unless text.language names another
        self.backend().prefetch(language=cfg.language or "en", align=cfg.word_timestamps)

    def _shared(self) -> bool:
        return self.config.text.diarization == SHARED_MODE

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, str(backend.version))

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        corrected = layout.corrected_transcript_path(recording.recording_id)
        heard = layout.artifacts_dir(recording.recording_id, SHARED_ASR)
        shared = [heard / SHARED_UTTERANCES_NAME, heard / SHARED_WORDS_NAME, heard / SHARED_OUTSIDE_NAME] if self._shared() else []
        return [
            *voice_inputs(self.config, recording, load_prep(self.config, recording.recording_id)), *shared,
            *([corrected] if corrected.is_file() else []),
        ]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        prep = load_prep(self.config, recording.recording_id)
        backend = self.backend()
        warnings: list[QCWarning] = []
        self._dropped: list[pd.DataFrame] = []
        self._crosstalk: list[pd.DataFrame] = []
        self._room: list[pd.DataFrame] = []
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
        bleed = self._bleed_checks(recording.recording_id)
        # ASR times are on the prepared track, whose sample 0 sits at t_origin_s
        for frame in (utterances, words, dropped, bleed):
            frame["t_start"] = frame["t_start"] + prep.t_origin_s
            frame["t_end"] = frame["t_end"] + prep.t_origin_s

        artifacts = {
            TRANSCRIPT_NAME: write_transcript(utterances, workspace / TRANSCRIPT_NAME),
            WORDS_NAME: _io.write_file(words, workspace / WORDS_NAME, WORD_SCHEMA, "words"),
            DROPPED_NAME: _io.write_file(dropped, workspace / DROPPED_NAME, DROPPED_SCHEMA, "dropped utterances"),
            BLEED_NAME: _io.write_file(bleed, workspace / BLEED_NAME, BLEED_SCHEMA, "bleed checks"),
        }

        warnings += self._quality_warnings(recording, utterances, prep)
        return ExtractorResult(
            rows=self._rows(recording, utterances, prep),
            specs=self._specs(backend),
            artifacts=artifacts,
            warnings=warnings,
        )

    def _from_correction(self, recording: Recording, prep, warnings: list[QCWarning]):
        """The transcript a person corrected, on the prepared-track clock like a recogniser's and cut to the analysed part;
        no words."""
        from macaw.interop.transcript_eaf import read_transcript_eaf

        layout = layout_for(self.config, create=False)
        rid = recording.recording_id
        corrected = read_transcript_eaf(layout.corrected_transcript_path(rid), turn_takers(self.config, recording))
        zero, recognised = corrected.attrs.get("zero", 0.0), layout.recognised_transcript_path(rid)
        if zero > 0 and recognised.is_file():
            # the recogniser's lines before ELAN's time 0, which the file could not hold
            earlier = read_transcript(recognised)
            earlier = earlier.loc[earlier["t_start"] < zero, list(corrected.columns)]
            corrected = pd.concat([earlier, corrected], ignore_index=True).sort_values(["t_start", "subject_id"], ignore_index=True)
        utterances = corrected.assign(
            recording_id=rid,
            utterance_index=corrected.groupby("subject_id").cumcount().astype("int64"),
            t_start=corrected["t_start"] - prep.t_origin_s,
            t_end=corrected["t_end"] - prep.t_origin_s,
            confidence=np.nan,
            language=self.config.text.language,
        )
        utterances = _concat([utterances], TRANSCRIPT_SCHEMA, drop={"n_words"})
        utterances["n_words"] = utterances["text"].fillna("").astype(str).str.split().str.len().astype("int64")
        kept = cut_transcript(utterances, _concat([], WORD_SCHEMA), 0.0, prep.duration_s)
        key = ["subject_id", "utterance_index"]
        outside = utterances.merge(kept[key], on=key, how="left", indicator=True).query("_merge == 'left_only'")
        if len(outside):
            self._keep_dropped(outside.drop(columns="_merge").assign(
                reason="outside the analysed part", detail=f"the analysed part is {prep.duration_s:.1f} s long",
            ))
            warnings.append(QCWarning(
                check="asr_outside_track", recording_id=rid, subject_id=None, severity="info",
                message=f"{len(outside)} utterance(s) of the corrected transcript lie outside the analysed part and were left out",
            ))
        stamp_path = layout.corrected_transcript_stamp_path(rid)
        stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.is_file() else {}
        warnings.append(
            QCWarning(
                check=CORRECTED_CHECK, recording_id=rid, subject_id=None, severity="info",
                message=f"the transcript is the one {stamp.get('by', 'a person')} corrected; speech was not recognised again",
            )
        )
        return kept.drop(columns="n_words"), _concat([], WORD_SCHEMA)

    def _keep_dropped(self, frame: pd.DataFrame) -> None:
        self.__dict__.setdefault("_dropped", []).append(frame)

    def _bleed_checks(self, recording_id: str) -> pd.DataFrame:
        """The crosstalk check's numbers for each own-microphone utterance, joined to the room bleed check's where it ran."""
        key = ["subject_id", "utterance_index"]
        checked = _concat(self.__dict__.get("_crosstalk", []), BLEED_SCHEMA)
        room = self.__dict__.get("_room", [])
        if room:
            found = pd.concat(room, ignore_index=True)
            checked = checked.drop(columns=_ROOM_COLUMNS).merge(found[[*key, *_ROOM_COLUMNS, "room"]], on=key, how="left")
            checked.loc[checked["room"].eq(True).to_numpy(), "dropped_as"] = "room bleed"
        checked["recording_id"] = recording_id
        return checked.reindex(columns=list(BLEED_SCHEMA))

    def _from_shared_track(self, recording: Recording, prep, backend, warnings: list[QCWarning]):
        """One transcript of the shared microphone, each word given to the person whose voice it falls in."""
        rid = recording.recording_id
        utterances, words = read_shared_transcript(self.config, rid)
        outside = layout_for(self.config, create=False).artifacts_dir(rid, SHARED_ASR) / SHARED_OUTSIDE_NAME
        if outside.is_file():
            self._keep_dropped(pd.read_parquet(outside).drop(columns="recording_id"))
        turns = _track_clock_turns(self.config, rid, prep, changed=True)
        utterances, words, unplaced = split_by_voice(
            utterances, words, turns, speaker_names(self.config, rid), elsewhere=frozenset(prep.tracks)
        )
        if len(unplaced):
            self._keep_dropped(unplaced_runs(unplaced))
            warnings.append(
                QCWarning(
                    check="asr_unplaced_words", recording_id=rid, subject_id=None, severity="info",
                    message=f"{len(unplaced)} word(s) fell outside every named voice and were left out; the dropped utterances list them.",
                )
            )
        for frame in (utterances, words):
            frame["recording_id"] = rid
        return _concat([utterances], TRANSCRIPT_SCHEMA, drop={"n_words"}), _concat([words], WORD_SCHEMA)

    def _drop_room_bleed(self, recording: Recording, prep, utterances: pd.DataFrame, warnings: list[QCWarning]) -> pd.DataFrame:
        """A person heard only on the shared microphone still leaks into the others' own microphones.
        An utterance is their bleed when it lies mostly in their voice and the owner's is all but absent."""
        names = speaker_names(self.config, recording.recording_id)
        turns = _track_clock_turns(self.config, recording.recording_id, prep, exclusive=False, changed=True)
        turns = turns.assign(subject_id=turns["label"].map(lambda label: names.get(label)))
        roomers = set(turns["subject_id"].dropna()) - set(prep.tracks)
        flags = np.zeros(len(utterances), dtype=bool)
        voice = [None] * len(utterances)
        room_share, own_share = np.full(len(utterances), np.nan), np.full(len(utterances), np.nan)
        for index, row in enumerate(utterances.itertuples(index=False)):
            length = row.t_end - row.t_start
            if length <= 0:
                continue
            overlap = (np.minimum(turns["t_end"], row.t_end) - np.maximum(turns["t_start"], row.t_start)).clip(lower=0)
            own = float(overlap[turns["subject_id"] == row.subject_id].sum())
            in_room = overlap[turns["subject_id"].isin(roomers)].groupby(turns["subject_id"]).sum()
            other = float(in_room.sum())
            flags[index] = other >= _ROOM_BLEED_SHARE * length and own < _OWNER_ABSENT_SHARE * length
            voice[index] = str(in_room.idxmax()) if other > 0 else None
            room_share[index], own_share[index] = other / length, own / length
        measured = utterances.assign(room_voice=voice, room_voice_share=room_share, own_voice_share=own_share, room=flags)
        self.__dict__.setdefault("_room", []).append(measured)
        self._keep_dropped(
            measured[flags].drop(columns="room").assign(
                reason="room bleed", detail=[f"in the voice of {voice[index]}" for index in np.flatnonzero(flags)]
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
            if len(outside):
                self._keep_dropped(outside.assign(
                    subject_id=subject.subject_id, reason="outside track", detail=f"the prepared track ends at {prep.duration_s:.1f} s",
                ))
                warnings.append(
                    QCWarning(
                        check="asr_outside_track",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="info",
                        message=f"{len(outside)} utterance(s) the backend returned lie outside the prepared track and were dropped.",
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

        flagged = flagged.assign(**{column: flagged[column] if column in flagged else None for column in _CROSSTALK_COLUMNS})
        crosstalk = flagged["is_crosstalk"].astype(bool)
        self.__dict__.setdefault("_crosstalk", []).append(
            flagged.assign(dropped_as=np.where(crosstalk, "crosstalk", None)).reindex(columns=list(BLEED_SCHEMA))
        )
        bleed = flagged[crosstalk]
        self._keep_dropped(
            bleed.assign(
                reason="crosstalk",
                detail=[f"{by:.1f} dB louder on {on}" if isinstance(on, str) else "" for on, by in zip(bleed["louder_on"], bleed["louder_by_db"])],
            )
        )
        extra = ["is_crosstalk", *_CROSSTALK_COLUMNS]
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
                    details={
                        "crosstalk_margin_db": self.config.text.crosstalk_margin_db,
                        **({"gain_db": round(float(group["gain_db"].iloc[0]), 1), "gain_raised": bool(group["gain_raised"].iloc[0])}
                           if "gain_db" in group and pd.notna(group["gain_db"].iloc[0]) else {}),
                    },
                )
            )
        return warnings

    def _low_confidence(self) -> float | None:
        # read off the backend's class, so checking a segment later needs no recogniser installed
        from macaw.extractors import get_backend

        return get_backend(BACKEND_GROUP, self.config.text.asr_backend).low_confidence

    def _quality_warnings(self, recording: Recording, utterances, prep, span=None, where: str = "") -> list[QCWarning]:
        warnings = []
        span = span or (prep.t_origin_s, prep.t_origin_s + prep.duration_s)

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
                        message=f"No speech was transcribed for them{where}.",
                        t_start=span[0],
                        t_end=span[1],
                    )
                )
                continue
            confidence = float(mine["confidence"].mean())
            threshold = self._low_confidence()
            if threshold is not None and math.isfinite(confidence) and confidence < threshold:
                warnings.append(
                    QCWarning(
                        check="asr_confidence",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message=(
                            f"Mean recogniser confidence is {confidence:.2f}{where}. The transcript "
                            "is unlikely to be usable as text, though the timings may still be."
                        ),
                        t_start=span[0],
                        t_end=span[1],
                    )
                )

        languages = {str(value) for value in utterances["language"].dropna().unique()} if not where else set()
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
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        return concat_long_tables([self._events(recording, utterances), self._summary(recording, utterances, window)])

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        if not transcript_saved(self.config, recording.recording_id):
            return None
        utterances = transcript_between(self.config, recording.recording_id, start_s, end_s, others=True)
        return self._summary(recording, utterances, (start_s, end_s))

    def segment_warnings(self, recording: Recording, segment) -> list[QCWarning]:
        if not transcript_saved(self.config, recording.recording_id):
            return []
        utterances = transcript_between(self.config, recording.recording_id, segment.start_s, segment.end_s, others=True)
        prep = load_prep(self.config, recording.recording_id)
        return self._quality_warnings(recording, utterances, prep, (segment.start_s, segment.end_s), f" in segment {segment.name}")

    def _events(self, recording: Recording, utterances: pd.DataFrame) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
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
        return concat_long_tables(frames)

    def _summary(self, recording: Recording, utterances: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        length = window[1] - window[0]
        frames = []
        for subject_id, group in utterances.groupby("subject_id", dropna=False, observed=True):
            duration = (group["t_end"] - group["t_start"]).to_numpy(dtype=float)
            total_speech = float(duration.sum())
            total_words = float(group["n_words"].to_numpy(dtype=float).sum())
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [window[0]],
                            "t_end": [window[1]],
                            "n_utterances": [float(len(group))],
                            "total_speech_s": [total_speech],
                            "speech_fraction": [total_speech / length if length > 0 else np.nan],
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
        params = {"asr_model": backend.model, "language": self.config.text.language}
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
                missing=_MISSING[feature],
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
                missing=_MISSING.get(feature, _MISSING["summary"]),
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clips to [0, duration_s] and also returns the utterances wholly outside; a backend that reads an existing
    transcript can return the whole session."""
    keep = (utterances["t_start"] < duration_s) & (utterances["t_end"] > 0)
    outside = utterances[~keep].copy()
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


@register_extractor
class SharedASRExtractor(Extractor):
    """Speech on a shared microphone recognised before anyone names the voices, so naming can read what each voice says."""

    name = SHARED_ASR
    version = "0.4.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = (
        "text.asr_backend", "text.asr_model", "text.language", "text.device", "text.compute_type", "text.diarization",
        "segment", "prosody.fallback_chain", "prosody.frame_step_s", "prosody.f0_min_hz", "prosody.f0_max_hz",
    )
    requires = (MediaPrepExtractor.name, DiarizationExtractor.name)
    resource_profile = "gpu_standard"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return config.text.diarization == SHARED_MODE

    @classmethod
    def resources(cls, config) -> str:
        return cls.resource_profile if config.text.diarization == SHARED_MODE else Extractor.resource_profile

    def _shared(self) -> bool:
        return self.config.text.diarization == SHARED_MODE

    def check_available(self) -> None:
        if self._shared():
            ASRExtractor(self.config).backend()

    def prefetch(self) -> None:
        if self._shared():
            ASRExtractor(self.config).prefetch()

    def backend_info(self) -> tuple[str | None, str | None]:
        return ASRExtractor(self.config).backend_info() if self._shared() else (None, None)

    def backends(self) -> list[Backend]:
        return ASRExtractor(self.config).backends()[:1] if self._shared() else []

    def inputs_for(self, recording: Recording) -> list[Path]:
        if not self._shared():
            return []
        rid = recording.recording_id
        prep = load_prep(self.config, rid)
        diarized = layout_for(self.config, create=False).artifacts_dir(rid, DiarizationExtractor.name) / DIARIZATION_NAME
        return [prep.shared_track_path(shared_track(prep, rid)), diarized]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        if not self._shared():
            return ExtractorResult(rows=pd.DataFrame())
        rid = recording.recording_id
        prep = load_prep(self.config, rid)
        path = prep.shared_track_path(shared_track(prep, rid))
        utterances, words = ASRExtractor(self.config).backend().transcribe(
            path, language=self.config.text.language, word_timestamps=True
        )
        utterances, words, outside = within_track(utterances, words, prep.duration_s)
        warnings = [] if outside.empty else [
            QCWarning(
                check="asr_outside_track", recording_id=rid, subject_id=None, severity="info",
                message=f"{len(outside)} utterance(s) the backend returned lie outside the prepared track and were dropped.",
            )
        ]
        outside = outside.assign(
            recording_id=rid, subject_id=None, reason="outside track", detail=f"the prepared track ends at {prep.duration_s:.1f} s",
        )
        utterances = utterances.assign(recording_id=rid, subject_id=None, n_words=_word_counts(utterances))
        words = words.assign(recording_id=rid, subject_id=None)
        turns = _track_clock_turns(self.config, rid, prep)
        pitch = voice_pitch(self.config, path, turns)
        artifacts = {
            SHARED_UTTERANCES_NAME: write_transcript(_concat([utterances], TRANSCRIPT_SCHEMA), workspace / SHARED_UTTERANCES_NAME),
            SHARED_WORDS_NAME: _io.write_file(_concat([words], WORD_SCHEMA), workspace / SHARED_WORDS_NAME, WORD_SCHEMA, "words"),
            SHARED_OUTSIDE_NAME: _io.write_file(
                _concat([outside], DROPPED_SCHEMA), workspace / SHARED_OUTSIDE_NAME, DROPPED_SCHEMA, "dropped utterances"
            ),
            VOICE_LINES_NAME: write_transcript(voice_lines(utterances, words, turns, rid, prep.t_origin_s), workspace / VOICE_LINES_NAME),
        }
        (workspace / VOICE_PITCH_NAME).write_text(json.dumps(pitch, indent=2), encoding="utf-8")
        artifacts[VOICE_PITCH_NAME] = workspace / VOICE_PITCH_NAME
        return ExtractorResult(rows=pd.DataFrame(), artifacts=artifacts, warnings=warnings)


def _word_counts(utterances: pd.DataFrame) -> pd.Series:
    return utterances["text"].fillna("").astype(str).str.split().str.len().astype("int64")


def read_shared_transcript(config, recording_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shared microphone's speech as the recogniser gave it, on the track clock."""
    root = layout_for(config, create=False).artifacts_dir(recording_id, SHARED_ASR)
    utterances = read_transcript(root / SHARED_UTTERANCES_NAME).drop(columns=["recording_id", "subject_id"])
    words = read_words(root / SHARED_WORDS_NAME).drop(columns=["recording_id", "subject_id"])
    return utterances, words


def voice_lines(utterances: pd.DataFrame, words: pd.DataFrame, turns: pd.DataFrame, recording_id: str, t_origin_s: float) -> pd.DataFrame:
    """Each voice's lines on the recording clock, its label standing in for the person until someone names it."""
    if utterances.empty or turns.empty:
        return empty_frame(TRANSCRIPT_SCHEMA)
    labels = {str(label): str(label) for label in turns["label"].unique()}
    lines, _, _ = split_by_voice(
        utterances.drop(columns=["recording_id", "subject_id"]), words.drop(columns=["recording_id", "subject_id"]), turns, labels
    )
    if lines.empty:
        return empty_frame(TRANSCRIPT_SCHEMA)
    lines = lines.assign(
        recording_id=recording_id, n_words=_word_counts(lines),
        t_start=lines["t_start"] + t_origin_s, t_end=lines["t_end"] + t_origin_s,
    )
    return _concat([lines], TRANSCRIPT_SCHEMA)


def voice_pitch(config, path: Path, turns: pd.DataFrame) -> dict[str, dict]:
    """Per voice its speech time and median f0 over the turns it speaks alone; turns on the track clock."""
    from macaw.extractors.audio.prosody.frame import f0_contour

    cfg = config.prosody
    contour = f0_contour(path, cfg, f0_min=cfg.f0_min_hz, f0_max=cfg.f0_max_hz)
    t, f0 = contour["t_start"].to_numpy(dtype=float), contour["f0_hz"].to_numpy(dtype=float)
    out = {}
    for label, own in turns.groupby("label", observed=True):
        voiced = inside(t, own) & np.isfinite(f0) & (f0 > 0)
        out[str(label)] = {
            "speech_s": float((own["t_end"] - own["t_start"]).sum()),
            "median_hz": float(np.median(f0[voiced])) if voiced.any() else None,
        }
    return out


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


def load_transcript(config, recording_id: str, *, others: bool = False) -> pd.DataFrame:
    """Where every downstream text extractor starts, the way audio ones start at load_prep.
    OTHER's utterances only with others, for turn-taking."""
    from macaw.project.layout import layout_for

    root = layout_for(config, create=False).artifacts_dir(
        recording_id, ASRExtractor.name
    )
    transcript = read_transcript(root / TRANSCRIPT_NAME)
    return transcript if others else transcript[transcript["subject_id"] != OTHER].reset_index(drop=True)


def transcript_saved(config, recording_id: str) -> bool:
    return (layout_for(config, create=False).artifacts_dir(recording_id, ASRExtractor.name) / TRANSCRIPT_NAME).is_file()


def load_words(config, recording_id: str) -> pd.DataFrame:
    path = layout_for(config, create=False).artifacts_dir(recording_id, ASRExtractor.name) / WORDS_NAME
    return read_words(path) if path.is_file() else empty_frame(WORD_SCHEMA)


def transcript_between(config, recording_id: str, start_s: float, end_s: float, *, others: bool = False) -> pd.DataFrame:
    return cut_transcript(load_transcript(config, recording_id, others=others), load_words(config, recording_id), start_s, end_s)


def cut_transcript(transcript: pd.DataFrame, words: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    """The utterances within start_s to end_s; one crossing an edge is cut there and keeps the words whose middle falls inside."""
    starts, ends = transcript["t_start"].to_numpy(dtype=float), transcript["t_end"].to_numpy(dtype=float)
    out = transcript[(ends > start_s) & (starts < end_s)].copy()
    crossing = ((out["t_start"] < start_s) | (out["t_end"] > end_s)).to_numpy()
    emptied = np.zeros(len(out), dtype=bool)
    for position, index in enumerate(out.index):
        if not crossing[position]:
            continue
        text = " ".join(_words_inside(out.loc[index], words, start_s, end_s))
        emptied[position] = int(out.at[index, "n_words"]) > 0 and not text
        out.at[index, "text"] = text
        out.at[index, "n_words"] = len(text.split())
    out["t_start"] = out["t_start"].clip(lower=start_s)
    out["t_end"] = out["t_end"].clip(upper=end_s)
    return out[~emptied].reset_index(drop=True)


def _words_inside(utterance, words: pd.DataFrame, start_s: float, end_s: float) -> list[str]:
    own = (
        words[(words["subject_id"] == utterance["subject_id"]) & (words["utterance_index"] == utterance["utterance_index"])]
        if not words.empty else words
    )
    if not own.empty:
        middle = (own["t_start"].to_numpy(dtype=float) + own["t_end"].to_numpy(dtype=float)) / 2.0
        kept = own[(middle >= start_s) & (middle < end_s)].sort_values("t_start", kind="stable")
        return [str(word).strip() for word in kept["word"] if str(word).strip()]
    # without word times (a corrected transcript) the words are taken as evenly spread over the utterance
    tokens = utterance["text"].split() if isinstance(utterance["text"], str) else []
    if not tokens:
        return []
    step = (float(utterance["t_end"]) - float(utterance["t_start"])) / len(tokens)
    return [token for i, token in enumerate(tokens) if start_s <= float(utterance["t_start"]) + (i + 0.5) * step < end_s]


def _track_clock_turns(config, recording_id: str, prep, *, exclusive: bool = True, changed: bool = False) -> pd.DataFrame:
    turns = read_diarization(config, recording_id, changed=changed)
    return turns[turns["exclusive"].astype(bool) == exclusive].assign(
        t_start=lambda frame: frame["t_start"] - prep.t_origin_s, t_end=lambda frame: frame["t_end"] - prep.t_origin_s
    )


def split_by_voice(
    utterances: pd.DataFrame, words: pd.DataFrame, turns: pd.DataFrame, names: dict[str, str | None],
    *, elsewhere: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Each word goes to the voice turn it overlaps most, or the nearest within reach; a run of
    one person's words inside a recogniser segment is one utterance. Words of people in elsewhere
    are left to their own microphone. Also returns the words, or without word times the segments, no named voice took,
    with the voice each fell in as label."""
    columns = [column for column in utterances.columns if column not in {"utterance_index", "t_start", "t_end", "text", "confidence"}]
    if words.empty:
        # a backend without word times can only be split at its segments
        placed = assign_voices(utterances, turns)
        placed["subject_id"] = placed["label"].map(lambda label: names.get(label))
        unplaced = placed[placed["subject_id"].isna()].rename(columns={"text": "word"})
        placed = placed.drop(columns="label")
        kept = placed[placed["subject_id"].notna() & ~placed["subject_id"].isin(elsewhere)].copy()
        kept["utterance_index"] = kept.sort_values("t_start").groupby("subject_id").cumcount()
        return kept, words.assign(subject_id=pd.Series(dtype="string")), unplaced

    placed = assign_voices(words, turns)
    placed["subject_id"] = placed["label"].map(lambda label: names.get(label))
    unplaced = placed[placed["subject_id"].isna()]
    placed = placed.drop(columns="label")
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
    # no word may be left for anyone to keep, as when every named voice has a microphone of its own
    split = pd.DataFrame(rows, columns=["subject_id", "run", "t_start", "t_end", "text", "confidence", *columns])
    split["utterance_index"] = split.groupby("subject_id").cumcount()
    index = split.set_index("run")["utterance_index"]
    placed["utterance_index"] = placed["run"].map(index)
    return split.drop(columns=["run"]), placed.drop(columns=["run"]), unplaced


def unplaced_runs(unplaced: pd.DataFrame) -> pd.DataFrame:
    """Words no named voice took as dropped utterances, one per run in one recogniser segment and one voice or none."""
    if unplaced.empty:
        return pd.DataFrame(columns=list(DROPPED_SCHEMA))
    frame = unplaced.sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    label = np.array(["" if pd.isna(value) else str(value) for value in frame["label"]], dtype=object)
    segment = frame["utterance_index"].to_numpy()
    run = np.cumsum(np.r_[True, (label[1:] != label[:-1]) | (segment[1:] != segment[:-1])])
    rows = []
    for _, part in frame.groupby(run, sort=True):
        voice = label[part.index[0]]
        rows.append({
            "subject_id": None, "t_start": float(part["t_start"].min()), "t_end": float(part["t_end"].max()),
            "text": " ".join(part["word"].fillna("").astype(str)).strip(), "reason": "no voice",
            "detail": f"in voice {voice}, which was given to no one" if voice else "in none of the voices",
        })
    return pd.DataFrame(rows)


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
