from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Backend, Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class ASRBackend(Backend):
    def transcribe(self, audio_path: Path, *, language: str | None, word_timestamps: bool) -> pd.DataFrame:
        """Returns segments with t_start, t_end, text, and words when requested."""
        raise NotImplementedError


class WhisperXBackend(ASRBackend):
    name = "whisperx"

    def check_available(self) -> None:
        raise NotImplementedError


class FasterWhisperBackend(ASRBackend):
    name = "faster_whisper"

    def check_available(self) -> None:
        raise NotImplementedError


class ASRExtractor(Extractor):
    name = "text.asr"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "text"
    config_sections = ("text", "segment")

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def write_transcript(df: pd.DataFrame, path: Path) -> Path:
    raise NotImplementedError


def read_transcript(path: Path) -> pd.DataFrame:
    raise NotImplementedError
