import importlib.metadata

import pytest

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors.audio.content.asr import WhisperXBackend
from macaw.extractors.audio.content.diarization import PyannoteBackend, refuse_older


def _installed(monkeypatch, versions: dict[str, str]) -> None:
    def version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)


def test_whisperx_before_pyannote_4_is_refused_by_name(monkeypatch):
    _installed(monkeypatch, {"whisperx": "3.7.2"})
    backend = WhisperXBackend.__new__(WhisperXBackend)
    with pytest.raises(BackendNotAvailableError, match=r"whisperx 3\.7\.2 is installed; MACAW needs 3\.8 or later"):
        backend.check_available()


def test_pyannote_3_is_refused_before_its_import_fails(monkeypatch):
    _installed(monkeypatch, {"pyannote.audio": "3.4.0"})
    with pytest.raises(BackendNotAvailableError, match=r"pyannote\.audio 3\.4\.0 is installed; MACAW needs 4\.0 or later"):
        PyannoteBackend().check_available()


@pytest.mark.parametrize("installed", ["3.8.4", "3.8.7rc1", "4.1"])
def test_a_release_at_or_past_the_floor_passes(monkeypatch, installed):
    _installed(monkeypatch, {"whisperx": installed})
    refuse_older("whisperx", (3, 8), "reason")


def test_a_missing_package_is_left_to_the_import(monkeypatch):
    _installed(monkeypatch, {})
    refuse_older("whisperx", (3, 8), "reason")


def test_the_word_timing_model_is_loaded_once_per_language(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np
    import soundfile

    from macaw.extractors.audio.content.asr import WhisperXBackend

    loaded = []
    segment = {"start": 0.0, "end": 1.0, "text": "hi", "words": [{"word": "hi", "start": 0.1, "end": 0.4, "score": 0.9}]}
    fake = SimpleNamespace(
        load_model=lambda *args, **kwargs: SimpleNamespace(transcribe=lambda audio, batch_size, language: {"language": language, "segments": [dict(segment)]}),
        load_align_model=lambda language_code, device: loaded.append(language_code) or ("aligner", {}),
        align=lambda segments, model, metadata, audio, device, return_char_alignments: {"segments": [dict(segment)]},
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    track = tmp_path / "track.wav"
    soundfile.write(track, np.zeros(16000, dtype=np.float32), 16000)
    backend = WhisperXBackend("large-v3", device="cpu")
    for language in ("en", "en", "de"):
        _, words = backend.transcribe(track, language=language, word_timestamps=True)
        assert words["word"].tolist() == ["hi"]
    assert loaded == ["en", "de"]


def test_a_word_the_aligner_gives_no_time_is_kept_where_the_word_before_it_ends(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np
    import soundfile

    from macaw.extractors.audio.content.asr import WhisperXBackend

    said = [{"word": "I", "start": 0.1, "end": 0.2, "score": 0.9}, {"word": "have", "start": 0.3, "end": 0.5, "score": 0.9},
            {"word": "3"}, {"word": "apples", "start": 0.7, "end": 1.0, "score": 0.8}]
    segment = {"start": 0.0, "end": 1.0, "text": "I have 3 apples", "words": said}
    fake = SimpleNamespace(
        load_model=lambda *args, **kwargs: SimpleNamespace(transcribe=lambda audio, batch_size, language: {"language": language, "segments": [dict(segment)]}),
        load_align_model=lambda language_code, device: ("aligner", {}),
        align=lambda segments, model, metadata, audio, device, return_char_alignments: {"segments": [dict(segment)]},
    )
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    track = tmp_path / "track.wav"
    soundfile.write(track, np.zeros(16000, dtype=np.float32), 16000)
    _, words = WhisperXBackend("large-v3", device="cpu").transcribe(track, language="en", word_timestamps=True)
    assert " ".join(words["word"]) == "I have 3 apples"
    three = words[words["word"] == "3"].iloc[0]
    assert three["t_start"] == three["t_end"] == 0.5 and np.isnan(three["score"])
