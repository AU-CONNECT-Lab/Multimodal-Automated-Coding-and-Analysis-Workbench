from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from macaw.config.schema import TextSection
from macaw.extractors.audio.content import asr as asr_module
from macaw.extractors.audio.content.asr import ASRExtractor, Qwen3ASRBackend, joined_stretches, sentences_of, track_language


class TestPieces:
    def test_stretches_join_while_the_piece_stays_short_enough(self):
        assert joined_stretches([(0.0, 4.0), (6.0, 12.0), (20.0, 31.0), (33.0, 40.0)], 30.0) == [(0.0, 12.0), (20.0, 40.0)]

    def test_a_long_stretch_stands_alone(self):
        assert joined_stretches([(0.0, 5.0), (6.0, 36.0)], 30.0) == [(0.0, 5.0), (6.0, 36.0)]

    def test_no_speech_no_pieces(self):
        assert joined_stretches([]) == []


def word(text, start):
    return {"t_start": start, "t_end": start + 0.3, "word": text}


class TestSentences:
    def count(self, text):
        return len(text.split())

    def test_each_sentence_takes_its_own_words(self):
        words = [word("Hi", 0.0), word("there", 0.4), word("How", 2.0), word("are", 2.3), word("you", 2.6)]
        parts = sentences_of("Hi there. How are you?", words, self.count)
        assert [(text, [w["word"] for w in own]) for text, own in parts] == [("Hi there.", ["Hi", "there"]), ("How are you?", ["How", "are", "you"])]

    def test_words_that_do_not_add_up_keep_the_piece_whole(self):
        words = [word("Hi", 0.0), word("there", 0.4)]
        assert sentences_of("Hi there. Yes.", words, self.count) == [("Hi there. Yes.", words)]


class TestTranscribe:
    @pytest.fixture
    def track(self, tmp_path):
        path = tmp_path / "track.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(np.zeros(16000 * 40, dtype="<i2").tobytes())
        return path

    def test_pieces_become_sentences_with_their_word_times(self, track, monkeypatch):
        monkeypatch.setattr(asr_module, "speech_pieces", lambda samples: [(1.0, 5.0), (20.0, 26.0)])
        backend = Qwen3ASRBackend(None, device="cpu")
        monkeypatch.setattr(backend, "_recognise", lambda samples, pieces, language: [("Hello there. Fine.", "en"), ("Bye.", "en")])
        monkeypatch.setattr(backend, "_align", lambda samples, pieces, heard, name: [
            [word("Hello", 1.2), word("there", 1.6), word("Fine", 3.0)], [word("Bye", 21.0)]])
        backend._aligner = (SimpleNamespace(split_words_for_alignment=lambda text, language: text.replace(".", "").split()), None)
        lines, words = backend.transcribe(track, language="en", word_timestamps=True)
        assert lines["text"].tolist() == ["Hello there.", "Fine.", "Bye."]
        assert lines["t_start"].tolist() == [1.2, 3.0, 21.0] and lines["language"].tolist() == ["en"] * 3
        assert words.groupby("utterance_index")["word"].apply(list).tolist() == [["Hello", "there"], ["Fine"], ["Bye"]]

    def test_without_word_times_each_piece_is_one_line(self, track, monkeypatch):
        monkeypatch.setattr(asr_module, "speech_pieces", lambda samples: [(1.0, 5.0)])
        backend = Qwen3ASRBackend(None, device="cpu")
        monkeypatch.setattr(backend, "_recognise", lambda samples, pieces, language: [("Hello there. Fine.", "en")])
        lines, words = backend.transcribe(track, language=None, word_timestamps=False)
        assert lines[["t_start", "t_end", "text"]].values.tolist() == [[1.0, 5.0, "Hello there. Fine."]] and words.empty

    def test_silence_gives_empty_tables(self, track, monkeypatch):
        monkeypatch.setattr(asr_module, "speech_pieces", lambda samples: [])
        lines, words = Qwen3ASRBackend(None, device="cpu").transcribe(track, language="en", word_timestamps=True)
        assert lines.empty and words.empty and "utterance_index" in lines.columns


class TestTrackLanguage:
    @pytest.fixture
    def backend(self, monkeypatch):
        backend = Qwen3ASRBackend(None, device="cpu")
        monkeypatch.setattr(backend, "_known_languages", lambda: {"en", "hi", "nl"})
        return backend

    def test_the_language_of_most_speech_wins(self):
        pieces = [(0.0, 20.0), (20.0, 45.0), (50.0, 60.0)]
        assert track_language(pieces, [("a", "en"), ("b", "nl"), ("c", "en")]) == "en"
        assert track_language(pieces, [("a", "en"), ("", "nl"), ("c", "nl")]) == "en"
        assert track_language(pieces, [("", "en")] * 3) is None

    def test_pieces_heard_as_another_language_are_heard_again_in_the_tracks(self, backend, monkeypatch):
        calls = []

        def heard(samples, pieces, indices, language):
            calls.append((indices, language))
            return [("Hello. Morning.", "en"), ("जी।", "hi"), ("", "nl"), ("Fine.", "en")] if language is None else [("Yes.", "en")]

        monkeypatch.setattr(backend, "_heard", heard)
        pieces = [(0.0, 20.0), (21.0, 22.0), (23.0, 24.0), (30.0, 40.0)]
        assert backend._recognise(None, pieces, None) == [("Hello. Morning.", "en"), ("Yes.", "en"), ("", "en"), ("Fine.", "en")]
        assert calls == [([0, 1, 2, 3], None), ([1], "en")]

    def test_a_given_language_is_heard_once(self, backend, monkeypatch):
        calls = []
        monkeypatch.setattr(backend, "_heard", lambda samples, pieces, indices, language: calls.append(language) or [("Hi.", "en")])
        assert backend._recognise(None, [(0.0, 5.0)], "en") == [("Hi.", "en")] and calls == ["en"]

    def test_a_language_the_model_cannot_be_told_is_not_asked_for(self, backend, monkeypatch):
        calls = []

        def heard(samples, pieces, indices, language):
            calls.append(language)
            return [("iyog", "Tagalog"), ("ok", "en")]

        monkeypatch.setattr(backend, "_heard", heard)
        assert backend._recognise(None, [(0.0, 9.0), (10.0, 11.0)], None) == [("iyog", "Tagalog"), ("ok", "Tagalog")] and calls == [None]


class TestWhichModel:
    def backend(self, **text):
        return ASRExtractor(SimpleNamespace(text=TextSection(**text)))._named_backend()

    def test_qwen3_by_default_with_its_own_model(self):
        backend = self.backend()
        assert isinstance(backend, Qwen3ASRBackend) and backend.model == "Qwen/Qwen3-ASR-1.7B-hf"

    def test_whisper_backends_take_large_v3(self):
        assert self.backend(asr_backend="faster_whisper").model == "large-v3"

    def test_a_named_model_is_used(self):
        assert self.backend(asr_model="Qwen/Qwen3-ASR-0.6B-hf").model == "Qwen/Qwen3-ASR-0.6B-hf"


def test_a_whisper_model_name_is_refused_with_what_to_write():
    from macaw.exceptions import BackendNotAvailableError

    with pytest.raises(BackendNotAvailableError, match="large-v3 is a Whisper model; with text.asr_backend qwen3 leave text.asr_model empty"):
        Qwen3ASRBackend("large-v3").check_available()


def test_the_old_shared_microphone_value_says_what_to_write():
    with pytest.raises(ValueError, match="write text.diarization: shared, and text.diarizer: pyannote"):
        TextSection(diarization="pyannote")
