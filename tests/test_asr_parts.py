from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.extractors.audio.content.asr import (
    ASRExtractor,
    TRANSCRIPT_SCHEMA,
    WORD_SCHEMA,
    _concat,
    _keep_utterances,
    resolve_compute_type,
)
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame


@pytest.mark.parametrize(
    "compute_type, device, expected",
    [
        ("float16", "cpu", "int8"),   # ctranslate2 refuses to load float16 on CPU
        ("float16", "cuda", "float16"),
        ("int8", "cpu", "int8"),
    ],
)
def test_compute_type_follows_the_device(compute_type, device, expected):
    assert resolve_compute_type(compute_type, device) == expected


class TestWordsFollowUtterances:
    @pytest.fixture
    def utterances(self):
        return pd.DataFrame(
            [
                {"subject_id": speaker, "utterance_index": index}
                for speaker, index in
                [("parent", 0), ("parent", 1), ("parent", 2), ("child", 0), ("child", 1)]
            ]
        )

    @pytest.fixture
    def words(self):
        return pd.DataFrame(
            [
                {"subject_id": speaker, "utterance_index": index, "word": f"{speaker}{index}"}
                for speaker, index in
                [("parent", 0), ("parent", 1), ("parent", 1), ("parent", 2), ("child", 0), ("child", 1)]
            ]
        )

    def test_dropped_utterances_take_their_words(self, utterances, words):
        kept = utterances.drop(index=[1, 3]).reset_index(drop=True)   # crosstalk removed these
        out = _keep_utterances(words, kept)
        assert sorted(out.word) == ["child1", "parent0", "parent2"]

    def test_indices_are_not_renumbered(self, utterances, words):
        """The two parquet files join on utterance_index, so gaps have to stay gaps."""
        kept = utterances.drop(index=[1, 3]).reset_index(drop=True)
        out = _keep_utterances(words, kept)
        assert sorted(zip(out.subject_id, out.utterance_index)) == [
            ("child", 1), ("parent", 0), ("parent", 2)
        ]

    def test_same_index_on_two_speakers_does_not_cross(self, utterances, words):
        out = _keep_utterances(words, utterances[utterances.subject_id == "child"])
        assert set(out.subject_id) == {"child"}

    def test_empty_words(self, utterances, words):
        assert len(_keep_utterances(words.iloc[:0], utterances)) == 0


class TestEmptyTranscript:
    def test_utterance_columns_survive(self):
        columns = _concat([], TRANSCRIPT_SCHEMA, drop={"n_words"}).columns
        assert list(columns) == [name for name in TRANSCRIPT_SCHEMA if name != "n_words"]

    def test_word_columns_survive(self):
        assert list(_concat([], WORD_SCHEMA).columns) == list(WORD_SCHEMA)


class TestOtherRecognisers:
    def test_columns_only_faster_whisper_gives_are_left_empty(self):
        from_csv = pd.DataFrame({"utterance_index": [0], "t_start": [1.0], "t_end": [2.0], "text": ["hi"],
                                 "confidence": [np.nan], "language": ["en"], "recording_id": ["r"], "subject_id": ["a"]})
        out = _concat([from_csv], TRANSCRIPT_SCHEMA, drop={"n_words"})
        assert list(out.columns) == [name for name in TRANSCRIPT_SCHEMA if name != "n_words"]
        assert out[["avg_logprob", "no_speech_prob"]].isna().all().all()


class TestLongTableRows:
    @pytest.fixture
    def extractor(self):
        text = SimpleNamespace(asr_model="large-v3", language=None)
        return ASRExtractor(SimpleNamespace(text=text))

    @pytest.fixture
    def rows(self, extractor, make_recording, make_prep):
        transcript = pd.DataFrame(
            [
                {"subject_id": "parent", "t_start": 0.0, "t_end": 2.0, "n_words": 4, "confidence": 0.9},
                {"subject_id": "parent", "t_start": 5.0, "t_end": 9.0, "n_words": 8, "confidence": 0.7},
            ]
        )
        return extractor._rows(
            make_recording(["parent"]), transcript, make_prep(duration_s=120.0)
        )

    def test_schema(self, rows):
        validate_frame(rows, LONG_TABLE_SCHEMA, "long table")

    def test_manifest_and_rows_agree(self, extractor, rows):
        backend = SimpleNamespace(name="fake", version="0")
        assert {spec.name for spec in extractor._specs(backend)} == set(rows.feature)

    def test_utterance_rate(self, rows):
        rates = rows[rows.feature == "speech_rate_wps"].value
        assert list(rates) == pytest.approx([2.0, 2.0])

    def test_summaries(self, rows):
        def value(feature):
            return float(rows[rows.feature == feature].value.iloc[0])

        assert value("n_utterances") == pytest.approx(2.0)
        assert value("total_speech_s") == pytest.approx(6.0)
        assert value("speech_fraction") == pytest.approx(6.0 / 120.0)
        assert value("mean_utterance_s") == pytest.approx(3.0)

    def test_words_per_minute_counts_speaking_time_only(self, rows):
        """Not words per minute of the session: 12 words over 6 s of speech is 120."""
        assert float(rows[rows.feature == "words_per_minute"].value.iloc[0]) == pytest.approx(120.0)


class TestLowConfidence:
    """Each backend's confidence has its own scale, so the warning threshold is the backend's."""

    def warnings(self, low_confidence):
        extractor = ASRExtractor(SimpleNamespace(text=SimpleNamespace()))
        extractor.backend = lambda: SimpleNamespace(low_confidence=low_confidence)
        recording = SimpleNamespace(recording_id="r", subjects=[SimpleNamespace(subject_id="parent")])
        prep = SimpleNamespace(t_origin_s=0.0, duration_s=60.0, tracks={"parent": "parent.wav"})
        utterances = pd.DataFrame([{"subject_id": "parent", "confidence": 0.41, "language": "en"}])
        return [warning.check for warning in extractor._quality_warnings(recording, utterances, prep)]

    def test_below_the_backends_threshold_warns(self):
        assert self.warnings(0.5) == ["asr_confidence"]

    def test_a_backend_without_a_measured_threshold_does_not(self):
        assert self.warnings(None) == []
