from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import TextSection
from macaw.exceptions import BackendNotAvailableError, InputNotFoundError
from macaw.extractors.audio.content import features as module
from macaw.extractors.audio.content.features import (
    EMBEDDINGS_NAME,
    TextFeatureExtractor,
    excluded_words,
    first_use_share,
    semantic_following,
    tokenize,
    type_token_ratio,
    unigram_entropy,
)
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame


def transcript(rows):
    frame = pd.DataFrame(rows, columns=["subject_id", "t_start", "text"])
    frame["t_end"] = frame["t_start"] + 1.0
    frame["utterance_index"] = frame.groupby("subject_id").cumcount()
    return frame


def test_tokens_are_lower_case_words_without_punctuation():
    assert tokenize("Okay, 'Kay. I'm the kick-off guy?") == ["okay", "kay", "i'm", "the", "kick", "off", "guy"]


class TestDiversity:
    def test_type_token_ratio(self):
        assert type_token_ratio(["a", "b", "a", "a"]) == 0.5
        assert np.isnan(type_token_ratio([]))

    def test_entropy_is_in_bits(self):
        assert unigram_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)
        assert unigram_entropy(["a", "a"]) == pytest.approx(0.0)


class TestFirstUse:
    def test_shared_words_go_to_whoever_said_them_first(self):
        spoken = transcript([("a", 0.0, "the red box"), ("b", 2.0, "the red one"), ("a", 4.0, "one more")])
        share = first_use_share(spoken.assign(tokens=spoken.text.map(tokenize)))
        # shared: the, red (a first), one (b first)
        assert share == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}

    def test_with_no_shared_words_nobody_has_a_share(self):
        spoken = transcript([("a", 0.0, "hello"), ("b", 1.0, "goodbye")])
        share = first_use_share(spoken.assign(tokens=spoken.text.map(tokenize)))
        assert np.isnan(share["a"]) and np.isnan(share["b"])

    def test_excluded_words_and_a_word_list_narrow_what_counts(self):
        spoken = transcript([("a", 0.0, "the red box"), ("b", 2.0, "the red box"), ("c", 4.0, "red")])
        tokens = spoken.assign(tokens=spoken.text.map(tokenize))
        assert first_use_share(tokens, exclude=frozenset({"the"}))["a"] == pytest.approx(1.0)
        assert first_use_share(tokens, min_speakers=3) == {"a": 1.0, "b": 0.0, "c": 0.0}
        assert np.isnan(first_use_share(tokens, only=frozenset({"green"}))["a"])

    def test_the_shipped_stopwords_drop_function_words_and_fillers_but_not_content(self):
        stopwords = excluded_words("english_stopwords")
        assert {"the", "i'm", "don't", "um", "yeah"} <= stopwords
        assert not {"remote", "button", "design"} & stopwords
        assert excluded_words("none") == frozenset()


def run(monkeypatch, tmp_path, rows, **text):
    monkeypatch.setattr(module, "load_transcript", lambda config, rid, others=False: transcript(rows))
    monkeypatch.setattr(module, "load_prep", lambda config, rid: SimpleNamespace(t_origin_s=0.0, duration_s=60.0))
    recording = SimpleNamespace(recording_id="r1", entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None})
    extractor = TextFeatureExtractor(SimpleNamespace(text=TextSection(**text)))
    return extractor, extractor.run(recording, tmp_path)


def test_the_extractor_writes_one_row_per_speaker_and_measure(monkeypatch, tmp_path):
    _, result = run(monkeypatch, tmp_path, [("a", 0.0, "the red box"), ("b", 2.0, "the red one ok"), ("a", 4.0, "one")])
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    assert {spec.name for spec in result.specs} == set(result.rows.feature)
    values = result.rows.set_index(["subject_id", "feature"]).value
    assert values["a", "mlu_words"] == 2.0 and values["b", "mlu_words"] == 4.0
    assert values["a", "n_types"] == 4.0
    assert result.artifacts == {}


def test_stopwords_are_dropped_by_default(monkeypatch, tmp_path):
    # without the stopword list "the" would give b no share at all; with it b brought in "box"
    _, result = run(monkeypatch, tmp_path, [("a", 0.0, "the red"), ("b", 2.0, "box the"), ("a", 4.0, "red box")])
    values = result.rows.set_index(["subject_id", "feature"]).value
    assert values["b", "first_use_share"] == 1.0
    assert result.specs[0].params["first_use_exclude"] == "english_stopwords"


def test_a_word_list_file_is_read_and_goes_into_the_cache_key(monkeypatch, tmp_path):
    words = tmp_path / "topic.txt"
    words.write_text("# task vocabulary\nRed\n", encoding="utf-8")
    extractor, result = run(monkeypatch, tmp_path, [("a", 0.0, "box red"), ("b", 2.0, "box red")], first_use_words=words)
    values = result.rows.set_index(["subject_id", "feature"]).value
    assert values["a", "first_use_share"] == 1.0
    assert words in extractor._word_lists()
    assert result.specs[0].params["first_use_words"] == "topic.txt"


def test_a_missing_word_list_stops_before_the_run(tmp_path):
    extractor = TextFeatureExtractor(SimpleNamespace(text=TextSection(first_use_exclude=tmp_path / "gone.txt")))
    with pytest.raises(InputNotFoundError, match="gone.txt"):
        extractor.check_available()


def test_embeddings_need_their_package_only_when_asked_for(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    TextFeatureExtractor(SimpleNamespace(text=TextSection())).check_available()
    with pytest.raises(BackendNotAvailableError, match="sentence-transformers"):
        TextFeatureExtractor(SimpleNamespace(text=TextSection(embedding_model="any"))).check_available()


def test_embeddings_are_one_row_per_utterance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module, "sentence_embeddings",
        lambda texts, model, device: pd.DataFrame(np.ones((len(texts), 3), dtype=np.float32), columns=["e0", "e1", "e2"]),
    )
    _, result = run(monkeypatch, tmp_path, [("a", 0.0, "hi"), ("b", 1.0, "hello")], embedding_model="any")
    table = pd.read_parquet(result.artifacts[EMBEDDINGS_NAME])
    assert list(table.columns) == ["subject_id", "utterance_index", "t_start", "t_end", "e0", "e1", "e2"]
    assert len(table) == 2


class TestSemanticFollowing:
    def test_each_direction_is_its_own_mean(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        found = semantic_following(["a", "b", "a", "b"], vectors)
        assert found[("b", "a")] == pytest.approx((1.0 + np.sqrt(0.5)) / 2)
        assert found[("a", "b")] == pytest.approx(0.0)

    def test_a_speaker_going_on_follows_no_one(self):
        assert semantic_following(["a", "a"], np.ones((2, 2))) == {}

    def test_a_line_not_measured_breaks_the_chain(self):
        vectors = np.array([[1.0, 0.0], [np.nan, np.nan], [1.0, 0.0]])
        assert semantic_following(["a", "other", "b"], vectors) == {}

    def test_an_empty_line_says_nothing(self):
        assert semantic_following(["a", "b"], np.array([[0.0, 0.0], [1.0, 0.0]])) == {}


def test_following_is_a_row_per_pair_when_utterances_are_embedded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module, "sentence_embeddings",
        lambda texts, model, device: pd.DataFrame([[1.0, 0.0] if "red" in text else [0.0, 1.0] for text in texts], columns=["e0", "e1"]),
    )
    _, result = run(monkeypatch, tmp_path, [("a", 0.0, "the red box"), ("b", 2.0, "red one"), ("a", 4.0, "ok then")], embedding_model="any")
    following = result.rows[result.rows.feature == "semantic_following"]
    assert dict(zip(zip(following.subject_id, following.subject_id_b), following.value)) == {("b", "a"): 1.0, ("a", "b"): 0.0}
    assert {spec.name for spec in result.specs} == set(result.rows.feature)
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
