from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import TextSection
from macaw.extractors.audio.content import syntax as module
from macaw.extractors.audio.content.syntax import POS_TAGS, SyntaxExtractor, speaker_syntax, utterance_syntax
from macaw.extractors.prep import PrepInfo

spacy = pytest.importorskip("spacy")
if not spacy.util.is_package("en_core_web_sm"):
    pytest.skip("needs the en_core_web_sm parser", allow_module_level=True)


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm", exclude=["ner"])


def test_a_complement_clause_is_counted_and_punctuation_is_not_a_word(nlp):
    parsed = utterance_syntax(nlp("I think that we should go."))
    assert parsed["n_words"] == 6 and parsed["n_subordinate"] == 1
    assert parsed["n_dependencies"] == 5 and parsed["distance_mean"] > 1.0


def test_an_utterance_without_words_is_left_out(nlp):
    assert utterance_syntax(nlp("...")) is None


def test_a_speaker_with_nothing_parsed_has_no_values():
    out = speaker_syntax([])
    assert out["n_utterances_parsed"] == 0.0 and np.isnan(out["dependency_distance_mean"])


def test_shares_are_of_all_words(nlp):
    out = speaker_syntax([utterance_syntax(nlp(text)) for text in ("the dog runs", "yeah", "she quickly ate a big apple")])
    assert out["n_utterances_parsed"] == 3.0
    assert sum(out[f"pos_share_{tag.lower()}"] for tag in POS_TAGS) == pytest.approx(1.0)
    assert 0.0 < out["lexical_density"] < 1.0


def test_the_extractor_writes_one_row_per_speaker_and_measure(tmp_path, monkeypatch):
    transcript = pd.DataFrame({"subject_id": ["A", "B", "A"], "text": ["I think we should go.", "okay", "you know what I mean"]})
    monkeypatch.setattr(module, "load_transcript", lambda config, rid: transcript)
    monkeypatch.setattr(module, "load_prep", lambda config, rid: PrepInfo(recording_id="r", t_origin_s=0.0, duration_s=60.0, sample_rate=16000))
    recording = SimpleNamespace(recording_id="r", entities=lambda: {})
    result = SyntaxExtractor(SimpleNamespace(text=TextSection(language="de"))).run(recording, tmp_path)
    values = result.rows.set_index(["subject_id", "feature"])["value"]
    assert values[("A", "n_utterances_parsed")] == 2.0 and values[("B", "n_utterances_parsed")] == 1.0
    assert {spec.name for spec in result.specs} == set(result.rows["feature"])
    assert [warning.severity for warning in result.warnings] == ["info", "warn"]
