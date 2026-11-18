from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.coding.llm.rate import part_lines, rating_lines
from macaw.config.schema import TextSection, TurnsSection
from macaw.extractors.video.pose.listening import alone_spans
from macaw.turns import talk as talk_module
from macaw.turns.segmentation import mark_within, talk_to_turns, utterances_to_turns
from macaw.turns.talk import said_units, talk_spans, voice_gaps
from macaw.turns.transitions import transitions_from

SHARED = SimpleNamespace(text=TextSection(diarization="pyannote"), turns=TurnsSection())


def words(rows):
    """(speaker, start, end, word, utterance) as text.asr gives words on a shared microphone."""
    return pd.DataFrame(rows, columns=["subject_id", "t_start", "t_end", "word", "utterance_index"])


def transcript(rows):
    return pd.DataFrame(
        [{"subject_id": who, "t_start": a, "t_end": b, "text": text, "n_words": len(text.split()), "utterance_index": i} for i, (who, a, b, text) in enumerate(rows)]
    )


@pytest.fixture
def voices(monkeypatch):
    """A diarization that lets voices overlap: the parent talks 0 to 6 and 6.3 to 10, the child murmurs 4 to 4.4 inside it,
    a 0.1 s crackle of the child's at 7, a 0.15 s child voice at 8 holding a word, and the child answers 10.3 to 13."""
    regular = pd.DataFrame(
        [
            ("SPEAKER_00", 0.0, 6.0), ("SPEAKER_00", 6.3, 10.0), ("SPEAKER_01", 4.0, 4.4), ("SPEAKER_01", 7.0, 7.1),
            ("SPEAKER_01", 8.0, 8.15), ("SPEAKER_01", 10.3, 13.0),
        ],
        columns=["label", "t_start", "t_end"],
    ).assign(exclusive=False)
    monkeypatch.setattr(talk_module, "voices_time_talk", lambda config, rid: True)
    monkeypatch.setattr(talk_module, "read_diarization", lambda config, rid, changed=False: regular)
    monkeypatch.setattr(talk_module, "speaker_names", lambda config, rid: {"SPEAKER_00": "parent", "SPEAKER_01": "child"})
    monkeypatch.setattr(talk_module, "load_prep", lambda config, rid: SimpleNamespace(tracks={}))
    said = words(
        [
            ("parent", 0.5, 1.0, "so", 0), ("parent", 1.0, 5.5, "tell", 0), ("parent", 6.5, 9.5, "me", 1),
            ("child", 8.02, 8.12, "no", 0), ("child", 10.4, 11.0, "people", 1), ("child", 11.0, 12.0, "stopping", 1),
            ("parent", 14.0, 14.3, "really", 2),
        ]
    )
    lines = transcript(
        [("parent", 0.5, 5.5, "so tell"), ("parent", 6.5, 9.5, "me"), ("child", 8.02, 8.12, "no"), ("child", 10.4, 12.0, "people stopping"), ("parent", 14.0, 14.3, "really")]
    )
    return said, lines


class TestTalkSpans:
    def test_voices_time_the_talk_and_noise_is_left_out(self, voices):
        said, lines = voices
        talk, left_out = talk_spans(SHARED, "r1", lines, said)
        child = talk[talk["subject_id"] == "child"]
        # the 0.1 s crackle goes; the 0.15 s voice stays since a word was heard in it; the murmur inside the parent's talk stays
        assert left_out == 1
        assert list(zip(child["t_start"], child["t_end"])) == [(4.0, 4.4), (8.0, 8.15), (10.3, 13.0)]
        # a word in none of its speaker's stretches stands as its own
        assert ((talk["subject_id"] == "parent") & (talk["t_start"] == 14.0)).any()

    def test_without_a_shared_microphone_the_utterances_time_it(self, voices, monkeypatch):
        monkeypatch.setattr(talk_module, "voices_time_talk", lambda config, rid: False)
        talk, left_out = talk_spans(SHARED, "r1", voices[1], voices[0])
        assert left_out == 0 and len(talk) == len(voices[1])


class TestTurnsFromVoices:
    def test_turns_hold_their_words_and_a_murmur_is_a_turn_without_any(self, voices):
        said, lines = voices
        talk, _ = talk_spans(SHARED, "r1", lines, said)
        turns = talk_to_turns(talk, said_units(lines, said), TurnsSection()).set_index(["subject_id", "t_start"])
        assert turns.loc[("parent", 0.0), "t_end"] == 10.0 and turns.loc[("parent", 0.0), "text"] == "so tell me"
        assert turns.loc[("child", 4.0), "n_words"] == 0 and turns.loc[("child", 4.0), "text"] == ""
        assert bool(turns.loc[("child", 4.0), "within"]) and bool(turns.loc[("child", 8.0), "within"])
        assert turns.loc[("child", 10.3), "text"] == "people stopping" and not bool(turns.loc[("child", 10.3), "within"])

    def test_the_murmur_is_a_backchannel_and_the_answer_a_gap(self, voices):
        said, lines = voices
        talk, _ = talk_spans(SHARED, "r1", lines, said)
        found = transitions_from(talk_to_turns(talk, said_units(lines, said), TurnsSection()), TurnsSection())
        assert list(zip(found["to_subject"], found["transition_type"])) == [
            ("child", "backchannel"), ("child", "backchannel"), ("child", "gap"), ("parent", "gap"),
        ]
        handover = found[found["transition_type"] == "gap"].iloc[0]
        assert handover["fto_s"] == pytest.approx(0.3) and bool(handover["is_switch"])


class TestWithin:
    def test_a_yeah_inside_a_long_utterance_leaves_the_floor_and_the_answer_after_it_is_the_handover(self, make_transcript):
        """The yeah leaves the floor with A, and the answer after A stops is the handover."""
        lines = make_transcript([("A", 0, 6, "a b c"), ("B", 4, 4.4, "yeah"), ("A", 6.3, 10, "d e"), ("B", 10.3, 13, "f g")])
        found = transitions_from(utterances_to_turns(lines, TurnsSection()), TurnsSection())
        assert list(found["transition_type"]) == ["backchannel", "gap"]
        assert found["fto_s"].tolist() == pytest.approx([-0.4, 0.3])
        assert found["is_switch"].tolist() == [False, True]

    def test_longer_talk_inside_a_turn_is_an_attempt_and_overlap_taking_the_floor_an_interruption(self, make_turns):
        turns = make_turns([("A", 0, 10), ("B", 2, 4.5), ("B", 9, 12)])
        found = transitions_from(turns, TurnsSection())
        assert list(found["transition_type"]) == ["attempt", "interruption"]
        assert bool(found["within"].iloc[0]) and not bool(found["within"].iloc[1])

    def test_stopping_together_is_not_inside(self, make_turns):
        assert not mark_within(make_turns([("A", 0, 2), ("B", 1.8, 2.0)]))["within"].any()


class TestTimingShownToTheModel:
    def test_a_line_said_over_the_talk_before_overlaps_it(self):
        lines = pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [0.0, 5.6], "t_end": [5.4, 7.0]})
        talk = pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [0.0, 5.2], "t_end": [6.0, 7.2]})
        assert voice_gaps(lines, talk)[1] == pytest.approx(-0.8)
        assert np.isnan(voice_gaps(lines, talk)[0])

    def test_a_line_is_not_widened_past_its_speaker_s_next(self):
        lines = pd.DataFrame({"subject_id": ["parent", "parent", "child"], "t_start": [0.0, 3.0, 4.5], "t_end": [2.0, 4.0, 5.0]})
        talk = pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [0.0, 4.5], "t_end": [6.0, 5.0]})
        gaps = voice_gaps(lines, talk)
        assert gaps[1] == pytest.approx(0.0) and gaps[2] == pytest.approx(-1.5)

    def test_the_rating_shows_the_overlap_from_the_start_of_the_part(self):
        lines = rating_lines(pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [10.0, 15.6], "t_end": [15.4, 17.0], "text": ["so", "no"]}))
        talk = pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [10.0, 15.2], "t_end": [16.0, 17.2]})
        shown = part_lines(lines, pd.DataFrame(), 10.0, 20.0, talk)
        assert shown["t_start"].tolist() == [0.0, 5.6] and shown["gap_before"].iloc[1] == pytest.approx(-0.8)
        assert part_lines(lines, pd.DataFrame(), 10.0, 20.0)["gap_before"].iloc[1] == pytest.approx(0.2)


def test_listening_leaves_out_the_listener_s_own_murmur():
    talk = pd.DataFrame({"subject_id": ["parent", "child"], "t_start": [0.0, 4.0], "t_end": [6.0, 4.4]})
    assert alone_spans(talk, 0.0, 6.0) == [("parent", (0.0, 4.0)), ("parent", (4.4, 6.0))]


def test_the_methods_say_how_talk_was_timed_and_what_talk_inside_a_turn_is():
    from macaw.qc.boilerplate import _transitions, _turns

    config = SimpleNamespace(turns=TurnsSection())
    assert "allows two voices at once" in _turns(config, SimpleNamespace(params={"talk_timed_by": "voices"}), set())
    assert "allows two voices" not in _turns(config, SimpleNamespace(params={"talk_timed_by": "utterances"}), set())
    assert "attempt to take the floor" in _transitions(config, SimpleNamespace(params={}), set())
