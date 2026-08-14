from __future__ import annotations

import pytest

from macaw.config.schema import TurnsSection
from macaw.turns.segmentation import dominance, silences, utterances_to_turns


@pytest.fixture
def cfg():
    return TurnsSection()


class TestMerging:
    def test_short_gap_merges(self, make_transcript, cfg):
        turns = utterances_to_turns(make_transcript([("A", 0, 1, "one"), ("A", 1.5, 2, "two")]), cfg)
        assert len(turns) == 1
        assert (turns.t_start[0], turns.t_end[0]) == (0.0, 2.0)
        assert turns.text[0] == "one two"
        assert turns.n_words[0] == 2
        assert turns.n_utterances[0] == 2

    def test_long_gap_splits(self, make_transcript, cfg):
        turns = utterances_to_turns(make_transcript([("A", 0, 1, "a"), ("A", 4, 5, "b")]), cfg)
        assert len(turns) == 2

    def test_interruption_in_the_gap_splits(self, make_transcript, cfg):
        turns = utterances_to_turns(
            make_transcript([("A", 0, 1, "a1"), ("B", 1.2, 1.4, "b1"), ("A", 1.6, 2.5, "a2")]), cfg
        )
        assert list(turns.subject_id) == ["A", "B", "A"]

    def test_self_overlap_merges(self, make_transcript, cfg):
        turns = utterances_to_turns(make_transcript([("A", 0, 2, "a1"), ("A", 1.5, 3, "a2")]), cfg)
        assert len(turns) == 1
        assert turns.t_end[0] == 3.0

    def test_simultaneous_speech_does_not_split(self, make_transcript, cfg):
        """B speaks while A is talking and stops before A's pause: A never lost the floor."""
        turns = utterances_to_turns(
            make_transcript([("A", 0, 1, "a1"), ("B", 0.5, 0.8, "mm"), ("A", 1.5, 2, "a2")]), cfg
        )
        assert (turns.subject_id == "A").sum() == 1

    def test_speech_carrying_into_the_gap_splits(self, make_transcript, cfg):
        """B started before A's pause but is still going during it, so the floor did change."""
        turns = utterances_to_turns(
            make_transcript([("A", 0, 1, "a1"), ("B", 0.9, 1.9, "b1"), ("A", 2.0, 3, "a2")]), cfg
        )
        assert (turns.subject_id == "A").sum() == 2

    def test_empty_transcript(self, make_transcript, cfg):
        turns = utterances_to_turns(make_transcript([]), cfg)
        assert len(turns) == 0
        assert "turn_index" in turns.columns


class TestJointSilence:
    def test_gap_bounds_and_threshold(self, make_transcript):
        quiet = silences(
            make_transcript([("A", 0, 1, "a"), ("B", 3, 4, "b"), ("A", 4.5, 5, "a")]), min_s=1.0
        )
        assert len(quiet) == 1
        assert (quiet.t_start[0], quiet.t_end[0]) == (1.0, 3.0)

    def test_gap_exactly_min_s_counts(self, make_transcript):
        quiet = silences(make_transcript([("A", 0, 1, "a"), ("B", 2, 3, "b")]), min_s=1.0)
        assert list(quiet.duration_s) == [1.0]

    def test_overlapping_talk_is_one_block(self, make_transcript):
        assert len(silences(make_transcript([("A", 0, 2, "a"), ("B", 1, 5, "b")]), min_s=1.0)) == 0

    def test_leading_and_trailing_silence_excluded(self, make_transcript):
        quiet = silences(make_transcript([("A", 2, 3, "a"), ("B", 7, 8, "b")]), min_s=1.0)
        assert list(quiet.t_start) == [3.0]

    def test_must_be_fed_utterances_not_turns(self, make_transcript, cfg):
        """A pause inside a turn is still nobody talking, and turns hide it by construction:
        max_gap_s is 2 s while silence_min_s is 1 s."""
        utterances = make_transcript([("A", 0, 1, "a1"), ("A", 2.5, 3.5, "a2")])
        turns = utterances_to_turns(utterances, cfg)

        assert len(silences(turns, min_s=cfg.silence_min_s)) == 0
        found = silences(utterances, min_s=cfg.silence_min_s)
        assert len(found) == 1
        assert found.duration_s[0] == pytest.approx(1.5)


class TestDominance:
    def test_shares(self, make_transcript, cfg):
        turns = utterances_to_turns(
            make_transcript([("A", 0, 3, "a"), ("B", 4, 5, "b"), ("A", 6, 8, "a")]), cfg
        )
        share = dominance(turns)
        assert sum(share.values()) == pytest.approx(1.0)
        assert share["A"] == pytest.approx(5 / 6)

    def test_empty(self, make_transcript, cfg):
        assert dominance(utterances_to_turns(make_transcript([]), cfg)) == {}
