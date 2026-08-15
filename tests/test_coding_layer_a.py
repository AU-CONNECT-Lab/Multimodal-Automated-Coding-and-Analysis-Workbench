from __future__ import annotations

import pandas as pd
import pytest

from macaw.config.schema import CodingSection, TurnsSection
from macaw.coding.layer_a import (
    code_backchannels,
    code_interruptions,
    code_pause_types,
)
from macaw.turns.transitions import classify_overlaps, detect_transitions


@pytest.fixture
def turns_cfg():
    return TurnsSection()


@pytest.fixture
def cfg():
    return CodingSection(backchannel_lexicon=["mm", "uh-huh", "yeah", "right"])


def with_text(turns: pd.DataFrame, texts: list[str]) -> pd.DataFrame:
    out = turns.copy()
    out["text"] = texts
    return out


class TestBackchannels:
    def test_all_three_rules_together(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.4), ("parent", 3.5, 6)]),
            ["telling a story", "mm", "and then"],
        )
        found = code_backchannels(turns, cfg)
        assert list(found.subject_id) == ["child"]
        assert list(found.label) == ["backchannel"]

    def test_the_floor_has_to_come_back(self, make_turns, cfg):
        """Same short token, but the other speaker never resumes, so it took the floor."""
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.4), ("child", 5, 8)]),
            ["telling a story", "mm", "my turn now"],
        )
        assert code_backchannels(turns, cfg).empty

    def test_too_long_to_be_one(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 5.5), ("parent", 5.6, 8)]),
            ["a story", "mm", "and then"],
        )
        assert code_backchannels(turns, cfg).empty

    def test_a_sentence_containing_the_token_is_not_one(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.9), ("parent", 4, 6)]),
            ["a story", "yeah but why", "because"],
        )
        assert code_backchannels(turns, cfg).empty

    def test_matching_ignores_case_and_punctuation(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.4), ("parent", 3.5, 6)]),
            ["a story", "Uh-huh.", "and then"],
        )
        assert len(code_backchannels(turns, cfg)) == 1

    def test_two_lexicon_tokens_still_count(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.6), ("parent", 3.7, 6)]),
            ["a story", "yeah right", "and then"],
        )
        assert len(code_backchannels(turns, cfg)) == 1

    def test_first_and_last_turn_can_never_qualify(self, make_turns, cfg):
        """Nothing outside them can show the floor came back."""
        turns = with_text(
            make_turns([("child", 0, 0.3), ("parent", 1, 4), ("child", 4.1, 4.4)]),
            ["mm", "a story", "mm"],
        )
        assert code_backchannels(turns, cfg).empty

    def test_empty_lexicon_codes_nothing(self, make_turns):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.4), ("parent", 3.5, 6)]),
            ["a story", "mm", "and then"],
        )
        assert code_backchannels(turns, CodingSection()).empty

    def test_empty_text_never_matches(self, make_turns, cfg):
        turns = with_text(
            make_turns([("parent", 0, 3), ("child", 3.1, 3.4), ("parent", 3.5, 6)]),
            ["a story", "", "and then"],
        )
        assert code_backchannels(turns, cfg).empty


class TestPauseTypes:
    @pytest.fixture
    def scene(self, make_turns, make_transcript):
        turns = make_turns(
            [("parent", 0.0, 5.0), ("child", 6.0, 7.0), ("child", 9.0, 10.0)]
        )
        # the parent turn is two utterances with a 1.5 s hole nobody else fills
        transcript = make_transcript(
            [
                ("parent", 0.0, 2.0, "first half"),
                ("parent", 3.5, 5.0, "second half"),
                ("child", 6.0, 7.0, "reply"),
                ("child", 9.0, 10.0, "again"),
            ]
        )
        return turns, transcript

    def test_the_three_kinds(self, scene, turns_cfg):
        turns, transcript = scene
        found = code_pause_types(turns, turns_cfg, transcript)
        assert sorted(found.label.unique()) == ["hold", "switching", "turn_internal"]

    def test_turn_internal_pause_is_invisible_to_the_turn_layer(self, scene, turns_cfg):
        """Turn boundaries hide it by construction; only the transcript shows it."""
        turns, transcript = scene
        found = code_pause_types(turns, turns_cfg, transcript)
        internal = found[found.label == "turn_internal"]
        assert list(internal.t_start) == [2.0]
        assert list(internal.t_end) == [3.5]
        assert list(internal.subject_id) == ["parent"]

    def test_switching_pause_belongs_to_whoever_ends_it(self, scene, turns_cfg):
        turns, transcript = scene
        found = code_pause_types(turns, turns_cfg, transcript)
        switching = found[found.label == "switching"]
        assert list(switching.subject_id) == ["child"]
        assert list(switching.t_start) == [5.0]

    def test_hold_pause_is_the_same_speaker_restarting(self, scene, turns_cfg):
        turns, transcript = scene
        held = code_pause_types(turns, turns_cfg, transcript)
        held = held[held.label == "hold"]
        assert list(held.subject_id) == ["child"]
        assert list(held.t_start) == [7.0]

    def test_no_threshold_is_applied(self, make_turns, make_transcript, turns_cfg):
        """How long a pause has to be to matter is the analyst's call, not ours."""
        turns = make_turns([("parent", 0.0, 1.0), ("child", 1.05, 2.0)])
        transcript = make_transcript([("parent", 0.0, 1.0, "a"), ("child", 1.05, 2.0, "b")])
        found = code_pause_types(turns, turns_cfg, transcript)
        assert len(found) == 1
        assert found.t_end[0] - found.t_start[0] == pytest.approx(0.05)

    def test_overlaps_are_not_pauses(self, make_turns, make_transcript, turns_cfg):
        turns = make_turns([("parent", 0.0, 2.0), ("child", 1.5, 4.0)])
        transcript = make_transcript([("parent", 0.0, 2.0, "a"), ("child", 1.5, 4.0, "b")])
        assert code_pause_types(turns, turns_cfg, transcript).empty

    def test_results_are_in_time_order(self, scene, turns_cfg):
        turns, transcript = scene
        found = code_pause_types(turns, turns_cfg, transcript)
        assert list(found.t_start) == sorted(found.t_start)


class TestInterruptions:
    def classified(self, turns, turns_cfg):
        return classify_overlaps(
            detect_transitions(turns, turns_cfg), interruption_s=turns_cfg.interruption_overlap_s
        )

    def test_span_is_the_overlap_and_owner_is_the_interrupter(self, make_turns, turns_cfg, cfg):
        turns = make_turns([("parent", 0.0, 4.0), ("child", 3.0, 6.0)])
        found = code_interruptions(self.classified(turns, turns_cfg), cfg)
        assert list(found.subject_id) == ["child"]
        assert (found.t_start[0], found.t_end[0]) == (3.0, 4.0)
        assert found.t_end[0] > found.t_start[0]

    def test_backchannels_and_gaps_are_not_interruptions(self, make_turns, turns_cfg, cfg):
        turns = make_turns([("parent", 0, 2), ("child", 1.8, 2.0), ("parent", 2.2, 4)])
        assert code_interruptions(self.classified(turns, turns_cfg), cfg).empty

    def test_empty_input(self, make_turns, turns_cfg, cfg):
        assert code_interruptions(self.classified(make_turns([]), turns_cfg), cfg).empty
