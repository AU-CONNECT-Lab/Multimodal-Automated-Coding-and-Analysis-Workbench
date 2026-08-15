from __future__ import annotations

import numpy as np
import pytest

from macaw.config.schema import TurnsSection
from macaw.turns.transitions import (
    classify_overlaps,
    detect_transitions,
    floor_transfer_efficiency,
    response_latency,
    transition_matrix,
)


@pytest.fixture
def cfg():
    return TurnsSection()


def classified(turns, cfg):
    return classify_overlaps(
        detect_transitions(turns, cfg), interruption_s=cfg.interruption_overlap_s
    )


class TestDetect:
    def test_gap_is_positive(self, make_turns, cfg):
        found = detect_transitions(make_turns([("A", 0, 1), ("B", 1.5, 2)]), cfg)
        assert len(found) == 1
        assert found.fto_s[0] == pytest.approx(0.5)
        assert bool(found.is_switch[0]) is True
        assert found.transition_type[0] == "gap"

    def test_overlap_is_negative(self, make_turns, cfg):
        found = detect_transitions(make_turns([("A", 0, 2), ("B", 1.5, 3)]), cfg)
        assert found.fto_s[0] == pytest.approx(-0.5)
        assert found.transition_type[0] == "overlap"

    def test_same_speaker_twice_is_a_hold(self, make_turns, cfg):
        """A long enough pause with nobody cutting in ends the turn without moving the floor."""
        found = detect_transitions(make_turns([("A", 0, 1), ("A", 4, 5)]), cfg)
        assert bool(found.is_switch[0]) is False
        assert found.transition_type[0] == "hold"
        assert found.fto_s[0] == pytest.approx(3.0)

    def test_one_row_per_adjacent_pair(self, make_turns, cfg):
        turns = make_turns([("A", 0, 1), ("B", 2, 3), ("A", 4, 5), ("B", 6, 7)])
        assert len(detect_transitions(turns, cfg)) == len(turns) - 1

    def test_turn_indices_link_back(self, make_turns, cfg):
        found = detect_transitions(make_turns([("A", 0, 1), ("B", 2, 3)]), cfg)
        assert (found.from_turn_index[0], found.to_turn_index[0]) == (0, 1)

    def test_pairs_follow_start_time_not_input_order(self, make_turns, cfg):
        turns = make_turns([("A", 5, 6), ("B", 0, 1)]).sort_index(ascending=False)
        found = detect_transitions(turns, cfg)
        assert (found.from_subject[0], found.to_subject[0]) == ("B", "A")

    @pytest.mark.parametrize("rows", [[], [("A", 0, 1)]])
    def test_needs_two_turns(self, make_turns, cfg, rows):
        found = detect_transitions(make_turns(rows), cfg)
        assert len(found) == 0
        assert "fto_s" in found.columns


class TestClassifyOverlaps:
    def test_long_overlap_is_an_interruption(self, make_turns, cfg):
        found = classified(make_turns([("A", 0, 2), ("B", 1.2, 3)]), cfg)
        assert found.transition_type[0] == "interruption"

    def test_short_overlap_that_gives_the_floor_back_is_a_backchannel(self, make_turns, cfg):
        """B says two words over A and A carries straight on: the floor never moved."""
        found = classified(make_turns([("A", 0, 2), ("B", 1.8, 2.0), ("A", 2.2, 4)]), cfg)
        assert list(found.transition_type) == ["backchannel", "gap"]

    def test_short_overlap_that_keeps_the_floor_is_not_a_backchannel(self, make_turns, cfg):
        """Same length, but A never speaks again, so B did take over."""
        found = classified(make_turns([("A", 0, 2), ("B", 1.8, 4), ("B", 5, 6)]), cfg)
        assert found.transition_type[0] == "overlap"

    def test_length_wins_over_the_floor_coming_back(self, make_turns, cfg):
        found = classified(make_turns([("A", 0, 2), ("B", 1.0, 2.0), ("A", 2.2, 4)]), cfg)
        assert found.transition_type[0] == "interruption"

    def test_boundary_is_strictly_greater(self, make_turns, cfg):
        """0.5 s of overlap is not yet an interruption; the check is on exceeding it."""
        found = classified(make_turns([("A", 0, 2), ("B", 1.5, 2.0), ("A", 2.2, 4)]), cfg)
        assert found.transition_type[0] == "backchannel"

    def test_a_last_overlap_cannot_be_a_backchannel(self, make_turns, cfg):
        """Nothing follows it, so there is no evidence the floor came back."""
        found = classified(make_turns([("A", 0, 2), ("B", 1.8, 3)]), cfg)
        assert found.transition_type[0] == "overlap"

    def test_gaps_and_holds_are_untouched(self, make_turns, cfg):
        found = classified(make_turns([("A", 0, 1), ("B", 2, 3), ("B", 6, 7)]), cfg)
        assert list(found.transition_type) == ["gap", "hold"]

    def test_empty_input(self, make_turns, cfg):
        assert len(classified(make_turns([]), cfg)) == 0


class TestFloorTransferEfficiency:
    @pytest.fixture
    def summary(self, make_turns, cfg):
        turns = make_turns(
            [
                ("A", 0.0, 3.0),
                ("B", 2.8, 3.0),
                ("A", 3.2, 5.0),
                ("B", 4.2, 6.0),
                ("A", 7.0, 8.0),
                ("A", 10.5, 11.0),
            ]
        )
        return floor_transfer_efficiency(classified(turns, cfg), cfg)

    def test_counts(self, summary):
        assert summary["n_transitions"] == 5
        assert summary["n_switches"] == 4
        assert summary["n_holds"] == 1

    def test_gap_and_overlap_shares_sum_to_one(self, summary):
        assert summary["gap_ratio"] == pytest.approx(0.5)
        assert summary["gap_ratio"] + summary["overlap_ratio"] == pytest.approx(1.0)

    def test_median_and_spread_over_switches(self, summary):
        assert summary["median_fto_s"] == pytest.approx(0.0)
        assert summary["sd_fto_s"] == pytest.approx(np.std([-0.2, 0.2, -0.8, 1.0], ddof=1))

    def test_smooth_transfers(self, summary):
        assert summary["smooth_transfer_ratio"] == pytest.approx(0.5)

    def test_gap_cv_ignores_overlaps(self, summary):
        """Overlaps drag the mean towards zero and the ratio blows up; only gaps go in."""
        assert summary["gap_cv"] == pytest.approx(np.std([0.2, 1.0], ddof=1) / 0.6)

    def test_hold_pause(self, summary):
        assert summary["median_hold_s"] == pytest.approx(2.5)

    def test_empty_still_reports_counts(self, make_turns, cfg):
        summary = floor_transfer_efficiency(classified(make_turns([]), cfg), cfg)
        assert summary["n_transitions"] == 0
        assert np.isnan(summary["median_fto_s"])

    def test_all_overlaps_leave_gap_cv_undefined(self, make_turns, cfg):
        turns = make_turns([("A", 0, 2), ("B", 1.0, 4), ("A", 3.0, 6)])
        assert np.isnan(floor_transfer_efficiency(classified(turns, cfg), cfg)["gap_cv"])


class TestResponseLatency:
    def test_per_direction_medians(self, make_turns, cfg):
        turns = make_turns([("A", 0, 1), ("B", 2, 3), ("A", 3.5, 4), ("B", 6, 7)])
        found = response_latency(classified(turns, cfg))
        rows = {(row.from_subject, row.to_subject): row for row in found.itertuples()}
        assert rows[("A", "B")].n == 2
        assert rows[("A", "B")].median_latency_s == pytest.approx(1.5)
        assert rows[("B", "A")].median_latency_s == pytest.approx(0.5)

    def test_overlaps_are_excluded(self, make_turns, cfg):
        """Somebody who started before the other finished was not waiting."""
        turns = make_turns([("A", 0, 2), ("B", 1.0, 4)])
        assert len(response_latency(classified(turns, cfg))) == 0

    def test_holds_are_excluded(self, make_turns, cfg):
        turns = make_turns([("A", 0, 1), ("A", 4, 5)])
        assert len(response_latency(classified(turns, cfg))) == 0

    def test_empty_input(self, make_turns, cfg):
        assert len(response_latency(classified(make_turns([]), cfg))) == 0


class TestTransitionMatrix:
    @pytest.fixture
    def turns(self, make_turns):
        return make_turns(
            [("A", 0, 1), ("B", 2, 3), ("A", 4, 5), ("B", 6, 7), ("B", 10, 11)]
        )

    def test_orientation_is_from_row_to_column(self, turns):
        matrix = transition_matrix(turns, ["A", "B"])
        assert matrix.loc["A", "B"] == 2
        assert matrix.loc["B", "A"] == 1

    def test_diagonal_counts_holds(self, turns):
        assert transition_matrix(turns, ["A", "B"])["B"]["B"] == 1

    def test_total_equals_the_number_of_pairs(self, turns):
        assert int(transition_matrix(turns, ["A", "B"]).to_numpy().sum()) == len(turns) - 1

    def test_shape_follows_the_subject_list(self, turns):
        assert transition_matrix(turns, ["A", "B", "C"]).shape == (3, 3)

    def test_unlisted_speakers_are_dropped(self, make_turns):
        turns = make_turns([("A", 0, 1), ("C", 2, 3), ("A", 4, 5)])
        assert int(transition_matrix(turns, ["A", "B"]).to_numpy().sum()) == 0

    def test_single_turn_gives_zeros(self, make_turns):
        assert int(transition_matrix(make_turns([("A", 0, 1)]), ["A"]).to_numpy().sum()) == 0
