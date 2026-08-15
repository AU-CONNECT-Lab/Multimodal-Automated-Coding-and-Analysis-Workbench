from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from macaw.turns.entropy import (
    _shannon,
    conditional_entropy,
    permutation_entropy,
    speaker_entropy,
    transition_entropy,
)

LN2 = math.log(2.0)


def matrix(cells: dict[tuple[str, str], int], names: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(0, index=names, columns=names, dtype="int64")
    for (source, target), count in cells.items():
        out.loc[source, target] = count
    return out


class TestShannon:
    def test_uniform_is_log2_of_the_support(self):
        assert _shannon([1, 1, 1, 1], correct=False) == pytest.approx(2.0)

    def test_deterministic_is_zero(self):
        assert _shannon([7, 0, 0], correct=False) == pytest.approx(0.0)

    def test_miller_madow_adds_the_leading_bias_term(self):
        counts = [3, 2, 2, 1]
        plug_in = _shannon(counts, correct=False)
        expected = plug_in + (4 - 1) / (2 * 8 * LN2)
        assert _shannon(counts, correct=True) == pytest.approx(expected)

    def test_correction_uses_observed_states_not_declared_ones(self):
        """Empty cells carry no information and must not inflate the correction."""
        assert _shannon([5, 5, 0, 0]) == pytest.approx(_shannon([5, 5]))

    def test_correction_always_raises_the_estimate(self):
        """Plug-in is biased low, so the fix can only go one way."""
        counts = [4, 3, 2, 1]
        assert _shannon(counts, correct=True) > _shannon(counts, correct=False)

    def test_bias_shrinks_as_the_sample_grows(self):
        small = _shannon([2, 2]) - _shannon([2, 2], correct=False)
        large = _shannon([200, 200]) - _shannon([200, 200], correct=False)
        assert small > large

    @pytest.mark.parametrize("counts", [[], [0, 0]])
    def test_no_observations(self, counts):
        assert np.isnan(_shannon(counts))


class TestTransitionEntropy:
    def test_two_speakers_top_out_at_two_bits(self):
        cells = {("A", "A"): 5, ("A", "B"): 5, ("B", "A"): 5, ("B", "B"): 5}
        assert transition_entropy(matrix(cells, ["A", "B"]), correct=False) == pytest.approx(2.0)

    def test_one_state_only_is_zero(self):
        assert transition_entropy(
            matrix({("A", "B"): 9}, ["A", "B"]), correct=False
        ) == pytest.approx(0.0)

    def test_three_speakers_top_out_higher(self):
        cells = {(a, b): 2 for a in "ABC" for b in "ABC"}
        found = transition_entropy(matrix(cells, ["A", "B", "C"]), correct=False)
        assert found == pytest.approx(math.log2(9))

    def test_empty_matrix(self):
        assert np.isnan(transition_entropy(matrix({}, ["A", "B"])))


class TestPermutationEntropy:
    def test_monotonic_series_has_one_pattern(self):
        assert permutation_entropy(np.arange(20.0), order=3) == pytest.approx(0.0)

    def test_alternating_series_fills_both_order_two_patterns(self):
        # an odd length so the two patterns come out in exactly equal counts
        series = np.array([0.0, 1.0] * 10 + [0.0])
        assert permutation_entropy(
            series, order=2, normalise=False, correct=False
        ) == pytest.approx(1.0)
        assert permutation_entropy(series, order=2) == pytest.approx(1.0)

    def test_unnormalised_is_in_bits(self):
        series = np.array([0.0, 1.0] * 10 + [0.0])
        ceiling = permutation_entropy(series, order=2, normalise=False)
        assert ceiling > 1.0
        assert permutation_entropy(series, order=2) == pytest.approx(min(ceiling, 1.0))

    def test_normalised_never_exceeds_one(self):
        """The bias correction can overshoot the ceiling on a series this short."""
        series = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        assert permutation_entropy(series, order=2) <= 1.0

    def test_noise_scores_higher_than_a_ramp(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(size=200)
        ramp = np.arange(200.0)
        assert permutation_entropy(noise, order=3) > permutation_entropy(ramp, order=3)

    @pytest.mark.parametrize("length", [0, 1, 2])
    def test_series_shorter_than_the_order(self, length):
        assert np.isnan(permutation_entropy(np.zeros(length), order=3))

    def test_order_below_two_is_meaningless(self):
        assert np.isnan(permutation_entropy(np.arange(20.0), order=1))


class TestSpeakerEntropy:
    def test_even_split_is_one_bit(self):
        assert speaker_entropy({"A": 10.0, "B": 10.0}) == pytest.approx(1.0)

    def test_one_speaker_holds_everything(self):
        assert speaker_entropy({"A": 10.0, "B": 0.0}) == pytest.approx(0.0)

    def test_durations_are_not_counts_so_nothing_is_corrected(self):
        """Talk time is measured, not sampled, so the Miller-Madow term must not appear."""
        assert speaker_entropy({"A": 3.0, "B": 1.0}) == pytest.approx(
            _shannon([3.0, 1.0], correct=False)
        )

    def test_scale_invariant(self):
        assert speaker_entropy({"A": 3.0, "B": 1.0}) == pytest.approx(
            speaker_entropy({"A": 30.0, "B": 10.0})
        )

    def test_empty(self):
        assert np.isnan(speaker_entropy({}))


class TestConditionalEntropy:
    def test_deterministic_response_is_zero(self):
        x = list("ABABA")
        y = list("BABAB")
        assert conditional_entropy(x, y, correct=False) == pytest.approx(0.0)

    def test_known_value(self):
        x = list("AAAB")
        y = list("BCDB")
        assert conditional_entropy(x, y, correct=False) == pytest.approx(2.0 - 0.81127812)

    def test_direction_matters(self):
        """The gap between the two directions is the whole point of this measure."""
        x = list("AAAB")
        y = list("BCDB")
        assert conditional_entropy(y, x, correct=False) == pytest.approx(0.5)

    def test_conditioning_never_increases_entropy(self):
        x = list("AABBABBA")
        y = list("BABAABAB")
        assert conditional_entropy(x, y, correct=False) <= _shannon(
            pd.Series(y).value_counts().to_numpy(), correct=False
        ) + 1e-9

    def test_length_mismatch(self):
        assert np.isnan(conditional_entropy(list("AAB"), list("BC")))

    def test_empty(self):
        assert np.isnan(conditional_entropy([], []))
