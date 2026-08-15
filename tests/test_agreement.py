from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from macaw.coding.agreement import (
    NO_CODE,
    agreement_report,
    build_units,
    cohen_kappa,
    format_report,
    krippendorff_alpha,
    pairwise_kappa,
    per_code_kappa,
    report_to_json,
)


def annotations(rows, *, tier="backchannel", layer="B", recording_id="r1", subject_id="parent"):
    """rows are (annotator, t_start, t_end, label) with an optional fifth abstained flag."""
    return pd.DataFrame(
        [
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "tier": tier,
                "layer": layer,
                "annotator": row[0],
                "t_start": float(row[1]),
                "t_end": float(row[2]),
                "label": row[3],
                "abstained": bool(row[4]) if len(row) > 4 else False,
            }
            for row in rows
        ]
    )


def units_from(pairs):
    """pairs are (unit_id, annotator, label); lets the coefficients be tested on their own."""
    return pd.DataFrame(pairs, columns=["unit_id", "annotator", "label"])


def two_coder_units(labels_a, labels_b):
    rows = []
    for index, (left, right) in enumerate(zip(labels_a, labels_b)):
        if left is not None:
            rows.append((f"u{index}", "A", left))
        if right is not None:
            rows.append((f"u{index}", "B", right))
    return units_from(rows)


# the worked example the two known values below come from
NINE_A = ["1", "2", "3", "3", "2", "1", "4", "1", "2"]
NINE_B = ["1", "2", "3", "3", "2", "2", "4", "1", "2"]
NINE_ALPHA = 1.0 - 17.0 * 2.0 / 230.0
NINE_KAPPA = 49.0 / 58.0


class TestBuildUnits:
    def test_identical_intervals_become_one_unit(self):
        units = build_units(annotations([("A", 0.0, 1.0, "yes"), ("B", 0.0, 1.0, "yes")]))
        assert units["unit_id"].nunique() == 1
        assert len(units) == 2

    def test_different_boundaries_never_pair_in_exact_mode(self):
        units = build_units(annotations([("A", 0.0, 1.0, "yes"), ("B", 0.0, 1.1, "yes")]))
        assert units["unit_id"].nunique() == 2

    def test_the_same_interval_in_two_tiers_is_two_units(self):
        frame = pd.concat(
            [
                annotations([("A", 0.0, 1.0, "yes")], tier="backchannel"),
                annotations([("A", 0.0, 1.0, "yes")], tier="pause"),
            ],
            ignore_index=True,
        )
        assert build_units(frame)["unit_id"].nunique() == 2

    def test_the_same_interval_for_two_subjects_is_two_units(self):
        frame = pd.concat(
            [
                annotations([("A", 0.0, 1.0, "yes")], subject_id="parent"),
                annotations([("A", 0.0, 1.0, "yes")], subject_id="child"),
            ],
            ignore_index=True,
        )
        assert build_units(frame)["unit_id"].nunique() == 2

    def test_abstentions_are_dropped(self):
        """Two models that both declined to answer are not agreeing about anything."""
        units = build_units(
            annotations([("A", 0.0, 1.0, "yes", True), ("B", 0.0, 1.0, "yes", False)])
        )
        assert units["annotator"].tolist() == ["B"]

    def test_a_second_label_from_one_annotator_is_dropped_and_counted(self):
        units = build_units(
            annotations([("A", 0.0, 1.0, "yes"), ("A", 0.0, 1.0, "no"), ("B", 0.0, 1.0, "yes")])
        )
        assert len(units) == 2
        assert units.attrs["duplicate_ratings"] == 1

    def test_empty_input_keeps_the_columns(self):
        units = build_units(pd.DataFrame())
        assert units.empty
        assert list(units.columns) == ["unit_id", "annotator", "label"]

    def test_missing_labels_are_dropped(self):
        frame = annotations([("A", 0.0, 1.0, "yes"), ("B", 0.0, 1.0, "yes")])
        frame.loc[0, "label"] = None
        assert len(build_units(frame)) == 1


class TestGridUnits:
    def test_one_interval_becomes_several_cells(self):
        units = build_units(
            annotations([("A", 0.0, 1.0, "x"), ("B", 0.0, 1.0, "x")]), grid_s=0.5
        )
        assert units["unit_id"].nunique() == 2
        assert len(units) == 4

    def test_a_cell_only_one_coder_marked_is_a_disagreement_not_a_gap(self):
        units = build_units(
            annotations([("A", 0.0, 0.5, "x"), ("B", 0.0, 1.0, "x")]), grid_s=0.5
        )
        assert units["unit_id"].nunique() == 2
        assert NO_CODE in units["label"].tolist()

    def test_cells_nobody_marked_are_not_units(self):
        units = build_units(
            annotations([("A", 0.0, 0.5, "x"), ("B", 0.0, 0.5, "x")]), grid_s=0.5
        )
        assert units["unit_id"].nunique() == 1

    def test_the_label_covering_most_of_the_cell_wins(self):
        units = build_units(
            annotations([("A", 0.0, 0.3, "x"), ("A", 0.3, 0.5, "y")]), grid_s=0.5
        )
        assert units["label"].tolist() == ["x"]

    def test_an_even_split_is_reported_and_resolved_alphabetically(self):
        units = build_units(
            annotations([("A", 0.0, 0.25, "x"), ("A", 0.25, 0.5, "y")]), grid_s=0.5
        )
        assert units["label"].tolist() == ["x"]
        assert units.attrs["grid_ties"] == 1

    def test_free_boundaries_that_never_pair_exactly_do_pair_on_a_grid(self):
        frame = annotations(
            [
                ("A", 0.0, 1.0, "x"),
                ("B", 0.05, 0.95, "x"),
                ("A", 2.0, 3.0, "y"),
                ("B", 2.05, 2.95, "y"),
            ]
        )
        assert build_units(frame)["unit_id"].nunique() == 4
        assert build_units(frame, grid_s=0.5)["unit_id"].nunique() == 4
        assert krippendorff_alpha(build_units(frame, grid_s=0.5)) == pytest.approx(1.0)

    def test_zero_width_intervals_cover_nothing(self):
        assert build_units(annotations([("A", 1.0, 1.0, "x")]), grid_s=0.5).empty

    @pytest.mark.parametrize("grid_s", [0.0, -1.0])
    def test_a_non_positive_grid_is_a_programming_error(self, grid_s):
        with pytest.raises(ValueError):
            build_units(annotations([("A", 0.0, 1.0, "x")]), grid_s=grid_s)


class TestKrippendorffAlpha:
    def test_perfect_agreement_is_one(self):
        assert krippendorff_alpha(two_coder_units(["a", "b"], ["a", "b"])) == pytest.approx(1.0)

    def test_known_value(self):
        assert krippendorff_alpha(two_coder_units(NINE_A, NINE_B)) == pytest.approx(NINE_ALPHA)

    def test_chance_level_is_zero(self):
        assert krippendorff_alpha(two_coder_units(["a", "a"], ["a", "b"])) == pytest.approx(0.0)

    def test_systematic_disagreement_goes_negative(self):
        """Below zero means worse than chance, which is information, not an error."""
        assert krippendorff_alpha(two_coder_units(["a", "b"], ["b", "a"])) == pytest.approx(-0.5)

    def test_one_category_everywhere_is_undefined(self):
        """With nothing to distinguish, expected disagreement is zero and the ratio has no value."""
        assert np.isnan(krippendorff_alpha(two_coder_units(["a", "a"], ["a", "a"])))

    def test_units_rated_once_drop_out(self):
        paired = two_coder_units(NINE_A, NINE_B)
        with_orphan = pd.concat(
            [paired, units_from([("u9", "B", "5")])], ignore_index=True
        )
        assert krippendorff_alpha(with_orphan) == pytest.approx(krippendorff_alpha(paired))

    def test_missing_ratings_are_tolerated(self):
        units = two_coder_units(["a", "b", "a", None], ["a", "b", "a", "b"])
        assert krippendorff_alpha(units) == pytest.approx(1.0)

    def test_three_annotators(self):
        rows = []
        for index, labels in enumerate([("a", "a", "a"), ("b", "b", "a")]):
            for name, label in zip("ABC", labels):
                rows.append((f"u{index}", name, label))
        assert np.isfinite(krippendorff_alpha(units_from(rows)))

    @pytest.mark.parametrize("units", [pd.DataFrame(), units_from([("u0", "A", "a")])])
    def test_nothing_pairable(self, units):
        assert np.isnan(krippendorff_alpha(units))


class TestCohenKappa:
    def test_perfect_agreement_is_one(self):
        assert cohen_kappa(two_coder_units(["a", "b"], ["a", "b"]), "A", "B") == pytest.approx(1.0)

    def test_known_value(self):
        assert cohen_kappa(two_coder_units(NINE_A, NINE_B), "A", "B") == pytest.approx(NINE_KAPPA)

    def test_chance_level_is_zero(self):
        assert cohen_kappa(two_coder_units(["a", "a"], ["a", "b"]), "A", "B") == pytest.approx(0.0)

    def test_total_reversal_is_minus_one(self):
        assert cohen_kappa(two_coder_units(["a", "b"], ["b", "a"]), "A", "B") == pytest.approx(-1.0)

    def test_alpha_and_kappa_stay_close_on_complete_data(self):
        units = two_coder_units(NINE_A, NINE_B)
        assert abs(krippendorff_alpha(units) - cohen_kappa(units, "A", "B")) < 0.02

    def test_only_units_both_rated_count(self):
        """Kappa has no way to use a unit one coder skipped; alpha does."""
        units = two_coder_units(["a", "b", "a"], ["a", "b", None])
        assert cohen_kappa(units, "A", "B") == pytest.approx(1.0)

    def test_an_unknown_annotator_has_no_value(self):
        assert np.isnan(cohen_kappa(two_coder_units(["a"], ["a"]), "A", "Z"))

    def test_an_annotator_against_itself_has_no_value(self):
        assert np.isnan(cohen_kappa(two_coder_units(["a", "b"], ["a", "b"]), "A", "A"))


class TestPairwiseKappa:
    def test_one_row_per_pair(self):
        rows = [(f"u{index}", name, "a") for index in range(2) for name in "ABC"]
        table = pairwise_kappa(units_from(rows))
        assert len(table) == 3
        assert table["annotator_a"].tolist() == ["A", "A", "B"]

    def test_n_units_counts_only_shared_units(self):
        units = two_coder_units(["a", "b", "a"], ["a", "b", None])
        assert pairwise_kappa(units)["n_units"].tolist() == [2]

    def test_observed_agreement_is_reported_alongside(self):
        units = two_coder_units(["a", "a"], ["a", "b"])
        assert pairwise_kappa(units)["observed_agreement"].tolist() == [pytest.approx(0.5)]

    def test_empty_input_keeps_the_columns(self):
        table = pairwise_kappa(pd.DataFrame())
        assert table.empty
        assert "kappa" in table.columns


class TestPerCodeKappa:
    def test_one_row_per_code(self):
        table = per_code_kappa(two_coder_units(["a", "b"], ["a", "b"]))
        assert table["code"].tolist() == ["a", "b"]

    def test_an_overall_value_can_hide_one_bad_code(self):
        """Three codes agree and one never does; the total stays high, the per-code row does not."""
        labels_a = ["a", "a", "b", "b", "c", "c", "d", "d"]
        labels_b = ["a", "a", "b", "b", "c", "c", "a", "b"]
        table = per_code_kappa(two_coder_units(labels_a, labels_b)).set_index("code")
        assert krippendorff_alpha(two_coder_units(labels_a, labels_b)) > 0.6
        assert table.loc["d", "alpha"] < 0.0

    def test_blank_grid_cells_are_not_a_code(self):
        units = build_units(
            annotations([("A", 0.0, 0.5, "x"), ("B", 0.0, 1.0, "x")]), grid_s=0.5
        )
        assert NO_CODE not in per_code_kappa(units)["code"].tolist()

    def test_counts_are_reported(self):
        table = per_code_kappa(two_coder_units(["a", "a"], ["a", "b"])).set_index("code")
        assert table.loc["a", "n_ratings"] == 3
        assert table.loc["a", "n_units"] == 2
        assert table.loc["b", "n_units"] == 1

    def test_kappa_is_left_out_when_there_are_not_exactly_two_annotators(self):
        rows = [(f"u{index}", name, "a") for index in range(2) for name in "ABC"]
        assert np.isnan(per_code_kappa(units_from(rows))["kappa"]).all()

    def test_empty_input_keeps_the_columns(self):
        table = per_code_kappa(pd.DataFrame())
        assert table.empty
        assert "alpha" in table.columns


class TestAgreementReport:
    def paired(self, n=40):
        rows = []
        for index in range(n):
            label = "yes" if index % 2 else "no"
            rows.append(("A", index, index + 0.5, label))
            rows.append(("B", index, index + 0.5, label))
        return annotations(rows)

    def test_counts_and_coefficients(self):
        report = agreement_report(self.paired())
        assert report["n_units"] == 40
        assert report["n_pairable"] == 40
        assert report["n_annotators"] == 2
        assert report["annotators"] == ["A", "B"]
        assert report["alpha"] == pytest.approx(1.0)
        assert report["coverage"] == pytest.approx(1.0)
        assert report["warnings"] == []

    def test_tier_filter(self):
        frame = pd.concat(
            [self.paired(), annotations([("C", 0.0, 1.0, "x")], tier="pause")], ignore_index=True
        )
        assert agreement_report(frame, tier="pause")["annotators"] == ["C"]

    def test_layer_filter(self):
        frame = pd.concat(
            [self.paired(), annotations([("C", 0.0, 1.0, "x")], layer="A")], ignore_index=True
        )
        assert agreement_report(frame, layer="A")["annotators"] == ["C"]

    def test_unit_mode_follows_the_argument(self):
        assert agreement_report(self.paired())["unit_mode"] == "exact"
        assert agreement_report(self.paired(), grid_s=0.25)["unit_mode"] == "grid"

    def test_abstention_rate_is_reported_per_annotator(self):
        frame = annotations(
            [("A", 0.0, 1.0, "x", True), ("A", 1.0, 2.0, "x", False), ("B", 0.0, 1.0, "x", False)]
        )
        rates = agreement_report(frame)["abstention_rate"]
        assert rates["A"] == pytest.approx(0.5)
        assert rates["B"] == pytest.approx(0.0)

    def test_one_annotator_warns(self):
        report = agreement_report(annotations([("A", 0.0, 1.0, "x")]))
        assert any("nothing to compare" in warning for warning in report["warnings"])

    def test_nothing_pairable_points_at_the_grid(self):
        """Near-miss boundaries are the failure mode exact mode has; say so instead of NaN."""
        report = agreement_report(annotations([("A", 0.0, 1.0, "x"), ("B", 0.1, 1.1, "x")]))
        assert report["n_units"] == 2
        assert report["n_pairable"] == 0
        assert any("grid_s" in warning for warning in report["warnings"])

    def test_a_small_sample_is_flagged_but_still_reported(self):
        report = agreement_report(self.paired(n=5))
        assert np.isfinite(report["alpha"])
        assert any("unstable" in warning for warning in report["warnings"])

    def test_partial_coverage_is_flagged(self):
        frame = pd.concat(
            [self.paired(), annotations([("A", 100.0, 101.0, "x")])], ignore_index=True
        )
        report = agreement_report(frame)
        assert report["coverage"] < 1.0
        assert any("rated by every annotator" in warning for warning in report["warnings"])

    def test_duplicate_ratings_are_flagged(self):
        frame = pd.concat(
            [self.paired(), annotations([("A", 0.0, 0.5, "maybe")])], ignore_index=True
        )
        assert any("more than one label" in warning for warning in agreement_report(frame)["warnings"])

    def test_empty_annotations_do_not_raise(self):
        report = agreement_report(pd.DataFrame())
        assert report["n_units"] == 0
        assert np.isnan(report["alpha"])

    def test_json_form_is_strictly_serialisable(self):
        text = json.dumps(report_to_json(agreement_report(self.paired(n=5))))
        assert "NaN" not in text
        assert json.loads(text)["n_units"] == 5

    def test_format_report_mentions_the_headline_number(self):
        text = format_report(agreement_report(self.paired()))
        assert "krippendorff_alpha=1.000" in text
