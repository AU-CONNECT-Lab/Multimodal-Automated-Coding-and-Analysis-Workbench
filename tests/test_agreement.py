from __future__ import annotations

import json
from types import SimpleNamespace

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
    per_code_alpha,
    per_code_pairs,
    report_to_json,
    segment_spans,
    unit_table,
    write_report,
)
from macaw.exceptions import MacawError


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
        assert list(units.columns) == ["unit_id", "annotator", "label", "recording_id", "subject_id", "tier", "t_start", "t_end"]

    @pytest.mark.parametrize("mode", [{}, {"grid_s": 0.5}, {"units_of": "A"}])
    def test_every_unit_says_where_and_when_it_is(self, mode):
        units = build_units(annotations([("A", 1.0, 2.0, "x"), ("B", 1.0, 2.0, "x")]), **mode)
        assert set(units["recording_id"]) == {"r1"}
        assert set(units["subject_id"]) == {"parent"}
        assert set(units["tier"]) == {"backchannel"}
        assert units["t_start"].min() == pytest.approx(1.0)
        assert units["t_end"].max() == pytest.approx(2.0)

    def test_a_grid_unit_is_its_cell(self):
        units = build_units(annotations([("A", 0.2, 0.9, "x")]), grid_s=0.5)
        assert sorted(zip(units["t_start"], units["t_end"])) == [(0.0, 0.5), (0.5, 1.0)]

    def test_missing_labels_are_dropped(self):
        frame = annotations([("A", 0.0, 1.0, "yes"), ("B", 0.0, 1.0, "yes")])
        frame.loc[0, "label"] = None
        assert len(build_units(frame)) == 1


class TestGridUnits:
    def test_an_interval_starting_on_a_cell_edge_does_not_reach_the_cell_before(self):
        # 0.3 / 0.1 falls just short of 3 and 3 * 0.1 just past 0.3, so cell 2 would seem to hold a hair of it
        units = build_units(annotations([("A", 0.3, 1.0, "x"), ("B", 0.3, 1.0, "x")]), grid_s=0.1)
        assert units["t_start"].min() == 0.3 and units["unit_id"].nunique() == 7

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


class TestPerCodeAlpha:
    def test_one_row_per_code(self):
        table = per_code_alpha(two_coder_units(["a", "b"], ["a", "b"]))
        assert table["code"].tolist() == ["a", "b"]

    def test_an_overall_value_can_hide_one_bad_code(self):
        """Three codes agree and one never does; the total stays high, the per-code row does not."""
        labels_a = ["a", "a", "b", "b", "c", "c", "d", "d"]
        labels_b = ["a", "a", "b", "b", "c", "c", "a", "b"]
        table = per_code_alpha(two_coder_units(labels_a, labels_b)).set_index("code")
        assert krippendorff_alpha(two_coder_units(labels_a, labels_b)) > 0.6
        assert table.loc["d", "alpha"] < 0.0

    def test_blank_grid_cells_are_not_a_code(self):
        units = build_units(
            annotations([("A", 0.0, 0.5, "x"), ("B", 0.0, 1.0, "x")]), grid_s=0.5
        )
        assert NO_CODE not in per_code_alpha(units)["code"].tolist()

    def test_counts_are_reported(self):
        table = per_code_alpha(two_coder_units(["a", "a"], ["a", "b"])).set_index("code")
        assert table.loc["a", "n_ratings"] == 3
        assert table.loc["a", "n_units"] == 2
        assert table.loc["b", "n_units"] == 1

    def test_empty_input_keeps_the_columns(self):
        table = per_code_alpha(pd.DataFrame())
        assert table.empty
        assert "alpha" in table.columns


def three_coder_units(labels):
    """labels are (A, B, M) per unit, None where that coder gave nothing."""
    return units_from(
        [(f"u{index}", name, label) for index, row in enumerate(labels) for name, label in zip("ABM", row) if label is not None]
    )


class TestPerCodePairs:
    def test_one_row_per_code_and_pair(self):
        table = per_code_pairs(three_coder_units([("a", "a", "a"), ("b", "b", "a")]))
        assert len(table) == 6
        assert table[["code", "annotator_a", "annotator_b"]].values.tolist()[:3] == [
            ["a", "A", "B"], ["a", "A", "M"], ["a", "B", "M"],
        ]

    def test_a_model_is_set_beside_the_people(self):
        """The people agree on every unit and the model misses each b; only the pairs with the model show it."""
        labels = [("a", "a", "a"), ("b", "b", "a")] * 5
        table = per_code_pairs(three_coder_units(labels)).set_index(["code", "annotator_a", "annotator_b"])
        assert table.loc[("b", "A", "B"), "kappa"] == pytest.approx(1.0)
        assert table.loc[("b", "A", "M"), "kappa"] == pytest.approx(0.0)
        assert table.loc[("b", "B", "M"), "n_a"] == 5
        assert table.loc[("b", "B", "M"), "n_b"] == 0

    def test_two_coders_match_cohen_kappa_on_a_split(self):
        units = two_coder_units(NINE_A, NINE_B)
        table = per_code_pairs(units).set_index("code")
        binary = units.assign(label=np.where(units["label"] == "2", "2", "rest"))
        assert table.loc["2", "kappa"] == pytest.approx(cohen_kappa(binary, "A", "B"))

    def test_only_units_both_rated_count(self):
        table = per_code_pairs(three_coder_units([("a", "a", None), ("b", "b", "b")])).set_index(["code", "annotator_a", "annotator_b"])
        assert table.loc[("a", "A", "B"), "n_units"] == 2
        assert table.loc[("a", "A", "M"), "n_units"] == 1

    def test_one_coder_has_no_pairs(self):
        assert per_code_pairs(units_from([("u0", "A", "a")])).empty


class TestUnitTable:
    def test_one_row_per_unit_and_a_column_per_coder(self):
        frame = annotations([("A", 0.0, 1.0, "x"), ("B", 0.0, 1.0, "y"), ("A", 1.0, 2.0, "x")])
        table = unit_table(build_units(frame))
        assert list(table.columns) == ["recording_id", "subject_id", "tier", "t_start", "t_end", "A", "B"]
        assert table["A"].tolist() == ["x", "x"]
        assert table["B"].iloc[0] == "y"
        assert pd.isna(table["B"].iloc[1])

    def test_rows_follow_time(self):
        frame = annotations([("A", 5.0, 6.0, "x"), ("A", 0.0, 1.0, "y"), ("B", 5.0, 6.0, "x"), ("B", 0.0, 1.0, "y")])
        assert unit_table(build_units(frame))["t_start"].tolist() == [0.0, 5.0]

    def test_coded_around_is_told_apart_from_not_rated(self):
        frame = annotations([("A", 0.0, 1.0, "x"), ("A", 3.0, 4.0, "y"), ("B", 0.0, 1.0, "x")])
        table = unit_table(build_units(frame, units_of="A"))
        assert table["B"].tolist() == ["x", NO_CODE]

    def test_empty_units_keep_the_place_columns(self):
        assert list(unit_table(build_units(pd.DataFrame())).columns) == ["recording_id", "subject_id", "tier", "t_start", "t_end"]


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
        assert "units" not in json.loads(text)

    def test_the_folder_holds_the_report_and_the_units(self, tmp_path):
        write_report(agreement_report(self.paired(n=5)), tmp_path / "out")
        assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["per_code_pairs"]
        assert len(pd.read_csv(tmp_path / "out" / "units.csv")) == 5

    def test_format_report_mentions_the_headline_number(self):
        text = format_report(agreement_report(self.paired()))
        assert "krippendorff_alpha=1.000" in text


class TestSegments:
    EARLY, LATE = {"r1": (0.0, 10.0)}, {"r1": (10.0, 20.0)}

    def test_a_unit_across_an_edge_is_cut_and_counted_on_both_sides(self):
        frame = annotations([(coder, *span) for coder in ("A", "B") for span in ((2.0, 4.0, "y"), (8.0, 12.0, "x"))])
        assert unit_table(build_units(frame, within=self.EARLY))[["t_start", "t_end"]].values.tolist() == [[2.0, 4.0], [8.0, 10.0]]
        late = unit_table(build_units(frame, within=self.LATE))
        assert late[["t_start", "t_end", "A", "B"]].values.tolist() == [[10.0, 12.0, "x", "x"]]

    def test_the_others_code_is_the_one_covering_most_of_the_cut_unit(self):
        frame = annotations([("A", 8.0, 12.0, "x"), ("B", 8.0, 9.5, "x"), ("B", 9.5, 12.0, "z")])
        assert unit_table(build_units(frame, units_of="A"))["B"].tolist() == ["z"]
        assert unit_table(build_units(frame, units_of="A", within=self.EARLY))["B"].tolist() == ["x"]

    def test_grid_cells_stay_inside_the_segment(self):
        frame = annotations([("A", 0.0, 20.0, "x"), ("B", 0.0, 20.0, "x")])
        assert unit_table(build_units(frame, grid_s=2.5, within=self.EARLY))["t_end"].max() == 10.0

    @pytest.mark.parametrize("mode", [{"grid_s": 1.0}, {"units_of": "A"}])
    def test_a_coder_who_coded_the_person_only_elsewhere_left_the_segment_blank(self, mode):
        frame = annotations([("A", 2.0, 4.0, "x"), ("A", 12.0, 14.0, "x"), ("B", 12.0, 14.0, "x")])
        table = unit_table(build_units(frame, within=self.EARLY, **mode))
        assert set(table["B"]) == {NO_CODE}

    # split on a cell edge, or on an edge of the units, so that nothing is cut
    @pytest.mark.parametrize("mode, middle", [({"grid_s": 0.1}, 5.0), ({"units_of": "A"}, 4.7)])
    def test_the_halves_together_are_the_whole(self, mode, middle):
        frame = annotations([("A", 0.3, 4.7, "x"), ("A", 4.7, 9.9, "y"), ("B", 0.0, 6.2, "x"), ("B", 6.2, 9.0, "y"), ("A", 12.0, 13.0, "x")])
        key = lambda units: set(zip(units["unit_id"], units["annotator"], units["label"]))
        halves = key(build_units(frame, within={"r1": (0.0, middle)}, **mode)) | key(build_units(frame, within={"r1": (middle, 20.0)}, **mode))
        assert halves == key(build_units(frame, **mode))

    def test_recordings_without_the_segment_are_left_out_and_named(self, tmp_path):
        table = tmp_path / "segments.csv"
        table.write_text("recording,segment,start_s,end_s\nr1,early,0,10\nr1,late,10,20\nr2,late,0,5\n", encoding="utf-8")
        config = SimpleNamespace(project=SimpleNamespace(segments=table))
        frame = pd.concat([annotations([("A", 0.0, 1.0, "x")]), annotations([("A", 0.0, 1.0, "x")], recording_id="r3")])
        assert segment_spans(config, frame, "early") == ({"r1": (0.0, 10.0)}, ["r3"])
        with pytest.raises(MacawError, match="the segments are: early, late"):
            segment_spans(config, frame, "middle")

    def test_the_report_names_the_segment(self):
        report = agreement_report(annotations([("A", 0.0, 1.0, "x"), ("B", 0.0, 1.0, "x")]), within=self.EARLY, segment="early")
        assert report_to_json(report)["segment"] == "early"
        assert "segment=early" in format_report(report)
