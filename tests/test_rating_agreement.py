from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.coding.ratings import (
    draws, icc_a1, known_scales, linear_kappa, pair_statistics, people_ratings, rating_agreement, ratings_table,
)
from macaw.coding.scales import RatingScale
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError, MacawError
from macaw.project.layout import layout_for

# twelve units two raters scored on a 1 to 5 scale, every score used by both
X = np.array([1, 2, 3, 4, 5, 3, 2, 4, 5, 1, 3, 4], dtype=float)
Y = np.array([2, 2, 3, 5, 4, 3, 1, 4, 5, 2, 2, 4], dtype=float)
SCALES = {
    "name": "talk",
    "scales": [
        {"scale": "warmth", "person": "parent", "range": [1, 5], "definition": "How warm."},
        {"scale": "on_task", "person": "child", "range": [0, 1], "definition": "Keeps to the task."},
    ],
}


class TestStatistics:
    def test_icc_and_weighted_kappa_match_an_independent_implementation(self):
        one = np.ones((1, len(X)))
        # ICC(2,1) and linearly weighted kappa computed elsewhere on the same scores
        assert icc_a1(one, X, Y)[0] == pytest.approx(0.8601694915, abs=1e-9)
        assert linear_kappa(one, X, Y)[0] == pytest.approx(0.6571428571, abs=1e-9)
        found = pair_statistics(one, X, Y, two_point=False)
        assert found["exact"][0] == pytest.approx(0.5) and found["within_one"][0] == 1.0

    def test_a_unit_drawn_twice_counts_as_two_copies(self):
        weights = np.array([[2, 1, 0, 1, 3, 1, 1, 0, 1, 1, 2, 1]], dtype=float)
        repeated = weights[0].astype(int)
        x, y = np.repeat(X, repeated), np.repeat(Y, repeated)
        one = np.ones((1, len(x)))
        for function in (icc_a1, linear_kappa):
            assert function(weights, X, Y)[0] == pytest.approx(function(one, x, y)[0], abs=1e-12)

    def test_kappa_counts_points_apart_not_categories_between(self):
        # a median between two answers can be a half point; it lies half a point from each neighbour
        x, y = np.array([1.0, 2.5, 4.0, 5.0]), np.array([1.0, 3.0, 4.0, 4.0])
        assert linear_kappa(np.ones((1, 4)), x, y)[0] == pytest.approx(1 - (1.5 / 4) / np.mean(np.abs(x[:, None] - y[None, :])))

    def test_a_whole_group_is_drawn_or_left_together(self):
        groups = np.array(["f1", "f1", "f2", "f2", "f3", "f3"])
        drawn = draws(groups, 500, np.random.default_rng(0))
        assert (drawn[:, 0] == drawn[:, 1]).all() and (drawn[:, 2] == drawn[:, 3]).all()
        assert (drawn.sum(axis=1) == 6).all()
        assert (draws(groups, 50, np.random.default_rng(3)) == draws(groups, 50, np.random.default_rng(3))).all()


def table_of(scores: dict[str, list[float]], scale="warmth", families=None):
    rows = []
    for rater, values in scores.items():
        for index, value in enumerate(values):
            rows.append({
                "recording_id": f"r{index}", "segment": "", "scale": scale, "person": "parent", "rater": rater,
                "score": value, "group_id": (families or {}).get(index, f"f{index}"),
            })
    return pd.DataFrame(rows)


class TestAgreement:
    SCALES = {"warmth": RatingScale("warmth", "parent", 1, 5), "on_task": RatingScale("on_task", "child", 0, 1)}

    def test_each_pair_is_compared_on_what_both_scored(self):
        table = table_of({"anna": list(X), "ben": list(Y), "k1": [*X[:6], np.nan, *X[7:]]})
        found = rating_agreement(table, self.SCALES, n_boot=200).set_index(["rater_a", "rater_b"])
        assert found.loc[("anna", "ben"), "n"] == 12 and found.loc[("anna", "k1"), "n"] == 11
        assert found.loc[("anna", "ben"), "icc"] == pytest.approx(0.8601694915, abs=1e-9)
        assert found.loc[("anna", "k1"), "exact"] == 1.0
        low, high = found.loc[("anna", "ben"), ["icc_low", "icc_high"]]
        assert low < 0.86 < high

    def test_the_same_seed_gives_the_same_intervals_and_raters_can_be_picked(self):
        table = table_of({"anna": list(X), "ben": list(Y), "k1": list(X)})
        first = rating_agreement(table, self.SCALES, raters=["anna", "ben"], n_boot=300, seed=4)
        again = rating_agreement(table, self.SCALES, raters=["anna", "ben"], n_boot=300, seed=4)
        assert len(first) == 1 and first.equals(again)

    def test_a_two_point_scale_has_no_within_one(self):
        table = table_of({"anna": [1, 0, 1, 1], "ben": [1, 0, 0, 1]}, scale="on_task")
        row = rating_agreement(table, self.SCALES, n_boot=0).iloc[0]
        assert row["exact"] == 0.75 and np.isnan(row["within_one"]) and np.isnan(row["icc_low"])

    def test_resampling_by_family_draws_its_recordings_together(self):
        families = {index: f"f{index // 2}" for index in range(12)}
        table = table_of({"anna": list(X), "ben": list(Y)}, families=families)
        by_family = rating_agreement(table, self.SCALES, cluster="group_id", n_boot=400).iloc[0]
        assert by_family["n"] == 12 and by_family["n_clusters"] == 6
        with pytest.raises(MacawError, match="no column family"):
            rating_agreement(table, self.SCALES, cluster="family", n_boot=10)


def project(tmp_path, people=None):
    for family in ("F1", "F2"):
        for task in ("plan", "play"):
            (tmp_path / "raw" / family / task).mkdir(parents=True, exist_ok=True)
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["recording_pattern"] = "{group_id}/{task}"
    data["coding"] = {"llm": {"model": "tiny"}, "ratings": {"k1": {"scales": "scales.yaml"}}}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    (tmp_path / "scales.yaml").write_text(yaml.safe_dump(SCALES), encoding="utf-8")
    if people is not None:
        pd.DataFrame(people).to_csv(tmp_path / "ratings.csv", index=False)
    return path


def model_scores(config, rid, scores: dict[str, float]):
    folder = layout_for(config).artifacts_dir(rid, "coding.rate.k1")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"recording_id": rid, "segment": "", "t_start": 0.0, "t_end": 60.0, "scale": scale, "person": "parent", "rater": "k1",
         "score": value, "score_sd": 0.5, "score_min": value, "score_max": value, "n_answers": 5, "n_asked": 5}
        for scale, value in scores.items()
    ]).to_parquet(folder / "ratings.parquet", index=False)
    pd.DataFrame([{"recording_id": rid, "segment": "", "scale": "warmth", "rater": "k1", "sample": 0, "score": 3.0, "problem": "", "reason": "", "answer": '{"score": 3}'}]).to_parquet(folder / "answers.parquet", index=False)


PEOPLE = [
    {"recording": "F1_plan", "segment": "", "scale": "warmth", "coder": "anna", "score": 4, "family": "a"},
    {"recording": "F1_play", "segment": "", "scale": "warmth", "coder": "anna", "score": 2, "family": "a"},
    {"recording": "F2_plan", "segment": "", "scale": "warmth", "coder": "anna", "score": 5, "family": "b"},
    {"recording": "F2_play", "segment": "", "scale": "warmth", "coder": "anna", "score": 3, "family": "b"},
]


class TestTables:
    def test_people_s_scores_are_read_and_checked_line_by_line(self, tmp_path):
        config = resolve_config(project(tmp_path, PEOPLE))
        found = people_ratings(config, known_scales(config), ["F1_plan", "F1_play", "F2_plan", "F2_play"])
        assert found["rater"].tolist() == ["anna"] * 4 and (found["rater_kind"] == "person").all() and found["family"].tolist() == ["a", "a", "b", "b"]

    @pytest.mark.parametrize("change,message", [
        ({"recording": "F9_plan"}, "line 2: F9_plan is not one of"),
        ({"scale": "cheer"}, "defines a scale cheer"),
        ({"coder": "k1"}, "name of a version"),
        ({"score": 6}, "whole scores from 1 to 5"),
        ({"score": 2.5}, "whole scores from 1 to 5"),
    ])
    def test_a_line_that_does_not_fit_says_where(self, tmp_path, change, message):
        config = resolve_config(project(tmp_path, [{**PEOPLE[0], **change}, *PEOPLE[1:]]))
        with pytest.raises(ConfigError, match=message):
            people_ratings(config, known_scales(config), ["F1_plan", "F1_play", "F2_plan", "F2_play"])

    def test_the_same_score_twice_is_refused(self, tmp_path):
        config = resolve_config(project(tmp_path, [*PEOPLE, PEOPLE[0]]))
        with pytest.raises(ConfigError, match="lines 2, 6"):
            people_ratings(config, known_scales(config), ["F1_plan", "F1_play", "F2_plan", "F2_play"])

    def test_the_model_and_people_side_by_side_with_each_recording_s_entities(self, tmp_path):
        config = resolve_config(project(tmp_path, PEOPLE))
        for rid, value in (("F1_plan", 4), ("F1_play", 3), ("F2_plan", 5)):
            model_scores(config, rid, {"warmth": value})
        table = ratings_table(config, ["F1_plan", "F1_play", "F2_plan", "F2_play"], ["F1_plan", "F1_play", "F2_plan", "F2_play"])
        assert len(table) == 7 and set(table["rater_kind"]) == {"model", "person"}
        model = table[table["rater"] == "k1"].set_index("recording_id")
        assert model.loc["F1_play", "group_id"] == "F1" and model.loc["F1_play", "task"] == "play" and model.loc["F1_play", "family"] == "a"

    def test_the_commands_write_the_tables_and_the_agreement(self, tmp_path):
        path = project(tmp_path, PEOPLE)
        config = resolve_config(path)
        for rid, value in (("F1_plan", 4), ("F1_play", 3), ("F2_plan", 5), ("F2_play", 3)):
            model_scores(config, rid, {"warmth": value})
        runner = CliRunner()
        result = runner.invoke(app, ["ratings", "export", str(tmp_path / "out_ratings"), "-c", str(path)])
        assert result.exit_code == 0, result.output
        assert "8 scores from 1 model version(s) and 1 coder(s), 4 answers" in result.output
        assert len(pd.read_csv(tmp_path / "out_ratings" / "scores.csv")) == 8
        result = runner.invoke(app, ["ratings", "agreement", "-c", str(path), "--cluster", "group_id", "--boot", "100", "--out", str(tmp_path / "agree")])
        assert result.exit_code == 0, result.output
        assert "warmth (parent)" in result.output and "anna" in result.output
        found = pd.read_csv(tmp_path / "agree" / "agreement.csv")
        assert found.loc[0, "n"] == 4 and found.loc[0, "n_clusters"] == 2 and found.loc[0, "exact"] == 0.75
        result = runner.invoke(app, ["ratings", "agreement", "-c", str(path), "--raters", "anna,carl"])
        assert result.exit_code != 0 and "no scores by carl" in str(result.exception or result.output)

    def test_the_export_says_what_each_column_holds_and_how_each_rater_rated(self, tmp_path):
        import json

        from macaw.project.manifest import FeatureManifest, FeatureSpec, write_manifest
        from macaw.store.schema import Scale

        path = project(tmp_path, PEOPLE)
        config = resolve_config(path)
        for rid, notes in (("F1_plan", "They plan."), ("F1_play", "They play."), ("F2_plan", "They plan.")):
            model_scores(config, rid, {"warmth": 4})
            fragment = FeatureManifest()
            fragment.add(FeatureSpec(
                name="share_unread_answers_k1", description="", unit="proportion", scale=Scale.RECORDING, group="coding",
                extractor="coding.rate.k1", extractor_version="0.1.0", backend="ollama", backend_version="qwen3:14b",
                params={"rating": "k1", "model": "qwen3:14b", "n_samples": 5, "notes": notes},
            ))
            write_manifest(fragment, layout_for(config).manifest_fragment_path(rid, "coding.rate.k1"))
        result = CliRunner().invoke(app, ["ratings", "export", str(tmp_path / "out"), "-c", str(path)])
        assert result.exit_code == 0, result.output
        scores = pd.read_csv(tmp_path / "out" / "scores_dictionary.csv", keep_default_na=False).set_index("column")
        assert list(scores.index) == list(pd.read_csv(tmp_path / "out" / "scores.csv").columns)
        assert scores.at["score", "unit"] == "points" and scores.at["score_sd", "missing"].startswith("Empty for a person's score")
        assert scores.at["family", "meaning"].startswith("A column added to the project's ratings table")
        assert scores.at["scale", "settings"] == "warmth.defined_in=[scales.yaml]; warmth.person=parent; warmth.range=[1, 5]"
        sidecar = json.loads((tmp_path / "out" / "scores_dictionary.json").read_text(encoding="utf-8"))
        raters = sidecar["rater"]["settings"]
        assert raters["anna"] == {"kind": "person", "table": "ratings.csv"}
        model = raters["k1"]
        assert model["kind"] == "model" and model["scales_file"] == "scales.yaml"
        assert [variant["recordings"] for variant in model["by_recording"]] == [["F1_plan", "F2_plan"], ["F1_play"]]
        assert model["by_recording"][0]["settings"]["step"] == "coding.rate.k1 0.1.0" and "rating" not in model["by_recording"][0]["settings"]
        assert set(sidecar["rater_kind"]["Levels"]) == {"model", "person"}
        answers = pd.read_csv(tmp_path / "out" / "answers_dictionary.csv", keep_default_na=False).set_index("column")
        assert list(answers.index) == list(pd.read_csv(tmp_path / "out" / "answers.csv").columns)
        assert answers.at["problem", "missing"] == "Empty where the answer gave a score."
