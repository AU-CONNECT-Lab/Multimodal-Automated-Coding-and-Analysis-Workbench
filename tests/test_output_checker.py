from __future__ import annotations

import pytest
import yaml

from macaw.exceptions import ConfigValidationError
from macaw.execution.output_checker import (
    OutputRule,
    check_task,
    completed_tasks,
    load_rules,
    rules_for,
)


@pytest.fixture
def recording_dir(tmp_path):
    def build(files):
        root = tmp_path / "recordings" / "rec1"
        for relative, size in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * size)
        root.mkdir(parents=True, exist_ok=True)
        return root

    return build


@pytest.fixture
def rules_file(tmp_path):
    def build(payload):
        path = tmp_path / "rules.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    return build


class TestLoading:
    def test_a_rule_reads_back(self, rules_file):
        path = rules_file({"prep.media": [{"pattern": "artifacts/prep.media/*.wav", "min_count": 2}]})
        rule = load_rules(path)[0]
        assert (rule.extractor, rule.min_count) == ("prep.media", 2)

    def test_an_empty_file_gives_no_rules(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_rules(path) == []

    def test_a_rule_without_a_pattern_is_refused(self, rules_file):
        with pytest.raises(ConfigValidationError, match="no pattern"):
            load_rules(rules_file({"prep.media": [{"min_count": 2}]}))

    def test_an_unknown_key_is_refused(self, rules_file):
        """A typo would otherwise be accepted and never checked."""
        with pytest.raises(ConfigValidationError, match="min_sizes_kb"):
            load_rules(rules_file({"prep.media": [{"pattern": "*.wav", "min_sizes_kb": 1}]}))

    def test_a_list_at_the_top_level_is_refused(self, tmp_path):
        path = tmp_path / "rules.yaml"
        path.write_text("- pattern: x\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError, match="mapping"):
            load_rules(path)


class TestDefaultRule:
    def test_an_extractor_without_a_rule_gets_the_store_default(self):
        """Most extractors write one parquet under features, so most need no rule at all."""
        rule = rules_for("prosody.frame", [])[0]
        assert rule.pattern == "features/prosody.frame.parquet"

    def test_a_declared_rule_replaces_the_default(self):
        declared = [OutputRule(extractor="prep.media", pattern="artifacts/prep.media/*.wav")]
        assert rules_for("prep.media", declared) == declared

    def test_another_extractors_rule_does_not_apply(self):
        declared = [OutputRule(extractor="prep.media", pattern="x")]
        assert rules_for("prosody.frame", declared)[0].pattern.endswith("prosody.frame.parquet")


class TestChecking:
    def test_a_written_output_passes(self, recording_dir):
        root = recording_dir({"features/prosody.frame.parquet": 100})
        assert check_task(root, "prosody.frame", [], "rec1:prosody.frame").complete

    def test_a_missing_output_fails_and_names_the_pattern(self, recording_dir):
        root = recording_dir({})
        result = check_task(root, "prosody.frame", [], "rec1:prosody.frame")
        assert not result.complete
        assert result.missing == ["features/prosody.frame.parquet"]

    def test_a_file_below_the_size_floor_fails(self, recording_dir):
        root = recording_dir({"features/x.parquet": 10})
        rules = [OutputRule(extractor="x", pattern="features/x.parquet", min_size_kb=1.0)]
        result = check_task(root, "x", rules, "rec1:x")
        assert not result.complete
        assert result.undersized

    def test_a_file_above_the_size_floor_passes(self, recording_dir):
        root = recording_dir({"features/x.parquet": 4096})
        rules = [OutputRule(extractor="x", pattern="features/x.parquet", min_size_kb=1.0)]
        assert check_task(root, "x", rules, "rec1:x").complete

    def test_too_few_matches_fails(self, recording_dir):
        root = recording_dir({"artifacts/prep.media/parent.wav": 100})
        rules = [OutputRule(extractor="prep.media", pattern="artifacts/prep.media/*.wav", min_count=2)]
        assert not check_task(root, "prep.media", rules, "rec1:prep.media").complete

    def test_enough_matches_passes(self, recording_dir):
        root = recording_dir(
            {"artifacts/prep.media/parent.wav": 100, "artifacts/prep.media/child.wav": 100}
        )
        rules = [OutputRule(extractor="prep.media", pattern="artifacts/prep.media/*.wav", min_count=2)]
        assert check_task(root, "prep.media", rules, "rec1:prep.media").complete

    def test_an_optional_output_may_be_absent(self, recording_dir):
        root = recording_dir({})
        rules = [OutputRule(extractor="x", pattern="features/x.parquet", optional=True)]
        assert check_task(root, "x", rules, "rec1:x").complete

    def test_a_directory_counts_by_the_files_inside_it(self, recording_dir):
        """The three tables are directories of parquet parts, not single files."""
        root = recording_dir({"features/prep.media.parquet": 10, "features/asr.parquet": 10})
        rules = [OutputRule(extractor="x", pattern="features", min_count=2)]
        assert check_task(root, "x", rules, "rec1:x").complete

    def test_the_task_id_is_carried_through(self, recording_dir):
        root = recording_dir({})
        assert check_task(root, "x", [], "rec1:x").task_id == "rec1:x"


class TestResume:
    def test_a_finished_task_is_reported(self, recording_dir):
        recording_dir({"features/prosody.frame.parquet": 100})
        output_dir = recording_dir({}).parent.parent
        assert "rec1:prosody.frame" in completed_tasks(output_dir, [])

    def test_nothing_written_means_nothing_to_skip(self, tmp_path):
        assert completed_tasks(tmp_path, []) == set()

    def test_a_missing_recordings_directory_is_not_an_error(self, tmp_path):
        assert completed_tasks(tmp_path / "nowhere", []) == set()

    def test_without_rules_the_written_parquet_files_are_the_evidence(self, recording_dir):
        recording_dir({"features/a.parquet": 10, "features/b.parquet": 10})
        output_dir = recording_dir({}).parent.parent
        assert completed_tasks(output_dir, []) == {"rec1:a", "rec1:b"}

    def test_declared_rules_decide_which_extractors_are_looked_for(self, recording_dir):
        recording_dir({"features/a.parquet": 10})
        output_dir = recording_dir({}).parent.parent
        rules = [OutputRule(extractor="b", pattern="features/b.parquet")]
        assert completed_tasks(output_dir, rules) == set()
