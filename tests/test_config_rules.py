from __future__ import annotations

import pytest

from macaw.config.hashing import hash_extractor_config
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return path


def test_each_person_needs_a_role(project):
    with pytest.raises(ConfigError, match="one role per person"):
        resolve_config(project, cli_overrides=["project.n_subjects=3"])
    assert resolve_config(project, cli_overrides=["project.n_subjects=3", "project.roles=[a, b, c]"]).project.roles == ["a", "b", "c"]


def test_a_language_model_setting_leaves_rule_based_coding_alone(project):
    before = resolve_config(project)
    after = resolve_config(project, cli_overrides=["coding.llm.n_samples=3"])
    assert hash_extractor_config(before, "coding.layer_a") == hash_extractor_config(after, "coding.layer_a")
    assert hash_extractor_config(before, "coding.judge") != hash_extractor_config(after, "coding.judge")


def test_a_backchannel_word_leaves_the_language_model_alone(project):
    before = resolve_config(project)
    after = resolve_config(project, cli_overrides=["coding.backchannel_lexicon=[mm]"])
    assert hash_extractor_config(before, "coding.judge") == hash_extractor_config(after, "coding.judge")
    assert hash_extractor_config(before, "coding.layer_a") != hash_extractor_config(after, "coding.layer_a")
