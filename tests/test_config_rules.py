from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from macaw.config import hashing
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


def test_where_the_language_model_answers_leaves_its_coding_alone(project):
    before = resolve_config(project)
    after = resolve_config(project, cli_overrides=["coding.llm.base_url=http://gpu-node:11434/v1", "coding.llm.timeout_s=900"])
    assert hash_extractor_config(before, "coding.judge") == hash_extractor_config(after, "coding.judge")


def test_model_weights_are_read_once_as_they_stand_and_a_codebook_every_time(project, tmp_path, monkeypatch):
    read = []
    content_hash = hashing._content_hash
    monkeypatch.setattr(hashing, "_content_hash", lambda path: read.append(Path(path).name) or content_hash(path))
    weights, codebook = tmp_path / "weights.pt", tmp_path / "acts.yaml"
    weights.write_bytes(b"a" * hashing._LARGE_BYTES)
    codebook.write_text("name: acts\n", encoding="utf-8")
    data = yaml.safe_load(project.read_text(encoding="utf-8"))
    data["pose"] = {"backend": "yolo", "model": "weights.pt"}
    data["coding"] = {"codebook": "acts.yaml"}
    project.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = resolve_config(project)
    first = [hash_extractor_config(config, "pose.detect") for _ in range(3)]
    assert len(set(first)) == 1 and read.count("weights.pt") == 1 and read.count("acts.yaml") == 3
    assert hashing.hash_referenced_file(weights) == hashlib.sha256(weights.read_bytes()).hexdigest()[: hashing.HASH_LENGTH]
    weights.write_bytes(b"b" * hashing._LARGE_BYTES)
    os.utime(weights, ns=(weights.stat().st_atime_ns, weights.stat().st_mtime_ns + 1_000_000_000))
    assert hash_extractor_config(config, "pose.detect") != first[0] and read.count("weights.pt") == 2


def test_how_many_requests_go_at_once_leaves_the_coding_alone(project):
    before = resolve_config(project)
    after = resolve_config(project, cli_overrides=["coding.llm.concurrency=8"])
    assert hash_extractor_config(before, "coding.judge") == hash_extractor_config(after, "coding.judge")
