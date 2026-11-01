from __future__ import annotations

import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.project.gaze_labels import GazeLabel, add_label, labels_frame, read_labels
from macaw.project.layout import layout_for

runner = CliRunner()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav", "003_room.mp4"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    with open(config, "a", encoding="utf-8") as handle:
        handle.write("\ngaze:\n  targets: [bottle]\n")
    return config


def label(project, *extra):
    return runner.invoke(app, ["label-gaze", "-c", str(project), "-r", "003", *extra])


def stored(project):
    return read_labels(layout_for(resolve_config(project)).gaze_labels_path("003"))


def test_a_label_is_kept_and_one_for_the_same_frame_replaces_it(project):
    assert label(project, "--at", "12.0", "--person", "child", "--looks-at", "parent").exit_code == 0
    assert label(project, "--at", "12.0", "--person", "child", "--looks-at", "bottle").exit_code == 0
    assert label(project, "--at", "12.0", "--person", "parent", "--looks-at", "unsure").exit_code == 0
    assert [(item.subject, item.target) for item in stored(project)] == [("child", "bottle"), ("parent", "unsure")]
    assert labels_frame(stored(project)).values.tolist() == [[12.0, "child", "bottle"]]


@pytest.mark.parametrize("extra, message", [
    (("--person", "nobody", "--looks-at", "parent"), "takes a role"),
    (("--person", "child", "--looks-at", "child"), "--looks-at takes one of"),
    (("--person", "child", "--looks-at", "toy"), "--looks-at takes one of"),
])
def test_only_roles_named_things_elsewhere_and_unsure_are_taken(project, extra, message):
    result = label(project, "--at", "1", *extra)
    assert result.exit_code != 0 and message in str(result.exception)


def test_clear_removes_them(project):
    label(project, "--at", "1", "--person", "child", "--looks-at", "elsewhere")
    assert label(project, "--clear").exit_code == 0 and stored(project) == []


def test_add_label_replaces_within_half_a_frame_only():
    labels = [GazeLabel(1.00, "child", "parent")]
    assert len(add_label(labels, GazeLabel(1.01, "child", "bottle"), period_s=0.04)) == 1
    assert len(add_label(labels, GazeLabel(1.05, "child", "bottle"), period_s=0.04)) == 2
