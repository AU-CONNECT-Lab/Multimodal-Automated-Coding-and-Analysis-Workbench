from __future__ import annotations

import pytest
import yaml

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import RecordingNotFoundError
from macaw.project.recording import build_recording, detect_recordings


def project(tmp_path, folders, pattern="{group_id}/{task}"):
    for folder in folders:
        (tmp_path / "raw" / folder).mkdir(parents=True, exist_ok=True)
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["project"]["recording_pattern"] = pattern
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return resolve_config(path)


def test_every_detected_recording_is_found_again_from_its_id_when_a_folder_name_holds_an_underscore(tmp_path):
    config = project(tmp_path, ["f1/task_a", "f1/task_b", "fam_2/task_a"])
    ids = detect_recordings(config.project.raw_dir, config.project.recording_pattern)
    assert ids == ["f1_task_a", "f1_task_b", "fam_2_task_a"]
    found = {rid: build_recording(rid, config).entities() for rid in ids}
    assert found["f1_task_a"]["group_id"] == "f1" and found["f1_task_a"]["task"] == "task_a"
    assert (found["fam_2_task_a"]["group_id"], found["fam_2_task_a"]["task"]) == ("fam_2", "task_a")


def test_an_id_two_existing_folders_could_stand_for_is_refused_with_both(tmp_path):
    config = project(tmp_path, ["f1/task_a", "f1_task/a"])
    with pytest.raises(RecordingNotFoundError, match=r"more than one recording folder \(f1/task_a, f1_task/a\)"):
        build_recording("f1_task_a", config)


def test_an_id_the_pattern_cannot_read_is_refused(tmp_path):
    config = project(tmp_path, ["f1"], pattern="{group_id}/{task}")
    with pytest.raises(RecordingNotFoundError, match="does not look like"):
        build_recording("f1", config)
