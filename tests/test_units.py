from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.exceptions import ConfigError
from macaw.interface.units import convert_entered, room_data
from macaw.project.heights import read_heights


def test_what_was_measured_for_a_camera_placed_by_the_calibration_changes_unit_with_it():
    room = {"cameras": {"front": {"at": [118.4, 31.2], "height": 151, "facing": 10, "measured": {"at": [100, 50], "height": 150, "facing": 10}}}}
    camera = room_data(room, 1 / 100)["cameras"]["front"]
    assert camera["measured"] == {"at": [1, 0.5], "height": 1.5, "facing": 10}
    assert camera["at"] == [1.184, 0.312] and camera["facing"] == 10


def test_the_heights_table_changes_unit_with_the_project(tmp_path):
    heights = tmp_path / "heights.csv"
    heights.write_text("recording,person,height,note\n003,parent,170,tall\n003,child,,\n", encoding="utf-8")
    config = SimpleNamespace(project=SimpleNamespace(recording_config_dir=tmp_path / "none", setups_dir=tmp_path / "none", heights=heights))
    project = tmp_path / "project.yaml"
    project.write_text("project:\n  units: m\n", encoding="utf-8")
    assert heights in convert_entered(config, project, "cm", "m")
    table = pd.read_csv(heights, dtype=str, keep_default_na=False)
    assert table["height"].tolist() == ["1.7", ""] and table["note"].tolist() == ["tall", ""]


def test_a_height_in_the_wrong_unit_is_refused(tmp_path):
    heights = tmp_path / "heights.csv"
    heights.write_text("recording,person,height\n003,parent,170\n", encoding="utf-8")
    config = SimpleNamespace(project=SimpleNamespace(heights=heights, roles=("parent", "child"), units="m"))
    with pytest.raises(ConfigError, match="not a person's"):
        read_heights(config)
    config.project.units = "cm"
    assert read_heights(config) == {("003", "parent"): 170.0}
