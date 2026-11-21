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
    config = SimpleNamespace(project=SimpleNamespace(
        recording_config_dir=tmp_path / "none", setups_dir=tmp_path / "none", heights=heights, calibration_dir=tmp_path / "none",
    ))
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


def test_every_length_typed_in_the_project_s_unit_changes_with_it_to_its_significant_digits(tmp_path):
    from macaw.project.placing import floor_said, write_floor

    project = tmp_path / "project.yaml"
    project.write_text(
        "project:\n  units: m\nscale:\n  per_px: 0.0523\nkinematics:\n  moving:\n    speed_3d: 30\n", encoding="utf-8",
    )
    (tmp_path / "calibration" / "a").mkdir(parents=True)
    write_floor(tmp_path / "calibration" / "a", 2.0, "floor", (120.0, 80.0), 90.0)
    config = SimpleNamespace(project=SimpleNamespace(
        recording_config_dir=tmp_path / "none", setups_dir=tmp_path / "none", heights=tmp_path / "none.csv",
        calibration_dir=tmp_path / "calibration",
    ))
    changed = convert_entered(config, project, "cm", "m")
    import yaml

    data = yaml.safe_load(project.read_text(encoding="utf-8"))
    assert data["scale"]["per_px"] == 0.000523 and data["kinematics"]["moving"]["speed_3d"] == 0.3
    assert floor_said(tmp_path / "calibration" / "a")["corner"] == (1.2, 0.8) and len(changed) == 2


def test_each_setting_said_to_be_in_the_project_s_unit_is_converted():
    from pathlib import Path

    import macaw.config
    from macaw.interface.units import BY_ROLE, LENGTHS

    # the dotted key of each line of the defaults whose comment gives project.units
    stack, typed = [], set()
    for line in (Path(macaw.config.__file__).parent / "defaults.yaml").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        depth = (len(line) - len(line.lstrip())) // 2
        stack[depth:] = [line.strip().split(":", 1)[0]]
        if "# project.units" in line:
            typed.add(".".join(stack))
    assert typed and typed <= set(LENGTHS) | set(BY_ROLE)
