from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from macaw.config.schema import MacawConfig, PlacedCamera, PlacedPerson, RoomSection, SetupSection
from macaw.project import room as room_module
from macaw.project.setup import setup_content


class TestCameraView:
    def test_a_level_camera_facing_along_the_room_sees_across_as_right(self):
        view = room_module.view_of(PlacedCamera(at=(0, 0), height=100, facing=0))
        assert np.allclose(view.local([[50, 200, 130]]), [[50, 200, 30]])

    def test_a_camera_turned_to_face_across_has_the_far_wall_on_its_right(self):
        view = room_module.view_of(PlacedCamera(at=(0, 0), height=100, facing=90))
        assert np.allclose(view.local([[200, -40, 100]]), [[40, 200, 0]])

    def test_a_tilted_camera_looks_down_its_line_of_sight(self):
        view = room_module.view_of(PlacedCamera(at=(0, 0), height=200, facing=0, tilt=45))
        right, ahead, up = view.local([[0, 100, 100]])[0]
        assert right == pytest.approx(0) and up == pytest.approx(0) and ahead == pytest.approx(100 * np.sqrt(2))

    def test_points_land_in_the_picture_by_the_lens_angle(self):
        view = room_module.view_of(PlacedCamera(at=(0, 0), height=100, facing=0, lens=90))
        assert np.allclose(view.pixels([[0, 100, 100], [100, 100, 100], [0, 100, 150]], (200, 100)), [[100, 50], [200, 50], [100, 0]])
        assert np.isnan(view.pixels([[0, -100, 100]], (200, 100))).all()

    def test_a_camera_without_its_place_height_or_facing_has_no_view(self):
        assert room_module.view_of(PlacedCamera(at=(0, 0), height=100)) is None


class TestSeats:
    def test_seats_are_seen_from_the_camera_with_people_at_their_eyes_and_things_at_their_tops(self):
        room = RoomSection(
            cameras={"front": {"at": [100, 0], "height": 120, "facing": 0}},
            people={"parent": {"at": [50, 150], "eyes": 115}, "child": {"at": [150, 150]}},
            things={"toy": {"at": [100, 100], "height": 70}},
        )
        seats = room_module.seats(room, "front")
        assert seats["camera"] == (0.0, 0.0, 0.0)
        assert np.allclose(seats["parent"], (-50, 150, -5)) and np.allclose(seats["toy"], (0, 100, -50))
        assert "child" not in seats
        assert room_module.seats(room, "side") == {}


class TestNotes:
    def test_angle_is_zero_facing_the_camera_and_180_with_it_behind(self):
        camera = PlacedCamera(at=(0, 0), height=100, facing=0)
        assert room_module.angle_to(camera, PlacedPerson(at=(0, 100), facing=180)) == pytest.approx(0)
        assert room_module.angle_to(camera, PlacedPerson(at=(0, 100), facing=0)) == pytest.approx(180)
        assert room_module.angle_to(camera, PlacedPerson(at=(0, 100), facing=90)) == pytest.approx(90)
        assert room_module.angle_to(camera, PlacedPerson(at=(0, 100))) is None

    def test_distance_and_whether_someone_is_in_the_picture(self):
        camera = PlacedCamera(at=(0, 0), height=100, facing=0, lens=60)
        assert room_module.distance(camera, PlacedPerson(at=(0, 300), eyes=100)) == pytest.approx(300)
        assert room_module.in_picture(camera, [50, 300, 100]) is True
        assert room_module.in_picture(camera, [300, 300, 100]) is False

    def test_units_convert_both_ways(self):
        assert room_module.to_unit(2.54, "in") == pytest.approx(1.0) and room_module.from_unit(1.0, "in") == pytest.approx(2.54)


class TestTheSetupFile:
    def test_the_room_is_written_without_what_is_left_blank(self):
        setup = SetupSection(
            cameras={"front": {"body": ["parent"]}},
            room={"people": {"parent": {"at": [1, 2]}, "child": {}}, "cameras": {"front": {}}},
        )
        assert setup_content(setup)["room"] == {"people": {"parent": {"at": [1, 2]}}}
        # a tilt of 0 says the camera is level, which a blank tilt does not
        level = SetupSection(cameras={"front": {"body": ["parent"]}}, room={"cameras": {"front": {"tilt": 0}}})
        assert setup_content(level)["room"] == {"cameras": {"front": {"tilt": 0}}}
        walled = SetupSection(cameras={"front": {"body": ["parent"]}}, room={"size": [400.0, 350.5]})
        assert setup_content(walled)["room"] == {"size": [400, 350.5]}
        assert "room" not in setup_content(SetupSection(cameras={"front": {"body": ["parent"]}}))

    def test_the_room_places_only_the_setups_devices_and_the_projects_people(self):
        with pytest.raises(ValidationError, match="the room places side, which is not one of the setup's cameras"):
            SetupSection(cameras={"front": {"body": ["parent"]}}, room={"cameras": {"side": {"at": [0, 0]}}})
        with pytest.raises(ValidationError, match="both a device and a thing"):
            SetupSection(cameras={"front": {"body": ["parent"]}}, room={"things": {"front": {"at": [0, 0]}}})
        base = {"project": {"name": "t", "raw_dir": "raw", "output_dir": "out"}}
        with pytest.raises(ValidationError, match="places teacher in the room"):
            MacawConfig.model_validate({**base, "setup": {"room": {"people": {"teacher": {"at": [0, 0]}}}}})
        with pytest.raises(ValidationError, match="both a role and a thing"):
            MacawConfig.model_validate({**base, "setup": {"room": {"things": {"child": {"at": [0, 0]}}}}})
        with pytest.raises(ValidationError, match="a thing's name"):
            RoomSection(things={"camera": {}})


def test_the_room_is_drawn_for_a_paper_at_300_dpi(tmp_path):
    import yaml
    from typer.testing import CliRunner

    from macaw.cli.main import app
    from macaw.config.loader import scaffold_project_config
    from setup_helpers import camera, write_setup

    pytest.importorskip("matplotlib")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    path = write_setup(config, cameras={"front": camera()})
    refused = CliRunner().invoke(app, ["draw-room", "-c", str(config)])
    assert refused.exit_code != 0 and "places nothing yet" in str(refused.exception)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["room"] = {
        "size": [400, 350],
        "people": {"parent": {"at": [100, 200], "eyes": 115, "facing": 90}, "child": {"at": [200, 200], "eyes": 70}},
        "cameras": {"front": {"at": [150, 0], "height": 120, "facing": 0, "tilt": 10, "lens": 70}},
        "things": {"table": {"at": [150, 200], "height": 72, "size": [60, 50]}},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = CliRunner().invoke(app, ["draw-room", "-c", str(config)])
    assert result.exit_code == 0, result.output
    drawn = path.with_suffix(".png")
    from matplotlib.image import imread

    png = drawn.read_bytes()
    # pixels per metre in the pHYs chunk: 300 dpi is 11811
    at = png.index(b"pHYs") + 4
    assert int.from_bytes(png[at:at + 4], "big") == 11811
    assert imread(drawn).shape[1] >= 1500


def test_a_name_too_long_for_its_circle_sits_behind_the_person():
    from macaw.project.room_figure import name_beside

    # facing the right, the arrow points right, so the name ends left of the circle
    at, ha, va = name_beside([100, 200], 90, 30)
    assert at[0] == pytest.approx(70) and at[1] == pytest.approx(200) and (ha, va) == ("right", "center")
    at, ha, va = name_beside([100, 200], 180, 30)
    assert at[1] == pytest.approx(230) and (ha, va) == ("center", "bottom")
    # no facing, no arrow: to the right as before
    at, ha, va = name_beside([100, 200], None, 30)
    assert at[0] == pytest.approx(130) and (ha, va) == ("left", "center")
