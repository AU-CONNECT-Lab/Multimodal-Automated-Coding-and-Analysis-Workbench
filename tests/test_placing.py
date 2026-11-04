from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from macaw.config.schema import RoomSection
from macaw.extractors.video.pose.cameras import Camera, CameraSetup
from macaw.project.placing import Floor, _level_axes, angles_of, lens_angle, place_cameras

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
# the calibration's frame against the room's: turned and shifted, in millimetres where the room is in centimetres
TURN = Rotation.from_euler("xyz", [101.0, -17.0, 33.0], degrees=True)
SHIFT = np.array([123.0, -45.0, 210.0])
# where each camera truly stands in the room: across, along, height, facing, tilt, turn about its line of sight
TRUTH = {
    "front": (200.0, 20.0, 150.0, 0.0, 15.0, 0.0),
    "side": (380.0, 150.0, 160.0, 270.0, 20.0, 0.0),
    "corner": (30.0, 280.0, 210.0, 145.0, 35.0, 0.0),
    "back": (220.0, 290.0, 120.0, 185.0, 5.0, 0.0),
}


def calibrated(truth=TRUTH) -> CameraSetup:
    """The cameras as a calibration would give them: in its own frame, in millimetres."""
    cameras = []
    for name, (x, y, height, facing, tilt, roll) in truth.items():
        axes = _level_axes(facing, tilt) @ Rotation.from_euler("y", roll, degrees=True).as_matrix()
        right, ahead, up = (TURN.inv().apply(axes[:, index]) for index in range(3))
        where = TURN.inv().apply((np.array([x, y, height]) - SHIFT) / 0.1)
        to_camera = np.vstack([right, -up, ahead])
        cameras.append(Camera(name, (1920, 1080), K, np.zeros(5), Rotation.from_matrix(to_camera).as_rotvec(), -to_camera @ where))
    return CameraSetup(cameras, "mm")


def room_with(given: dict[str, tuple[str, ...]], truth=TRUTH, **things) -> RoomSection:
    cameras = {}
    for name in truth:
        x, y, height, facing, tilt, _ = truth[name]
        keys = given.get(name, ())
        entry = {}
        if "at" in keys:
            entry["at"] = [x, y]
        for key, value in (("height", height), ("facing", facing), ("tilt", tilt)):
            if key in keys:
                entry[key] = value
        cameras[name] = entry
    return RoomSection.model_validate({"cameras": cameras, "things": things})


def floor_on(on: str, height: float) -> Floor:
    """The board lying flat with its middle at this height, as the calibration's cameras would see it."""
    middle = TURN.inv().apply((np.array([180.0, 160.0, height]) - SHIFT) / 0.1)
    return Floor(TURN.inv().apply([0.0, 0.0, 1.0]), middle, on, {"front": 24})


def close_to_truth(placing, names=TRUTH, length=1.0, angle=0.5):
    for name in names:
        x, y, height, facing, tilt, _ = TRUTH[name]
        got = placing.values[name]
        assert abs(got["x"] - x) <= length and abs(got["y"] - y) <= length and abs(got["height"] - height) <= length, (name, got)
        assert abs((got["facing"] - facing + 180) % 360 - 180) <= angle and abs(got["tilt"] - tilt) <= angle, (name, got)


def test_the_lens_angle_is_the_pictures_width():
    camera = calibrated().cameras[0]
    assert lens_angle(camera) == pytest.approx(np.degrees(2 * np.arctan(960 / 1000)))


def test_a_level_cameras_angles_read_back():
    axes = _level_axes(250.0, 12.0)
    assert angles_of(axes[:, 0], axes[:, 1]) == pytest.approx((250.0, 12.0, 0.0))


def test_three_cameras_places_and_heights_are_enough_without_any_angle():
    placing = place_cameras(calibrated(), room_with({name: ("at", "height") for name in ("front", "side", "corner")}), "cm")
    assert placing.tied and not placing.missing
    close_to_truth(placing)
    assert placing.values["back"]["lens"] == pytest.approx(np.degrees(2 * np.arctan(0.96)), abs=0.06)
    assert all(check["agrees"] for check in placing.checks)


def test_two_cameras_places_leave_up_open_and_only_the_lens_is_given():
    placing = place_cameras(calibrated(), room_with({"front": ("at",), "side": ("at",)}), "cm")
    assert not placing.tied and placing.values["front"] == {"lens": pytest.approx(np.degrees(2 * np.arctan(0.96)), abs=0.06)}
    assert any(text.startswith("which way is up") for text in placing.missing)


def test_nothing_measured_names_both_things_missing():
    placing = place_cameras(calibrated(), room_with({}), "cm")
    assert not placing.tied and len(placing.missing) == 2


def test_the_board_on_the_table_with_two_cameras_places():
    room = room_with({"front": ("at",), "side": ("at",)}, table={"at": [180, 160], "height": 70})
    placing = place_cameras(calibrated(), room, "cm", floor_on("table", 70.0))
    assert placing.tied
    close_to_truth(placing)


def test_the_board_on_a_thing_without_a_height_leaves_the_level_to_the_rest():
    room = room_with({"front": ("at", "height"), "side": ("at",)}, table={"at": [180, 160]})
    placing = place_cameras(calibrated(), room, "cm", floor_on("table", 70.0))
    assert placing.tied and any("no height" in note for note in placing.notes)
    close_to_truth(placing)


def test_one_camera_fully_laid_out_places_the_others():
    placing = place_cameras(calibrated(), room_with({"front": ("at", "height", "facing", "tilt")}), "cm")
    assert placing.tied
    close_to_truth(placing)


def test_a_wrong_height_is_among_those_that_do_not_agree():
    room = room_with({name: ("at", "height") for name in TRUTH})
    room.cameras["side"].height = 190.0
    placing = place_cameras(calibrated(), room, "cm")
    wrong = {(check["who"], check["what"]) for check in placing.checks if not check["agrees"]}
    assert ("side", "height") in wrong
    if len(wrong) > 1:
        assert any("may be any of those" in note for note in placing.notes)


def test_with_the_board_flat_a_wrong_height_stands_alone():
    room = room_with({name: ("at", "height") for name in ("front", "side", "corner")})
    room.cameras["side"].height = 190.0
    placing = place_cameras(calibrated(), room, "cm", floor_on("floor", 0.0))
    wrong = [(check["who"], check["what"]) for check in placing.checks if not check["agrees"]]
    assert wrong == [("side", "height")]
    close_to_truth(placing, names=("front", "corner", "back"))


def test_a_camera_turned_about_its_line_of_sight_is_said():
    truth = {**TRUTH, "corner": (*TRUTH["corner"][:5], 9.0)}
    placing = place_cameras(calibrated(truth), room_with({name: ("at", "height") for name in TRUTH}, truth), "cm")
    assert set(placing.rolled) == {"corner"} and abs(placing.rolled["corner"]) == pytest.approx(9.0, abs=0.5)


def test_metres_are_the_projects_unit_too():
    room = RoomSection.model_validate({"cameras": {
        name: {"at": [x / 100, y / 100], "height": height / 100} for name, (x, y, height, *_rest) in TRUTH.items()
    }})
    placing = place_cameras(calibrated(), room, "m")
    assert placing.tied and placing.values["front"]["height"] == pytest.approx(1.5, abs=0.01)


def test_cameras_on_one_side_only_are_said():
    room = room_with({name: ("at", "height") for name in TRUTH})
    room.cameras["extra"] = room.cameras["front"].model_copy()
    setup = calibrated()
    setup.cameras = [camera for camera in setup.cameras if camera.name != "back"] + [
        Camera("loose", (1920, 1080), K, np.zeros(5), np.zeros(3), np.zeros(3))
    ]
    del room.cameras["back"]
    placing = place_cameras(setup, room, "cm")
    assert any("loose" in note and "not in this setup" in note for note in placing.notes)
    assert any("extra" in note and "not in the calibration" in note for note in placing.notes)
    assert "extra" not in placing.values


SMALL = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
# the calibration's y points down: the board lies flat 1.2 m below the first camera, its middle here
MIDDLE = np.array([300.0, 1200.0, 1500.0])


def flat_cameras():
    from test_cameras import look_at

    return [look_at(name, where, MIDDLE, matrix=SMALL, size=(640, 480)) for name, where in (("left", [0, 0, 0]), ("right", [700, -100, 200]))]


def flat_board_clips(folder: Path, cameras) -> dict[str, Path]:
    """Each camera's clip of the board lying flat and still, face up, drawn through its lens-free pinhole."""
    cv2 = pytest.importorskip("cv2")
    from macaw.extractors.video.pose.board import make_board
    from test_cameras import board_cfg

    board = make_board(board_cfg())
    per_mm, margin = 1.5, 40
    picture = board.generateImage((round(525 * per_mm), round(375 * per_mm)), marginSize=0, borderBits=1)
    page = np.full((picture.shape[0] + 2 * margin, picture.shape[1] + 2 * margin), 255, np.uint8)
    page[margin:-margin, margin:-margin] = picture
    to_page = np.array([[per_mm, 0.0, margin - 0.5], [0.0, per_mm, margin - 0.5], [0.0, 0.0, 1.0]])
    # the board's z, which points away from a camera seeing its face, points down
    pose = Rotation.from_matrix(np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))
    shift = MIDDLE - pose.apply([525 / 2, 375 / 2, 0.0])
    clips = {}
    for camera in cameras:
        (folder / camera.name).mkdir(parents=True, exist_ok=True)
        clips[camera.name] = folder / camera.name / "clip.mp4"
        writer = cv2.VideoWriter(str(clips[camera.name]), cv2.VideoWriter_fourcc(*"mp4v"), 10, camera.size)
        rotation = Rotation.from_rotvec(camera.rotation)
        plane = np.column_stack([(rotation * pose).apply([1.0, 0, 0]), (rotation * pose).apply([0, 1.0, 0]), rotation.apply(shift) + camera.translation])
        frame = cv2.warpPerspective(page, camera.matrix @ plane @ np.linalg.inv(to_page), camera.size, borderValue=170)
        for _ in range(10):
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()
    return clips


def test_the_board_lying_flat_is_found_in_the_board_clips(tmp_path):
    from macaw.extractors.video.pose.board import make_board
    from macaw.project.placing import find_floor
    from test_cameras import board_cfg

    cameras = flat_cameras()
    clips = flat_board_clips(tmp_path, cameras)
    board = make_board(board_cfg())
    setup = CameraSetup(cameras, "mm", {"offsets_s": {"left": 0.0, "right": 0.0}})
    floor = find_floor(setup, clips, board, 0.5, "floor", normalise=False)
    assert floor is not None and set(floor.seen_by) == {"left", "right"} and min(floor.seen_by.values()) >= 6
    assert np.degrees(np.arccos(np.dot(floor.up, [0.0, -1.0, 0.0]))) < 1.0
    assert np.linalg.norm(floor.middle - MIDDLE) < 10.0
    assert find_floor(setup, clips, board, 5.0, "floor", normalise=False) is None


@pytest.fixture
def calibrated_project(tmp_path):
    """A project whose calibration lab has board clips ending with the board lying flat, and its cameras found."""
    import yaml

    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.extractors.video.pose.cameras import write_setup as write_calibration
    from macaw.project.layout import layout_for
    from setup_helpers import camera, mics, put, write_setup

    raw = tmp_path / "raw"
    for device, name in (("parent_mic", "a.wav"), ("left", "a.mp4"), ("right", "a.mp4")):
        put(raw, "003", device, name)
    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    setup_file = write_setup(project, cameras={"left": camera(), "right": camera(uses=("body",))}, microphones=mics(["parent"]))
    with open(project, "a", encoding="utf-8") as handle:
        handle.write("\nvideo:\n  normalise: false\npose3d:\n  calibration: lab\n")
    cameras = flat_cameras()
    flat_board_clips(tmp_path / "calibration" / "lab", cameras)
    write_calibration(CameraSetup(cameras, "mm", {"made_by": "macaw calibrate-cameras", "offsets_s": {"left": 0.0, "right": 0.0}}),
                      layout_for(resolve_config(project)).camera_setup_path("lab"))
    # the room as measured with a tape: each camera's place on the floor, nothing else
    data = yaml.safe_load(setup_file.read_text(encoding="utf-8"))
    data["room"] = {"cameras": {"left": {"at": [100, 50]}, "right": {"at": [170, 70]}}}
    setup_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return project


# in the room, x is the calibration's x, y its z and up its -y; left stands at (100, 50, 120) cm
LEFT_FACING, LEFT_TILT = np.degrees(np.arctan2(30, 150)), np.degrees(np.arctan2(120, np.hypot(30, 150)))


def test_the_commands_take_the_floor_and_place_the_cameras(calibrated_project):
    import yaml
    from typer.testing import CliRunner

    from macaw.cli.main import app

    runner = CliRunner()
    placed = runner.invoke(app, ["place-cameras", "-c", str(calibrated_project)])
    assert placed.exit_code == 0, placed.output + str(placed.exception)
    assert "Only the lens angles" in placed.output and "which way is up" in placed.output
    said = runner.invoke(app, ["calibration-floor", "lab", "--at", "0.5", "-c", str(calibrated_project)])
    assert said.exit_code == 0, said.output + str(said.exception)
    assert "left: board found" in said.output and (calibrated_project.parent / "calibration" / "lab" / "floor.yaml").is_file()
    placed = runner.invoke(app, ["place-cameras", "-c", str(calibrated_project)])
    assert placed.exit_code == 0 and "Placed left, right from the calibration lab." in placed.output, placed.output
    room = yaml.safe_load((calibrated_project.parent / "setups" / "main.yaml").read_text(encoding="utf-8"))["room"]["cameras"]
    assert room["left"]["height"] == pytest.approx(120, abs=1) and room["right"]["height"] == pytest.approx(130, abs=1)
    assert room["left"]["facing"] == pytest.approx(LEFT_FACING, abs=0.5) and room["left"]["tilt"] == pytest.approx(LEFT_TILT, abs=0.5)
    assert room["left"]["lens"] == pytest.approx(np.degrees(2 * np.arctan(320 / 600)), abs=0.1) and room["left"]["calibration"] == "lab"


def test_no_camera_seeing_the_board_is_said(calibrated_project):
    from typer.testing import CliRunner

    from macaw.cli.main import app

    said = CliRunner().invoke(app, ["calibration-floor", "lab", "--at", "30", "-c", str(calibrated_project)])
    assert said.exit_code != 0 and "no camera finds the board at 30 s" in str(said.exception)


def test_the_setup_page_shows_saves_and_uses_the_board_lying_flat(calibrated_project):
    pytest.importorskip("dash")
    from macaw.config.loader import resolve_config
    from macaw.interface.pages import cameras, room
    from test_gui import text_of

    shown = text_of(cameras.floor_view(calibrated_project, "lab", 0.5))
    assert "left: board found" in shown and "right: board found" in shown
    assert "Saved: the board lay on the floor at 0.5 s" in text_of(cameras.save_floor(calibrated_project, "lab", 0.5, "floor"))
    length, at, things, on, saved = cameras.floor_tools(calibrated_project, "lab", "main")
    assert length == pytest.approx(1.0) and at == 0.5 and things == ["floor"] and "saved: 0.5 s, on floor" in text_of(saved)
    model = room.model_of(resolve_config(calibrated_project), "main")
    after, message = room.from_calibration(calibrated_project, model)
    assert "Placed left, right from the calibration lab." in text_of(message)
    assert after["cameras"]["right"]["height"] == pytest.approx(130, abs=1) and after["cameras"]["right"]["calibration"] == "lab"
    # a number changed by hand is no longer the calibration's; one left as it was keeps the mark
    kept = room.moved(after, "cameras", "right", {"height": after["cameras"]["right"]["height"]})
    assert kept["cameras"]["right"]["calibration"] == "lab"
    assert room.moved(after, "cameras", "right", {"height": 140})["cameras"]["right"]["calibration"] is None
    assert room.room_of(after)["cameras"]["left"]["calibration"] == "lab"
