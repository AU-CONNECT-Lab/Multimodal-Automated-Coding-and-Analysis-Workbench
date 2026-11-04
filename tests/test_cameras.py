from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError, ExternalToolError, MacawError
from macaw.extractors.video.pose.cameras import Camera, CameraSetup, read_setup, triangulate, write_setup
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.triangulate import (
    Kinematics3DExtractor,
    TriangulationExtractor,
    bone_spread,
    points_file,
    triangulate_people,
)
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store.keypoints import write_keypoints
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, validate_frame
from setup_helpers import camera as setup_camera, mics, put, write_setup as write_devices

runner = CliRunner()
K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
NO_LENS = np.zeros(5)


def look_at(name, position, target, matrix=K, distortions=NO_LENS, size=(1920, 1080)):
    """A camera at position facing target, with image x to the room's x and image y down the room's y."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    return Camera(name, size, matrix, distortions, Rotation.from_matrix(rotation).as_rotvec(), -rotation @ position)


def trio():
    target = [900.0, 0.0, 2000.0]
    return [look_at("left", [0, 0, 0], target), look_at("middle", [900, -200, 300], target), look_at("right", [1800, 0, 0], target)]


class TestLens:
    def test_undistorting_undoes_the_lens(self):
        camera = Camera("c", (1920, 1080), K, np.array([-0.2, 0.05, 0.001, -0.001, 0.01]), np.zeros(3), np.zeros(3))
        normal = np.random.default_rng(0).uniform(-0.5, 0.5, (200, 2))
        assert np.abs(camera.undistort(camera.distort(normal)) - normal).max() < 1e-8

    def test_the_lens_is_opencvs(self):
        cv2 = pytest.importorskip("cv2")
        camera = look_at("c", [100, 50, 0], [0, 0, 2000], distortions=np.array([-0.1, 0.02, 0.001, 0.002, 0.0]))
        points = np.random.default_rng(1).uniform([-500, -300, 1500], [500, 300, 2500], (50, 3))
        theirs, _ = cv2.projectPoints(points, camera.rotation, camera.translation, camera.matrix, camera.distortions)
        assert np.abs(camera.project(points) - theirs.reshape(-1, 2)).max() < 1e-6


class TestTriangulate:
    points = np.random.default_rng(2).uniform([300, -400, 1500], [1500, 400, 2500], (300, 3))

    def seen(self, cameras):
        return np.stack([camera.project(self.points) for camera in cameras], axis=1)

    def test_clean_views_meet_at_the_point(self):
        cameras = trio()
        found, used, error = triangulate(cameras, self.seen(cameras), np.ones((300, 3)), max_error_px=15)
        assert np.abs(found - self.points).max() < 1e-6 and used.all() and np.nanmax(error) < 1e-6

    def test_a_view_far_off_is_left_out(self):
        cameras = trio()
        pixels = self.seen(cameras)
        pixels[:, 2] += 40.0
        found, used, _ = triangulate(cameras, pixels, np.ones((300, 3)), max_error_px=15)
        assert not used[:, 2].any() and np.abs(found - self.points).max() < 1e-6

    def test_two_views_that_miss_each_other_place_nothing(self):
        cameras = trio()[:2]
        pixels = self.seen(cameras)
        pixels[:150, 1] += 40.0
        found, used, error = triangulate(cameras, pixels, np.ones((300, 2)), max_error_px=15)
        assert np.isnan(found[:150]).all() and not used[:150].any() and np.isnan(error[:150]).all()
        assert np.abs(found[150:] - self.points[150:]).max() < 1e-6

    def test_a_point_behind_the_cameras_is_a_mismatch(self):
        cameras = trio()
        behind = np.array([[900.0, 0.0, -1500.0], [600.0, 100.0, -900.0]])
        pixels = np.stack([camera.project(behind) for camera in cameras], axis=1)
        found, used, _ = triangulate(cameras, pixels, np.ones((2, 3)), max_error_px=15)
        assert np.isnan(found).all() and not used.any()

    def test_one_view_is_not_enough(self):
        cameras = trio()
        weights = np.zeros((300, 3))
        weights[:, 0] = 1.0
        found, used, error = triangulate(cameras, self.seen(cameras), weights, max_error_px=15)
        assert np.isnan(found).all() and not used.any() and np.isnan(error).all()

    def test_a_missing_point_counts_as_no_view(self):
        cameras = trio()
        pixels = self.seen(cameras)
        pixels[:, 1] = np.nan
        found, used, _ = triangulate(cameras, pixels, np.ones((300, 3)), max_error_px=15)
        assert np.abs(found - self.points).max() < 1e-6 and not used[:, 1].any()


class TestFile:
    def test_a_setup_reads_back_as_written(self, tmp_path):
        setup = CameraSetup(trio(), "mm", {"made_by": "test", "error_px": {"left": 0.4}})
        back = read_setup(write_setup(setup, tmp_path / "lab.toml"))
        assert back.units == "mm" and back.metadata["error_px"] == {"left": 0.4}
        for written, read in zip(setup.cameras, back.cameras):
            assert read.name == written.name and read.size == written.size
            assert np.allclose(read.matrix, written.matrix) and np.allclose(read.translation, written.translation)

    def test_an_anipose_file_without_units_takes_the_setting(self, tmp_path):
        path = tmp_path / "calibration.toml"
        path.write_text(
            '[cam_0]\nname = "A"\nsize = [640, 480]\nmatrix = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]\n'
            "distortions = [0.0, 0.0, 0.0, 0.0, 0.0]\nrotation = [0, 0, 0]\ntranslation = [0, 0, 0]\n\n"
            '[cam_1]\nname = "B"\nsize = [640, 480]\nmatrix = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]\n'
            "distortions = [0.0, 0.0, 0.0, 0.0, 0.0]\nrotation = [0, 0.5, 0]\ntranslation = [-50, 0, 10]\n\n"
            '[metadata]\nadjusted = true\n',
            encoding="utf-8",
        )
        setup = read_setup(path, default_units="cm")
        assert setup.units == "cm" and [camera.name for camera in setup.cameras] == ["A", "B"]

    @pytest.mark.parametrize("text, reason", [
        ('[cam_0]\nname = "A"\nsize = [640, 480]\nmatrix = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]\nrotation = [0, 0, 0]\ntranslation = [0, 0, 0]\n', "two cameras"),
        ("[metadata]\nunits = \"inch\"\n", "units"),
        ("not toml [", "cannot read"),
    ])
    def test_a_file_that_cannot_place_cameras_is_refused(self, tmp_path, text, reason):
        path = tmp_path / "bad.toml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match=reason):
            read_setup(path)


def board_cfg(**changes):
    return SimpleNamespace(**{"squares_x": 7, "squares_y": 5, "square_mm": 75.0, "marker_mm": 56.0, "dictionary": "5X5_100", **changes})


class TestCalibration:
    def sightings(self, cameras, steps=80, seed=3, skip=()):
        from macaw.extractors.video.pose.board import CameraClip, Sighting, make_board

        rng = np.random.default_rng(seed)
        board = make_board(board_cfg())
        corners = np.asarray(board.getChessboardCorners(), float)
        clips = [CameraClip(camera.name, camera.size) for camera in cameras]
        for step in range(steps):
            centre = np.array([900.0, 0.0, 1700.0]) + rng.normal(0, [350, 200, 250])
            pose = Rotation.from_euler("xyz", rng.normal(0, [0.6, 0.6, 0.3]))
            room = pose.apply(corners - corners.mean(axis=0)) + centre
            for clip, camera in zip(clips, cameras):
                if (camera.name, step) in skip:
                    continue
                pixels = camera.project(room)
                inside = (pixels[:, 0] > 0) & (pixels[:, 0] < 1920) & (pixels[:, 1] > 0) & (pixels[:, 1] < 1080)
                if inside.sum() >= 6:
                    ids = np.flatnonzero(inside)
                    clip.sightings.append(Sighting(step, ids, pixels[ids] + rng.normal(0, 0.3, (len(ids), 2))))
        return board, clips

    def test_lenses_and_places_come_back(self):
        pytest.importorskip("cv2")
        from macaw.extractors.video.pose.board import calibrate

        target = [900.0, 0.0, 1800.0]
        truth = [look_at("left", [0, 0, 0], target), look_at("middle", [900, 0, 300], target), look_at("right", [1800, 0, 0], target)]
        board, clips = self.sightings(truth)
        fit = calibrate(clips, board)
        first = truth[0].to_camera
        for found, real in zip(fit.setup.cameras, truth):
            assert np.linalg.norm(found.position() - first(real.position()[None])[0]) < 25.0
            assert found.matrix[0, 0] == pytest.approx(1000.0, rel=0.01)
        assert max(fit.error_px.values()) < 0.6 and fit.setup.units == "mm"

    def test_a_camera_that_never_sees_the_board_with_another_is_said(self):
        pytest.importorskip("cv2")
        from macaw.extractors.video.pose.board import calibrate

        target = [900.0, 0.0, 1800.0]
        truth = [look_at("left", [0, 0, 0], target), look_at("right", [1800, 0, 0], target)]
        alone = {("right", step) for step in range(0, 80, 2)} | {("left", step) for step in range(1, 80, 2)}
        board, clips = self.sightings(truth, skip=alone)
        with pytest.raises(MacawError, match="right never saw the board at the same moment"):
            calibrate(clips, board)


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for device, name in (("parent_mic", "a.wav"), ("left", "a.mp4"), ("right", "a.mp4")):
        put(raw, "003", device, name)
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_devices(config, cameras={"left": setup_camera(), "right": setup_camera()}, microphones=mics(["parent"]))
    return config


def settings(project, text):
    with open(project, "a", encoding="utf-8") as handle:
        handle.write("\n" + text)


def said(result) -> str:
    return result.output + str(result.exception or "")


def render_clips(folder, cameras, steps=40, fps=10, seed=4):
    """Each camera's video of a board moved about in front of them, drawn through its lens-free pinhole."""
    cv2 = pytest.importorskip("cv2")
    from macaw.extractors.video.pose.board import make_board

    board = make_board(board_cfg())
    per_mm, margin = 1.5, 40
    picture = board.generateImage((round(525 * per_mm), round(375 * per_mm)), marginSize=0, borderBits=1)
    page = np.full((picture.shape[0] + 2 * margin, picture.shape[1] + 2 * margin), 255, np.uint8)
    page[margin:-margin, margin:-margin] = picture
    # board millimetres to page pixels, pixel centres at the half
    to_page = np.array([[per_mm, 0.0, margin - 0.5], [0.0, per_mm, margin - 0.5], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(seed)
    poses = [(Rotation.from_euler("xyz", rng.normal(0, [0.5, 0.5, 0.2])), np.array([300.0, 0.0, 1100.0]) + rng.normal(0, [120, 80, 120])) for _ in range(steps)]
    folder.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        (folder / camera.name).mkdir(exist_ok=True)
        writer = cv2.VideoWriter(str(folder / camera.name / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, camera.size)
        rotation = Rotation.from_rotvec(camera.rotation)
        for pose, centre in poses:
            shift = centre - pose.apply([525 / 2, 375 / 2, 0.0])
            plane = np.column_stack([(rotation * pose).apply([1.0, 0, 0]), (rotation * pose).apply([0, 1.0, 0]), rotation.apply(shift) + camera.translation])
            homography = camera.matrix @ plane @ np.linalg.inv(to_page)
            frame = cv2.warpPerspective(page, homography, camera.size, borderValue=170)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        writer.release()


def test_calibrate_cameras_finds_them_from_board_clips(project):
    small = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    truth = [look_at(name, position, [300, 0, 1100], matrix=small, size=(640, 480)) for name, position in (("left", [0, 0, 0]), ("right", [600, 0, 100]))]
    render_clips(project.parent / "calibration" / "lab", truth)
    settings(project, "pose3d:\n  board_every_s: 0.1\n")
    try:
        result = runner.invoke(app, ["calibrate-cameras", "lab", "-c", str(project)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    if isinstance(result.exception, ExternalToolError):
        pytest.skip("needs ffmpeg")
    assert result.exit_code == 0, said(result)
    assert "left and right saw the board together" in result.output
    setup = read_setup(layout_for(resolve_config(project)).camera_setup_path("lab"))
    left, right = setup.cameras
    assert (left.name, right.name) == ("left", "right") and setup.metadata["made_by"] == "macaw calibrate-cameras"
    baseline = np.linalg.norm(right.position() - left.position())
    assert baseline == pytest.approx(np.hypot(600, 100), rel=0.03)
    assert left.matrix[0, 0] == pytest.approx(600, rel=0.03) and max(setup.metadata["error_px"].values()) < 1.5


def test_calibrate_cameras_needs_two_clips(project):
    (project.parent / "calibration" / "lab").mkdir(parents=True)
    result = runner.invoke(app, ["calibrate-cameras", "lab", "-c", str(project)])
    assert result.exit_code != 0 and "board clip in a folder named after it" in said(result)


def test_the_board_prints_at_its_size(project):
    pytest.importorskip("cv2")
    result = runner.invoke(app, ["calibration-board", "-c", str(project)])
    assert result.exit_code == 0, said(result)
    data = (project.parent / "calibration" / "board_A2.png").read_bytes()
    found = data.find(b"pHYs")
    assert found > 0 and struct.unpack(">IIB", data[found + 4 : found + 13]) == (11811, 11811, 1)
    too_small = runner.invoke(app, ["calibration-board", "-c", str(project), "--paper", "A3"])
    assert too_small.exit_code != 0 and "does not fit on A3" in said(too_small)


def person(head, facing, frames, *, wrist_speed=0.0, fps=30.0, hips=False):
    """Joints in mm: a seated upper body whose nose points along facing; the left forearm swings about the
    elbow at wrist_speed metres per second, so no bone changes length. hips adds them 700 mm below the head."""
    facing = np.asarray(facing, float) / np.linalg.norm(facing)
    side = np.cross(facing, [0.0, 1.0, 0.0])
    head, down = np.asarray(head, float), np.array([0.0, 1.0, 0.0])
    body = {
        "nose": head + 100 * facing, "left_eye": head + 80 * facing - 20 * down + 30 * side,
        "right_eye": head + 80 * facing - 20 * down - 30 * side, "left_ear": head + 70 * side, "right_ear": head - 70 * side,
        "left_shoulder": head + 250 * down + 180 * side, "right_shoulder": head + 250 * down - 180 * side,
        "left_elbow": head + 500 * down + 220 * side, "right_elbow": head + 500 * down - 220 * side,
        "right_wrist": head + 750 * down - 220 * side,
    }
    if hips:
        body.update({"left_hip": head + 700 * down + 150 * side, "right_hip": head + 700 * down - 150 * side})
    joints = {joint: np.tile(xyz, (frames, 1)) for joint, xyz in body.items()}
    angle = np.arange(frames) / fps * wrist_speed * 1000.0 / 250.0
    joints["left_wrist"] = body["left_elbow"] + 250.0 * (np.cos(angle)[:, None] * down + np.sin(angle)[:, None] * facing)
    return joints


def keypoints_of(cameras, videos, people, frames, fps=30.0):
    rows = []
    for camera, video in zip(cameras, videos):
        for subject_id, joints in people.items():
            for joint, xyz in joints.items():
                pixels = camera.project(xyz)
                rows.append(pd.DataFrame({
                    "recording_id": "003", "subject_id": subject_id, "source": "pose.keypoints", "video": video,
                    "frame": np.arange(frames), "t": np.arange(frames) / fps, "joint": joint, "x": pixels[:, 0], "y": pixels[:, 1],
                    "z": np.nan, "confidence": 0.9, "track_id": 1, "units": "pixel", "metric_scale_valid": None,
                }))
    return coerce_frame(pd.concat(rows, ignore_index=True), KEYPOINT_SCHEMA, "keypoints")


@pytest.fixture
def seated(project):
    """Two people facing each other a metre apart, filmed by two calibrated cameras."""
    target = [500.0, 300.0, 2000.0]
    cameras = [look_at("left", [-300, -200, 0], target), look_at("right", [1300, -200, 0], target)]
    write_setup(CameraSetup(cameras, "mm"), project.parent / "calibration" / "lab.toml")
    settings(project, "pose3d:\n  calibration: lab.toml\n")
    people = {"parent": person([0, 0, 2000], [1, 0, 0], 60, wrist_speed=0.5), "child": person([1000, 100, 2000], [-1, 0, 0], 60)}
    config = resolve_config(project)
    write_keypoints(keypoints_of(cameras, ["left", "right"], people, 60), layout_for(config).keypoints_path("003"))
    return SimpleNamespace(config=config, people=people, cameras=cameras)


class TestSteps:
    def test_people_are_placed_in_metres(self, seated, tmp_path):
        result = TriangulationExtractor(seated.config).run(build_recording("003", seated.config), tmp_path)
        points = result.keypoints
        assert set(points["units"]) == {"metre"} and set(points["video"]) == {"3d"} and len(points) == 2 * 11 * 60
        nose = points[(points.subject_id == "child") & (points.joint == "nose")]
        assert np.allclose(nose[["x", "y", "z"]].to_numpy(), seated.people["child"]["nose"] / 1000.0, atol=1e-6)
        messages = [warning.message for warning in result.warnings]
        assert any("in 3D in 100%" in message for message in messages), messages
        assert not [warning.message for warning in result.warnings if warning.severity == "warn"]

    def test_cameras_are_matched_by_name(self, seated, tmp_path):
        renamed = [Camera(f"cam{index}", camera.size, camera.matrix, camera.distortions, camera.rotation, camera.translation) for index, camera in enumerate(seated.cameras)]
        write_setup(CameraSetup(renamed, "mm"), seated.config.project.calibration_dir / "lab.toml")
        with pytest.raises(MacawError, match="match a camera people were measured on"):
            TriangulationExtractor(seated.config).run(build_recording("003", seated.config), tmp_path)

    def test_a_folder_of_board_clips_wants_calibrating_first(self, project):
        (project.parent / "calibration" / "lab").mkdir(parents=True)
        settings(project, "pose3d:\n  calibration: lab\n")
        with pytest.raises(ConfigError, match="macaw calibrate-cameras lab"):
            TriangulationExtractor(resolve_config(project)).check_available()

    def test_the_3d_measures_are_in_metres_and_degrees(self, seated, tmp_path):
        placed = TriangulationExtractor(seated.config).run(build_recording("003", seated.config), tmp_path).keypoints
        write_keypoints(placed, points_file(seated.config, "003").parent)
        result = Kinematics3DExtractor(seated.config).run(build_recording("003", seated.config), tmp_path)
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        rows = result.rows

        def values(feature, subject="parent", other=None):
            picked = rows[(rows.feature == feature) & (rows.subject_id == subject)]
            picked = picked[picked.subject_id_b.isna()] if other is None else picked[picked.subject_id_b == other]
            return picked["value"].to_numpy(dtype=float)

        assert np.allclose(values("interpersonal_distance_3d", "child", "parent"), np.hypot(1.0, 0.1))
        assert np.allclose(values("head_toward_deg_3d", "parent", "child"), np.degrees(np.arctan2(0.1, 1.0)), atol=1e-6)
        assert len(values("interpersonal_distance_3d", "parent", "child")) == 0
        assert np.nanmedian(values("left_wrist_speed_3d")) == pytest.approx(0.5, rel=1e-3)
        units = {spec.name: spec.unit for spec in result.specs}
        assert units["left_wrist_speed_3d"] == "m/s" and units["head_toward_deg_3d"] == "deg"


OWN_MEASURES = """kinematics:
  measures:
    - {name: hand_to_head, person: parent, joints: [left_wrist, right_wrist], to_person: child, to_joints: [nose]}
    - {name: closing, person: parent, joints: [nose], to_person: child, to_joints: [nose], method: approach_speed}
    - {name: lean, person: parent, joints: [nose], to_person: child, to_joints: [nose], method: offset_toward, per_body_size: true}
    - {name: depth, person: parent, to_person: child, method: mine.depth, in_3d: true}
    - {name: flat, person: parent, to_person: self, method: mine.flat}
"""


def test_the_projects_own_measures_have_3d_versions(seated, project, tmp_path):
    from macaw.project.manifest import FeatureManifest
    from macaw.qc.boilerplate import describe, load_references, render

    (project.parent / "scripts").mkdir()
    (project.parent / "scripts" / "mine.py").write_text(
        "def depth(person, other):\n    return other['nose_z'] - person['nose_y']\n\n\ndef flat(person, other):\n    return person['nose_x']\n",
        encoding="utf-8",
    )
    settings(project, OWN_MEASURES)
    config = resolve_config(project)
    placed = TriangulationExtractor(config).run(build_recording("003", config), tmp_path).keypoints
    write_keypoints(placed, points_file(config, "003").parent)
    step = Kinematics3DExtractor(config)
    step.check_available()
    result = step.run(build_recording("003", config), tmp_path)
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    rows = result.rows

    def values(feature):
        return rows[rows.feature == feature]["value"].to_numpy(dtype=float)

    parent, child = seated.people["parent"], seated.people["child"]
    nearest = np.minimum(*(np.linalg.norm(parent[wrist] - child["nose"], axis=1) for wrist in ("left_wrist", "right_wrist"))) / 1000.0
    assert np.allclose(values("hand_to_head_3d"), nearest, atol=1e-6)
    assert np.nanmax(np.abs(values("closing_3d"))) < 1e-6
    centre = (parent["left_shoulder"][0] + parent["right_shoulder"][0]) / 2
    toward = child["nose"][0] - centre
    lean = (parent["nose"][0] - centre) @ toward / np.linalg.norm(toward) / np.linalg.norm(parent["left_shoulder"][0] - parent["right_shoulder"][0])
    assert np.allclose(values("lean_3d"), lean, atol=1e-6)
    assert np.allclose(values("depth_3d"), child["nose"][0, 2] / 1000.0 - parent["nose"][0, 1] / 1000.0, atol=1e-6)
    assert "flat_3d" not in set(rows.feature)
    assert set(rows[rows.feature == "hand_to_head_3d"].subject_id_b) == {"child"}
    units = {spec.name: spec.unit for spec in result.specs}
    assert (units["hand_to_head_3d"], units["closing_3d"], units["lean_3d"], units["depth_3d"]) == ("m", "m/s", None, None)

    manifest = FeatureManifest()
    for spec in result.specs:
        manifest.add(spec)
    text, _ = render(describe(manifest, config), load_references())
    assert "hand_to_head_3d: Distance in the room from parent's left_wrist, right_wrist to child's nose" in text
    assert "flat_3d" not in text


def test_a_3d_version_may_not_take_a_name_in_use(seated, project):
    settings(project, "kinematics:\n  measures:\n    - {name: head_distance, person: parent, joints: [nose], to_person: child, to_joints: [nose]}\n")
    with pytest.raises(ConfigError, match="head_distance_3d"):
        Kinematics3DExtractor(resolve_config(project)).check_available()


def test_a_stretching_bone_shows_in_its_spread():
    frames = 40
    rows = []
    for frame in range(frames):
        stretch = 1.0 + (0.4 if frame % 2 else 0.0)
        for joint, xyz in {"left_shoulder": [0, 0, 0], "left_elbow": [0, 0.3 * stretch, 0], "left_wrist": [0, 0.3 * stretch + 0.25, 0]}.items():
            rows.append({"subject_id": "parent", "frame": frame, "joint": joint, "x": xyz[0], "y": xyz[1], "z": xyz[2]})
    spread = bone_spread(pd.DataFrame(rows))
    assert spread[("left_shoulder", "left_elbow")] > 0.15 and spread[("left_elbow", "left_wrist")] == pytest.approx(0.0)


def test_the_methods_describe_the_cameras_and_the_measures(seated, tmp_path):
    from macaw.project.manifest import FeatureManifest, FeatureSpec
    from macaw.qc.boilerplate import describe, load_references, render
    from macaw.store.schema import Scale

    setup = read_setup(seated.config.project.calibration_dir / "lab.toml")
    setup.metadata["made_by"] = "macaw calibrate-cameras"
    write_setup(setup, seated.config.project.calibration_dir / "lab.toml")
    manifest = FeatureManifest()
    manifest.add(FeatureSpec(name="interpersonal_distance_3d", description="", unit="m", scale=Scale.FRAME, group="pose", extractor="pose.kinematics3d", extractor_version="0.1.0"))
    text, used = render(describe(manifest, seated.config), load_references())
    assert "from the 2 calibrated cameras" in text and "ChArUco board" in text and "75 mm squares" in text
    assert {"hartley2003", "garrido2014", "zhang2000", "triggs2000"} <= set(used)


def test_the_viewer_draws_each_person_and_the_cameras(seated, tmp_path):
    pytest.importorskip("dash")
    from macaw.interface.pages import results

    placed = TriangulationExtractor(seated.config).run(build_recording("003", seated.config), tmp_path).keypoints
    graph = results.view_3d(seated.config.project.calibration_dir.parent / "project.yaml", "003", 0.5, points=placed)
    names = {trace.name for trace in graph.figure.data}
    assert {"parent", "child", "cameras"} <= names
    shown = next(trace for trace in graph.figure.data if trace.name == "cameras")
    assert list(shown.text) == ["left", "right"]


GESTURES = """kinematics:
  gestures:
    tier: gesture
    still_speed: 10
    peak_prominence: 20
"""


def code_gestures(config, segments):
    from macaw.store._io import write_file
    from macaw.store.schema import ANNOTATION_SCHEMA

    rows = pd.DataFrame([
        {"recording_id": "003", "subject_id": who, "tier": "gesture", "t_start": start, "t_end": end, "label": "iconic",
         "annotator": "ann", "layer": "C", "abstained": False, "needs_review": False, "group_id": "003", "session_id": None,
         "task": None, "run": None, "confidence": np.nan, "execution_id": None, "config_hash": None}
        for who, start, end in segments
    ])
    write_file(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(config).imported_codes_path("003"), ANNOTATION_SCHEMA, "annotations")


@pytest.fixture
def gesturing(project):
    """The parent, filmed from their side, raises the left hand and moves it up and inward twice with a pause between."""
    target = [500.0, 300.0, 2000.0]
    cameras = [look_at("left", [-300, -200, 0], target), look_at("right", [1300, -200, 0], target)]
    write_setup(CameraSetup(cameras, "mm"), project.parent / "calibration" / "lab.toml")
    frames = 105
    parent = person([0, 0, 2000], [1, 0, 0], frames, hips=True)
    t = np.arange(frames) / 30.0
    rise = lambda start: np.clip(t - start, 0.0, 1.0)
    shift = 100.0 * ((1 - np.cos(np.pi * rise(0.5))) / 2 - (1 - np.cos(np.pi * rise(2.1))) / 2)
    # forearm raised forward; the parent's side is the room's z, so the hand moves in depth for both cameras
    raised = parent["left_shoulder"][0] + np.array([250.0, 50.0, 40.0])
    parent["left_wrist"] = raised + shift[:, None] * np.array([0.0, -1.0, -1.0]) / np.sqrt(2)
    people = {"parent": parent, "child": person([1000, 100, 2000], [-1, 0, 0], frames, hips=True)}
    settings(project, "pose3d:\n  calibration: lab.toml\n" + GESTURES)
    config = resolve_config(project)
    write_keypoints(keypoints_of(cameras, ["left", "right"], people, frames), layout_for(config).keypoints_path("003"))
    code_gestures(config, [("parent", 0.0, 3.4)])
    return SimpleNamespace(project=project, people=people)


def triangulated(project, tmp_path):
    config = resolve_config(project)
    placed = TriangulationExtractor(config).run(build_recording("003", config), tmp_path).keypoints
    write_keypoints(placed, points_file(config, "003").parent)
    return config


class TestGestures3D:
    def test_each_gesture_is_measured_in_the_room(self, gesturing, tmp_path):
        from macaw.extractors.video.pose.gestures import Gesture3DExtractor

        settings(gesturing.project, "    still_speed_3d: 0.02\n    peak_prominence_3d: 0.05\n")
        config = triangulated(gesturing.project, tmp_path)
        assert Gesture3DExtractor.wanted_by_default(config)
        result = Gesture3DExtractor(config).run(build_recording("003", config), tmp_path)
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        rows = result.rows[result.rows.subject_id == "parent"]
        value = lambda feature: float(rows[rows.feature == feature]["value"].iloc[0])
        assert value("gesture_left_submovements_3d") == 2.0 and value("gesture_left_holds_3d") == 1.0
        assert 0.6 < value("gesture_left_hold_s_3d") < 0.85
        assert value("gesture_left_peak_speed_3d") == pytest.approx(0.1 * np.pi / 2, rel=0.05)
        assert value("gesture_left_extent_3d") == pytest.approx((0.1 / np.sqrt(2)) ** 2, rel=1e-3)
        # from the side the picture shows no frontal plane; in the room the hand reaches above the shoulders
        assert value("gesture_left_max_height_3d") == 3.0 and value("gesture_left_space_max_3d") == 2.0
        assert value("gesture_right_submovements_3d") == 0.0
        specs = {spec.name: spec for spec in result.specs}
        assert specs["gesture_left_peak_speed_3d"].unit == "m/s" and specs["gesture_left_extent_3d"].unit == "m^2"
        assert "peak_prominence_3d" in specs["gesture_left_submovements_3d"].description
        assert not any("back to it" in spec.description for spec in result.specs)
        assert all(spec.comparable_across_subjects for spec in result.specs)

    def test_without_its_thresholds_it_says_what_to_set(self, gesturing, tmp_path):
        from macaw.extractors.video.pose.gestures import Gesture3DExtractor

        config = triangulated(gesturing.project, tmp_path)
        result = Gesture3DExtractor(config).run(build_recording("003", config), tmp_path)
        assert result.rows.empty and "still_speed_3d and peak_prominence_3d" in result.warnings[0].message

    def test_the_frontal_plane_turns_with_the_person(self):
        from macaw.extractors.video.pose.gestures import frontal_plane

        body = person([0, 0, 2000], [0, 0, -1], 3, hips=True)
        turn = Rotation.from_euler("xyz", [0.3, 1.9, -0.4])
        turned = {joint: turn.apply(xyz) + [400.0, -100.0, 50.0] for joint, xyz in body.items()}
        frames = np.arange(3)
        assert np.allclose(frontal_plane(turned, frames).to_numpy(), frontal_plane(body, frames).to_numpy())

    def test_the_methods_give_the_thresholds_in_metres(self, gesturing, tmp_path):
        from macaw.extractors.video.pose.gestures import Gesture3DExtractor
        from macaw.project.manifest import FeatureManifest
        from macaw.qc.boilerplate import describe, load_references, render

        settings(gesturing.project, "    still_speed_3d: 0.02\n    peak_prominence_3d: 0.05\n")
        config = triangulated(gesturing.project, tmp_path)
        manifest = FeatureManifest()
        for spec in Gesture3DExtractor(config).run(build_recording("003", config), tmp_path).specs:
            manifest.add(spec)
        text, _ = render(describe(manifest, config), load_references())
        assert "at least 0.05 m/s" in text and "slower than 0.02 m/s" in text and "frontal plane" in text
