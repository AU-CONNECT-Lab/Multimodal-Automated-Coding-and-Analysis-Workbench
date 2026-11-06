from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from macaw.extractors.video.pose.cameras import Camera
from macaw.extractors.video.pose.depth import (
    LENS_FILE, DepthLens, DepthPoseExtractor, DepthPrepExtractor, Surfaces, frame_pairs, read_lens, write_lens,
)

COLOUR = Camera("colour", (640, 480), np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]]), np.array([0.05, -0.02, 0, 0, 0]), np.zeros(3), np.zeros(3))
# the depth camera 5 cm to the colour camera's right, turned 2 degrees about its up axis
TO_COLOUR = Rotation.from_euler("y", 2.0, degrees=True)
SHIFT = np.array([0.05, 0.0, 0.0])
DEPTH_SIZE = (320, 240)
DEPTH_MATRIX = np.array([[290.0, 0, 160], [0, 290.0, 120], [0, 0, 1]])
# a body square 1 m across at 2 m before a wall at 3 m, in the colour camera's frame
BODY_Z, WALL_Z, BODY_HALF = 2.0, 3.0, 0.5


def lens() -> DepthLens:
    back = TO_COLOUR.inv()
    depth = Camera("depth", DEPTH_SIZE, DEPTH_MATRIX, np.array([0.1, -0.05, 0, 0, 0]), back.as_rotvec(), -back.apply(SHIFT))
    return DepthLens(depth, COLOUR, 0.001)


def depth_image(missing: tuple[slice, slice] | None = None) -> np.ndarray:
    """What the depth camera sees of the body and the wall, in millimetres along its own axis."""
    found = lens()
    grid = np.stack(np.meshgrid(np.arange(DEPTH_SIZE[0], dtype=float), np.arange(DEPTH_SIZE[1], dtype=float)), axis=-1).reshape(-1, 2)
    rays = np.column_stack([found.depth.undistort(grid), np.ones(len(grid))])
    origin, ways = SHIFT, TO_COLOUR.apply(rays)
    image = np.zeros(len(grid))
    for plane, square in ((WALL_Z, False), (BODY_Z, True)):
        scale = (plane - origin[2]) / ways[:, 2]
        hit = origin + ways * scale[:, None]
        inside = (np.abs(hit[:, 0]) <= BODY_HALF) & (np.abs(hit[:, 1]) <= BODY_HALF) if square else np.ones(len(grid), bool)
        # the depth value is along the depth camera's own axis
        image[inside] = scale[inside] * rays[inside, 2] * 1000.0
    image = image.reshape(DEPTH_SIZE[1], DEPTH_SIZE[0]).astype(np.uint16)
    if missing is not None:
        image[missing] = 0
    return image


JOINTS = np.array([[0.0, 0.0, BODY_Z], [0.3, -0.2, BODY_Z], [-0.35, 0.25, BODY_Z]])


def pixels_of(points: np.ndarray) -> np.ndarray:
    return COLOUR.distort(points[:, :2] / points[:, 2:3])


def test_a_point_on_the_body_is_placed_at_the_body_not_the_wall_behind():
    surfaces = Surfaces(lens())
    placed = surfaces.place(surfaces.nearest(depth_image()), pixels_of(JOINTS))
    assert np.abs(placed - JOINTS).max() < 0.01
    # where the depth camera saw nothing there is no place
    hole = surfaces.place(surfaces.nearest(np.zeros((DEPTH_SIZE[1], DEPTH_SIZE[0]), np.uint16)), pixels_of(JOINTS))
    assert np.isnan(hole).all()


def test_the_lens_file_reads_back_what_was_written(tmp_path):
    written = lens()
    read = read_lens(write_lens(written, tmp_path / LENS_FILE))
    assert np.allclose(read.depth.rotation, written.depth.rotation) and np.allclose(read.depth.translation, written.depth.translation)
    assert read.colour.size == COLOUR.size and read.unit_m == pytest.approx(0.001)
    surfaces = Surfaces(read)
    assert np.abs(surfaces.place(surfaces.nearest(depth_image()), pixels_of(JOINTS)) - JOINTS).max() < 0.01


def test_each_depth_image_goes_to_the_prepared_frame_taken_with_it():
    same = {"source_fps": 30.0, "fps": 30.0, "first_picture_s": 0.2, "frames_start_s": 0.2}
    assert frame_pairs(same, 4) == {0: 0, 1: 1, 2: 2, 3: 3}
    # prepared at half the rate from the start of the sound, 0.1 s before the first picture
    halved = {"source_fps": 30.0, "fps": 15.0, "first_picture_s": 0.1, "frames_start_s": 0.0}
    assert frame_pairs(halved, 7) == {2: 1, 3: 3, 4: 5}


@pytest.fixture
def depth_project(tmp_path):
    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.project.layout import layout_for
    from macaw.project.recording import build_recording
    from setup_helpers import camera, put, write_setup

    raw = tmp_path / "raw"
    put(raw, "003", "kinect", "kinect.mp4")
    folder = raw / "003" / "kinect" / "depth"
    folder.mkdir()
    for index in range(3):
        from macaw.extractors.video.pose.board import _cv2

        _cv2().imwrite(str(folder / f"{index:06d}.png"), depth_image(missing=(slice(0, 240), slice(0, 110)) if index == 2 else None))
    write_lens(lens(), folder / LENS_FILE)
    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(project, cameras={"kinect": {**camera(), "depth": True}})
    config = resolve_config(project)
    layout = layout_for(config)
    video = layout.artifacts_dir("003", "prep.video")
    video.mkdir(parents=True)
    entry = {"name": "kinect", "file": None, "fps": 30.0, "frames_start_s": 0.0, "first_picture_s": 0.0, "source_fps": 30.0,
             "size": [640, 480], "prepared_size": None}
    (video / "video.json").write_text(json.dumps([entry]), encoding="utf-8")
    return config, layout, build_recording("003", config)


def test_the_steps_place_each_body_point_from_the_depth_beside_the_video(depth_project, tmp_path):
    pytest.importorskip("cv2")
    from macaw.extractors.prep import VIDEO_INFO_NAME
    from macaw.extractors.video.pose.kinematics import _keypoints_file
    from macaw.store.keypoints import write_keypoints

    config, layout, recording = depth_project
    assert VIDEO_INFO_NAME == "video.json"
    prep = DepthPrepExtractor(config).run(recording, layout.artifacts_dir("003", "prep.depth"))
    assert not prep.warnings
    pixels = pixels_of(JOINTS)
    rows = [
        {"recording_id": "003", "subject_id": "parent", "source": "pose.keypoints", "video": "kinect", "frame": frame, "t": frame / 30,
         "joint": joint, "x": pixels[index, 0], "y": pixels[index, 1], "z": np.nan, "confidence": 0.9, "track_id": 1, "units": "pixel",
         "metric_scale_valid": None}
        for frame in range(4) for index, joint in enumerate(("nose", "left_shoulder", "right_wrist"))
    ]
    write_keypoints(pd.DataFrame(rows), _keypoints_file(config, "003").parent)
    result = DepthPoseExtractor(config).run(recording, tmp_path / "work")
    placed = result.keypoints
    assert set(placed["frame"]) == {0, 1, 2} and (placed["video"] == "kinect").all() and (placed["units"] == "metre").all()
    first = placed[placed["frame"] == 0].set_index("joint").loc[["nose", "left_shoulder", "right_wrist"], ["x", "y", "z"]].to_numpy()
    assert np.abs(first - JOINTS).max() < 0.01
    # frame 2's depth lacks its left half, where the right wrist is; frame 3 has no depth image at all
    assert set(placed.loc[placed["frame"] == 2, "joint"]) == {"nose", "left_shoulder"}
    said = {w.subject_id: w.message for w in result.warnings}
    assert said["parent"] == "placed in 3D for 67% of the body points seen; the rest had no depth around them"
    assert "child" not in said


def test_the_3d_measures_follow_the_depth_camera(depth_project):
    from macaw.execution.dag import build_dag
    from macaw.extractors.video.pose.triangulate import points_file, three_d_step

    config, _, _ = depth_project
    assert three_d_step(config) == "pose.depth" and points_file(config, "003").name == "pose.depth.parquet"
    dag = build_dag(config, ["pose.kinematics3d"])
    assert {"pose.depth", "prep.depth", "pose.review"} <= set(dag.nodes)


def test_the_methods_say_the_points_come_from_the_depth_camera_and_lie_on_the_surface(depth_project):
    from macaw.qc.boilerplate import _placed_3d

    config, _, _ = depth_project
    text = _placed_3d(config)
    assert "from the depth camera their body was filmed by (kinect)" in text and "on the body's surface facing the camera" in text


def test_a_rational_lens_projects_as_opencv_does_and_reads_back():
    cv2 = pytest.importorskip("cv2")
    matrix = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    eight = np.array([0.4, -0.1, 0.001, -0.002, 0.02, 0.7, -0.05, 0.01])
    camera = Camera("kinect", (1280, 720), matrix, eight, np.zeros(3), np.zeros(3))
    points = np.random.default_rng(1).uniform([-0.6, -0.4, 1.5], [0.6, 0.4, 3.0], size=(50, 3))
    theirs, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, eight)
    mine = camera.distort(points[:, :2] / points[:, 2:3])
    assert np.abs(mine - theirs.reshape(-1, 2)).max() < 1e-6
    assert np.abs(camera.undistort(mine) - points[:, :2] / points[:, 2:3]).max() < 1e-6


def test_an_imported_recording_keeps_one_depth_image_per_frame_even_where_frames_were_lost(tmp_path):
    pytest.importorskip("cv2")
    from macaw.extractors.video.pose.board import _cv2
    from macaw.media.depth_import import Shot, Source, write_folder

    picture = np.zeros((48, 64, 3), np.uint8)
    times = [0.0, 0.1, 0.3, 0.3, 0.4]   # 0.2 s is lost; 0.3 s comes twice
    shots = (Shot(t, picture + i, np.full((48, 64), 1000 + i, np.uint16)) for i, t in enumerate(times))
    source = Source(COLOUR, 0.001, 10.0, shots, [])
    done = write_folder(source, tmp_path / "cam.mp4", tmp_path)
    assert (done["frames"], done["repeated"], done["size"]) == (5, 1, (64, 48))
    read = [_cv2().imread(str(tmp_path / f"{index:06d}.png"), -1) for index in range(5)]
    assert [int(image.max()) for image in read] == [1000, 1001, 0, 1002, 1004]
    lens = read_lens(tmp_path / LENS_FILE)
    assert lens.colour.size == (64, 48) and np.allclose(lens.depth.matrix, COLOUR.matrix) and np.allclose(lens.depth.rotation, 0)


def test_a_kinect_recording_is_read_through_the_makers_library(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    import sys
    import types

    from macaw.media import depth_import

    class ImageFormat:
        COLOR_MJPG, COLOR_NV12, COLOR_YUY2, COLOR_BGRA32 = range(4)

    picture = np.zeros((48, 64, 3), np.uint8)
    picture[:, :, 2] = 200          # red, in OpenCV's order
    encoded = cv2.imencode(".jpg", picture)[1]

    class Capture:
        def __init__(self, index):
            self.color, self.depth = encoded, np.ones((10, 10), np.uint16)
            self.transformed_depth = np.full((48, 64), 900 + index, np.uint16)
            self.color_timestamp_usec = index * 33_333

    class Playback:
        def __init__(self, path):
            self.left = 3
            self.calibration = types.SimpleNamespace(
                get_camera_matrix=lambda kind: COLOUR.matrix, get_distortion_coefficients=lambda kind: np.arange(8) / 100,
            )
            self.configuration = {"color_format": ImageFormat.COLOR_MJPG, "camera_fps": 2}

        def open(self):
            pass

        def close(self):
            pass

        def get_next_capture(self):
            if not self.left:
                raise EOFError
            self.left -= 1
            return Capture(2 - self.left)

    fake = types.SimpleNamespace(PyK4APlayback=Playback, CalibrationType=types.SimpleNamespace(COLOR=1), ImageFormat=ImageFormat)
    monkeypatch.setitem(sys.modules, "pyk4a", fake)
    done = depth_import.import_recording(tmp_path / "session.mkv", tmp_path / "kinect")
    assert (done["frames"], done["fps"], done["size"]) == (3, 30.0, (64, 48))
    assert int(cv2.imread(str(tmp_path / "kinect" / "depth" / "000002.png"), -1).max()) == 902
    lens = read_lens(tmp_path / "kinect" / "depth" / LENS_FILE)
    assert len(lens.colour.distortions) == 8
