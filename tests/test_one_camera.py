from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from macaw.exceptions import ConfigError
from macaw.extractors.video.pose.one_camera import (
    PYTHON_ENV_VAR, MeshExtractor, OneCameraExtractor, match, mesh_file, read_mesh,
)
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS

K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
# two people as the model fits them, in the camera's frame: each a small figure, the second further away
FIGURES = {
    "parent": np.column_stack([np.linspace(-0.6, -0.3, 17), np.linspace(-0.4, 0.4, 17), np.full(17, 2.5)]),
    "child": np.column_stack([np.linspace(0.3, 0.5, 17), np.linspace(-0.1, 0.3, 17), np.full(17, 3.0)]),
}


def pixels(points: np.ndarray) -> np.ndarray:
    flat = points @ K.T
    return flat[:, :2] / flat[:, 2:3]


# the runner contract: python runner.py job.json, writing these arrays per body found; guarded, since the project's
# scripts folder is also read for measure functions
FAKE_RUNNER = f"""
import json, sys
import numpy as np


def main():
  job = json.loads(open(sys.argv[1], encoding="utf-8").read())
  figures = {json.dumps({name: points.tolist() for name, points in FIGURES.items()})}
  K = np.array({K.tolist()})
  frames, points, pixels, heights = [], [], [], []
  for frame in range(0, 4, job["every"]):
    for name, figure in figures.items():
      figure = np.array(figure)
      flat = figure @ K.T
      frames.append(frame); points.append(figure); pixels.append(flat[:, :2] / flat[:, 2:3])
      heights.append(1.7 if name == "parent" else 1.4)
  np.savez_compressed(job["out"], frame=np.array(frames), points=np.array(points), pixels=np.array(pixels),
                      height=np.array(heights), focal=np.full(len(frames), 1000.0))


if __name__ == "__main__":
  main()
"""


@pytest.fixture
def one_camera_project(tmp_path, monkeypatch):
    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.project.layout import layout_for
    from macaw.project.recording import build_recording
    from setup_helpers import camera, put, write_setup

    raw = tmp_path / "raw"
    put(raw, "003", "room", "room.mp4")
    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(project, cameras={"room": camera()})
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fake_runner.py").write_text(FAKE_RUNNER, encoding="utf-8")
    with open(project, "a", encoding="utf-8") as handle:
        handle.write("\npose3d:\n  one_camera: fake_runner.py\n  one_camera_every: 1\n")
    (tmp_path / "heights.csv").write_text("recording,person,height\n003,child,98\n", encoding="utf-8")
    monkeypatch.setenv(PYTHON_ENV_VAR, sys.executable)
    config = resolve_config(project)
    layout = layout_for(config)
    video = layout.artifacts_dir("003", "prep.video")
    video.mkdir(parents=True)
    (video / "video.json").write_text(json.dumps([{"name": "room", "file": None, "fps": 10.0}]), encoding="utf-8")
    return config, layout, build_recording("003", config)


def test_the_heights_table_is_read_and_checked(one_camera_project, tmp_path):
    from macaw.project.heights import read_heights

    config, _, _ = one_camera_project
    assert read_heights(config) == {("003", "child"): 98.0}
    for text, reason in (("003,uncle,170", "not one of project.roles"), ("003,child,tall", "is not a length"), ("003,child,0", "is not a length"),
                         ("003,child,98\n003,child,99", "given twice")):
        (tmp_path / "heights.csv").write_text("recording,person,height\n" + text + "\n", encoding="utf-8")
        with pytest.raises(ConfigError, match=reason):
            read_heights(config)


def test_each_named_person_takes_the_fitted_body_nearest_their_keypoints():
    mine = {"parent": pixels(FIGURES["parent"]), "child": pixels(FIGURES["child"])}
    mine["child"][:5] = np.nan
    found = np.stack([pixels(FIGURES["child"]) + 3, pixels(FIGURES["parent"]) - 2, pixels(FIGURES["parent"]) + 900])
    assert match(mine, found) == {"parent": 1, "child": 0}
    # a body far from anyone's keypoints is nobody's
    assert match({"parent": mine["parent"]}, found[2:]) == {}


def test_one_camera_and_another_way_to_3d_are_refused_together(one_camera_project):
    from macaw.extractors.video.pose.triangulate import ONE_CAMERA_STEP, three_d_step

    config, _, _ = one_camera_project
    assert three_d_step(config) == ONE_CAMERA_STEP
    both = config.model_copy(update={"pose3d": config.pose3d.model_copy(update={"calibration": "lab"})})
    with pytest.raises(ConfigError, match="3D comes one way per project"):
        three_d_step(both)


def test_the_steps_place_each_named_person_from_the_body_fitted_to_them(one_camera_project, tmp_path):
    from macaw.extractors.video.pose.kinematics import _keypoints_file
    from macaw.store.keypoints import write_keypoints

    config, layout, recording = one_camera_project
    mesh = MeshExtractor(config).run(recording, layout.artifacts_dir("003", MeshExtractor.name))
    written = read_mesh(layout.artifacts_dir("003", MeshExtractor.name) / mesh_file("room"))
    assert len(written["frame"]) == 8 and "the model estimated the picture's angle itself" in mesh.warnings[0].message
    job = json.loads((layout.artifacts_dir("003", MeshExtractor.name) / "room.job.json").read_text(encoding="utf-8"))
    assert job["threshold"] == 0.1 and job["every"] == 1
    rows = []
    for person, figure in FIGURES.items():
        seen = pixels(figure)
        for frame in range(3):
            for index, joint in enumerate(STANDARD_JOINTS):
                rows.append({"recording_id": "003", "subject_id": person, "source": "pose.keypoints", "video": "room", "frame": frame,
                             "t": frame / 10, "joint": joint, "x": seen[index, 0], "y": seen[index, 1], "z": np.nan, "confidence": 0.9,
                             "track_id": 1, "units": "pixel", "metric_scale_valid": None})
    write_keypoints(pd.DataFrame(rows), _keypoints_file(config, "003").parent)
    result = OneCameraExtractor(config).run(recording, tmp_path / "work")
    placed = result.keypoints
    assert set(placed["frame"]) == {0, 1, 2} and (placed["video"] == "room").all()
    parent = placed[(placed["subject_id"] == "parent") & (placed["frame"] == 0)].set_index("joint").loc[list(STANDARD_JOINTS), ["x", "y", "z"]]
    assert np.allclose(parent.to_numpy(), FIGURES["parent"]) and not placed.loc[placed["subject_id"] == "parent", "metric_scale_valid"].any()
    # the child's stated 98 cm against the fitted body's 1.4 m
    child = placed[(placed["subject_id"] == "child") & (placed["frame"] == 0)].set_index("joint").loc[list(STANDARD_JOINTS), ["x", "y", "z"]]
    assert np.allclose(child.to_numpy(), FIGURES["child"] * 0.98 / 1.4) and placed.loc[placed["subject_id"] == "child", "metric_scale_valid"].all()
    said = {}
    for warning in result.warnings:
        said.setdefault(warning.subject_id, []).append(warning.message)
    assert any(m.startswith("no height in the heights table, so placed at the size of the body the model fitted (1.70 m tall)") for m in said["parent"])
    assert "set to the stated height, 0.98 m, from the fitted body's 1.40 m (scaled by 0.70)" in said["child"]
    assert "placed in 3D in 100% of the frames they were seen in" in said["child"]


def test_how_often_someone_was_placed_counts_only_the_frames_the_model_read(one_camera_project, tmp_path):
    from macaw.extractors.video.pose.kinematics import _keypoints_file
    from macaw.store.keypoints import write_keypoints

    config, layout, recording = one_camera_project
    config = config.model_copy(update={"pose3d": config.pose3d.model_copy(update={"one_camera_every": 2})})
    MeshExtractor(config).run(recording, layout.artifacts_dir("003", MeshExtractor.name))
    seen = pixels(FIGURES["child"])
    rows = [
        {"recording_id": "003", "subject_id": "child", "source": "pose.keypoints", "video": "room", "frame": frame, "t": frame / 10,
         "joint": joint, "x": seen[index, 0], "y": seen[index, 1], "z": np.nan, "confidence": 0.9, "track_id": 1, "units": "pixel",
         "metric_scale_valid": None}
        for frame in range(4) for index, joint in enumerate(STANDARD_JOINTS)
    ]
    write_keypoints(pd.DataFrame(rows), _keypoints_file(config, "003").parent)
    result = OneCameraExtractor(config).run(recording, tmp_path / "work")
    assert set(result.keypoints["frame"]) == {0, 2}
    said = [w for w in result.warnings if w.subject_id == "child" and w.message.startswith("placed in 3D")]
    assert said[0].severity == "info"
    assert said[0].message == "placed in 3D in 100% of the frames they were seen in, counting the one frame in 2 the body model read"


def test_speeds_in_3d_are_taken_over_the_frames_the_model_read(one_camera_project):
    from macaw.extractors.video.pose.triangulate import frame_stride, subject_values_3d

    config, _, _ = one_camera_project
    assert frame_stride(config) == 1
    config = config.model_copy(update={"pose3d": config.pose3d.model_copy(update={"one_camera_every": 3})})
    assert frame_stride(config) == 3
    # the left wrist moves 1 m/s along x at 30 fps, placed on one frame in three
    frames = np.arange(0, 91, 3)
    points = pd.DataFrame([
        {"frame": frame, "t": frame / 30, "joint": joint, "x": frame / 30 if joint == "left_wrist" else 0.0, "y": 0.0, "z": 2.0}
        for frame in frames for joint in ("left_wrist", "right_wrist")
    ])
    speed = subject_values_3d(points, 0.25, frame_stride(config))["left_wrist_speed_3d"]
    assert np.allclose(speed, 1.0)
    # read as if the frames were consecutive, every one of them is a gap
    assert subject_values_3d(points, 0.25)["left_wrist_speed_3d"].isna().all()


def test_people_placed_from_different_cameras_are_not_measured_against_each_other():
    from macaw.extractors.video.pose.triangulate import apart, frame_of

    one = pd.DataFrame({"video": ["room", "room"]})
    other = pd.DataFrame({"video": ["side"]})
    room = pd.DataFrame({"video": ["3d"]})
    assert frame_of(one) == "room" and not apart({"a": one, "b": one}, "a", "b")
    assert apart({"a": one, "b": other}, "a", "b") and not apart({"a": room, "b": room}, "a", "b")


def test_the_methods_say_how_size_was_set(one_camera_project):
    from macaw.qc.boilerplate import _placed_3d

    config, _, _ = one_camera_project
    text = _placed_3d(config)
    assert "a body model run by the study's own script fake_runner.py" in text and "scaled to it" in text
    assert "too large for a small child" in text and "only for people placed from the same camera" in text
    assert "detection threshold" not in text
    built_in = config.model_copy(update={"pose3d": config.pose3d.model_copy(update={"one_camera": "multihmr", "one_camera_threshold": 0.2})})
    assert "fitting the Anny body model [bregier2026] (detection threshold 0.2) to every person" in _placed_3d(built_in)
