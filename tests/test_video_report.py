from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("jinja2")

from macaw.config.loader import resolve_config, scaffold_project_config  # noqa: E402
from macaw.extractors.prep import VIDEO_INFO_NAME, VideoPrepExtractor  # noqa: E402
from macaw.project.layout import layout_for  # noqa: E402
from macaw.qc.report import _common_rate, _overview_features, write_project_report, write_recording_report  # noqa: E402
from macaw.store.longtable import build_long_rows, write_long_table  # noqa: E402
from macaw.store.schema import Scale  # noqa: E402


def video(name, fps, *, made=True, variable=False, repeated=0, dropped=0):
    return {
        "name": name, "file": f"{name}.mp4" if made else None, "fps": fps, "variable_rate": variable,
        "frames_repeated": repeated if made else None, "frames_dropped": dropped if made else None,
    }


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for recording in ("001", "002"):
        (raw / recording).mkdir(parents=True)
        for person in ("parent", "child"):
            (raw / recording / f"{recording}_{person}.mp4").write_bytes(b"")
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    config = resolve_config(path)

    def prepared(recording, entries):
        directory = layout_for(config).artifacts_dir(recording, VideoPrepExtractor.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / VIDEO_INFO_NAME).write_text(json.dumps(entries), encoding="utf-8")

    return config, prepared


def test_the_recording_page_says_what_was_done_to_its_videos(project):
    config, prepared = project
    prepared("001", [video("001_parent.mp4", 30.0, variable=True, repeated=12, dropped=3), video("001_child.mp4", 30.0)])
    page, _ = write_recording_report(config, "001")
    text = page.read_text(encoding="utf-8")
    assert "2 file(s) re-encoded at a constant 30 fps" in text
    assert "1 had a variable frame rate; 12 frame(s) repeated and 3 left out" in text


def test_videos_read_as_recorded_say_so(project):
    config, prepared = project
    prepared("001", [video("001_parent.mp4", 29.6, made=False)])
    page, _ = write_recording_report(config, "001")
    assert "read as recorded (video.normalise: false)" in page.read_text(encoding="utf-8")


def test_the_project_page_lists_frame_rates_when_they_differ(project):
    config, prepared = project
    prepared("001", [video("001_parent.mp4", 30.0), video("001_child.mp4", 30.0)])
    prepared("002", [video("002_parent.mp4", 60.0), video("002_child.mp4", 60.0)])
    pages = {recording: write_recording_report(config, recording) for recording in ("001", "002")}
    text = write_project_report(config, pages).read_text(encoding="utf-8")
    assert "Frame rates" in text and "video.fps: 30" in text


def test_one_frame_rate_needs_no_table(project):
    config, prepared = project
    for recording in ("001", "002"):
        prepared(recording, [video(f"{recording}_parent.mp4", 25.0)])
    pages = {recording: write_recording_report(config, recording) for recording in ("001", "002")}
    assert "Frame rates" not in write_project_report(config, pages).read_text(encoding="utf-8")


@pytest.mark.parametrize(("rates", "expected"), [([30.0, 60.0], 30.0), ([29.97, 59.94], 29.97), ([25.0, 30.0], None)])
def test_a_rate_every_video_can_be_brought_to(rates, expected):
    assert _common_rate(rates) == expected


def kinematics_rows(config, recording):
    t = np.arange(0.0, 20.0, 1 / 30)
    common = dict(recording_id=recording, source="pose.kinematics", scale=Scale.FRAME)
    people = [
        build_long_rows(subject_id=who, values=pd.DataFrame({"t_start": t, "left_wrist_speed": np.cos(t), "right_wrist_speed": np.sin(t)}), **common)
        for who in ("parent", "child")
    ]
    pair = build_long_rows(
        subject_id="parent", subject_id_b="child", values=pd.DataFrame({"t_start": t, "interpersonal_distance": 200 + t}), **common
    )
    write_long_table(pd.concat([*people, pair], ignore_index=True), layout_for(config).long_table_path(recording))


def test_the_overview_shows_movement_and_distance_when_nothing_is_chosen(project):
    config, _ = project
    kinematics_rows(config, "001")
    page, _ = write_recording_report(config, "001")
    assert "Overview" in page.read_text(encoding="utf-8")
    assert (layout_for(config).renders_dir("001") / "report" / "overview.png").is_file()


def test_the_overview_draws_what_is_chosen_or_movement_distance_and_the_projects_measures():
    measures = [type("M", (), {"name": "hand_to_head"})()]
    default = type("C", (), {"qc": type("Q", (), {"overview_features": []})(), "kinematics": type("K", (), {"measures": measures})()})()
    assert _overview_features(default, ["motion.mea", "pose.kinematics"]) == ["motion_energy", "interpersonal_distance", "hand_to_head"]
    assert _overview_features(default, ["pose.kinematics"])[:2] == ["left_wrist_speed", "right_wrist_speed"]
    assert _overview_features(default, ["text.asr"]) == []
    default.qc.overview_features = ["f0_hz"]
    assert _overview_features(default, ["pose.kinematics"]) == ["f0_hz"]
