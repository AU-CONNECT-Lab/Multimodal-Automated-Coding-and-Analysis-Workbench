from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import PoseSection
from macaw.extractors.video.pose import keypoints as pose_module
from macaw.extractors.video.pose.backends import BACKEND_KEYPOINT_SCHEMA, PoseFrames
from macaw.extractors.video.pose.keypoints import PoseDetectExtractor, PoseExtractor
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.track_repair import suggested_people
from macaw.extractors.video.pose.voice_motion import bin_edges
from macaw.store.schema import coerce_frame
from test_pose_rows import clicks, fake_prep_timing, media, offsets, recording  # noqa: F401
from test_voice_motion import talking, three_at_a_table

RNG = np.random.default_rng(11)
FPS = 10.0
SECONDS = 120.0


class TestReferences:
    def test_clear_matches_to_own_cameras_settle_who_is_who(self):
        moves = {"a": talking(), "b": talking(), "c": talking()}
        keypoints = three_at_a_table([moves["c"], moves["a"], moves["b"]])
        choice = suggested_people(keypoints, ["a", "b", "c"], references=moves, edges=bin_edges(0.0, 120.0))
        assert choice.people == {"a": [2], "b": [3], "c": [1]}
        assert choice.doubt is None

    def test_a_weak_match_still_places_by_motion_but_waits(self):
        rng = np.random.default_rng(3)
        moves = {name: (rng.random(240) < 0.3).astype(float) for name in "abc"}
        keypoints = three_at_a_table([moves["c"], moves["a"], moves["b"]])
        references = {k: v + 4.0 * rng.random(v.size) for k, v in moves.items()}
        choice = suggested_people(keypoints, ["a", "b", "c"], references=references, edges=bin_edges(0.0, 120.0))
        assert choice.people == {"a": [2], "b": [3], "c": [1]}
        assert "weak" in choice.doubt and "not yet checked by a person" in choice.doubt

    def test_someone_present_a_little_less_can_still_be_the_one_matched(self):
        """Two references, three people; the one they pair with is in view 70% of the time, the others always."""
        rng = np.random.default_rng(5)
        moves = {name: (rng.random(240) < 0.3).astype(float) for name in "abc"}
        keypoints = three_at_a_table([moves["c"], moves["a"], moves["b"]])
        keypoints = keypoints[~((keypoints.track_id == 3) & (keypoints.frame % 10 >= 7))]
        choice = suggested_people(keypoints, ["a", "b"], references={"a": moves["a"], "b": moves["b"]}, edges=bin_edges(0.0, 120.0))
        assert choice.people == {"a": [2], "b": [3]}

    def test_too_little_time_leaves_it_to_a_person(self):
        moves = {"a": talking(), "b": talking(), "c": talking()}
        keypoints = three_at_a_table([moves["c"], moves["a"], moves["b"]], seconds=12.0)
        choice = suggested_people(
            keypoints, ["a", "b", "c"], references={k: v[:24] for k, v in moves.items()}, edges=bin_edges(0.0, 12.0)
        )
        assert "not yet checked by a person" in choice.doubt


def nodding(activity, x, size, track):
    """One person whose head moves hard in the half seconds where activity is on."""
    rows = []
    for frame in range(int(SECONDS * FPS)):
        on = activity[min(int(frame / FPS / 0.5), activity.size - 1)] > 0
        nod = RNG.normal(0, 0.06 if on else 0.005) * size
        for index, joint in enumerate(STANDARD_JOINTS):
            rows.append({"frame": frame, "joint": joint, "x": x + (index % 5) * size / 5 + nod,
                         "y": 100.0 + (index // 5) * size / 4, "z": float("nan"), "confidence": 0.9,
                         "track_id": track, "units": "pixel", "metric_scale_valid": None})
    return rows


class PerFileBackend:
    name, version, joint_set = "fake", "9.9", "coco17"

    def __init__(self, by_file):
        self.by_file = by_file

    def check_available(self):
        return None

    def infer(self, video_path, **kwargs):
        return PoseFrames(keypoints=self.by_file[video_path.name])


@pytest.fixture
def room_and_own(monkeypatch, offsets, recording, tmp_path):  # noqa: F811
    """Each person on a camera of their own, and both on the room camera, the child on the left."""
    activity = {"parent": talking(n=240), "child": talking(n=240)}
    frame = lambda rows: coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
    backend = PerFileBackend({
        "parent.mp4": frame(nodding(activity["parent"], 200.0, 300.0, 1)),
        "child.mp4": frame(nodding(activity["child"], 200.0, 300.0, 1)),
        "room.mp4": frame(nodding(activity["child"], 100.0, 60.0, 1) + nodding(activity["parent"], 500.0, 60.0, 2)),
    })
    monkeypatch.setattr(pose_module, "resolve_backend", lambda name: backend)
    monkeypatch.setattr(pose_module, "apply_prep_timing", fake_prep_timing)
    monkeypatch.setattr(pose_module, "load_prep", lambda config, rid: SimpleNamespace(tracks={}, t_origin_s=0.0, duration_s=SECONDS))

    def run(suggest):
        config = SimpleNamespace(pose=PoseSection(suggest_names=suggest))
        files = [media(name, subject_id=subject) for name, subject in
                 (("parent.mp4", "parent"), ("child.mp4", "child"), ("room.mp4", None))]
        rec = recording(files, t_task_end=SECONDS - 1 / FPS)
        found = PoseDetectExtractor(config).run(rec, tmp_path / "detect")
        artifacts = tmp_path / "artifacts" / PoseDetectExtractor.name
        artifacts.mkdir(parents=True, exist_ok=True)
        for name, source in found.artifacts.items():
            (artifacts / name).write_bytes(source.read_bytes())
        return found, PoseExtractor(config).run(rec, tmp_path)

    return run


class TestNamingTheRoomFromOwnCameras:
    def test_without_the_option_the_room_waits_for_a_person_to_click(self, room_and_own):
        _, result = room_and_own(False)
        room = result.keypoints[result.keypoints.video == "room.mp4"]
        assert room.subject_id.isna().all()
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert "nobody has said who is who on room.mp4" in owners["room.mp4"]["doubt"]

    def test_with_it_the_room_people_are_named_by_their_own_camera_and_the_run_goes_on(self, room_and_own):
        _, result = room_and_own(True)
        room = result.keypoints[result.keypoints.video == "room.mp4"]
        assert room.groupby("subject_id")["x"].min().lt(300).to_dict() == {"child": True, "parent": False}
        owners = json.loads(result.artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["room.mp4"]["doubt"] is None and owners["room.mp4"]["chosen_by"] == {"child": "suggested", "parent": "suggested"}

    def test_quality_rows_come_from_each_persons_own_camera(self, room_and_own):
        _, result = room_and_own(True)
        rows = result.rows[result.rows.feature == "detection_rate"]
        assert rows.value.astype(float).tolist() == [1.0, 1.0]


@pytest.fixture
def two_rooms(monkeypatch, offsets, recording, tmp_path):  # noqa: F811
    """Two room cameras facing each other: the child is left on one and right on the other."""
    activity = {"parent": talking(n=240), "child": talking(n=240)}
    frame = lambda rows: coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")
    backend = PerFileBackend({
        "room_a.mp4": frame(nodding(activity["child"], 100.0, 60.0, 1) + nodding(activity["parent"], 500.0, 60.0, 2)),
        "room_b.mp4": frame(nodding(activity["parent"], 100.0, 60.0, 1) + nodding(activity["child"], 500.0, 60.0, 2)),
    })
    monkeypatch.setattr(pose_module, "resolve_backend", lambda name: backend)
    monkeypatch.setattr(pose_module, "apply_prep_timing", fake_prep_timing)
    monkeypatch.setattr(pose_module, "load_prep", lambda config, rid: SimpleNamespace(tracks={}, t_origin_s=0.0, duration_s=SECONDS))
    config = SimpleNamespace(pose=PoseSection(suggest_names=True))
    clicks(tmp_path, "room_a.mp4", 1.0, child=(100.0, 125.0), parent=(500.0, 125.0))
    rec = recording([media("room_a.mp4"), media("room_b.mp4")], t_task_end=SECONDS - 1 / FPS)
    found = PoseDetectExtractor(config).run(rec, tmp_path / "detect")
    artifacts = tmp_path / "artifacts" / PoseDetectExtractor.name
    artifacts.mkdir(parents=True, exist_ok=True)
    for name, source in found.artifacts.items():
        (artifacts / name).write_bytes(source.read_bytes())
    return PoseExtractor(config).run(rec, tmp_path)


class TestTwoRoomCameras:
    def test_the_second_camera_is_named_by_motion_against_the_one_clicked(self, two_rooms):
        left = two_rooms.keypoints[two_rooms.keypoints.x < 300].groupby("video")["subject_id"].agg(lambda s: set(s.dropna()))
        assert {video: set(who) for video, who in left.items()} == {"room_a.mp4": {"child"}, "room_b.mp4": {"parent"}}
        owners = json.loads(two_rooms.artifacts["owners"].read_text(encoding="utf-8"))
        assert owners["room_a.mp4"]["doubt"] is None and owners["room_b.mp4"]["doubt"] is None
