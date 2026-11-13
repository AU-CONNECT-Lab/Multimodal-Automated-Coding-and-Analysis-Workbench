from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import GazeSection, RoomSection
from macaw.extractors.video.gaze import angles
from macaw.project.room import seats as room_seats
from macaw.extractors.video.gaze import target as target_module
from macaw.extractors.video.gaze.heads import head_boxes
from macaw.extractors.video.gaze.target import (
    ELSEWHERE,
    OFF_SCREEN,
    GazeTargetExtractor,
    label_runs,
    mutual_gaze,
    name_targets,
    settle_looks,
)
from macaw.project.recording import MediaFile
from macaw.store.keypoints import write_keypoints
from macaw.store.schema import ANNOTATION_SCHEMA, KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, validate_frame
from setup_helpers import video_file, recording_of

FPS = 10.0
FACE = {"nose": (0, 0), "left_eye": (8, -6), "right_eye": (-8, -6), "left_ear": (20, -2), "right_ear": (-20, -2),
        "left_shoulder": (45, 60), "right_shoulder": (-45, 60)}


def keypoints(people, n_frames=10, video="room.mp4", joints=FACE):
    """people: subject -> where their nose is; the other points sit around it as in FACE."""
    rows = [
        {"recording_id": "r1", "subject_id": subject, "source": "pose.keypoints", "video": video, "frame": frame,
         "t": frame / FPS, "joint": joint, "x": x + dx, "y": y + dy, "z": float("nan"), "confidence": 0.9, "track_id": 0,
         "units": "pixel", "metric_scale_valid": None}
        for frame in range(n_frames) for subject, (x, y) in people.items() for joint, (dx, dy) in joints.items()
    ]
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def box(x0, y0, x1, y1, subject, frames=range(10)):
    return pd.DataFrame([{"subject_id": subject, "frame": f, "x0": x0, "y0": y0, "x1": x1, "y1": y1} for f in frames])


def gaze(subject, x, y, in_frame=0.9, frames=range(10)):
    return pd.DataFrame([{"frame": f, "subject_id": subject, "gaze_x": x, "gaze_y": y, "in_frame": in_frame, "t": f / FPS} for f in frames])


class TestHeadBoxes:
    def test_the_box_holds_the_face_points_and_is_square(self):
        boxes = head_boxes(keypoints({"parent": (100, 100)}))
        first = boxes.iloc[0]
        assert len(boxes) == 10 and first.x0 <= 80 and first.x1 >= 120 and first.y0 <= 94 and first.y1 >= 100
        assert first.x1 - first.x0 == pytest.approx(first.y1 - first.y0)

    def test_a_turned_head_with_bunched_points_takes_its_size_from_the_shoulders(self):
        profile = {"nose": (0, 0), "left_eye": (-3, -5), "left_ear": (-10, -3), "left_shoulder": (40, 60), "right_shoulder": (-40, 60)}
        boxes = head_boxes(keypoints({"parent": (100, 100)}, joints=profile))
        assert boxes.iloc[0].x1 - boxes.iloc[0].x0 > 30

    def test_one_face_point_is_not_a_head(self):
        boxes = head_boxes(keypoints({"parent": (100, 100)}, joints={"nose": (0, 0), "left_shoulder": (40, 60)}))
        assert boxes.empty


class TestNameTargets:
    def test_a_point_on_someone_s_head_names_them(self):
        named = name_targets(gaze("parent", 410, 110), box(380, 80, 440, 140, "child"))
        assert set(named.target) == {"child"} and set(named.to_child) == {0.0}

    def test_a_point_beside_every_head_is_elsewhere_with_its_distance(self):
        named = name_targets(gaze("parent", 500, 110), box(380, 80, 440, 140, "child"))
        assert set(named.target) == {ELSEWHERE}
        assert named.to_child.iloc[0] == pytest.approx(60 / 60)

    def test_a_target_out_of_the_picture_is_off_screen_even_on_a_head(self):
        named = name_targets(gaze("parent", 410, 110, in_frame=0.2), box(380, 80, 440, 140, "child"))
        assert set(named.target) == {OFF_SCREEN}

    def test_no_one_is_measured_against_their_own_head(self):
        heads = pd.concat([box(380, 80, 440, 140, "child"), box(80, 80, 140, 140, "parent")])
        named = name_targets(gaze("parent", 110, 110), heads)
        assert named.to_parent.isna().all() and set(named.target) == {ELSEWHERE}


def test_mutual_gaze_needs_both_looking_at_each_other_in_the_same_frame():
    points = pd.concat([
        gaze("parent", 0, 0).assign(target=["child"] * 5 + [ELSEWHERE] * 5),
        gaze("child", 0, 0).assign(target=["parent"] * 3 + [ELSEWHERE] * 7),
    ])
    mutual = mutual_gaze(points, "child", "parent")
    assert mutual["mutual"].tolist() == [True] * 3 + [False] * 7


def test_people_read_from_cameras_with_different_offsets_are_paired_by_time():
    # the child's camera started 0.3 s later, so its frame 0 is the parent's frame 3
    parent = gaze("parent", 0, 0).assign(target=["child"] * 10)
    child = gaze("child", 0, 0, frames=range(7)).assign(t=lambda f: f["frame"] / FPS + 0.3, target=["parent"] * 2 + [ELSEWHERE] * 5)
    mutual = mutual_gaze(pd.concat([parent, child]), "child", "parent")
    assert mutual["t"].tolist() == pytest.approx([0.3 + f / FPS for f in range(7)])
    assert mutual["mutual"].tolist() == [True] * 2 + [False] * 5


class TestJointAttention:
    def points(self, **targets):
        return pd.concat([gaze(who, 0, 0, frames=range(len(seen))).assign(target=seen) for who, seen in targets.items()])

    def test_the_one_who_looked_first_leads_and_the_lag_is_how_much_later_the_other_came(self):
        points = self.points(child=["toy"] * 10, parent=[ELSEWHERE] * 3 + ["toy"] * 5 + ["book"] * 2)
        episodes = target_module.joint_episodes(target_module.joint_attention(points, "child", "parent", ["toy", "book"]), "child", "parent")
        assert episodes[["thing", "first", "follower", "leader"]].values.tolist() == [["toy", "child", "parent", "child"]]
        assert episodes["lag_s"].tolist() == pytest.approx([0.3])
        assert episodes[["t_start", "t_end"]].values.tolist() == [pytest.approx([0.3, 0.8])]

    def test_looks_that_began_together_have_no_leader(self):
        points = self.points(child=[ELSEWHERE] + ["toy"] * 4, parent=[ELSEWHERE] + ["toy"] * 4)
        episodes = target_module.joint_episodes(target_module.joint_attention(points, "child", "parent", ["toy"]), "child", "parent")
        assert episodes["first"].tolist() == [None] and episodes["lag_s"].tolist() == [0.0]

    def test_both_looking_at_a_third_person_is_not_joint_attention(self):
        points = self.points(child=["sibling"] * 5, parent=["sibling"] * 5)
        assert target_module.joint_attention(points, "child", "parent", ["toy"])["thing"].isna().all()


class TestSettleLooks:
    def looks(self, targets, frames=None):
        frames = list(range(len(targets))) if frames is None else frames
        return pd.DataFrame({"frame": frames, "subject_id": "parent", "t": [f / FPS for f in frames], "target": targets})

    def test_a_flicker_joins_the_look_before_it_and_what_was_read_is_kept(self):
        settled = settle_looks(self.looks(["child"] * 4 + [ELSEWHERE] + ["child"] * 4 + ["toy"] * 3), 0.2)
        assert settled["target"].tolist() == ["child"] * 9 + ["toy"] * 3
        assert settled["target_read"].tolist()[4] == ELSEWHERE

    def test_a_short_first_look_takes_the_one_after_it(self):
        assert settle_looks(self.looks(["toy"] + ["child"] * 4), 0.2)["target"].tolist() == ["child"] * 5

    def test_a_look_exactly_as_long_as_the_shortest_stays(self):
        targets = ["child"] * 3 + ["toy"] * 2 + ["child"] * 3
        assert settle_looks(self.looks(targets), 0.2)["target"].tolist() == targets

    def test_looks_on_either_side_of_a_gap_are_not_joined(self):
        settled = settle_looks(self.looks(["child"] * 4 + ["toy"], frames=[0, 1, 2, 3, 7]), 0.2)
        assert settled["target"].tolist() == ["child"] * 4 + ["toy"]

    def test_zero_keeps_every_frame_as_read(self):
        targets = ["child", ELSEWHERE, "child"]
        assert settle_looks(self.looks(targets), 0)["target"].tolist() == targets


def test_label_runs_join_consecutive_frames_and_break_at_a_gap():
    frame = pd.DataFrame({"frame": [0, 1, 2, 5, 6], "t": [0.0, 0.1, 0.2, 0.5, 0.6], "target": ["a", "a", "b", "b", "b"]})
    runs = label_runs(frame, "target")
    assert runs["label"].tolist() == ["a", "b", "b"]
    assert runs["t_start"].tolist() == [0.0, 0.2, 0.5]
    assert runs["t_end"].tolist() == pytest.approx([0.2, 0.3, 0.7])


class FakeGazeBackend:
    name, version, default_model, finds_points = "fake", "1.0", "m", True

    def __init__(self, points):
        self.points = points
        self.calls = []

    def infer(self, video_path, *, heads, **kwargs):
        self.calls.append((Path(video_path).name, sorted(set(heads["subject_id"]))))
        return self.points[self.points["subject_id"].isin(set(heads["subject_id"]))].drop(columns="t")


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(kp, points, files, targets=()):
        stored = tmp_path / "pose.keypoints.parquet"
        write_keypoints(kp, stored)
        backend = FakeGazeBackend(points)
        monkeypatch.setattr(target_module, "_keypoints_file", lambda config, rid: stored)
        monkeypatch.setattr(target_module, "resolve_backend", lambda name: backend)
        monkeypatch.setattr(target_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=0.9))
        recording = recording_of(files, recording_id="r1", group_id="1", session_id=None)
        return backend, GazeTargetExtractor(SimpleNamespace(gaze=GazeSection(targets=list(targets)))).run(recording, tmp_path)

    return build


def video(name, subject_id=None):
    return video_file(name, (subject_id,) if subject_id else ("parent", "child"), fps=FPS, offset_s=0.0)


class TestExtractor:
    def test_two_people_looking_at_each_other(self, run):
        people = keypoints({"parent": (400, 100), "child": (100, 100)})
        points = pd.concat([gaze("parent", 100, 98), gaze("child", 400, 98).assign(gaze_x=[400] * 6 + [700] * 4)])
        _, result = run(people, points, [video("room.mp4")])
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        validate_frame(result.annotations, ANNOTATION_SCHEMA, "annotations")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        share = result.rows[result.rows.feature == "looks_at_share"].set_index(["subject_id", "subject_id_b"])["value"]
        assert share.to_dict() == {("child", "parent"): pytest.approx(0.6), ("parent", "child"): 1.0}
        mutual = result.rows[result.rows.feature == "mutual_gaze_share"]
        assert mutual.value.tolist() == [pytest.approx(0.6)] and mutual.subject_id.tolist() == ["child"]
        child = result.annotations[(result.annotations.tier == "gaze_target") & (result.annotations.subject_id == "child")]
        assert child.label.tolist() == ["parent", ELSEWHERE]
        assert result.artifacts["gaze_points.parquet"].is_file()

    def test_joint_attention_on_a_named_thing_gives_its_share_stretches_and_who_led(self, run, monkeypatch):
        people = keypoints({"parent": (400, 100), "child": (100, 100)})
        points = pd.concat([gaze("parent", 250, 300), gaze("child", 250, 300)], ignore_index=True)
        things = pd.DataFrame([{"video": "room.mp4", "frame": f, "label": "toy", "x0": 200, "y0": 250, "x1": 300, "y1": 350, "score": 0.9} for f in range(10)])
        monkeypatch.setattr(target_module, "load_objects", lambda config, rid: things)
        _, result = run(people, points, [video("room.mp4")], targets=["toy"])
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        value = result.rows[result.rows.scale != "frame"].set_index(["feature", "subject_id", "subject_id_b"])["value"]
        assert value[("joint_attention_share", "child", "parent")] == 1.0 and value[("joint_attention_n", "child", "parent")] == 1.0
        assert value[("joint_attention_led_n", "child", "parent")] == 0.0 and value[("joint_attention_led_n", "parent", "child")] == 0.0
        assert value[("joint_attention_lag_s", "child", "parent")] == 0.0
        tier = result.annotations[result.annotations.tier == "joint_attention_with_parent"]
        assert tier.label.tolist() == ["toy"] and tier.subject_id.tolist() == ["child"]

    def test_without_named_things_there_is_no_joint_attention(self, run):
        _, result = run(keypoints({"parent": (400, 100), "child": (100, 100)}), pd.concat([gaze("parent", 250, 300), gaze("child", 250, 300)]), [video("room.mp4")])
        assert not any(name.startswith("joint_attention") for name in set(result.rows.feature))

    def test_someone_alone_on_their_own_camera_is_reported(self, run):
        kp = pd.concat([keypoints({"parent": (100, 100)}, video="parent.mp4"), keypoints({"child": (100, 100)}, video="child.mp4")])
        points = pd.concat([gaze("parent", 300, 100), gaze("child", 300, 100)])
        backend, result = run(kp, points, [video("parent.mp4", "parent"), video("child.mp4", "child")])
        assert sorted(backend.calls) == [("child.mp4", ["child"]), ("parent.mp4", ["parent"])]
        assert {w.subject_id for w in result.warnings if "cannot be read off it" in w.message} == {"parent", "child"}
        assert "looks_at" not in set(result.rows.feature)
        # neither can be seen looking at the other, so mutual gaze is unknown rather than 0
        assert not {"mutual_gaze", "mutual_gaze_share"} & set(result.rows.feature)


class TestNameObjects:
    def things(self, frames=(0, 5)):
        return pd.DataFrame(
            [{"video": "room.mp4", "frame": f, "label": label, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": 0.9}
             for f in frames for label, (x0, y0, x1, y1) in {"table": (0, 200, 600, 400), "bottle": (280, 250, 320, 330)}.items()]
        )

    def test_the_smallest_box_holding_the_point_names_it(self):
        points = name_targets(gaze("parent", 300, 300, frames=range(3)), box(380, 80, 440, 140, "child", frames=range(3)))
        named = target_module.name_objects(points, self.things(), within_frames=2)
        assert set(named.target) == {"bottle"}

    def test_a_head_comes_before_a_thing_and_a_look_too_far_in_time_takes_nothing(self):
        heads = box(280, 250, 320, 330, "child", frames=range(10))
        points = name_targets(gaze("parent", 300, 300), heads)
        assert set(target_module.name_objects(points, self.things(), within_frames=2).target) == {"child"}
        free = name_targets(gaze("parent", 100, 300, frames=[2, 9]), heads)
        assert target_module.name_objects(free, self.things(frames=(0,)), within_frames=2).target.tolist() == ["table", ELSEWHERE]


@pytest.fixture
def run_angles(tmp_path, monkeypatch):
    from macaw.store.longtable import build_long_rows, write_long_table

    def build(angles_by_subject, *, room, labels=(), tolerance=15.0):
        stored = tmp_path / "pose.keypoints.parquet"
        write_keypoints(keypoints({"parent": (400, 100), "child": (100, 100)}), stored)
        rows = pd.concat([
            build_long_rows(values=pd.DataFrame({"t_start": np.arange(len(yaws)) / FPS, "face_yaw_deg": yaws, "face_pitch_deg": 0.0}),
                            recording_id="r1", source="face.au", scale="frame", subject_id=subject)
            for subject, yaws in angles_by_subject.items()
        ])
        face = tmp_path / "face.au.parquet"
        write_long_table(rows, face)
        label_file = tmp_path / "gaze_labels.json"
        from macaw.project.gaze_labels import GazeLabel, write_labels
        write_labels([GazeLabel(*label) for label in labels], label_file)
        monkeypatch.setattr(target_module, "_keypoints_file", lambda config, rid: stored)
        monkeypatch.setattr(target_module, "_face_rows_file", lambda config, rid: face)
        monkeypatch.setattr(target_module, "layout_for", lambda config, create=False: SimpleNamespace(gaze_labels_path=lambda rid: label_file))
        monkeypatch.setattr(target_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=0.9))
        recording = recording_of([video("room.mp4")], recording_id="r1", group_id="1", session_id=None)
        config = SimpleNamespace(gaze=GazeSection(backend="angles", tolerance_deg=tolerance), setup=SimpleNamespace(room=RoomSection(**room)))
        return GazeTargetExtractor(config).run(recording, tmp_path)

    return build


class TestAngles:
    # the camera level at eye height, the parent on its left and the child on its right, both a metre ahead
    ROOM = {
        "cameras": {"room.mp4": {"at": [0, 0], "height": 100, "facing": 0}},
        "people": {"parent": {"at": [-100, 100], "eyes": 100}, "child": {"at": [100, 100], "eyes": 100}},
    }

    def test_the_room_names_whom_each_looks_at(self, run_angles):
        result = run_angles({"parent": [-90.0] * 6 + [0.0] * 4, "child": [90.0] * 10}, room=self.ROOM)
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        share = result.rows[result.rows.feature == "looks_at_share"].set_index("subject_id")["value"]
        assert share.to_dict() == {"parent": pytest.approx(0.6), "child": 1.0}
        off = result.rows[(result.rows.feature == "gaze_off_deg") & (result.rows.subject_id == "parent")]["value"]
        assert off.iloc[0] == pytest.approx(0.0) and off.iloc[-1] == pytest.approx(90.0)
        assert "gaze_x" not in set(result.rows.feature)

    def test_labelled_frames_name_a_thing_and_no_room_and_no_labels_warn(self, run_angles):
        labels = [(t / FPS, "child", "bottle") for t in range(3)]
        result = run_angles({"parent": [0.0] * 10, "child": [40.0] * 10}, room={}, labels=labels)
        child = result.annotations[result.annotations.subject_id == "child"]
        assert child.label.tolist() == ["bottle"]
        assert any(w.subject_id == "parent" and "neither the room laid out in the setup" in w.message for w in result.warnings)

    def test_a_thing_the_room_places_is_looked_at_without_labelled_frames(self, run_angles):
        # half a metre nearer the camera than the people, between them: 63 degrees to each one's side
        room = {**self.ROOM, "things": {"toy": {"at": [0, 50], "height": 100}}}
        result = run_angles({"parent": [-63.4] * 10, "child": [63.4] * 10}, room=room)
        looks = result.annotations[result.annotations.tier == "gaze_target"]
        assert looks.groupby("subject_id")["label"].agg(list).to_dict() == {"child": ["toy"], "parent": ["toy"]}
        joint = result.annotations[result.annotations.tier == "joint_attention_with_parent"]
        assert joint.subject_id.tolist() == ["child"] and joint.label.tolist() == ["toy"]

    def test_a_camera_turned_and_tilted_puts_the_people_where_it_sees_them(self, run_angles):
        """The same people seen by a camera at the parent's right shoulder, looking down at the child."""
        room = {
            "cameras": {"room.mp4": {"at": [0, 0], "height": 200, "facing": 45, "tilt": 30}},
            "people": {"parent": {"at": [0, 100], "eyes": 100}, "child": {"at": [100, 0], "eyes": 100}},
        }
        expected = angles.expected_directions("parent", ["child"], room_seats(RoomSection(**room), "room.mp4"), pd.DataFrame())
        yaw = expected["child"][0]
        result = run_angles({"parent": [yaw] * 10, "child": [0.0] * 10}, room=room)
        assert result.annotations[result.annotations.subject_id == "parent"].label.tolist() == ["child"]
        assert abs(yaw) > 30

    def test_the_room_without_labelled_frames_can_be_keyed(self, tmp_path, monkeypatch):
        from macaw.store.cache import hash_inputs

        labels = tmp_path / "gaze_labels.json"
        monkeypatch.setattr(target_module, "_keypoints_file", lambda config, rid: tmp_path / "pose.parquet")
        monkeypatch.setattr(target_module, "_face_rows_file", lambda config, rid: tmp_path / "face.parquet")
        monkeypatch.setattr(target_module, "layout_for", lambda config, create=False: SimpleNamespace(gaze_labels_path=lambda rid: labels))
        for name in ("pose.parquet", "face.parquet"):
            (tmp_path / name).write_bytes(b"rows")
        step = GazeTargetExtractor(SimpleNamespace(gaze=GazeSection(backend="angles", tolerance_deg=20), setup=SimpleNamespace(room=RoomSection(**self.ROOM))))
        before = hash_inputs(step.inputs_for(SimpleNamespace(recording_id="r1")))
        labels.write_text("[]", encoding="utf-8")
        assert hash_inputs(step.inputs_for(SimpleNamespace(recording_id="r1"))) != before

    def test_the_angles_backend_needs_a_tolerance(self):
        from macaw.exceptions import ConfigError

        with pytest.raises(ConfigError, match="tolerance_deg"):
            GazeTargetExtractor(SimpleNamespace(gaze=GazeSection(backend="angles"))).check_available()


def test_people_measured_on_cameras_of_their_own_are_each_seen_from_theirs(tmp_path, monkeypatch):
    """Each camera faces its person across the table; the other person is on that camera's left in both."""
    from macaw.store.longtable import build_long_rows, write_long_table

    stored = tmp_path / "pose.keypoints.parquet"
    write_keypoints(pd.concat([keypoints({"parent": (100, 100)}, video="parent.mp4"), keypoints({"child": (100, 100)}, video="child.mp4")]), stored)
    rows = pd.concat([
        build_long_rows(values=pd.DataFrame({"t_start": np.arange(10) / FPS, "face_yaw_deg": yaw, "face_pitch_deg": 0.0}),
                        recording_id="r1", source="face.au", scale="frame", subject_id=subject)
        for subject, yaw in (("parent", 90.0), ("child", 90.0))
    ])
    face = tmp_path / "face.au.parquet"
    write_long_table(rows, face)
    monkeypatch.setattr(target_module, "_keypoints_file", lambda config, rid: stored)
    monkeypatch.setattr(target_module, "_face_rows_file", lambda config, rid: face)
    monkeypatch.setattr(target_module, "layout_for", lambda config, create=False: SimpleNamespace(gaze_labels_path=lambda rid: tmp_path / "none.json"))
    monkeypatch.setattr(target_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=0.9))
    recording = recording_of([video("parent.mp4", "parent"), video("child.mp4", "child")], recording_id="r1", group_id="1", session_id=None)
    room = RoomSection(
        cameras={"parent.mp4": {"at": [0, 200], "height": 100, "facing": 180}, "child.mp4": {"at": [100, 0], "height": 100, "facing": 0}},
        people={"parent": {"at": [0, 100], "eyes": 100}, "child": {"at": [100, 100], "eyes": 100}},
    )
    config = SimpleNamespace(gaze=GazeSection(backend="angles", tolerance_deg=20), setup=SimpleNamespace(room=room))
    result = GazeTargetExtractor(config).run(recording, tmp_path)
    assert {who: set(group.label) for who, group in result.annotations.groupby("subject_id")} == {"child": {"parent"}, "parent": {"child"}}
