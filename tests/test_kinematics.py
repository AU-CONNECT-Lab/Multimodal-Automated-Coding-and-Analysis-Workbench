from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw import project_scripts
from macaw.config.schema import KinematicsSection, MeasureSection
from macaw.exceptions import ConfigError, SchemaViolationError
from macaw.extractors.video.pose import kinematics as kinematics_module
from macaw.extractors.video.pose.kinematics import (
    KinematicsExtractor,
    acceleration,
    envelope_overlap,
    hand_distance,
    head_orientation,
    interpersonal_distance,
    jerk,
    measure_values,
    postural_openness,
    velocity,
)
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.store.keypoints import write_keypoints, write_no_keypoints
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, empty_frame, validate_frame

FPS = 30.0

# upright and facing the camera, so the person's left is the image's right
TEMPLATE = {
    "nose": (0.0, -60.0),
    "left_eye": (8.0, -68.0),
    "right_eye": (-8.0, -68.0),
    "left_ear": (16.0, -64.0),
    "right_ear": (-16.0, -64.0),
    "left_shoulder": (40.0, 0.0),
    "right_shoulder": (-40.0, 0.0),
    "left_elbow": (50.0, 60.0),
    "right_elbow": (-50.0, 60.0),
    "left_wrist": (55.0, 120.0),
    "right_wrist": (-55.0, 120.0),
    "left_hip": (25.0, 150.0),
    "right_hip": (-25.0, 150.0),
    "left_knee": (25.0, 230.0),
    "right_knee": (-25.0, 230.0),
    "left_ankle": (25.0, 310.0),
    "right_ankle": (-25.0, 310.0),
}


def track(xy, *, joint="left_wrist", fps=FPS, frames=None):
    """One joint's path; a NaN row is a frame where the joint was seen but not usable."""
    xy = np.asarray(xy, dtype=float)
    frames = np.arange(len(xy)) if frames is None else np.asarray(frames)
    return pd.DataFrame(
        {"frame": frames, "t": frames / fps, "joint": joint, "x": xy[:, 0], "y": xy[:, 1]}
    )


def person(subject_id, *, centre=(300.0, 200.0), n_frames=20, fps=FPS, scale=1.0, moves=None):
    """A full KEYPOINT_SCHEMA person; moves maps a joint to an (n_frames, 2) path override."""
    rows = []
    for frame in range(n_frames):
        for joint in STANDARD_JOINTS:
            dx, dy = TEMPLATE[joint]
            x, y = centre[0] + scale * dx, centre[1] + scale * dy
            if moves and joint in moves:
                x, y = moves[joint][frame]
            rows.append(
                {
                    "recording_id": "sub-003_ses-12mo",
                    "subject_id": subject_id,
                    "source": "pose.keypoints",
                    "video": f"{subject_id}.mp4",
                    "frame": frame,
                    "t": frame / fps,
                    "joint": joint,
                    "x": x,
                    "y": y,
                    "z": float("nan"),
                    "confidence": 0.9,
                    "track_id": 0,
                    "units": "pixel",
                    "metric_scale_valid": None,
                }
            )
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def times(n, fps=FPS):
    return np.arange(n) / fps


class TestDerivatives:
    def test_constant_velocity_gives_its_speed(self):
        t = times(30)
        speed = velocity(track(np.column_stack([30.0 * t, 40.0 * t])), "left_wrist")
        assert np.allclose(speed, 50.0)

    def test_constant_acceleration_gives_its_magnitude(self):
        t = times(30)
        path = track(np.column_stack([10.0 * t**2, np.zeros_like(t)]))
        assert np.allclose(acceleration(path, "left_wrist"), 20.0)
        assert np.allclose(velocity(path, "left_wrist"), 20.0 * t)

    def test_a_cubic_has_constant_jerk(self):
        t = times(30)
        assert np.allclose(jerk(track(np.column_stack([t**3, np.zeros_like(t)])), "left_wrist"), 6.0)

    def test_a_still_joint_does_not_move(self):
        still = track(np.tile([100.0, 100.0], (20, 1)))
        for derivative in (velocity, acceleration, jerk):
            assert np.allclose(derivative(still, "left_wrist"), 0.0)

    def test_time_comes_from_t_not_from_the_frame_count(self):
        """The same displacement per frame is twice the speed at twice the frame rate."""
        path = np.column_stack([np.arange(20.0), np.zeros(20)])
        slow = velocity(track(path, fps=10.0), "left_wrist")
        fast = velocity(track(path, fps=20.0), "left_wrist")
        assert np.allclose(slow, 10.0)
        assert np.allclose(fast, 20.0)

    def test_no_fit_spans_missing_frames(self):
        """A person who reappears somewhere else did not move there at infinite speed."""
        path = np.vstack([np.tile([0.0, 0.0], (10, 1)), np.tile([500.0, 0.0], (10, 1))])
        frames = np.concatenate([np.arange(10), np.arange(15, 25)])
        assert np.allclose(velocity(track(path, frames=frames), "left_wrist"), 0.0)

    def test_an_unusable_frame_breaks_the_run_too(self):
        path = np.vstack([np.tile([0.0, 0.0], (10, 1)), [[np.nan, np.nan]], np.tile([500.0, 0.0], (10, 1))])
        speed = velocity(track(path), "left_wrist")
        assert np.isnan(speed.iloc[10])
        assert np.allclose(speed.drop(speed.index[10]), 0.0)

    def test_a_run_shorter_than_the_window_is_nan_not_a_guess(self):
        t = times(4)
        speed = velocity(track(np.column_stack([30.0 * t, np.zeros_like(t)])), "left_wrist")
        assert speed.isna().all()

    def test_the_window_is_never_shorter_than_the_cubic_needs(self):
        """A tiny smoothing_s still gets five frames, so jerk is fitted rather than undefined."""
        t = times(5)
        path = track(np.column_stack([t**3, np.zeros_like(t)]))
        assert np.allclose(jerk(path, "left_wrist", smoothing_s=0.001), 6.0)

    def test_the_result_is_indexed_by_frame(self):
        speed = velocity(track(np.zeros((10, 2)), frames=np.arange(100, 110)), "left_wrist")
        assert list(speed.index) == list(range(100, 110))


class TestHandDistance:
    def test_is_the_distance_between_the_wrists(self):
        kp = pd.concat(
            [
                track([[0.0, 0.0]], joint="left_wrist"),
                track([[3.0, 4.0]], joint="right_wrist"),
            ]
        )
        assert hand_distance(kp).tolist() == [5.0]

    def test_a_missing_wrist_is_nan(self):
        kp = pd.concat(
            [track([[0.0, 0.0]], joint="left_wrist"), track([[np.nan, np.nan]], joint="right_wrist")]
        )
        assert hand_distance(kp).isna().all()


class TestHeadOrientation:
    def head(self, nose, left_eye=(8.0, -68.0), right_eye=(-8.0, -68.0)):
        return pd.concat(
            [
                track([nose], joint="nose"),
                track([left_eye], joint="left_eye"),
                track([right_eye], joint="right_eye"),
            ]
        )

    def test_a_centred_nose_faces_the_camera(self):
        head = head_orientation(self.head((0.0, -60.0)))
        assert head["yaw_index"].iloc[0] == pytest.approx(0.0)
        assert head["roll_deg"].iloc[0] == pytest.approx(0.0)

    def test_turning_to_the_own_left_is_positive(self):
        """Facing the camera the left eye is on the image right, and the nose moves toward it."""
        assert head_orientation(self.head((5.0, -60.0)))["yaw_index"].iloc[0] > 0
        assert head_orientation(self.head((-5.0, -60.0)))["yaw_index"].iloc[0] < 0

    def test_the_yaw_index_stays_within_one(self):
        """The nose on top of an eye is the extreme, not a blow-up."""
        assert head_orientation(self.head((8.0, -68.0)))["yaw_index"].iloc[0] == pytest.approx(1.0)

    def test_roll_is_the_eye_line_angle_clockwise(self):
        angle = np.radians(30.0)
        left = (10.0 * np.cos(angle), 10.0 * np.sin(angle))
        right = (-left[0], -left[1])
        roll = head_orientation(self.head((0.0, 5.0), left, right))["roll_deg"].iloc[0]
        assert roll == pytest.approx(30.0)

    def test_a_person_facing_away_is_not_at_180_degrees(self):
        """The eye line has no direction; swapping which eye is on which side is not a half turn."""
        roll = head_orientation(self.head((0.0, -60.0), (-8.0, -68.0), (8.0, -68.0)))["roll_deg"]
        assert roll.iloc[0] == pytest.approx(0.0)

    def test_a_missing_nose_loses_yaw_but_keeps_roll(self):
        head = head_orientation(self.head((np.nan, np.nan)))
        assert np.isnan(head["yaw_index"].iloc[0])
        assert head["roll_deg"].iloc[0] == pytest.approx(0.0)


class TestPairs:
    def test_distance_is_between_shoulder_midpoints(self):
        distance = interpersonal_distance(person("a", centre=(100.0, 200.0)), person("b", centre=(400.0, 200.0)))
        assert np.allclose(distance, 300.0)

    def test_only_frames_where_both_are_seen_count(self):
        a = person("a", n_frames=10)
        b = person("b", n_frames=10)
        b = b[b.frame >= 4]
        assert list(interpersonal_distance(a, b).index) == list(range(4, 10))
        assert list(envelope_overlap(a, b).index) == list(range(4, 10))

    def test_people_apart_do_not_overlap(self):
        overlap = envelope_overlap(person("a", centre=(100.0, 200.0)), person("b", centre=(600.0, 200.0)))
        assert np.allclose(overlap, 0.0)

    def test_a_box_inside_another_overlaps_fully(self):
        """Over the smaller box, so a child in front of a parent reads as full overlap."""
        overlap = envelope_overlap(person("a", scale=1.0), person("b", scale=0.5))
        assert np.allclose(overlap, 1.0)

    def test_half_overlap(self):
        """Each body box is 110 wide; shifting by 55 leaves half of it shared."""
        overlap = envelope_overlap(person("a", centre=(300.0, 200.0)), person("b", centre=(355.0, 200.0)))
        assert np.allclose(overlap, 0.5)


class TestPosturalOpenness:
    def test_spread_arms_are_more_open_than_arms_at_the_sides(self):
        n = 10
        spread = {
            "left_elbow": np.tile([300.0 + 90.0, 200.0], (n, 1)),
            "right_elbow": np.tile([300.0 - 90.0, 200.0], (n, 1)),
            "left_wrist": np.tile([300.0 + 140.0, 200.0 - 20.0], (n, 1)),
            "right_wrist": np.tile([300.0 - 140.0, 200.0 - 20.0], (n, 1)),
        }
        closed = postural_openness(person("a", n_frames=n))
        opened = postural_openness(person("a", n_frames=n, moves=spread))
        assert (opened > closed).all()

    def test_it_does_not_depend_on_how_far_away_the_person_is(self):
        near = postural_openness(person("a", scale=2.0))
        far = postural_openness(person("a", scale=1.0))
        assert np.allclose(near, far)

    def test_a_frame_turned_sideways_keeps_the_same_yardstick(self):
        """Shoulders narrowed by a turn must not inflate openness; the size is the upper
        envelope of shoulder width over the recording, not this frame's width."""
        n = 20
        narrowed = {
            "left_shoulder": np.array([[340.0, 200.0]] * 19 + [[310.0, 200.0]]),
            "right_shoulder": np.array([[260.0, 200.0]] * 19 + [[290.0, 200.0]]),
        }
        baseline = postural_openness(person("a", n_frames=n))
        turned = postural_openness(person("a", n_frames=n, moves=narrowed))
        assert np.allclose(turned, baseline)

    def test_arms_inside_the_shoulders_read_as_one(self):
        """Crossed arms reach no further than the shoulders, whatever the shoulder width."""
        n = 5
        crossed = {
            "left_elbow": np.tile([290.0, 260.0], (n, 1)),
            "right_elbow": np.tile([310.0, 260.0], (n, 1)),
            "left_wrist": np.tile([270.0, 250.0], (n, 1)),
            "right_wrist": np.tile([330.0, 250.0], (n, 1)),
        }
        assert np.allclose(postural_openness(person("a", n_frames=n, moves=crossed)), 1.0)

    def test_a_missing_arm_joint_is_nan(self):
        n = 5
        gone = {"left_wrist": np.full((n, 2), np.nan)}
        assert postural_openness(person("a", n_frames=n, moves=gone)).isna().all()


def video(name, subject_id=None):
    return SimpleNamespace(path=Path(name), kind="video", subject_id=subject_id)


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(keypoints, files, subjects=("parent", "child")):
        stored = tmp_path / "keypoints"
        if keypoints.empty:
            write_no_keypoints(stored, "pose.keypoints")
        else:
            write_keypoints(keypoints, stored)
        monkeypatch.setattr(
            kinematics_module, "_keypoints_file", lambda config, rid: stored / "pose.keypoints.parquet"
        )
        recording = SimpleNamespace(
            recording_id="sub-003_ses-12mo",
            subjects=[SimpleNamespace(subject_id=name) for name in subjects],
            media=list(files),
            entities=lambda: {"group_id": "003", "session_id": "12mo", "task": None, "run": None},
        )
        extractor = KinematicsExtractor(SimpleNamespace(kinematics=KinematicsSection()))
        return extractor.run(recording, tmp_path)

    return build


@pytest.fixture
def shared(run):
    keypoints = pd.concat(
        [person("parent", centre=(150.0, 200.0)), person("child", centre=(450.0, 220.0))]
    ).assign(video="room.mp4")
    return run(keypoints, [video("room.mp4")])


class TestExtractor:
    def test_rows_satisfy_the_long_table_schema(self, shared):
        validate_frame(shared.rows, LONG_TABLE_SCHEMA, "long table")

    def test_manifest_and_rows_agree(self, shared):
        assert {spec.name for spec in shared.specs} == set(shared.rows.feature)

    def test_every_person_gets_frame_rows(self, shared):
        rows = shared.rows[shared.rows.subject_id_b.isna()]
        assert set(rows.subject_id) == {"parent", "child"}
        assert (rows.scale == "frame").all()
        assert rows.groupby(["subject_id", "feature"]).size().eq(20).all()

    def test_frame_rows_sit_on_the_recording_clock(self, shared):
        rows = shared.rows
        assert rows.t_start.max() == pytest.approx(19 / FPS)
        assert (rows.t_start == rows.t_end).all()

    def test_a_pair_is_stored_once_in_subject_order(self, shared):
        pairs = shared.rows[shared.rows.subject_id_b.notna()]
        assert set(zip(pairs.subject_id, pairs.subject_id_b)) == {("parent", "child")}
        assert set(pairs.feature) == {"interpersonal_distance", "envelope_overlap"}

    def test_lengths_carry_pixel_units(self, shared):
        units = {spec.name: spec.unit for spec in shared.specs}
        assert units["interpersonal_distance"] == "px"
        assert units["left_wrist_speed"] == "px/s"
        assert units["left_wrist_jerk"] == "px/s^3"
        assert units["head_roll_deg"] == "deg"
        assert units["postural_openness"] is None

    def test_the_smoothing_window_is_recorded(self, shared):
        assert all(spec.params["smoothing_s"] == 0.3 for spec in shared.specs)

    def test_per_subject_cameras_get_no_pairwise_rows(self, run):
        keypoints = pd.concat([person("parent"), person("child")])
        result = run(keypoints, [video("parent.mp4", "parent"), video("child.mp4", "child")])
        assert result.rows.subject_id_b.isna().all()
        assert "interpersonal_distance" not in {spec.name for spec in result.specs}
        assert any("own camera" in warning.message for warning in result.warnings)

    def test_a_shared_camera_next_to_own_cameras_gives_the_pairs_and_only_the_pairs(self, run):
        in_room = [person(name, centre=centre, scale=0.2).assign(video="room.mp4")
                   for name, centre in (("parent", (100.0, 200.0)), ("child", (300.0, 200.0)))]
        keypoints = pd.concat([person("parent"), person("child"), *in_room])
        files = [video("parent.mp4", "parent"), video("child.mp4", "child"), video("room.mp4")]
        rows = run(keypoints, files).rows
        assert set(rows.subject_id_b.dropna()) == {"child"}
        alone = run(pd.concat([person("parent"), person("child")]), files[:2]).rows
        mine = rows[rows.subject_id_b.isna()].reset_index(drop=True)
        pd.testing.assert_series_equal(mine.value, alone.value.reset_index(drop=True))

    def test_two_room_cameras_measure_each_person_where_they_are_seen_most(self, run):
        left = pd.concat([person("parent", n_frames=20), person("child", centre=(450.0, 220.0), n_frames=5)]).assign(video="left.mp4")
        right = pd.concat([person("parent", n_frames=8), person("child", centre=(450.0, 220.0), n_frames=20, scale=2.0)]).assign(video="right.mp4")
        rows = run(pd.concat([left, right]), [video("left.mp4"), video("right.mp4")]).rows
        hands = rows[rows.feature == "hand_distance"].groupby("subject_id").size()
        assert hands.to_dict() == {"child": 20, "parent": 20}
        pairs = rows[rows.feature == "interpersonal_distance"]
        assert len(pairs) == 8

    def test_a_subject_without_keypoints_is_reported(self, run):
        result = run(person("parent"), [video("room.mp4")])
        assert set(result.rows.subject_id) == {"parent"}
        assert any(w.subject_id == "child" and w.severity == "warn" for w in result.warnings)

    def test_a_pose_run_that_found_nobody_reports_everyone(self, run):
        result = run(empty_frame(KEYPOINT_SCHEMA), [video("room.mp4")])
        assert result.rows.empty
        assert result.specs == []
        assert {w.subject_id for w in result.warnings if w.severity == "warn"} == {"parent", "child"}

    def test_unassigned_tracks_are_ignored(self, run):
        stray = person("parent").assign(subject_id=pd.NA)
        result = run(pd.concat([person("parent"), person("child"), stray]), [video("room.mp4")])
        assert set(result.rows.subject_id.dropna()) == {"parent", "child"}

    def test_the_cache_key_is_the_keypoints_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(kinematics_module, "_keypoints_file", lambda config, rid: tmp_path / rid)
        extractor = KinematicsExtractor(SimpleNamespace(kinematics=KinematicsSection()))
        assert extractor.inputs_for(SimpleNamespace(recording_id="r1")) == [tmp_path / "r1"]
        assert extractor.config_sections == ("kinematics",)


def measure(**fields):
    return MeasureSection(**{"name": "m", "person": "parent", "to_person": "child", **fields})


# the parent sits left of the child; the parent's left wrist is the one nearer the child
PARENT = (150.0, 200.0)
CHILD = (450.0, 220.0)


class TestMeasureValues:
    def test_distance_is_between_the_nearest_pair(self):
        values = measure_values(
            measure(joints=["left_wrist", "right_wrist"], to_joints=["nose"]), person("parent", centre=PARENT), person("child", centre=CHILD)
        )
        assert len(values) == 20
        assert values.iloc[0] == pytest.approx(np.hypot(450.0 - 205.0, 160.0 - 320.0))

    def test_a_centre_is_a_midpoint(self):
        values = measure_values(
            measure(joints=["shoulder_center"], to_joints=["hip_center"]), person("parent", centre=PARENT), person("child", centre=CHILD)
        )
        assert values.iloc[0] == pytest.approx(np.hypot(300.0, 170.0))

    def test_per_body_size_divides_by_the_persons_shoulder_width(self):
        a, b = person("parent", centre=PARENT, scale=2.0), person("child", centre=CHILD)
        raw = measure_values(measure(joints=["nose"], to_joints=["nose"]), a, b)
        scaled = measure_values(measure(joints=["nose"], to_joints=["nose"], per_body_size=True), a, b)
        assert scaled.iloc[0] == pytest.approx(raw.iloc[0] / 160.0)

    def test_approach_speed_is_positive_while_closing_in(self):
        path = np.column_stack([205.0 + 3.0 * np.arange(20), np.full(20, 160.0)])
        a = person("parent", centre=PARENT, moves={"left_wrist": path})
        values = measure_values(measure(joints=["left_wrist"], to_joints=["nose"], method="approach_speed"), a, person("child", centre=CHILD))
        assert values.to_numpy() == pytest.approx(np.full(20, 3.0 * FPS))

    def test_offset_toward_is_signed_by_where_the_other_is(self):
        toward = np.tile([CHILD[0] - 10.0, CHILD[1] - 60.0], (20, 1))
        away = np.tile([CHILD[0] + 10.0, CHILD[1] - 60.0], (20, 1))
        head = measure(person="child", to_person="parent", joints=["nose"], to_joints=["nose"], method="offset_toward")
        assert measure_values(head, person("child", centre=CHILD, moves={"nose": toward}), person("parent", centre=PARENT)).iloc[0] == pytest.approx(10.0)
        assert measure_values(head, person("child", centre=CHILD, moves={"nose": away}), person("parent", centre=PARENT)).iloc[0] == pytest.approx(-10.0)

    def test_only_frames_with_both_count(self):
        values = measure_values(measure(joints=["nose"], to_joints=["nose"]), person("parent", n_frames=20), person("child", n_frames=8))
        assert list(values.index) == list(range(8))

    def test_a_script_function_gets_both_skeletons_row_by_row(self):
        def wrist_height_gap(a, b):
            assert list(a.columns[:3]) == ["t", "nose_x", "nose_y"]
            return a["left_wrist_y"] - b["left_wrist_y"]

        values = measure_values(measure(method="lab.gap"), person("parent", centre=PARENT), person("child", centre=CHILD), function=wrist_height_gap)
        assert values.iloc[0] == pytest.approx(-20.0)

    def test_a_script_returning_the_wrong_length_is_refused(self):
        with pytest.raises(SchemaViolationError, match="returned 1 values for 20 rows"):
            measure_values(measure(method="lab.one"), person("parent"), person("child"), function=lambda a, b: [1.0])


def measure_config(measures, scripts_dir=Path("scripts"), roles=("parent", "child")):
    return SimpleNamespace(
        kinematics=KinematicsSection(measures=measures),
        project=SimpleNamespace(roles=list(roles), scripts_dir=scripts_dir),
    )


class TestMeasureChecks:
    @pytest.mark.parametrize(
        "fields, message",
        [
            ({"name": "hand_distance", "joints": ["nose"], "to_joints": ["nose"]}, "already has a feature"),
            ({"person": "teacher", "joints": ["nose"], "to_joints": ["nose"]}, "name a role"),
            ({"to_person": "teacher", "joints": ["nose"], "to_joints": ["nose"]}, "name a role"),
            ({"joints": ["nose"]}, "needs joints and to_joints"),
            ({"joints": ["thumb"], "to_joints": ["nose"]}, "no joint called thumb"),
            ({"joints": ["nose", "left_eye"], "to_joints": ["nose"], "method": "offset_toward"}, "one joint and one to_joint"),
            ({"method": "angle"}, "or file.function"),
            ({"method": "missing.angle"}, "there is no"),
        ],
    )
    def test_a_bad_measure_is_refused_before_running(self, tmp_path, fields, message):
        extractor = KinematicsExtractor(measure_config([measure(**fields)], scripts_dir=tmp_path))
        with pytest.raises(ConfigError, match=message):
            extractor.check_available()

    def test_self_is_a_target(self):
        KinematicsExtractor(measure_config([measure(to_person="self", joints=["left_wrist"], to_joints=["nose"])])).check_available()

    def test_two_measures_cannot_share_a_name(self):
        with pytest.raises(ValueError, match="two measures are called m"):
            KinematicsSection(measures=[measure(), measure()])


@pytest.fixture
def measured(tmp_path, monkeypatch):
    def build(measures, keypoints=None, scripts_dir=None):
        stored = tmp_path / "keypoints"
        if keypoints is None:
            keypoints = pd.concat([person("p1", centre=PARENT), person("c1", centre=CHILD)]).assign(video="room.mp4")
        write_keypoints(keypoints, stored)
        monkeypatch.setattr(kinematics_module, "_keypoints_file", lambda config, rid: stored / "pose.keypoints.parquet")
        recording = SimpleNamespace(
            recording_id="sub-003_ses-12mo",
            subjects=[SimpleNamespace(subject_id="p1", role="parent"), SimpleNamespace(subject_id="c1", role="child")],
            media=[video("room.mp4")],
            entities=lambda: {"group_id": "003", "session_id": "12mo", "task": None, "run": None},
        )
        extractor = KinematicsExtractor(measure_config(measures, scripts_dir=scripts_dir or tmp_path))
        extractor.check_available()
        return extractor.run(recording, tmp_path)

    return build


class TestMeasuresInTheRun:
    def test_a_measure_between_people_is_a_directed_pair(self, measured):
        result = measured([measure(name="hand_to_head", joints=["left_wrist", "right_wrist"], to_joints=["nose"])])
        rows = result.rows[result.rows.feature == "hand_to_head"]
        assert set(zip(rows.subject_id, rows.subject_id_b)) == {("p1", "c1")}
        assert len(rows) == 20
        spec = next(spec for spec in result.specs if spec.name == "hand_to_head")
        assert spec.unit == "px"
        assert spec.params["joints"] == ["left_wrist", "right_wrist"]
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")

    def test_a_measure_within_a_person_has_no_second_subject(self, measured):
        result = measured([measure(name="hand_to_face", person="child", to_person="self", joints=["left_wrist"], to_joints=["nose"])])
        rows = result.rows[result.rows.feature == "hand_to_face"]
        assert set(rows.subject_id) == {"c1"}
        assert rows.subject_id_b.isna().all()
        assert rows.value.iloc[0] == pytest.approx(np.hypot(55.0, 180.0))

    def test_people_on_separate_cameras_are_not_measured_against_each_other(self, measured):
        keypoints = pd.concat([person("p1", centre=PARENT), person("c1", centre=CHILD)])
        result = measured([measure(name="apart", joints=["nose"], to_joints=["nose"])], keypoints=keypoints)
        assert "apart" not in set(result.rows.feature)
        assert any("never on one camera together" in warning.message for warning in result.warnings)

    def test_a_script_measure_runs_from_the_scripts_folder(self, measured, tmp_path, monkeypatch):
        monkeypatch.setattr(project_scripts, "_loaded", {})
        monkeypatch.setattr(project_scripts, "_failed", {})
        (tmp_path / "lab.py").write_text("def gap(a, b):\n    return a['nose_x'] - b['nose_x']\n", encoding="utf-8")
        result = measured([measure(name="nose_gap", method="lab.gap")], scripts_dir=tmp_path)
        rows = result.rows[result.rows.feature == "nose_gap"]
        assert rows.value.iloc[0] == pytest.approx(-300.0)
        extractor = KinematicsExtractor(measure_config([measure(name="nose_gap", method="lab.gap")], scripts_dir=tmp_path))
        assert tmp_path / "lab.py" in extractor.inputs_for(SimpleNamespace(recording_id="r1"))
