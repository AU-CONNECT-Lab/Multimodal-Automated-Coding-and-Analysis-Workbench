from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.extractors.video.gaze.angles import (
    angle_between,
    directions,
    expected_directions,
    labelled_directions,
    name_by_angles,
    seat_direction,
)

SEATS = {"camera": [0, 0, 110], "mother": [-50, 100, 115], "child": [50, 100, 85]}


class TestSeatDirection:
    def test_a_person_looking_into_the_camera_has_level_zero_angles(self):
        assert seat_direction({"camera": [0, 0, 110], "p": [0, 150, 110]}, "p", "camera") == pytest.approx((0.0, 0.0))

    def test_turning_to_someone_on_the_camera_s_right_is_turning_to_one_s_own_left(self):
        yaw, pitch = seat_direction(SEATS, "mother", "child")
        assert yaw == pytest.approx(-90.0)
        assert pitch == pytest.approx(-np.degrees(np.arctan2(30, 100)))

    def test_the_two_look_along_opposite_lines(self):
        there, back = seat_direction(SEATS, "mother", "child"), seat_direction(SEATS, "child", "mother")
        assert there[0] == pytest.approx(-back[0]) and there[1] == pytest.approx(-back[1])


def test_angle_between_counts_yaw_and_pitch_together():
    assert angle_between(0, 0, 30, 0) == pytest.approx(30.0)
    assert angle_between(0, 0, 0, -20) == pytest.approx(20.0)
    assert angle_between(10, 5, 10, 5) == pytest.approx(0.0, abs=1e-6)


def test_eyes_are_used_where_read_and_the_head_elsewhere():
    rows = pd.DataFrame(
        [("child", 0.0, "face_yaw_deg", 10.0), ("child", 0.0, "face_pitch_deg", 1.0), ("child", 0.0, "gaze_yaw_deg", 20.0),
         ("child", 0.0, "gaze_pitch_deg", 2.0), ("child", 0.1, "face_yaw_deg", 11.0), ("child", 0.1, "face_pitch_deg", 1.5)],
        columns=["subject_id", "t_start", "feature", "value"],
    )
    found = directions(rows)
    assert found[["yaw", "pitch", "source"]].values.tolist() == [[20.0, 2.0, "eyes"], [11.0, 1.5, "head"]]


def measured(subject, yaws, pitch=0.0, fps=10.0):
    return pd.DataFrame({"subject_id": subject, "t": np.arange(len(yaws)) / fps, "yaw": yaws, "pitch": pitch})


class TestExpected:
    def test_seats_alone_give_the_direction(self):
        expected = expected_directions("mother", ["child"], SEATS, pd.DataFrame(columns=["subject", "target", "yaw", "pitch"]))
        assert expected["child"] == pytest.approx(seat_direction(SEATS, "mother", "child"))

    def test_labels_move_the_seats_by_the_person_s_own_offset(self):
        labels = pd.DataFrame({"subject": ["mother"], "target": ["camera"], "yaw": [5.0], "pitch": [-3.0]})
        seat_yaw, seat_pitch = seat_direction(SEATS, "mother", "camera")
        expected = expected_directions("mother", ["child"], SEATS, labels)
        child_yaw, child_pitch = seat_direction(SEATS, "mother", "child")
        assert expected["child"] == pytest.approx((child_yaw + 5.0 - seat_yaw, child_pitch - 3.0 - seat_pitch))

    def test_enough_labels_name_a_target_no_seat_holds(self):
        labels = pd.DataFrame({"subject": ["child"] * 3, "target": ["bottle"] * 3, "yaw": [30.0, 32.0, 40.0], "pitch": [-30.0, -28.0, -25.0]})
        assert expected_directions("child", ["bottle"], {}, labels) == {"bottle": (32.0, -28.0)}

    def test_too_few_labels_and_no_seat_leave_a_target_out(self):
        labels = pd.DataFrame({"subject": ["child"] * 2, "target": ["bottle"] * 2, "yaw": [30.0, 32.0], "pitch": [-30.0, -28.0]})
        assert expected_directions("child", ["bottle"], {}, labels) == {}


def test_labels_take_the_direction_of_the_nearest_frame_within_half_a_frame():
    labels = pd.DataFrame({"t": [0.21, 0.5], "subject": ["child", "child"], "target": ["mother", "mother"]})
    found = labelled_directions(measured("child", [0, 10, 20, 30]), labels, period_s=0.1)
    assert found["yaw"].tolist() == [20.0]


def test_the_nearest_expected_direction_within_tolerance_is_the_target():
    named = name_by_angles(measured("child", [0.0, 85.0, 50.0]), {"mother": (90.0, 0.0), "camera": (0.0, 0.0)}, 15.0, "elsewhere")
    assert named["target"].tolist() == ["camera", "mother", "elsewhere"]
    assert named["off_mother"].tolist() == pytest.approx([90.0, 5.0, 40.0])
