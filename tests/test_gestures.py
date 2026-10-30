from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import GestureSection, KinematicsSection
from macaw.extractors.video.pose import gestures as gestures_module
from macaw.extractors.video.pose import kinematics as kinematics_module
from macaw.extractors.video.pose.gestures import (
    GestureExtractor,
    gesture_kinematics,
    gesture_spaces,
    height_levels,
    holds,
    submovement_peaks,
)
from macaw.store.annotations import write_annotations
from macaw.store.keypoints import write_keypoints
from macaw.store.schema import ANNOTATION_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, validate_frame

from test_kinematics import TEMPLATE, person

FPS = 30.0


def hump(t, start, length, height):
    """A speed profile rising from rest to height and back, as one stroke of a hand."""
    inside = (t >= start) & (t <= start + length)
    return np.where(inside, height * np.sin(np.pi * (t - start) / length), 0.0)


def body(n=1):
    """TEMPLATE's landmarks, image y down, for n frames."""
    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose", "left_eye", "right_eye")
    columns = pd.MultiIndex.from_product([["x", "y"], joints])
    return pd.DataFrame([[TEMPLATE[j][0] for j in joints] + [TEMPLATE[j][1] for j in joints]] * n, columns=columns)


class TestSubmovements:
    def test_each_stroke_is_one_peak(self):
        t = np.arange(0, 3, 1 / FPS)
        speed = hump(t, 0.2, 0.8, 100.0) + hump(t, 1.5, 0.8, 60.0)
        assert submovement_peaks(t, speed, 20.0) == pytest.approx([0.6, 1.9], abs=1 / FPS)

    def test_a_ripple_smaller_than_the_prominence_is_no_submovement(self):
        t = np.arange(0, 2, 1 / FPS)
        speed = hump(t, 0.2, 1.4, 100.0) + 5.0 * np.sin(2 * np.pi * 6 * t)
        assert len(submovement_peaks(t, speed, 20.0)) == 1

    def test_a_tracking_gap_makes_no_peak(self):
        t = np.arange(0, 2, 1 / FPS)
        speed = np.full(t.size, 50.0)
        speed[25:35] = np.nan
        assert len(submovement_peaks(t, speed, 20.0)) == 0


class TestHolds:
    def test_a_hold_lies_between_the_first_and_last_stroke(self):
        t = np.arange(0, 4, 1 / FPS)
        wrist = hump(t, 0.5, 1.0, 100.0) + hump(t, 2.1, 1.0, 100.0)
        found = holds(t, wrist, np.zeros_like(t), 5.0, np.array([1.0, 2.6]))
        assert len(found) == 1
        start, end = found[0]
        assert start == pytest.approx(1.5, abs=0.05) and end == pytest.approx(2.1, abs=0.05)

    def test_a_moving_elbow_is_no_hold(self):
        t = np.arange(0, 4, 1 / FPS)
        wrist = hump(t, 0.5, 1.0, 100.0) + hump(t, 2.1, 1.0, 100.0)
        assert holds(t, wrist, np.full_like(t, 50.0), 5.0, np.array([1.0, 2.6])) == []

    def test_one_stroke_has_nothing_between(self):
        t = np.arange(0, 2, 1 / FPS)
        assert holds(t, np.zeros_like(t), np.zeros_like(t), 5.0, np.array([1.0])) == []


class TestOneGesture:
    def test_peaks_holds_rhythm_and_extent(self):
        t = np.arange(0, 4, 1 / FPS)
        speed = hump(t, 0.0, 0.6, 80.0) + hump(t, 1.0, 0.6, 120.0) + hump(t, 2.6, 0.6, 80.0)
        xy = np.column_stack([np.linspace(0, 30, t.size), np.linspace(0, 20, t.size)])
        measured = gesture_kinematics(t, speed, np.zeros_like(t), xy, 5.0, 20.0)
        assert measured["peak_speed"] == pytest.approx(120.0, rel=0.01)
        assert measured["submovements"] == 3.0 and measured["holds"] == 2.0
        # peaks at 0.3, 1.3 and 2.9 s: intervals of 1.0 and 1.6 s
        assert measured["rhythm_sd_s"] == pytest.approx(0.3, abs=1 / FPS)
        assert measured["extent"] == pytest.approx(600.0)

    def test_an_untracked_hand_gives_nothing(self):
        t = np.arange(0, 1, 1 / FPS)
        nan = np.full(t.size, np.nan)
        measured = gesture_kinematics(t, nan, nan, np.column_stack([nan, nan]), 5.0, 20.0)
        assert all(np.isnan(value) for value in measured.values())


class TestPlaces:
    def test_height_levels_climb_from_the_midline_to_above_the_head(self):
        # in TEMPLATE the midline is at y 75, mid-chest 37.5, shoulders 0, nose -60, top of head -84
        wrists = np.array([[0.0, y] for y in (100.0, 50.0, 10.0, -30.0, -70.0, -100.0)])
        assert height_levels(wrists, body(len(wrists))).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_gesture_spaces_widen_from_the_centre(self):
        wrists = np.array([[0.0, 75.0], [40.0, 75.0], [70.0, 75.0], [100.0, 75.0], [0.0, -100.0]])
        assert gesture_spaces(wrists, body(len(wrists))).tolist() == [1.0, 2.0, 3.0, 4.0, 4.0]

    def test_without_hips_neither_is_known(self):
        hidden = body(1)
        hidden[("y", "left_hip")] = np.nan
        wrist = np.array([[0.0, 50.0]])
        assert np.isnan(height_levels(wrist, hidden)).all() and np.isnan(gesture_spaces(wrist, hidden)).all()


def test_a_tier_needs_both_thresholds():
    with pytest.raises(ValueError, match="still_speed and peak_prominence"):
        GestureSection(tier="gesture", still_speed=10.0)
    assert GestureSection(tier="gesture", still_speed=10.0, peak_prominence=20.0).tier == "gesture"


def stroke_path(n):
    """The left wrist at mid-chest height moves right, holds, and moves back."""
    t = np.arange(n) / FPS
    rise = lambda start: np.clip((t - start) / 1.0, 0.0, 1.0)
    shift = 30.0 * ((1 - np.cos(np.pi * rise(0.5))) / 2 - (1 - np.cos(np.pi * rise(2.1))) / 2)
    return np.column_stack([300.0 + 10.0 + shift, np.full(n, 200.0 + 60.0)])


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(segments, *, annotator=None):
        stored = tmp_path / "keypoints"
        write_keypoints(person("parent", n_frames=105, moves={"left_wrist": stroke_path(105)}), stored)
        monkeypatch.setattr(kinematics_module, "_keypoints_file", lambda config, rid: stored / "pose.keypoints.parquet")
        coded = tmp_path / "annotations"
        rows = pd.DataFrame(
            [
                {"recording_id": "sub-003_ses-12mo", "subject_id": who, "tier": "gesture", "t_start": start, "t_end": end,
                 "label": "iconic", "annotator": by, "layer": "C", "abstained": False, "needs_review": False,
                 "group_id": "003", "session_id": None, "task": None, "run": None, "confidence": np.nan, "execution_id": None, "config_hash": None}
                for who, start, end, by in segments
            ]
        )
        write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), coded)
        monkeypatch.setattr(gestures_module, "layout_for", lambda config, create=False: SimpleNamespace(annotations_path=lambda rid: coded))
        gestures = GestureSection(tier="gesture", annotator=annotator, still_speed=10.0, peak_prominence=20.0)
        recording = SimpleNamespace(
            recording_id="sub-003_ses-12mo", subjects=[SimpleNamespace(subject_id="parent")], media=[],
            entities=lambda: {"group_id": "003", "session_id": "12mo", "task": None, "run": None},
        )
        return GestureExtractor(SimpleNamespace(kinematics=KinematicsSection(gestures=gestures))).run(recording, tmp_path)

    return build


def value(result, feature):
    rows = result.rows[result.rows.feature == feature]
    assert len(rows) == 1
    return float(rows.value.iloc[0])


class TestExtractor:
    def test_each_coded_gesture_is_one_event_per_measure_and_hand(self, run):
        result = run([("parent", 0.0, 3.4, "ann")])
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        assert set(result.rows.scale) == {"event"} and set(result.rows.t_end) == {3.4}
        assert value(result, "gesture_left_submovements") == 2.0 and value(result, "gesture_left_holds") == 1.0
        assert 0.6 < value(result, "gesture_left_hold_s") < 0.85
        assert value(result, "gesture_left_peak_speed") == pytest.approx(30.0 * np.pi / 2, rel=0.05)
        assert value(result, "gesture_left_max_height") == 1.0 and value(result, "gesture_left_space_max") == 2.0
        assert value(result, "gesture_right_submovements") == 0.0

    def test_speeds_carry_pixel_units(self, run):
        specs = {spec.name: spec for spec in run([("parent", 0.0, 3.4, "ann")]).specs}
        assert specs["gesture_left_peak_speed"].unit == "px/s" and specs["gesture_left_extent"].unit == "px^2"

    def test_two_coders_are_not_mixed(self, run):
        both = [("parent", 0.0, 3.4, "ann"), ("parent", 0.1, 3.3, "bob")]
        mixed = run(both)
        assert mixed.rows.empty and "set kinematics.gestures.annotator" in mixed.warnings[0].message
        assert set(run(both, annotator="bob").rows.t_start) == {0.1}

    def test_a_gesture_of_nobody_is_left_out_and_said(self, run):
        result = run([("parent", 0.0, 3.4, "ann"), (None, 1.0, 2.0, "ann")])
        assert set(result.rows.t_start) == {0.0} and "belong to no person" in result.warnings[0].message

    def test_it_runs_only_when_a_tier_is_set(self):
        assert not GestureExtractor.wanted_by_default(SimpleNamespace(kinematics=KinematicsSection()))
