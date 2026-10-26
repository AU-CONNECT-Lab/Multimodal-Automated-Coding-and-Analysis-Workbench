from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from macaw.extractors.video.pose.backends import (
    BACKEND_BODY_MODEL_SCHEMA,
    BACKEND_KEYPOINT_SCHEMA,
    MEDIAPIPE_LANDMARKS,
    PROVIDES_2D_ONLY,
    PoseFrames,
    landmark_rows,
)
from macaw.extractors.video.pose.skeleton import (
    STANDARD_JOINTS,
    extra_joint,
    is_standard,
)
from macaw.store.schema import (
    BODY_MODEL_SCHEMA,
    KEYPOINT_SCHEMA,
    UNITS_PIXEL,
    coerce_frame,
    empty_frame,
    validate_frame,
)

WIDTH, HEIGHT = 1920, 1080
EXTRACTOR_OWNED = {"recording_id", "subject_id", "source", "t"}


@dataclass
class FakeLandmark:
    x: float
    y: float
    visibility: float


def pose(x=0.5, y=0.25, visibility=0.9, n=len(MEDIAPIPE_LANDMARKS)):
    return [FakeLandmark(x, y, visibility) for _ in range(n)]


@pytest.fixture
def rows():
    return landmark_rows(
        [pose(), pose(x=0.1, y=0.2, visibility=0.1)],
        frame_index=7,
        width=WIDTH,
        height=HEIGHT,
        min_confidence=0.3,
    )


@pytest.fixture
def frame(rows):
    return coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


class TestSkeleton:
    def test_the_standard_set_is_coco_17(self):
        assert len(STANDARD_JOINTS) == 17
        assert len(set(STANDARD_JOINTS)) == 17

    def test_an_extra_joint_can_never_pass_as_a_standard_one(self):
        assert extra_joint("mediapipe", "left_thumb") == "mediapipe:left_thumb"
        assert is_standard("left_wrist")
        assert not is_standard("mediapipe:left_wrist")


class TestBackendSchemas:
    def test_the_keypoint_subset_is_the_stored_table_minus_what_the_extractor_owns(self):
        """A backend sees one file and its own tracks, so it has no business naming a subject."""
        assert set(BACKEND_KEYPOINT_SCHEMA) == set(KEYPOINT_SCHEMA) - EXTRACTOR_OWNED
        for column, spec in BACKEND_KEYPOINT_SCHEMA.items():
            assert spec == KEYPOINT_SCHEMA[column]

    def test_the_body_model_subset_adds_a_track_id_the_stored_table_lacks(self):
        """Parameters have to be keyed to a track until assign_subjects names the person."""
        assert "track_id" in BACKEND_BODY_MODEL_SCHEMA
        assert "track_id" not in BODY_MODEL_SCHEMA
        assert set(BACKEND_BODY_MODEL_SCHEMA) - {"track_id"} == set(BODY_MODEL_SCHEMA) - EXTRACTOR_OWNED

    @pytest.mark.parametrize("schema", [BACKEND_KEYPOINT_SCHEMA, BACKEND_BODY_MODEL_SCHEMA])
    def test_an_empty_backend_frame_is_valid(self, schema):
        validate_frame(empty_frame(schema), schema, "backend output")

    def test_a_2d_backend_declares_only_2d(self):
        assert PoseFrames(keypoints=empty_frame(BACKEND_KEYPOINT_SCHEMA)).provides == PROVIDES_2D_ONLY


class TestLandmarkRows:
    def test_every_landmark_of_every_pose_becomes_a_row(self, rows):
        assert len(rows) == 2 * len(MEDIAPIPE_LANDMARKS)

    def test_the_rows_satisfy_the_backend_schema(self, frame):
        validate_frame(frame, BACKEND_KEYPOINT_SCHEMA, "backend keypoints")

    def test_seventeen_standard_joints_and_the_rest_behind_a_prefix(self, frame):
        joints = set(frame.joint)
        assert {joint for joint in joints if is_standard(joint)} == set(STANDARD_JOINTS)
        extras = {joint for joint in joints if not is_standard(joint)}
        assert len(extras) == len(MEDIAPIPE_LANDMARKS) - len(STANDARD_JOINTS)
        assert all(joint.startswith("mediapipe:") for joint in extras)

    def test_normalised_coordinates_become_pixels(self, frame):
        detected = frame[frame.track_id == 0]
        assert detected.x.iloc[0] == pytest.approx(0.5 * WIDTH)
        assert detected.y.iloc[0] == pytest.approx(0.25 * HEIGHT)
        assert set(frame.units) == {UNITS_PIXEL}

    def test_a_point_below_the_threshold_loses_its_coordinates_but_keeps_its_confidence(self, frame):
        """Dropping the row instead would make "not measured" and "not detected" the same thing,
        and the per-joint missing rate is exactly what tells those apart."""
        faint = frame[frame.track_id == 1]
        assert faint.x.isna().all()
        assert faint.y.isna().all()
        assert faint.confidence.iloc[0] == pytest.approx(0.1)

    def test_z_is_never_filled_in(self, frame):
        """MediaPipe reports a depth in units of torso width, which is neither metres nor
        pixels; feeding it to interpersonal_distance would look right and be wrong."""
        assert frame.z.isna().all()

    def test_metric_scale_does_not_apply_on_the_2d_path(self, frame):
        assert frame.metric_scale_valid.isna().all()

    def test_the_track_id_is_the_detection_slot(self, frame):
        assert sorted(frame.track_id.unique()) == [0, 1]

    def test_the_frame_number_is_carried_through(self, frame):
        assert set(frame.frame) == {7}

    def test_a_missing_visibility_counts_as_below_threshold(self):
        @dataclass
        class Bare:
            x: float
            y: float

        rows = landmark_rows(
            [[Bare(0.5, 0.5)]], frame_index=0, width=WIDTH, height=HEIGHT, min_confidence=0.0
        )
        assert pd.isna(rows[0]["x"])

    def test_a_point_extrapolated_out_of_frame_is_not_kept(self):
        """A seated person's elbows under the table come back visible but not present."""
        @dataclass
        class Scored:
            x: float
            y: float
            visibility: float
            presence: float | None

        rows = landmark_rows(
            [[Scored(0.5, 1.3, 0.53, 0.02), Scored(0.5, 0.8, 0.95, 0.85)]],
            frame_index=0, width=WIDTH, height=HEIGHT, min_confidence=0.3,
        )
        assert pd.isna(rows[0]["y"])
        assert rows[0]["confidence"] == pytest.approx(0.02)
        assert rows[1]["y"] == pytest.approx(0.8 * HEIGHT)
        assert rows[1]["confidence"] == pytest.approx(0.85)

    def test_a_model_without_presence_falls_back_to_visibility(self):
        @dataclass
        class Unscored:
            x: float
            y: float
            visibility: float
            presence: None = None

        rows = landmark_rows([[Unscored(0.5, 0.5, 0.9)]], frame_index=0, width=WIDTH, height=HEIGHT, min_confidence=0.3)
        assert rows[0]["confidence"] == pytest.approx(0.9)

    def test_a_model_with_more_points_than_we_know_stays_readable(self):
        rows = landmark_rows(
            [pose(n=len(MEDIAPIPE_LANDMARKS) + 5)],
            frame_index=0,
            width=WIDTH,
            height=HEIGHT,
            min_confidence=0.3,
        )
        assert len(rows) == len(MEDIAPIPE_LANDMARKS)

    def test_no_detections_is_no_rows(self):
        assert landmark_rows([], frame_index=0, width=WIDTH, height=HEIGHT, min_confidence=0.3) == []
