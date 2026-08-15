from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.exceptions import SchemaViolationError
from macaw.store.body_model import (
    CAMERA_COLUMNS,
    POSE_COLUMNS,
    SHAPE_COLUMNS,
    build_body_model_rows,
    has_body_model,
    to_arrays,
)
from macaw.store.schema import (
    BODY_MODEL_SCHEMA,
    BODY_MODELS,
    KEYPOINT_SCHEMA,
    UNITS_METRE,
    UNITS_PIXEL,
    coerce_frame,
    empty_frame,
    validate_frame,
)

N_FRAMES = 4


def make_block(n_columns: int, offset: float = 0.0) -> np.ndarray:
    """Distinct value per cell so a transposed or shifted block cannot pass unnoticed."""
    return offset + np.arange(N_FRAMES * n_columns, dtype=float).reshape(N_FRAMES, n_columns)


@pytest.fixture
def rows():
    return build_body_model_rows(
        recording_id="sub-003_ses-12mo",
        subject_id="child",
        source="pose.rtmpose",
        body_model="smil",
        frames=np.arange(N_FRAMES),
        times=np.arange(N_FRAMES) / 30.0,
        shape=make_block(len(SHAPE_COLUMNS)),
        pose=make_block(len(POSE_COLUMNS), offset=1000.0),
        camera=make_block(len(CAMERA_COLUMNS), offset=2000.0),
    )


class TestSchema:
    def test_parameter_counts_match_the_body_model(self):
        assert len(SHAPE_COLUMNS) == 10
        assert len(POSE_COLUMNS) == 72
        assert len(CAMERA_COLUMNS) == 6

    def test_a_person_frame_is_88_floats(self):
        """The whole point of storing parameters instead of mesh vertices."""
        parameters = SHAPE_COLUMNS + POSE_COLUMNS + CAMERA_COLUMNS
        assert len(parameters) == 88
        assert all(BODY_MODEL_SCHEMA[name] == "float64" for name in parameters)

    def test_columns_are_ordered_shape_then_pose_then_camera(self):
        names = [n for n in BODY_MODEL_SCHEMA if n.split("_")[0] in {"shape", "pose", "camera"}]
        assert names == SHAPE_COLUMNS + POSE_COLUMNS + CAMERA_COLUMNS

    def test_empty_frame_is_valid(self):
        validate_frame(empty_frame(BODY_MODEL_SCHEMA), BODY_MODEL_SCHEMA, "body_model")

    def test_body_model_vocabulary(self):
        assert BODY_MODELS == ("smpl", "smil", "none")


class TestKeypointSchemaAdditions:
    def test_units_and_metric_scale_valid_are_present(self):
        assert KEYPOINT_SCHEMA["units"] == "string"
        assert KEYPOINT_SCHEMA["metric_scale_valid"] == "bool"

    def test_metric_scale_valid_is_nullable(self):
        """On the 2D path it is neither true nor false but not applicable."""
        frame = coerce_frame(
            pd.DataFrame(
                {
                    "recording_id": ["sub-003"], "subject_id": ["child"],
                    "source": ["pose.mediapipe"], "frame": [0], "t": [0.0],
                    "joint": ["nose"], "x": [640.0], "y": [360.0], "z": [np.nan],
                    "confidence": [0.9], "track_id": [0],
                    "units": [UNITS_PIXEL], "metric_scale_valid": [None],
                }
            ),
            KEYPOINT_SCHEMA,
            "keypoints",
        )
        validate_frame(frame, KEYPOINT_SCHEMA, "keypoints")
        assert pd.isna(frame.loc[0, "metric_scale_valid"])

    def test_unit_vocabulary(self):
        assert (UNITS_PIXEL, UNITS_METRE) == ("pixel", "metre")


class TestBuildRows:
    def test_result_satisfies_the_schema(self, rows):
        validate_frame(rows, BODY_MODEL_SCHEMA, "body_model")

    def test_one_row_per_frame(self, rows):
        assert len(rows) == N_FRAMES

    def test_identity_columns_are_broadcast(self, rows):
        assert set(rows["subject_id"]) == {"child"}
        assert set(rows["body_model"]) == {"smil"}
        assert set(rows["source"]) == {"pose.rtmpose"}

    def test_parameters_land_in_the_right_columns(self, rows):
        """Column-major mistakes are the likely bug, so check a cell in each block."""
        assert rows.loc[0, "shape_0"] == 0.0
        assert rows.loc[1, "shape_0"] == float(len(SHAPE_COLUMNS))
        assert rows.loc[0, "pose_0"] == 1000.0
        assert rows.loc[0, "pose_71"] == 1000.0 + 71
        assert rows.loc[0, "camera_5"] == 2000.0 + 5

    def test_wrong_shape_width_is_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            build_body_model_rows(
                recording_id="r", subject_id="child", source="s", body_model="smpl",
                frames=np.arange(N_FRAMES), times=np.zeros(N_FRAMES),
                shape=make_block(9),                     # one column short
                pose=make_block(len(POSE_COLUMNS)),
                camera=make_block(len(CAMERA_COLUMNS)),
            )

    def test_frame_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="pose"):
            build_body_model_rows(
                recording_id="r", subject_id="child", source="s", body_model="smpl",
                frames=np.arange(N_FRAMES), times=np.zeros(N_FRAMES),
                shape=make_block(len(SHAPE_COLUMNS)),
                pose=np.zeros((N_FRAMES + 1, len(POSE_COLUMNS))),
                camera=make_block(len(CAMERA_COLUMNS)),
            )


class TestRoundTrip:
    def test_to_arrays_recovers_the_blocks(self, rows):
        out = to_arrays(rows)
        assert out["shape"].shape == (N_FRAMES, len(SHAPE_COLUMNS))
        assert out["pose"].shape == (N_FRAMES, len(POSE_COLUMNS))
        assert out["camera"].shape == (N_FRAMES, len(CAMERA_COLUMNS))
        np.testing.assert_allclose(out["shape"], make_block(len(SHAPE_COLUMNS)))
        np.testing.assert_allclose(out["pose"], make_block(len(POSE_COLUMNS), 1000.0))

    def test_to_arrays_sorts_by_frame(self, rows):
        """Rows can arrive in any order once several writers have appended."""
        shuffled = rows.iloc[::-1].reset_index(drop=True)
        out = to_arrays(shuffled)
        assert list(out["frame"]) == sorted(out["frame"])
        np.testing.assert_allclose(out["shape"], make_block(len(SHAPE_COLUMNS)))

    def test_extra_column_is_rejected(self, rows):
        with pytest.raises(SchemaViolationError):
            to_arrays(rows.assign(unexpected=1.0))


class TestHasBodyModel:
    def test_true_when_parameters_were_fitted(self, rows):
        assert has_body_model(rows) is True

    def test_false_when_every_row_is_none(self, rows):
        assert has_body_model(rows.assign(body_model="none")) is False

    def test_false_on_empty_input(self):
        assert has_body_model(empty_frame(BODY_MODEL_SCHEMA)) is False

    def test_false_when_the_column_is_absent(self):
        assert has_body_model(pd.DataFrame({"frame": [0, 1]})) is False
