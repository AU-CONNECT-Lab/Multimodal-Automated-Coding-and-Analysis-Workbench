from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import FaceSection
from macaw.extractors.video.face import movement
from macaw.extractors.video.face.movement import (
    FaceMovementExtractor,
    eye_opening,
    find_blinks,
    mouth_opening,
    per_second_change,
)
from macaw.store.longtable import build_long_rows, write_long_table
from macaw.store.schema import LONG_TABLE_SCHEMA, Scale, validate_frame

FPS = 25.0


def face(n_frames, *, eye_gap=4.0, eye_width=12.0, lip_gap=2.0, subject="parent"):
    """Eyes at y 40, nose base 20 px below them; eye_gap and lip_gap may be per-frame arrays."""
    eye_gap = np.broadcast_to(np.asarray(eye_gap, dtype=float), (n_frames,))
    lip_gap = np.broadcast_to(np.asarray(lip_gap, dtype=float), (n_frames,))
    columns = {"subject_id": [subject] * n_frames, "frame": np.arange(n_frames), "t": np.arange(n_frames) / FPS}
    points = {i: (np.zeros(n_frames), np.zeros(n_frames)) for i in range(68)}
    for left in (30.0, 60.0):
        corner_a, top_a, top_b, corner_b, bottom_b, bottom_a = (36, 37, 38, 39, 40, 41) if left == 30.0 else (42, 43, 44, 45, 46, 47)
        points[corner_a] = (np.full(n_frames, left), np.full(n_frames, 40.0))
        points[corner_b] = (np.full(n_frames, left + eye_width), np.full(n_frames, 40.0))
        for top, bottom in ((top_a, bottom_a), (top_b, bottom_b)):
            points[top] = (np.full(n_frames, left + eye_width / 2), 40.0 - eye_gap / 2)
            points[bottom] = (np.full(n_frames, left + eye_width / 2), 40.0 + eye_gap / 2)
    for top, bottom in ((61, 67), (62, 66), (63, 65)):
        points[top] = (np.full(n_frames, 50.0), 70.0 - lip_gap / 2)
        points[bottom] = (np.full(n_frames, 50.0), 70.0 + lip_gap / 2)
    eye_x = np.mean([points[i][0] for i in range(36, 48)], axis=0)
    points[33] = (eye_x, np.full(n_frames, 60.0))
    for i, (x, y) in points.items():
        columns[f"x_{i}"], columns[f"y_{i}"] = x, y
    return pd.DataFrame(columns)


def with_blinks(n_frames, at, *, closed_frames=3, open_gap=4.0):
    gap = np.full(n_frames, open_gap)
    for start in at:
        gap[start:start + closed_frames] = 0.5
    return gap


def blinks_of(table):
    eyes, _ = eye_opening(table)
    return find_blinks(eyes, table["frame"], table["t"])


class TestMeasures:
    def test_mouth_opening_is_lip_gap_over_eyes_to_nose(self):
        assert mouth_opening(face(3, lip_gap=6.0)).tolist() == pytest.approx([0.3] * 3)

    def test_eye_opening_is_gap_over_width_and_width_in_pixels(self):
        ratio, width = eye_opening(face(2, eye_gap=3.0, eye_width=12.0))
        assert ratio.tolist() == pytest.approx([0.25, 0.25]) and width.tolist() == pytest.approx([12.0, 12.0])

    def test_change_per_second_is_missing_across_a_gap_in_frames(self):
        values = pd.Series([0.1, 0.2, 0.2, 0.5])
        change = per_second_change(values, pd.Series([0, 1, 2, 5]), pd.Series([0, 1, 2, 5]) / FPS)
        assert np.isnan(change[0]) and change[1] == pytest.approx(2.5) and change[2] == 0 and np.isnan(change[3])


class TestBlinks:
    def test_short_closures_with_the_eye_open_around_them_are_blinks(self):
        found = blinks_of(face(500, eye_gap=with_blinks(500, [100, 250, 400])))
        assert found["t_start"].tolist() == pytest.approx([4.0, 10.0, 16.0])
        assert (found["t_end"] - found["t_start"]).tolist() == pytest.approx([3 / FPS] * 3)

    def test_a_closure_longer_than_a_blink_is_not_one(self):
        assert blinks_of(face(500, eye_gap=with_blinks(500, [200], closed_frames=20))).empty

    def test_lids_that_lower_slowly_and_stay_down_are_not_a_blink(self):
        gap = np.full(500, 4.0)
        gap[200:215] = np.linspace(4.0, 0.5, 15)
        gap[215:] = 0.5
        assert blinks_of(face(500, eye_gap=gap)).empty

    def test_a_closure_at_a_missing_face_is_not_counted(self):
        table = face(500, eye_gap=with_blinks(500, [200]))
        assert blinks_of(table[(table.frame < 197) | (table.frame > 205)]).empty

    def test_steady_eyes_have_no_blinks(self):
        rng = np.random.default_rng(0)
        assert blinks_of(face(500, eye_gap=4.0 + rng.normal(0, 0.2, 500))).empty


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(landmarks, yaw=None, min_eye_px=5.0):
        landmarks_path = tmp_path / "landmarks.parquet"
        landmarks.to_parquet(landmarks_path, index=False)
        angles_path = tmp_path / "features"
        parts = []
        for subject, part in landmarks.groupby("subject_id"):
            parts.append(build_long_rows(
                values=pd.DataFrame({"t_start": part["t"], "face_yaw_deg": yaw if yaw is not None else 0.0}),
                scale=Scale.FRAME, recording_id="r1", source="face.au", entities={}, subject_id=subject,
            ))
        write_long_table(pd.concat(parts, ignore_index=True), angles_path / "face.au.parquet")
        monkeypatch.setattr(FaceMovementExtractor, "_files", lambda self, rid: (landmarks_path, angles_path / "face.au.parquet"))
        monkeypatch.setattr(movement, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=0.0, duration_s=20.0))
        recording = SimpleNamespace(
            recording_id="r1", entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None}
        )
        config = SimpleNamespace(face=FaceSection(blink_min_eye_px=min_eye_px))
        return FaceMovementExtractor(config).run(recording, tmp_path)

    return build


def feature(result, name, subject="parent"):
    rows = result.rows
    return rows[(rows.feature == name) & (rows.subject_id == subject)]


class TestReadable:
    def test_a_head_tilted_away_from_its_usual_pitch_or_a_narrow_face_is_not_read(self):
        table = face(10).assign(yaw=0.0, pitch=[0.0] * 8 + [-30.0, 0.0])
        table.loc[9, [f"x_{i}" for i in range(68)]] = table.loc[9, [f"x_{i}" for i in range(68)]] * 0.3
        assert movement.eyes_in_view(table).tolist() == [True] * 8 + [False, False]

    def test_missing_angles_do_not_count_against_a_frame(self):
        assert movement.eyes_in_view(face(3).assign(yaw=np.nan, pitch=np.nan)).all()


class TestExtractor:
    def test_rows_and_manifest_agree(self, run):
        result = run(face(500, eye_gap=with_blinks(500, [100, 250, 400])))
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature)
        assert len(feature(result, "blink_duration_s")) == 3
        assert feature(result, "blinks_per_min")["value"].item() == pytest.approx(3 / 20.0 * 60.0)
        assert feature(result, "eyes_measurable_s")["value"].item() == pytest.approx(20.0)
        assert not result.warnings

    def test_eyes_too_small_are_not_read_and_say_so(self, run):
        result = run(face(500, eye_gap=with_blinks(500, [100]), eye_width=4.0))
        assert feature(result, "eye_opening")["value"].isna().all()
        assert feature(result, "blink_duration_s").empty and np.isnan(feature(result, "blinks_per_min")["value"].item())
        assert any("4.0 px wide" in warning.message for warning in result.warnings)

    def test_a_turned_head_is_not_read_for_blinks(self, run):
        yaw = np.where(np.arange(500) < 250, 60.0, 0.0)
        result = run(face(500, eye_gap=with_blinks(500, [100, 400])), yaw=yaw)
        assert feature(result, "blink_duration_s")["t_start"].tolist() == pytest.approx([16.0])
        assert feature(result, "eyes_measurable_s")["value"].item() == pytest.approx(10.0)
        assert not result.warnings

    def test_each_person_is_measured_on_their_own(self, run):
        both = pd.concat([face(500, eye_gap=with_blinks(500, [100])), face(500, subject="child", lip_gap=6.0)], ignore_index=True)
        result = run(both)
        assert len(feature(result, "blink_duration_s")) == 1 and feature(result, "blink_duration_s", "child").empty
        assert feature(result, "mouth_opening", "child")["value"].median() == pytest.approx(0.3)

    def test_a_minute_of_eyes_without_one_blink_is_no_rate_not_zero(self, run):
        result = run(face(1600))
        assert np.isnan(feature(result, "blinks_per_min")["value"].item())
        assert any("do not follow the lids" in warning.message for warning in result.warnings)
        short = run(face(1000))
        assert feature(short, "blinks_per_min")["value"].item() == 0.0 and not short.warnings
