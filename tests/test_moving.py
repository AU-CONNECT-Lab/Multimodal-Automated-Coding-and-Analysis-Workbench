from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.schema import KinematicsSection, MovingSection
from macaw.exceptions import ConfigError
from macaw.extractors.video.pose import kinematics as kinematics_module
from macaw.extractors.video.pose.moving import MovingExtractor, either_hand, hands_moving, moving_frames, moving_spans
from macaw.store.keypoints import write_keypoints
from macaw.store.schema import ANNOTATION_SCHEMA, LONG_TABLE_SCHEMA, validate_frame
from setup_helpers import recording_of

from test_gestures import stroke_path
from test_kinematics import person

FPS = 30.0


def speeds(n, fast):
    """A speed of 50 in each (start, end) of fast, in seconds, and 0 elsewhere."""
    t = np.arange(n) / FPS
    speed = np.zeros(n)
    for start, end in fast:
        speed[(t >= start - 1e-9) & (t < end - 1e-9)] = 50.0
    return t, speed


def spans(found):
    return [(round(start, 3), round(end, 3)) for start, end in zip(found["t_start"], found["t_end"])]


class TestSpans:
    def test_close_stretches_are_one_and_short_ones_are_left_out(self):
        t, speed = speeds(180, [(1.0, 1.5), (1.6, 2.0), (3.0, 3.0 + 2 / FPS), (4.0, 4.5)])
        found, dropped = moving_spans(t, speed, above=10.0, min_s=0.15, join_s=0.15)
        assert spans(found) == [(1.0, 2.0), (4.0, 4.5)]
        assert len(dropped) == 1 and dropped["t_start"].iloc[0] == pytest.approx(3.0)

    def test_stretches_are_joined_before_the_short_ones_go(self):
        t, speed = speeds(90, [(1.0, 1.1), (1.2, 1.3)])
        found, dropped = moving_spans(t, speed, above=10.0, min_s=0.25, join_s=0.15)
        assert spans(found) == [(1.0, 1.3)] and dropped.empty

    def test_the_short_stretches_of_both_hands_are_kept_beside_the_kept_ones(self):
        t, left = speeds(180, [(1.0, 1.5), (3.0, 3.0 + 2 / FPS)])
        _, right = speeds(180, [(3.0, 3.0 + 2 / FPS)])
        _, kept, short, dropped = hands_moving(t, {"left": left, "right": right}, above=10.0, min_s=0.15, join_s=0.15)
        assert kept["label"].tolist() == ["left"] and short["label"].tolist() == ["both"] and dropped == 1
        assert short["t_start"].iloc[0] == pytest.approx(3.0)

    def test_nothing_faster_gives_no_stretch(self):
        t, speed = speeds(60, [])
        found, dropped = moving_spans(t, speed, above=10.0, min_s=0.15, join_s=0.15)
        assert found.empty and list(found.columns) == ["t_start", "t_end"] and dropped.empty

    def test_a_frame_without_a_speed_is_not_moving(self):
        t, speed = speeds(90, [(1.0, 2.0)])
        speed[(t >= 1.4 - 1e-9) & (t < 1.6 - 1e-9)] = np.nan
        found, _ = moving_spans(t, speed, above=10.0, min_s=0.15, join_s=0.1)
        assert spans(found) == [(1.0, 1.4), (1.6, 2.0)]

    def test_frames_the_person_was_out_of_view_end_a_stretch(self):
        t, speed = speeds(120, [(1.0, 3.0)])
        seen = (t < 1.5 - 1e-9) | (t >= 2.5 - 1e-9)
        found, _ = moving_spans(t[seen], speed[seen], above=10.0, min_s=0.15, join_s=0.15)
        assert spans(found) == [(1.0, 1.5), (2.5, 3.0)]

    def test_each_frame_says_whether_it_lies_in_a_kept_stretch(self):
        t, speed = speeds(90, [(1.0, 1.5), (1.6, 2.0)])
        speed[:3] = np.nan
        found, _ = moving_spans(t, speed, above=10.0, min_s=0.15, join_s=0.15)
        frames = moving_frames(t, speed, found)
        assert np.isnan(frames[:3]).all()
        assert frames[(t >= 1.0 - 1e-9) & (t < 2.0 - 1e-9)].tolist() == [1.0] * 30
        assert set(frames[t >= 2.0 - 1e-9]) == {0.0}

    def test_the_hands_stretches_are_laid_over_each_other(self):
        left = pd.DataFrame({"t_start": [1.0, 5.0], "t_end": [2.0, 6.0]})
        right = pd.DataFrame({"t_start": [1.5, 8.0], "t_end": [3.0, 9.0]})
        found = either_hand({"left": left, "right": right})
        assert spans(found) == [(1.0, 3.0), (5.0, 6.0), (8.0, 9.0)]
        assert found["label"].tolist() == ["both", "left", "right"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    def build(**moving):
        stored = tmp_path / "keypoints"
        write_keypoints(person("parent", n_frames=105, moves={"left_wrist": stroke_path(105)}), stored)
        monkeypatch.setattr(kinematics_module, "_keypoints_file", lambda config, rid: stored / "pose.keypoints.parquet")
        config = SimpleNamespace(kinematics=KinematicsSection(moving=MovingSection(**{"speed": 10.0, **moving})))
        return MovingExtractor(config).run(recording_of([], ["parent"]), tmp_path)

    return build


class TestExtractor:
    def test_two_strokes_are_two_stretches_on_the_persons_tier(self, run):
        result = run()
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        validate_frame(result.annotations, ANNOTATION_SCHEMA, "annotations")
        assert {spec.name for spec in result.specs} == set(result.rows.feature) == {"left_hand_moving", "right_hand_moving"}
        coded = result.annotations
        assert set(coded.tier) == {"moving"} and set(coded.subject_id) == {"parent"} and set(coded.annotator) == {"pose.moving"}
        assert coded.label.tolist() == ["left", "left"]
        assert 0.5 < coded.t_start.iloc[0] < 0.8 and 1.2 < coded.t_end.iloc[0] < 1.5
        assert 2.1 < coded.t_start.iloc[1] < 2.4 and 2.8 < coded.t_end.iloc[1] < 3.1

    def test_the_frames_agree_with_the_tier(self, run):
        result = run()
        left = result.rows[result.rows.feature == "left_hand_moving"]
        moving = left[left.value == 1.0].t_start
        inside = pd.concat([moving.between(start, end, inclusive="left") for start, end in zip(result.annotations.t_start, result.annotations.t_end)], axis=1).any(axis=1)
        assert len(left) == 105 and inside.all() and len(moving) > 30
        assert set(result.rows[result.rows.feature == "right_hand_moving"].value) == {0.0}

    def test_the_report_says_how_many_stretches_there_are_and_how_many_were_left_out(self, run):
        notes = [warning.message for warning in run().warnings if warning.severity == "info"]
        assert notes == ["2 stretch(es) of hand movement on the moving tier to pick gestures from; 0 shorter than 0.15 s left out, on the moving_short tier"]

    def test_a_stretch_shorter_than_the_least_length_is_left_out_on_a_tier_of_its_own(self, run):
        result = run(min_s=5.0)
        assert set(result.annotations["tier"]) == {"moving_short"} and len(result.annotations) == 2
        assert result.rows.loc[result.rows.feature == "left_hand_moving", "value"].max() == 0
        assert "0 stretch(es)" in result.warnings[0].message and "2 shorter than 5 s left out" in result.warnings[0].message

    def test_the_step_runs_only_with_a_speed(self):
        config = SimpleNamespace(kinematics=KinematicsSection())
        assert not MovingExtractor.wanted_by_default(config)
        assert MovingExtractor.wanted_by_default(SimpleNamespace(kinematics=KinematicsSection(moving=MovingSection(speed=10.0))))
        with pytest.raises(ConfigError, match="kinematics.moving.speed"):
            MovingExtractor(config).check_available()


def test_the_methods_text_for_3d_leans_on_the_2d_sentence_only_when_there_is_one():
    from macaw.qc.boilerplate import _moving_3d

    alone = SimpleNamespace(kinematics=KinematicsSection(moving=MovingSection(speed_3d=8.0)), project=SimpleNamespace(units="cm"))
    both = SimpleNamespace(kinematics=KinematicsSection(moving=MovingSection(speed=12.0, speed_3d=8.0)), project=SimpleNamespace(units="cm"))
    assert "8 cm/s" in _moving_3d(alone, None, []) and "[ripperda2020]" in _moving_3d(alone, None, [])
    assert "as above" in _moving_3d(both, None, []) and "[ripperda2020]" not in _moving_3d(both, None, [])


def test_the_methods_text_names_the_speed_and_the_source():
    from macaw.qc.boilerplate import _moving

    config = SimpleNamespace(kinematics=KinematicsSection(moving=MovingSection(speed=12.0, join_s=0.2)))
    text = _moving(config, SimpleNamespace(params={"length_unit": "px"}), [])
    assert "faster than 12 px/s" in text and "less than 0.2 s apart" in text and "[ripperda2020]" in text
