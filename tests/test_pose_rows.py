from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.config.schema import PoseSection
from macaw.extractors.video.pose import keypoints as pose_module
from macaw.extractors.video.pose.backends import BACKEND_KEYPOINT_SCHEMA, PoseFrames
from macaw.extractors.video.pose.keypoints import PoseExtractor
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.store.schema import (
    KEYPOINT_SCHEMA,
    LONG_TABLE_SCHEMA,
    coerce_frame,
    empty_frame,
    validate_frame,
)

FPS = 10.0
N_FRAMES = 20
SWAP_AT = 10


def backend_frame(positions, *, n_frames=N_FRAMES, swap_at=None, confidence=0.9):
    rows = []
    for frame in range(n_frames):
        order = list(positions)
        if swap_at is not None and frame >= swap_at:
            order = order[::-1]
        for track, x in enumerate(order):
            for joint in STANDARD_JOINTS:
                rows.append(
                    {
                        "frame": frame,
                        "joint": joint,
                        "x": float(x),
                        "y": 200.0,
                        "z": float("nan"),
                        "confidence": confidence,
                        "track_id": track,
                        "units": "pixel",
                        "metric_scale_valid": None,
                    }
                )
    if not rows:
        return empty_frame(BACKEND_KEYPOINT_SCHEMA)
    return coerce_frame(pd.DataFrame(rows), BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


class FakeBackend:
    name = "fake"
    version = "9.9"
    joint_set = "coco17"

    def __init__(self, frames=None):
        self.frames = frames if frames is not None else backend_frame([100.0, 500.0], swap_at=SWAP_AT)
        self.calls = []

    def check_available(self):
        return None

    def infer(self, video_path, *, device, min_confidence, model=None, max_people=1):
        self.calls.append({"path": video_path, "max_people": max_people})
        return PoseFrames(keypoints=self.frames)


def media(name="room.mp4", *, subject_id=None, fps=FPS, offset_s=0.0, kind="video"):
    return SimpleNamespace(
        path=Path(name),
        kind=kind,
        subject_id=subject_id,
        fps=fps,
        offset_s=offset_s,
        duration_s=N_FRAMES / FPS,
    )


@pytest.fixture
def recording():
    def build(files, *, t_task_start=0.0, t_task_end=None, subject_ids=("parent", "child")):
        return SimpleNamespace(
            recording_id="sub-003_ses-12mo",
            subjects=[SimpleNamespace(subject_id=name, role=name) for name in subject_ids],
            media=list(files),
            t_task_start=t_task_start,
            t_task_end=t_task_end,
            entities=lambda: {
                "group_id": "003",
                "session_id": "12mo",
                "task": None,
                "run": None,
            },
        )

    return build


@pytest.fixture
def extractor(monkeypatch):
    def build(backend=None):
        backend = backend or FakeBackend()
        monkeypatch.setattr(pose_module, "resolve_backend", lambda name: backend)
        instance = PoseExtractor(SimpleNamespace(pose=PoseSection()))
        instance.fake = backend
        return instance

    return build


@pytest.fixture
def shared(extractor, recording, tmp_path):
    instance = extractor()
    return instance, instance.run(recording([media()]), tmp_path)


class TestSharedCamera:
    def test_the_keypoints_satisfy_the_stored_schema(self, shared):
        validate_frame(shared[1].keypoints, KEYPOINT_SCHEMA, "keypoints")

    def test_the_rows_satisfy_the_long_table_schema(self, shared):
        validate_frame(shared[1].rows, LONG_TABLE_SCHEMA, "long table")

    def test_manifest_and_rows_agree(self, shared):
        """Every feature emitted is documented, and nothing documented is never emitted."""
        extractor, result = shared
        assert {spec.name for spec in result.specs} == set(result.rows.feature)

    def test_the_backend_is_asked_for_as_many_people_as_there_are_subjects(self, shared):
        assert shared[0].fake.calls[0]["max_people"] == 2

    def test_both_people_are_named(self, shared):
        assert set(shared[1].keypoints.subject_id.dropna()) == {"parent", "child"}

    def test_the_crossover_is_repaired(self, shared):
        """Each person stayed at one x, so one x per subject means the swap was undone."""
        assert shared[1].keypoints.groupby("subject_id")["x"].nunique().to_dict() == {
            "parent": 1,
            "child": 1,
        }

    def test_the_raw_track_id_survives_the_repair(self, shared):
        """The repair shows up in subject_id; track_id stays what the model actually said, so
        a disputed assignment can still be traced back."""
        keypoints = shared[1].keypoints
        assert sorted(keypoints.track_id.dropna().unique()) == [0, 1]
        assert keypoints[keypoints.frame == 0].track_id.nunique() == 2

    def test_time_comes_from_the_frame_rate(self, shared):
        keypoints = shared[1].keypoints
        assert keypoints.t.min() == pytest.approx(0.0)
        assert keypoints.t.max() == pytest.approx((N_FRAMES - 1) / FPS)

    def test_the_swap_count_belongs_to_nobody(self, shared):
        rows = shared[1].rows
        swaps = rows[rows.feature == "n_track_swaps"]
        assert len(swaps) == 1
        assert swaps.subject_id.isna().all()
        assert float(swaps.value.iloc[0]) == 1.0

    def test_the_crossovers_are_written_out_as_evidence(self, shared):
        """The identity review step needs to show why a track was relabelled."""
        assert shared[1].artifacts["swaps"].is_file()

    def test_a_crossover_is_reported(self, shared):
        assert any(warning.severity == "info" for warning in shared[1].warnings)


class TestPerSubjectCameras:
    @pytest.fixture
    def result(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0])))
        files = [media("a.mp4", subject_id="parent"), media("b.mp4", subject_id="child")]
        return instance, instance.run(recording(files), tmp_path)

    def test_one_person_is_asked_for_per_file(self, result):
        assert [call["max_people"] for call in result[0].fake.calls] == [1, 1]

    def test_the_subject_comes_from_the_file_not_from_a_guess(self, result):
        assert set(result[1].keypoints.subject_id.dropna()) == {"parent", "child"}

    def test_no_repair_is_attempted(self, result):
        """A camera pointed at one person has already answered the identity question."""
        rows = result[1].rows
        assert float(rows[rows.feature == "n_track_swaps"].value.iloc[0]) == 0.0
        assert "swaps" not in result[1].artifacts

    def test_a_shared_camera_alongside_them_is_skipped_with_a_warning(
        self, extractor, recording, tmp_path
    ):
        instance = extractor(FakeBackend(backend_frame([100.0])))
        files = [media("a.mp4", subject_id="parent"), media("room.mp4")]
        result = instance.run(recording(files), tmp_path)
        assert len(instance.fake.calls) == 1
        assert any("shared" in warning.message for warning in result.warnings)


class TestWindowAndClock:
    def test_frames_outside_the_task_window_are_dropped(self, extractor, recording, tmp_path):
        instance = extractor()
        result = instance.run(recording([media()], t_task_start=0.5, t_task_end=1.0), tmp_path)
        assert result.keypoints.t.min() >= 0.5
        assert result.keypoints.t.max() <= 1.0

    def test_the_offset_is_applied_once_here(self, extractor, recording, tmp_path):
        """prep applies it once for audio; doing it again downstream would misalign the two
        modalities against each other rather than against the clock."""
        instance = extractor()
        result = instance.run(
            recording([media(offset_s=100.0)], t_task_start=100.0, t_task_end=102.0), tmp_path
        )
        assert result.keypoints.t.min() == pytest.approx(100.0)


class TestNothingDetected:
    @pytest.fixture
    def result(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0], n_frames=0)))
        return instance.run(recording([media()]), tmp_path)

    def test_an_empty_keypoint_table_is_still_valid(self, result):
        validate_frame(result.keypoints, KEYPOINT_SCHEMA, "keypoints")
        assert result.keypoints.empty

    def test_it_fails_loudly_rather_than_returning_nothing(self, result):
        assert any(warning.severity == "fail" for warning in result.warnings)

    def test_no_video_at_all_is_an_error(self, extractor, recording, tmp_path):
        from macaw.exceptions import UnreadableMediaError

        instance = extractor()
        with pytest.raises(UnreadableMediaError):
            instance.run(recording([media(kind="audio")]), tmp_path)


class TestQualityWarnings:
    def test_a_track_that_is_mostly_missing_is_reported(self, extractor, recording, tmp_path):
        frames = backend_frame([100.0, 500.0])
        frames.loc[frames.frame > 3, "x"] = None
        instance = extractor(FakeBackend(frames))
        result = instance.run(recording([media()]), tmp_path)
        assert any("detected in only" in warning.message for warning in result.warnings)

    def test_an_unmatched_track_is_reported(self, extractor, recording, tmp_path):
        instance = extractor(FakeBackend(backend_frame([100.0, 300.0, 500.0])))
        result = instance.run(recording([media()]), tmp_path)
        assert result.keypoints.subject_id.isna().any()
        assert any("no subject_id" in warning.message for warning in result.warnings)
