from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from macaw.config.schema import MotionSection  # noqa: E402
from macaw.exceptions import ConfigError  # noqa: E402
from macaw.extractors.video.motion import mea as mea_module  # noqa: E402
from macaw.extractors.video.motion.mea import MEAExtractor, frame_difference_energy, smooth  # noqa: E402
from macaw.project.recording import MediaFile  # noqa: E402
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame  # noqa: E402

FPS = 10.0
SIZE = (320, 240)


def write_video(path: Path, paint, n_frames=20):
    """paint(frame, image) draws on a black picture; MJPG is lossy, so drawings are solid blocks."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for frame in range(n_frames):
        image = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
        paint(frame, image)
        writer.write(image)
    writer.release()
    return path


def blinking_left(frame, image):
    """A 40x40 block inside the left region, white on even frames."""
    if frame % 2 == 0:
        image[40:80, 40:80] = 255


class TestFrameDifference:
    def test_only_the_region_with_movement_moves(self, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        energy, size = frame_difference_energy(video, {"a": (0, 0, 160, 240), "b": (160, 0, 160, 240)}, threshold=10)
        assert size == SIZE
        assert np.isnan(energy["a"].iloc[0])
        assert energy["a"].iloc[1:].min() > 0
        assert energy["b"].iloc[1:].max() == 0

    def test_the_value_is_the_share_of_the_region_that_changed(self, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        energy, _ = frame_difference_energy(video, {"a": (0, 0, 160, 240)}, threshold=10)
        assert energy["a"].iloc[1:].median() == pytest.approx(1600 / (160 * 240), rel=0.15)

    def test_a_repeated_picture_is_marked_identical(self, tmp_path):
        video = write_video(tmp_path / "room.avi", lambda frame, image: None)
        energy, _ = frame_difference_energy(video, {"a": (0, 0, 160, 240)}, threshold=10)
        assert energy["identical"].iloc[1:].all() and not energy["identical"].iloc[0]

    def test_only_the_asked_frames_are_read_and_keep_their_numbers(self, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        energy, _ = frame_difference_energy(video, {"a": (0, 0, 160, 240)}, threshold=10, frames=(5, 9))
        assert energy["frame"].tolist() == [5, 6, 7, 8]

    def test_a_region_outside_the_picture_is_refused(self, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        with pytest.raises(ConfigError, match="outside"):
            frame_difference_energy(video, {"a": (400, 0, 50, 50)}, threshold=10)


class TestSmooth:
    def test_under_two_frames_leaves_it_alone(self):
        series = pd.Series([0.0, 1.0, 0.0])
        assert smooth(series, 1).tolist() == [0.0, 1.0, 0.0]

    def test_a_centred_mean(self):
        assert smooth(pd.Series([0.0, 3.0, 0.0, 0.0]), 3).tolist() == pytest.approx([1.5, 1.0, 1.0, 0.0])


def media(path, subject_id=None, offset_s=0.0):
    return MediaFile(path=path, kind="video", subject_id=subject_id, fps=FPS, offset_s=offset_s)


def recording(files, rois):
    return SimpleNamespace(
        recording_id="sub-003_ses-12mo",
        subjects=[SimpleNamespace(subject_id=name, roi=rois.get(name)) for name in ("parent", "child")],
        media=list(files),
        entities=lambda: {"group_id": "003", "session_id": "12mo", "task": None, "run": None},
    )


@pytest.fixture
def run(monkeypatch, tmp_path):
    def build(files, rois, *, start=0.0, duration=1.9, smoothing_s=0.0):
        monkeypatch.setattr(
            mea_module, "apply_prep_timing", lambda config, rec: SimpleNamespace(t_origin_s=start, duration_s=duration)
        )
        extractor = MEAExtractor(SimpleNamespace(motion=MotionSection(threshold=10, smoothing_s=smoothing_s)))
        return extractor.run(recording(files, rois), tmp_path)

    return build


class TestExtractor:
    def test_every_person_needs_a_region(self, run, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        with pytest.raises(ConfigError, match="child"):
            run([media(video)], {"parent": (0, 0, 160, 240)})

    def test_rows_and_manifest(self, run, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        result = run([media(video)], {"parent": (0, 0, 160, 240), "child": (160, 0, 160, 240)})
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert {spec.name for spec in result.specs} == set(result.rows.feature) == {"motion_energy"}
        moved = result.rows.groupby("subject_id")["value"].max()
        assert moved["parent"] > 0 and moved["child"] == 0
        assert any(w.subject_id == "child" and "nothing moved" in w.message for w in result.warnings)

    def test_a_region_is_read_on_the_persons_own_camera(self, run, tmp_path):
        room = write_video(tmp_path / "room.avi", lambda frame, image: None)
        own = write_video(tmp_path / "parent.avi", blinking_left)
        files = [media(room), media(own, "parent")]
        result = run(files, {"parent": (0, 0, 160, 240), "child": (0, 0, 160, 240)})
        moved = result.rows.groupby("subject_id")["value"].max()
        assert moved["parent"] > 0 and moved["child"] == 0

    def test_time_is_on_the_recording_clock_and_cut_to_the_window(self, run, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        rois = {"parent": (0, 0, 160, 240), "child": (160, 0, 160, 240)}
        rows = run([media(video, offset_s=1.0)], rois, start=1.5, duration=0.5).rows
        assert rows.t_start.min() == pytest.approx(1.5)
        assert rows.t_start.max() == pytest.approx(2.0)

    def test_a_region_reaching_outside_is_cut_and_reported(self, run, tmp_path):
        video = write_video(tmp_path / "room.avi", blinking_left)
        result = run([media(video)], {"parent": (-20, 0, 180, 240), "child": (160, 0, 200, 240)})
        clipped = {w.subject_id for w in result.warnings if "outside" in w.message}
        assert clipped == {"parent", "child"}

    def test_the_lights_going_on_is_reported(self, run, tmp_path):
        def flash(frame, image):
            if frame == 10:
                image[:] = 255

        video = write_video(tmp_path / "room.avi", flash)
        result = run([media(video)], {"parent": (0, 0, 160, 240), "child": (160, 0, 160, 240)})
        assert any("whole picture" in w.message and "2 frame" in w.message for w in result.warnings)

    def test_smoothing_spreads_a_single_change(self, run, tmp_path):
        def once(frame, image):
            if frame >= 10:
                image[40:80, 40:80] = 255

        video = write_video(tmp_path / "room.avi", once)
        rois = {"parent": (0, 0, 160, 240), "child": (160, 0, 160, 240)}
        raw = run([media(video)], rois).rows
        smoothed = run([media(video)], rois, smoothing_s=0.5).rows
        count = lambda rows: int((rows[rows.subject_id == "parent"].value > 0).sum())
        assert count(raw) == 1 and count(smoothed) == 5
