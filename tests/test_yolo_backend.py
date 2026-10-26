from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors.video.pose import backends
from macaw.extractors.video.pose.backends import (
    COCO_KEYPOINT_ORDER,
    YOLO_ENV_VAR,
    YOLOPoseBackend,
    yolo_device,
    yolo_label_rows,
)
from macaw.media.probe import MediaInfo, StreamInfo


def label_line(*, x=0.5, y=0.25, confidence=0.9, track_id=None):
    points = " ".join(f"{x} {y} {confidence}" for _ in COCO_KEYPOINT_ORDER)
    tail = "0.8" if track_id is None else f"0.8 {track_id}"
    return f"0 0.5 0.5 0.1 0.2 {points} {tail}"


class TestLabelRows:
    def read(self, line, min_confidence=0.3):
        return yolo_label_rows([line], frame_index=7, width=640, height=480, min_confidence=min_confidence)

    def test_a_tracked_person_is_seventeen_named_points_in_pixels(self):
        rows = self.read(label_line(track_id=3))
        assert [row["joint"] for row in rows] == list(COCO_KEYPOINT_ORDER)
        assert (rows[0]["x"], rows[0]["y"], rows[0]["track_id"], rows[0]["frame"]) == (320.0, 120.0, 3, 7)
        assert math.isnan(rows[0]["z"])

    def test_without_a_tracker_the_track_id_is_empty_not_zero(self):
        assert {row["track_id"] for row in self.read(label_line())} == {None}

    def test_a_doubtful_point_keeps_its_confidence_but_loses_its_place(self):
        row = self.read(label_line(confidence=0.2))[0]
        assert math.isnan(row["x"]) and row["confidence"] == pytest.approx(0.2)

    def test_a_point_written_at_the_corner_is_missing(self):
        assert math.isnan(self.read(label_line(x=0.0, y=0.0))[0]["x"])

    def test_a_model_with_other_points_is_refused(self):
        with pytest.raises(ExternalToolError, match="17-point"):
            self.read("0 0.5 0.5 0.1 0.2 0.1 0.1 0.9 0.8")


@pytest.mark.parametrize("device, expected", [("auto", None), ("cuda", "0"), ("cuda:1", "1"), ("cpu", "cpu")])
def test_device_names(device, expected):
    assert yolo_device(device) == expected


class TestExecutable:
    def test_missing_says_how_to_install_it_apart(self, monkeypatch):
        monkeypatch.delenv(YOLO_ENV_VAR, raising=False)
        monkeypatch.setattr(backends.shutil, "which", lambda name: None)
        with pytest.raises(BackendNotAvailableError, match=YOLO_ENV_VAR):
            YOLOPoseBackend().check_available()

    def test_the_environment_variable_wins(self, monkeypatch, tmp_path):
        executable = tmp_path / "yolo.exe"
        executable.write_bytes(b"")
        monkeypatch.setenv(YOLO_ENV_VAR, str(executable))
        assert YOLOPoseBackend()._executable() == str(executable)


@pytest.fixture
def fake_yolo(monkeypatch, tmp_path):
    """A yolo that writes three frames of labels for whatever clip it is given, and records how it was called."""
    executable = tmp_path / "yolo.exe"
    executable.write_bytes(b"")
    weights = tmp_path / "pose.pt"
    weights.write_bytes(b"")
    monkeypatch.setenv(YOLO_ENV_VAR, str(executable))
    info = MediaInfo(
        path=tmp_path / "room.mp4",
        duration_s=100.0,
        streams=[StreamInfo(index=0, codec_type="video", codec_name="h264", fps=25.0, width=640, height=480)],
    )
    monkeypatch.setattr(backends, "probe", lambda path: info)
    clips = []

    def clip(video_path, first, stop, scratch):
        clips.append((first, stop))
        return scratch / "window.mp4"

    monkeypatch.setattr(backends, "_window_clip", clip)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs.get("env") or {}))
        args = dict(part.split("=", 1) for part in command[3:])
        labels = Path(args["project"]) / "run" / "labels"
        labels.mkdir(parents=True)
        stem = Path(args["source"]).stem
        for k in (1, 2, 3):
            (labels / f"{stem}_{k}.txt").write_text(label_line(track_id=k) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", run)
    return type("Fake", (), {"calls": calls, "clips": clips, "video": info.path, "weights": str(weights)})


class TestInfer:
    def test_frames_are_numbered_in_the_file_not_the_clip(self, fake_yolo):
        frames = YOLOPoseBackend().infer(
            fake_yolo.video, device="cuda", min_confidence=0.3, model=fake_yolo.weights, frames=(100, 102),
            input_size=1280,
        )
        assert sorted(frames.keypoints["frame"].unique()) == [100, 101]
        assert fake_yolo.clips == [(100, 102)]
        command, _ = fake_yolo.calls[0]
        assert command[1:3] == ["pose", "track"]
        assert {"imgsz=1280", "device=0", f"model={fake_yolo.weights}", "save=False"} <= set(command)

    def test_it_runs_offline_unless_the_user_said_otherwise(self, fake_yolo, monkeypatch):
        monkeypatch.delenv("YOLO_OFFLINE", raising=False)
        YOLOPoseBackend().infer(fake_yolo.video, device="auto", min_confidence=0.3, model=fake_yolo.weights)
        assert fake_yolo.calls[0][1]["YOLO_OFFLINE"] == "1"

    def test_the_whole_file_is_read_without_cutting(self, fake_yolo):
        YOLOPoseBackend().infer(
            fake_yolo.video, device="auto", min_confidence=0.3, model=fake_yolo.weights, frames=(0, 2500)
        )
        assert fake_yolo.clips == []
        assert not any(part.startswith("device=") for part in fake_yolo.calls[0][0])

    @pytest.mark.parametrize("model", [None, "yolo11x-pose.pt"])
    def test_weights_have_to_be_a_file_since_nothing_is_downloaded(self, fake_yolo, model):
        with pytest.raises(ExternalToolError, match="pose.model"):
            YOLOPoseBackend().infer(fake_yolo.video, device="auto", min_confidence=0.3, model=model)

    def test_a_failed_run_shows_what_yolo_said(self, fake_yolo, monkeypatch):
        monkeypatch.setattr(
            backends.subprocess, "run",
            lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "CUDA error: out of memory"),
        )
        with pytest.raises(ExternalToolError, match="CUDA error: out of memory"):
            YOLOPoseBackend().infer(fake_yolo.video, device="auto", min_confidence=0.3, model=fake_yolo.weights)
