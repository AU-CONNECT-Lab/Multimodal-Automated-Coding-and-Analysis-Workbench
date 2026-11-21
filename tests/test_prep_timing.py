from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from macaw.extractors import prep as prep_module
from macaw.extractors.prep import PrepInfo, apply_prep_timing
from macaw.project.recording import MediaFile


def entry(name, *, fps=None, offset_s=0.0, duration_s=60.0):
    return {
        "path": f"/somewhere/else/{name}", "name": name, "kind": "video", "subject_id": None,
        "duration_s": duration_s, "sample_rate": None, "fps": fps, "offset_s": offset_s, "clock_scale": 1.0,
        "drift_ppm": None, "offset_confidence": 0.9,
    }


@pytest.fixture
def prepared(monkeypatch):
    info = PrepInfo(
        recording_id="r1", t_origin_s=10.0, duration_s=120.0, sample_rate=16000,
        media=[entry("room.mp4", fps=25.0, offset_s=1.5), entry("mic.wav")],
    )
    monkeypatch.setattr(prep_module, "load_prep", lambda config, rid: info)
    monkeypatch.setattr(prep_module, "load_video_prep", lambda config, rid: {})
    return info


def video_entry(*, fps, frames_start_s, copy):
    return {"fps": fps, "frames_start_s": frames_start_s, "path": Path(copy) if copy else None}


def test_media_get_the_timing_prep_found(prepared):
    """A fresh process builds the recording without probing, so fps would otherwise be None."""
    room = MediaFile(path=Path("raw/r1/room.mp4"), kind="video")
    apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
    assert (room.fps, room.offset_s, room.duration_s, room.offset_confidence) == (25.0, 1.5, 60.0, 0.9)


def test_files_are_matched_by_name_not_by_absolute_path(prepared):
    """prep may have run on another machine, where the raw directory sits somewhere else."""
    room = MediaFile(path=Path("D:/elsewhere/room.mp4"), kind="video")
    apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
    assert room.fps == 25.0


def test_a_file_prep_never_saw_is_left_alone(prepared):
    late = MediaFile(path=Path("raw/r1/added_later.mp4"), kind="video")
    apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[late]))
    assert late.fps is None


def test_the_prep_record_is_returned_for_its_window(prepared):
    assert apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[])).t_origin_s == 10.0


def test_a_prepared_video_is_read_from_its_copy_at_its_rate(prepared, monkeypatch):
    monkeypatch.setattr(
        prep_module, "load_video_prep", lambda config, rid: {"room.mp4": video_entry(fps=30.0, frames_start_s=0.0, copy="prep/room.mp4.mp4")}
    )
    room = MediaFile(path=Path("raw/r1/room.mp4"), kind="video")
    apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
    assert (room.frames_path, room.fps, room.offset_s, room.path.name) == (Path("prep/room.mp4.mp4"), 30.0, 1.5, "room.mp4")


def test_a_video_read_as_recorded_counts_from_its_first_picture(prepared, monkeypatch):
    """Its pictures start 0.2 s after its sound, and the file's offset was measured on the sound."""
    monkeypatch.setattr(prep_module, "load_video_prep", lambda config, rid: {"room.mp4": video_entry(fps=25.0, frames_start_s=0.2, copy=None)})
    room = MediaFile(path=Path("raw/r1/room.mp4"), kind="video")
    apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
    assert room.frames_path == room.path
    assert room.offset_s == pytest.approx(1.7)


def test_applying_the_timing_twice_does_not_add_the_start_twice(prepared, monkeypatch):
    monkeypatch.setattr(prep_module, "load_video_prep", lambda config, rid: {"room.mp4": video_entry(fps=25.0, frames_start_s=0.2, copy=None)})
    room = MediaFile(path=Path("raw/r1/room.mp4"), kind="video")
    for _ in range(2):
        apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
    assert room.offset_s == pytest.approx(1.7)


def copy_entry(**changes):
    entry = {
        "file": "room.mp4.mp4", "fps": 30.0, "source_fps": 60.0, "first_picture_s": 0.0, "size": [200, 100],
        "rotation": None, "prepared_size": [400, 200],
    }
    return {**entry, **changes}


def test_the_original_behind_a_copy_puts_its_points_on_the_copys_frames_and_pixels():
    from macaw.extractors.prep import SourceVideo

    source = SourceVideo.of(copy_entry(fps=20.0, first_picture_s=0.1), Path("raw/room.mp4"))
    table = pd.DataFrame({"frame": [0, 1, 2, 3], "track_id": 1, "joint": "nose", "x": 10.0, "y": 20.0})
    moved = source.to_copy(table)
    # 0.1 s in at 20 fps is copy frame 2; of the original frames on one copy frame the nearest stays
    assert moved["frame"].tolist() == [2, 3] and moved["x"].tolist() == [20.0] * 2 and moved["y"].tolist() == [40.0] * 2


def test_a_turned_original_is_measured_upright_and_none_is_given_without_a_copy():
    from macaw.extractors.prep import SourceVideo

    assert SourceVideo.of(copy_entry(rotation=-90), Path("room.mp4")).size == (100, 200)
    assert SourceVideo.of(copy_entry(file=None), Path("room.mp4")) is None
    assert SourceVideo.of(None, Path("room.mp4")) is None
