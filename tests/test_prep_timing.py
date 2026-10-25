from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from macaw.extractors import prep as prep_module
from macaw.extractors.prep import PrepInfo, apply_prep_timing
from macaw.project.recording import MediaFile


def entry(name, *, fps=None, offset_s=0.0, duration_s=60.0):
    return {
        "path": f"/somewhere/else/{name}", "name": name, "kind": "video", "subject_id": None,
        "duration_s": duration_s, "sample_rate": None, "fps": fps, "offset_s": offset_s,
        "drift_ppm": None, "offset_confidence": 0.9,
    }


@pytest.fixture
def prepared(monkeypatch):
    info = PrepInfo(
        recording_id="r1", t_origin_s=10.0, duration_s=120.0, sample_rate=16000,
        media=[entry("room.mp4", fps=25.0, offset_s=1.5), entry("mic.wav")],
    )
    monkeypatch.setattr(prep_module, "load_prep", lambda config, rid: info)
    return info


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
