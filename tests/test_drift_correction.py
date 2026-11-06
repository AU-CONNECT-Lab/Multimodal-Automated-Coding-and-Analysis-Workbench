from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import fftconvolve

from macaw.config.schema import AlignSection
from macaw.extractors import prep as prep_module
from macaw.extractors.prep import MediaPrepExtractor, _extract_track, apply_prep_timing
from macaw.extractors.video.pose.keypoints import PoseExtractor
from macaw.media.align import (
    GCC_PHAT,
    MANUAL,
    SAMPLE_RATE as SR,
    AlignResult,
    align_signals,
    apply_alignment,
    clock_map,
    describe_drift,
    drift_corrected,
    gcc_phat,
    write_offset_table,
)
from macaw.project.recording import MediaFile, Recording

DRIFT_PPM = 300.0
LATER_S = 12.0


def speechlike(seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    gate = np.zeros(n)
    t = 0
    while t < n:
        on, off = int(rng.uniform(0.1, 0.6) * SR), int(rng.uniform(0.05, 0.8) * SR)
        gate[t : t + on] = rng.uniform(0.2, 1.0)
        t += on + off
    return rng.standard_normal(n) * gate


def heard_by(source, offset_s, length_s, drift_ppm):
    """Another recorder: its sample j is the room at recording time offset_s + j/SR * (1 + drift), through a duller mic."""
    t = offset_s + np.arange(int(length_s * SR)) / SR * (1 + drift_ppm * 1e-6)
    dull = fftconvolve(source, np.exp(-np.arange(200) / 40.0), mode="full")[: source.size]
    signal = np.interp(t, np.arange(source.size) / SR, dull, left=0.0, right=0.0)
    return 0.5 * signal / np.abs(signal).max()


@pytest.fixture(scope="module")
def drifting():
    room = speechlike(210, 1)
    reference = 0.5 * room[: 200 * SR] / np.abs(room).max()
    target = heard_by(room, LATER_S, 180, DRIFT_PPM)
    return reference, target, align_signals(reference, target, AlignSection(drift_segment_s=30.0), names=("room.wav", "mic.wav"))


def result_for(offset_s=4.0, **extra):
    fields = dict(drift_ppm=150.0, drift_r2=0.99, overlap_s=600.0, drift_intercept_s=offset_s)
    return AlignResult(reference="room.wav", target="mic.wav", offset_s=offset_s, confidence=0.9, **{**fields, **extra})


class TestClockMap:
    def test_the_fit_gives_the_offset_and_the_rate(self, drifting):
        _, _, result = drifting
        assert drift_corrected(result)
        offset, scale = clock_map(result)
        assert offset == pytest.approx(LATER_S, abs=0.002)
        # a second pass on the target played at the first fit's rate sharpens every segment
        assert (scale - 1) * 1e6 == pytest.approx(DRIFT_PPM, abs=2.0)
        # the file's last second lands where the room heard it
        assert offset + 179 * scale == pytest.approx(LATER_S + 179 * (1 + DRIFT_PPM * 1e-6), abs=0.001)

    def test_a_steady_or_small_drift_keeps_the_offset_as_it_is(self):
        for result in (result_for(drift_ppm=0.5), result_for(drift_r2=0.2), result_for(overlap_s=60.0)):
            assert clock_map(result) == (4.0, 1.0)

    def test_a_hand_set_offset_is_used_as_it_is(self):
        assert clock_map(result_for(method=MANUAL)) == (4.0, 1.0)

    def test_a_table_without_the_fit_is_not_corrected(self):
        assert not drift_corrected(result_for(drift_intercept_s=None))

    def test_the_description_says_it_was_corrected(self):
        assert describe_drift(result_for()).endswith(", corrected")
        assert "corrected" not in describe_drift(result_for(method=MANUAL))

    def test_applying_it_sets_the_files_scale(self):
        item = MediaFile(path=Path("raw/mic.wav"), kind="audio")
        apply_alignment(SimpleNamespace(media=[item]), [result_for(offset_s=2.0, drift_intercept_s=2.0)])
        assert item.clock_scale == pytest.approx(1 / (1 - 150e-6))
        assert item.own_time(item.offset_s + 100 * item.clock_scale) == pytest.approx(100.0)
        item.duration_s = 100.0
        assert item.span_s == pytest.approx(100.0 * item.clock_scale)


@pytest.fixture(scope="module")
def tracks(drifting, tmp_path_factory):
    """The task window cut from the drifting file with the correction and with the single offset alone."""
    from macaw.media.probe import require_ffmpeg

    try:
        require_ffmpeg()
    except Exception:
        pytest.skip("needs ffmpeg")
    reference, target, result = drifting
    folder = tmp_path_factory.mktemp("drift")
    wavfile.write(folder / "mic.wav", SR, (target * 32767).astype("int16"))
    start, duration = 15.0, 170.0
    item = MediaFile(path=folder / "mic.wav", kind="audio")
    apply_alignment(SimpleNamespace(media=[item]), [result])
    made = {}
    for name, (offset, scale) in {"corrected": (item.offset_s, item.clock_scale), "as_before": (result.offset_s, 1.0)}.items():
        item.offset_s, item.clock_scale = offset, scale
        path = _extract_track(
            item.path, folder / f"{name}.wav", start_s=item.own_time(start), duration_s=duration, sample_rate=SR,
            clock_scale=scale,
        )
        made[name] = wavfile.read(path)[1].astype(float)
    return reference[int(start * SR) : int((start + duration) * SR)], made


class TestPreparedTrack:
    @staticmethod
    def lag_ms(reference, track, at_s):
        piece = slice(int(at_s * SR), int((at_s + 20) * SR))
        return gcc_phat(reference[piece], track[piece], SR, 0.5)[0] * 1000

    def test_the_corrected_track_lines_up_at_both_ends(self, tracks):
        reference, made = tracks
        assert made["corrected"].size == reference.size
        assert abs(self.lag_ms(reference, made["corrected"], 0)) < 1.0
        assert abs(self.lag_ms(reference, made["corrected"], 145)) < 1.0

    def test_without_it_one_end_walks_off(self, tracks):
        """The single offset is the average over the overlap, so the walk shows at the ends."""
        reference, made = tracks
        assert max(abs(self.lag_ms(reference, made["as_before"], at)) for at in (0, 145)) > 15.0


class TestTiming:
    def test_frames_are_counted_on_the_recording_clock(self, monkeypatch):
        scale = 1.0002
        entry = {
            "path": "/x/room.mp4", "name": "room.mp4", "kind": "video", "subject_id": None, "duration_s": 60.0,
            "sample_rate": None, "fps": 25.0, "offset_s": 1.5, "clock_scale": scale, "drift_ppm": 200.0,
            "offset_confidence": 0.9,
        }
        info = prep_module.PrepInfo(recording_id="r1", t_origin_s=0.0, duration_s=60.0, sample_rate=16000, media=[entry])
        monkeypatch.setattr(prep_module, "load_prep", lambda config, rid: info)
        monkeypatch.setattr(
            prep_module, "load_video_prep",
            lambda config, rid: {"room.mp4": {"fps": 25.0, "frames_start_s": 0.2, "path": Path("prep/room.mp4.mp4")}},
        )
        room = MediaFile(path=Path("raw/r1/room.mp4"), kind="video")
        apply_prep_timing(None, SimpleNamespace(recording_id="r1", media=[room]))
        assert room.fps == pytest.approx(25.0 / scale)
        # frame 1000 is 40 s into the copy, 40.2 s into the file, 1.5 + 40.2 x scale on the recording clock
        assert 1000 / room.fps + room.offset_s == pytest.approx(1.5 + 40.2 * scale)

    def test_a_naming_click_is_placed_on_the_recording_clock(self):
        item = MediaFile(path=Path("raw/r1/room.mp4"), kind="video", offset_s=2.0, clock_scale=1.001)
        video = {"first_picture_s": 0.0, "frames_start_s": 0.0, "size": [640, 480], "prepared_size": None}
        click = PoseExtractor._click(item, video, {"person": "child", "at": 100.0, "x": 0.5, "y": 0.5})
        assert click.t == pytest.approx(2.0 + 100.0 * 1.001)


class TestPrepWarnings:
    def warnings(self, tmp_path, monkeypatch, result):
        path = write_offset_table([AlignResult(reference="room.wav", target="room.wav", offset_s=0.0, confidence=1.0), result], tmp_path / "o.json")
        extractor = MediaPrepExtractor(SimpleNamespace(align=AlignSection()))
        monkeypatch.setattr(extractor, "_offset_table", lambda recording: path)
        recording = Recording(
            recording_id="r1", root=tmp_path, media=[MediaFile(path=tmp_path / name, kind="audio") for name in ("room.wav", "mic.wav")],
        )
        found = []
        extractor._apply_offsets(recording, found)
        return [(w.check, w.severity) for w in found]

    def test_a_corrected_drift_is_reported_for_a_look(self, tmp_path, monkeypatch):
        assert self.warnings(tmp_path, monkeypatch, result_for()) == [("clock_drift", "info")]

    def test_a_drift_left_uncorrected_is_a_warning(self, tmp_path, monkeypatch):
        assert self.warnings(tmp_path, monkeypatch, result_for(drift_intercept_s=None)) == [("clock_drift", "warn")]


class TestElan:
    @pytest.fixture
    def exported(self, tmp_path, caplog):
        pytest.importorskip("pympi")
        import pandas as pd

        from macaw.interop.eaf import export_eaf
        from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame

        media = [
            MediaFile(path=tmp_path / "room.wav", kind="audio"),
            MediaFile(path=tmp_path / "early.mp4", kind="video", offset_s=-3.25),
            MediaFile(path=tmp_path / "late.wav", kind="audio", offset_s=2.0),
            MediaFile(path=tmp_path / "slow.wav", kind="audio", offset_s=-1.0, clock_scale=1.0002, duration_s=600.0),
        ]
        empty = coerce_frame(pd.DataFrame(columns=list(ANNOTATION_SCHEMA)), ANNOTATION_SCHEMA, "annotations")
        with caplog.at_level(logging.WARNING):
            path = export_eaf(empty, SimpleNamespace(recording_id="r1", media=media), tmp_path / "r1.eaf")
        return path.read_text(encoding="utf-8"), caplog.text

    def test_a_file_that_started_first_skips_its_start(self, exported):
        text, _ = exported
        assert 'TIME_ORIGIN="3250"' in text and 'TIME_ORIGIN="1000"' in text
        assert text.count("TIME_ORIGIN") == 2

    def test_what_elan_cannot_show_is_said(self, exported):
        _, log = exported
        assert "late.wav starts 2.000 s after" in log
        assert "slow.wav's clock was corrected for drift" in log and "120 ms" in log


def test_methods_say_how_many_files_were_corrected():
    from macaw.config.schema import MacawConfig
    from macaw.project.manifest import FeatureManifest, FeatureSpec
    from macaw.qc.boilerplate import describe, load_references, render
    from macaw.store.schema import Scale

    config = MacawConfig.model_validate({"project": {"name": "t", "raw_dir": "raw", "output_dir": "out"}})
    ran = FeatureManifest()
    ran.extend([FeatureSpec(
        name="n_media", description="", unit=None, scale=Scale.RECORDING, group="prep", extractor="prep.media",
        extractor_version="0.5.0",
    )])
    text, _ = render(describe(ran, config, drifted=(1, 1, 4)), load_references())
    assert "more than 20 ms over its overlap with the reference (1 file, in 1 of the 4 recordings), its clock was rescaled" in text
    text, _ = render(describe(ran, config, drifted=(3, 2, 2)), load_references())
    assert "(three files, in each recording)" in text
    for drifted in ((0, 0, 4), None):
        assert "rescaled" not in render(describe(ran, config, drifted=drifted), load_references())[0]
