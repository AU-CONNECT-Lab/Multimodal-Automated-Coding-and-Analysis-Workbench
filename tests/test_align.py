from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import scaffold_project_config
from macaw.config.schema import AlignSection
from macaw.exceptions import UnreadableMediaError
from macaw.extractors.prep import MediaPrepExtractor
from macaw.media import align as align_module
from macaw.media.align import (
    GCC_PHAT,
    MANUAL,
    SAMPLE_RATE as SR,
    UNALIGNED,
    AlignResult,
    align_recording,
    align_signals,
    apply_alignment,
    gcc_phat,
    keep_manual,
    metadata_offset,
    read_offset_table,
    write_offset_table,
)
from macaw.media.drift import drift_is_real
from macaw.media.probe import MediaInfo, StreamInfo
from macaw.project.layout import make_layout
from macaw.project.recording import MediaFile, Recording

CFG = AlignSection(drift_segment_s=30.0)


def speechlike(seconds, seed):
    """Noise gated into syllable-length bursts of varying loudness."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    gate = np.zeros(n)
    t = 0
    while t < n:
        on, off = int(rng.uniform(0.1, 0.6) * SR), int(rng.uniform(0.05, 0.8) * SR)
        gate[t : t + on] = rng.uniform(0.2, 1.0)
        t += on + off
    return rng.standard_normal(n) * gate


def other_mic(signal, seed=2):
    """The same room heard through another microphone: reverberant, duller, noisier."""
    rng = np.random.default_rng(seed)
    response = rng.standard_normal(800) * np.exp(-np.arange(800) / 160.0)
    response[0] = 3.0
    wet = np.convolve(np.convolve(signal, response)[: signal.size], np.ones(4) / 4, mode="same")
    return wet / np.abs(wet).max() + 0.05 * rng.standard_normal(wet.size)


def started_later(source, offset_s, length_s, drift_ppm=0.0):
    """Sample j heard the room at recording time offset_s + j/SR * (1 + drift)."""
    t = offset_s + np.arange(int(length_s * SR)) / SR * (1 + drift_ppm * 1e-6)
    return np.interp(t, np.arange(source.size) / SR, source, left=0.0, right=0.0)


@pytest.fixture(scope="module")
def room():
    return speechlike(330, 1)


class TestGccPhat:
    def test_a_later_start_is_a_positive_offset(self):
        signal = speechlike(60, 3)
        assert gcc_phat(signal, signal[2 * SR :], SR, 5.0)[0] == pytest.approx(2.0, abs=1 / SR)

    def test_an_earlier_start_is_a_negative_offset(self):
        signal = speechlike(60, 3)
        assert gcc_phat(signal[2 * SR :], signal, SR, 5.0)[0] == pytest.approx(-2.0, abs=1 / SR)

    def test_a_pure_tone_matches_everywhere_and_says_so(self):
        tone = np.sin(2 * np.pi * 440 * np.arange(30 * SR) / SR)
        assert gcc_phat(tone, tone[SR:], SR, 5.0)[1] < 0.1

    def test_unrelated_signals_are_not_confident(self):
        assert gcc_phat(speechlike(60, 5), speechlike(60, 6), SR, 5.0)[1] < 0.1

    def test_room_reflections_do_not_count_as_rivals(self, room):
        """A reverberant second mic is the normal case and must not be flagged."""
        assert gcc_phat(room[: 60 * SR], other_mic(room)[SR : 61 * SR], SR, 5.0)[1] > 0.6


class TestAlignSignals:
    @pytest.mark.parametrize("offset_s", [95.0, 3.25])
    def test_an_offset_beyond_the_old_thirty_second_window_is_found(self, room, offset_s):
        reference = room[: 300 * SR]
        result = align_signals(reference, started_later(other_mic(room), offset_s, 200), CFG, names=("a", "b"))
        assert result.offset_s == pytest.approx(offset_s, abs=0.002)
        assert result.confidence > CFG.min_confidence
        assert result.coarse_offset_s == pytest.approx(offset_s, abs=0.1)

    def test_a_target_that_started_first(self, room):
        lead = 0.05 * np.random.default_rng(3).standard_normal(int(12.5 * SR))
        target = np.concatenate([lead, other_mic(room)[: 200 * SR]])
        result = align_signals(room[: 300 * SR], target, CFG, names=("a", "b"))
        assert result.offset_s == pytest.approx(-12.5, abs=0.002)

    def test_drift_is_measured(self, room):
        target = started_later(other_mic(room), 20.0, 280, drift_ppm=150.0)
        result = align_signals(room[: 320 * SR], target, CFG, names=("a", "b"))
        assert result.drift_ppm == pytest.approx(150.0, abs=15.0)
        assert drift_is_real(result.drift_ppm, result.drift_r2, 100.0)

    def test_a_steady_clock_shows_no_real_drift(self, room):
        result = align_signals(room[: 300 * SR], started_later(other_mic(room), 20.0, 280), CFG, names=("a", "b"))
        assert not drift_is_real(result.drift_ppm, result.drift_r2, 100.0)

    def test_a_different_recording_is_not_confident(self, room):
        result = align_signals(room[: 300 * SR], speechlike(280, 9), CFG, names=("a", "b"))
        assert result.confidence < CFG.min_confidence


class TestMetadataOffset:
    def test_same_sign_as_offset(self):
        assert metadata_offset("2024-03-01T10:00:00Z", "2024-03-01T10:00:02.5Z") == pytest.approx(2.5)

    @pytest.mark.parametrize(
        "reference, target",
        [(None, "2024-03-01T10:00:00Z"), ("2024-03-01T10:00:00", "2024-03-01T10:00:00Z"), ("garbage", "x")],
    )
    def test_anything_unreadable_is_no_hint(self, reference, target):
        assert metadata_offset(reference, target) is None


def info(name, *, audio=True, duration=300.0, created=None):
    streams = [StreamInfo(index=0, codec_type="audio", codec_name="pcm")] if audio else []
    return MediaInfo(path=Path(name), duration_s=duration, streams=streams, creation_time=created)


@pytest.fixture
def three_files(tmp_path, room, monkeypatch):
    """A long camera without audio, a room mic, and a person's mic that started 4 s late."""
    signals = {"room.wav": room[: 300 * SR], "parent.wav": started_later(other_mic(room), 4.0, 250)}

    def fake_load(path, limit_s=None):
        if Path(path).name not in signals:
            raise UnreadableMediaError(f"{Path(path).name} cannot be decoded")
        return signals[Path(path).name]

    monkeypatch.setattr(align_module, "load_normalised", fake_load)
    media = [
        MediaFile(path=tmp_path / "camera.avi", kind="video", duration_s=900.0),
        MediaFile(path=tmp_path / "room.wav", kind="audio", duration_s=300.0),
        MediaFile(path=tmp_path / "parent.wav", kind="audio", subject_id="parent", duration_s=250.0),
    ]
    infos = {
        "camera.avi": info("camera.avi", audio=False, duration=900.0, created="2024-03-01T10:00:07Z"),
        "room.wav": info("room.wav", created="2024-03-01T10:00:00Z"),
        "parent.wav": info("parent.wav", duration=250.0),
    }
    return Recording(recording_id="r1", root=tmp_path, media=media), infos


class TestAlignRecording:
    def test_the_reference_has_audio_even_when_a_silent_camera_is_longer(self, three_files):
        recording, infos = three_files
        results = align_recording(recording, CFG, infos)
        assert {result.reference for result in results} == {"room.wav"}

    def test_a_file_without_audio_is_listed_with_its_metadata_hint(self, three_files):
        recording, infos = three_files
        camera = next(r for r in align_recording(recording, CFG, infos) if r.target == "camera.avi")
        assert camera.method == UNALIGNED
        assert camera.confidence is None
        assert camera.metadata_offset_s == pytest.approx(7.0)

    def test_a_correlated_pair_is_found_and_handed_to_the_drawer(self, three_files):
        recording, infos = three_files
        seen = []
        results = align_recording(recording, CFG, infos, on_pair=lambda a, b, r: seen.append(r.target))
        parent = next(r for r in results if r.target == "parent.wav")
        assert parent.offset_s == pytest.approx(4.0, abs=0.002)
        assert seen == ["parent.wav"]

    def test_a_file_that_cannot_be_decoded_is_reported_not_fatal(self, three_files):
        recording, infos = three_files
        recording.media.append(MediaFile(path=recording.root / "broken.wav", kind="audio", duration_s=10.0))
        infos["broken.wav"] = info("broken.wav", duration=10.0)
        broken = next(r for r in align_recording(recording, CFG, infos) if r.target == "broken.wav")
        assert broken.method == UNALIGNED
        assert "cannot be decoded" in broken.note


def entry(target, offset_s=0.0, *, method=GCC_PHAT, reference="room.wav", confidence=0.9, **extra):
    return AlignResult(reference=reference, target=target, offset_s=offset_s, confidence=confidence, method=method, **extra)


class TestOffsetTable:
    def test_the_table_holds_names_not_paths(self, tmp_path):
        """A table made on the cluster has to apply on a laptop, and hash the same on both."""
        path = write_offset_table([entry("parent.wav", 4.0, segment_offsets=[(30.0, 4.0)])], tmp_path / "o.json")
        assert "\\" not in path.read_text(encoding="utf-8") and "/" not in path.read_text(encoding="utf-8")
        assert read_offset_table(path)[0].segment_offsets == [(30.0, 4.0)]

    def test_files_are_matched_by_name_wherever_they_sit(self):
        item = MediaFile(path=Path("D:/elsewhere/parent.wav"), kind="audio")
        apply_alignment(SimpleNamespace(media=[item]), [entry("parent.wav", 4.0)])
        assert item.offset_s == 4.0

    def test_a_rerun_keeps_what_a_person_set(self):
        fresh = [entry("room.wav"), entry("parent.wav", 4.0)]
        merged, stale = keep_manual(fresh, [entry("parent.wav", 4.2, method=MANUAL, confidence=None)])
        assert next(r for r in merged if r.target == "parent.wav").offset_s == 4.2
        assert stale == []

    def test_a_manual_offset_against_another_reference_is_dropped_and_named(self):
        fresh = [entry("room.wav"), entry("parent.wav", 4.0)]
        old = [entry("parent.wav", 1.0, method=MANUAL, reference="camera.avi", confidence=None)]
        merged, stale = keep_manual(fresh, old)
        assert next(r for r in merged if r.target == "parent.wav").offset_s == 4.0
        assert stale == ["parent.wav"]


class TestPrepWarnings:
    def warnings(self, tmp_path, monkeypatch, table, names=("room.wav", "parent.wav", "camera.avi")):
        path = write_offset_table(table, tmp_path / "offsets.json")
        extractor = MediaPrepExtractor(SimpleNamespace(align=AlignSection(max_drift_ppm=100.0)))
        monkeypatch.setattr(extractor, "_offset_table", lambda recording: path)
        recording = Recording(
            recording_id="r1", root=tmp_path,
            media=[MediaFile(path=tmp_path / name, kind="audio") for name in names],
        )
        found = []
        extractor._apply_offsets(recording, found)
        return {(w.check, Path(w.message.split(" ")[0]).name) for w in found}

    def test_each_problem_is_named_once(self, tmp_path, monkeypatch):
        table = [
            entry("room.wav", confidence=1.0),
            entry("parent.wav", 4.0, confidence=0.3, drift_ppm=180.0, drift_r2=0.95),
            entry("camera.avi", method=UNALIGNED, confidence=None, note="no audio stream"),
        ]
        assert self.warnings(tmp_path, monkeypatch, table) == {
            ("alignment_confidence", "parent.wav"),
            ("clock_drift", "parent.wav"),
            ("alignment_missing", "camera.avi"),
        }

    def test_a_hand_set_offset_and_a_noisy_slope_raise_nothing(self, tmp_path, monkeypatch):
        table = [
            entry("room.wav", confidence=1.0),
            entry("parent.wav", 4.0, method=MANUAL, confidence=None, drift_ppm=500.0, drift_r2=0.1),
        ]
        assert self.warnings(tmp_path, monkeypatch, table, names=("room.wav", "parent.wav")) == set()

    def test_a_file_added_after_aligning_is_reported(self, tmp_path, monkeypatch):
        assert ("alignment_missing", "camera.avi") in self.warnings(tmp_path, monkeypatch, [entry("room.wav")])


class TestFigure:
    def test_a_pair_is_drawn_to_its_own_file(self, room, tmp_path):
        pytest.importorskip("matplotlib")
        from macaw.qc.figures import alignment_figure

        reference, target = room[: 120 * SR], started_later(other_mic(room), 4.0, 110)
        result = align_signals(reference, target, CFG, names=("room.wav", "parent.wav"))
        path = alignment_figure(reference, target, SR, result, CFG.search_window_s, tmp_path / "parent.png")
        assert path.stat().st_size > 10_000


class TestSetOffsetCommand:
    @pytest.fixture
    def project(self, tmp_path):
        recording = tmp_path / "raw" / "003"
        recording.mkdir(parents=True)
        for name in ("room.wav", "parent.wav", "camera.avi"):
            (recording / name).write_bytes(b"")
        config = tmp_path / "project.yaml"
        scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        table = make_layout(tmp_path / "out").offset_table_path("003")
        write_offset_table([entry("room.wav", confidence=1.0), entry("parent.wav", 4.0)], table)
        return SimpleNamespace(config=config, table=table)

    def invoke(self, project, *args):
        return CliRunner().invoke(app, ["set-offset", "-c", str(project.config), "-r", "003", *args])

    def test_a_hand_set_offset_replaces_the_estimate_and_keeps_the_rest(self, project):
        result = self.invoke(project, "--media", "camera.avi", "--offset", "7.0")
        assert result.exit_code == 0, result.output
        table = {r.target: r for r in read_offset_table(project.table)}
        assert (table["camera.avi"].offset_s, table["camera.avi"].method) == (7.0, MANUAL)
        assert table["parent.wav"].offset_s == 4.0

    def test_the_reference_cannot_be_moved(self, project):
        assert self.invoke(project, "--media", "room.wav", "--offset", "1.0").exit_code != 0

    def test_an_unknown_file_is_refused_with_the_known_ones(self, project):
        result = self.invoke(project, "--media", "nope.wav", "--offset", "1.0")
        assert result.exit_code != 0
        assert "parent.wav" in str(result.exception) + result.output
