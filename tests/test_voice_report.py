from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("parselmouth")

from macaw.config.schema import ProsodySection  # noqa: E402
from macaw.extractors.audio.prosody import voice_report as module  # noqa: E402
from macaw.extractors.audio.prosody.voice_report import (  # noqa: E402
    VoiceReportExtractor,
    f0_slope,
    pitch_statistics,
    voice_report,
)
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame  # noqa: E402

RATE = 16000


def glide(seconds, start_hz, end_hz, *, jitter=0.0, seed=0):
    """A harmonic buzz whose pitch moves linearly, with optional cycle-to-cycle period noise."""
    n = int(seconds * RATE)
    hz = np.linspace(start_hz, end_hz, n)
    if jitter:
        periods = np.random.default_rng(seed).normal(1.0, jitter, n // 50 + 1)
        hz = hz / np.repeat(periods, 50)[:n]
    phase = 2 * np.pi * np.cumsum(hz) / RATE
    return sum(0.3 / k * np.sin(k * phase) for k in range(1, 12))


def report(audio):
    return voice_report(audio, RATE, f0_min=75.0, f0_max=500.0, step_s=0.01)


class TestPitch:
    def test_a_rise_has_a_positive_slope_in_semitones(self):
        values = report(glide(1.0, 150.0, 200.0))
        assert values["turn_f0_slope_st_per_s"] == pytest.approx(12 * np.log2(200 / 150), rel=0.15)
        assert values["turn_f0_median_hz"] == pytest.approx(175.0, rel=0.05)
        assert values["turn_f0_range_st"] == pytest.approx(12 * np.log2(200 / 150), rel=0.15)

    def test_a_fall_has_a_negative_slope(self):
        assert report(glide(1.0, 220.0, 160.0))["turn_f0_slope_st_per_s"] < 0

    def test_unvoiced_frames_leave_the_statistics_and_count_against_voicing(self):
        f0 = np.array([0.0, 100.0, 100.0, 0.0, 200.0, np.nan])
        stats = pitch_statistics(f0, np.arange(6) * 0.01)
        assert stats["turn_voiced_fraction"] == pytest.approx(0.5)
        assert stats["turn_f0_min_hz"] == 100.0 and stats["turn_f0_max_hz"] == 200.0
        assert stats["turn_f0_range_st"] == pytest.approx(12.0)

    def test_too_few_voiced_frames_have_no_slope(self):
        assert np.isnan(f0_slope(np.array([0.0, 120.0, 130.0]), np.arange(3) * 0.01))


class TestVoiceQuality:
    def test_period_noise_raises_jitter(self):
        steady = report(glide(1.0, 150.0, 150.0))
        noisy = report(glide(1.0, 150.0, 150.0, jitter=0.02))
        assert noisy["jitter_local"] > 3 * steady["jitter_local"]
        assert noisy["jitter_ddp"] == pytest.approx(3 * noisy["jitter_rap"], rel=0.01)

    def test_added_noise_lowers_hnr_and_cpps(self):
        clean = glide(1.0, 150.0, 150.0)
        noisy = clean + np.random.default_rng(1).normal(0, 0.1, clean.size)
        a, b = report(clean), report(noisy)
        assert b["hnr_db"] < a["hnr_db"] - 5
        assert b["cpps_db"] < a["cpps_db"]
        assert b["shimmer_local"] > a["shimmer_local"]

    def test_silence_and_a_sliver_measure_nothing(self):
        for audio in (np.zeros(RATE), glide(0.02, 150.0, 150.0)):
            values = report(audio)
            assert all(np.isnan(values[name]) for name in values if name != "turn_voiced_fraction")


def test_the_extractor_writes_one_row_per_turn_and_measure(tmp_path, monkeypatch):
    path = tmp_path / "parent.wav"
    audio = np.concatenate([glide(1.0, 150.0, 200.0), np.zeros(RATE), glide(1.0, 220.0, 160.0)])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    prep = SimpleNamespace(t_origin_s=10.0, tracks={"parent": "parent.wav"}, track_path=lambda subject: path)
    turns = pd.DataFrame(
        {"subject_id": ["parent", "parent", "parent", "child"], "t_start": [10.0, 11.2, 12.0, 11.0], "t_end": [11.0, 11.8, 13.0, 12.0]}
    )
    monkeypatch.setattr(module, "load_prep", lambda config, rid: prep)
    monkeypatch.setattr(module, "load_turns", lambda config, rid: turns)
    recording = SimpleNamespace(
        recording_id="r1",
        subjects=[SimpleNamespace(subject_id="parent", role="parent"), SimpleNamespace(subject_id="child", role="child")],
        entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None},
    )
    result = VoiceReportExtractor(SimpleNamespace(prosody=ProsodySection())).run(recording, tmp_path)
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    assert {spec.name for spec in result.specs} == set(result.rows.feature)
    slope = result.rows[result.rows.feature == "turn_f0_slope_st_per_s"].sort_values("t_start")
    assert slope.t_start.tolist() == [10.0, 11.2, 12.0]
    assert slope.value.iloc[0] > 0 and np.isnan(slope.value.iloc[1]) and slope.value.iloc[2] < 0
    assert [w.message.startswith("1 of 3") for w in result.warnings] == [True]


def test_without_parselmouth_it_says_how_to_install(monkeypatch):
    import sys

    from macaw.exceptions import BackendNotAvailableError

    monkeypatch.setitem(sys.modules, "parselmouth", None)
    with pytest.raises(BackendNotAvailableError, match="parselmouth"):
        VoiceReportExtractor(SimpleNamespace(prosody=ProsodySection())).check_available()
