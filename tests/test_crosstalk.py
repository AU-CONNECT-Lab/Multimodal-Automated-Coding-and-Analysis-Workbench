from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macaw.extractors.audio.content.diarization import suppress_crosstalk

from conftest import SAMPLE_RATE

# what ASR run on each track separately returns: every event appears on both tracks
SEGMENTS = pd.DataFrame(
    [
        {"subject_id": "A", "t_start": 0.0, "t_end": 2.0, "tag": "A solo, own track"},
        {"subject_id": "B", "t_start": 0.0, "t_end": 2.0, "tag": "A solo, bleed"},
        {"subject_id": "A", "t_start": 3.0, "t_end": 5.0, "tag": "B solo, bleed"},
        {"subject_id": "B", "t_start": 3.0, "t_end": 5.0, "tag": "B solo, own track"},
        {"subject_id": "A", "t_start": 6.0, "t_end": 7.0, "tag": "both talking, A"},
        {"subject_id": "B", "t_start": 6.0, "t_end": 7.0, "tag": "both talking, B"},
    ]
)


def _voice(rng, duration_s, amplitude):
    """Noise shaped into syllable-sized bursts, close enough to speech for an energy test."""
    n = int(duration_s * SAMPLE_RATE)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * np.arange(n) / SAMPLE_RATE)
    return amplitude * rng.standard_normal(n) * envelope


def _place(track, signal, start_s):
    first = int(start_s * SAMPLE_RATE)
    track[first : first + signal.size] += signal


@pytest.fixture
def two_tracks(tmp_path, write_wav):
    """A talks 0-2, B talks 3-5, both talk 6-7. Each track carries the other person's
    voice attenuated by bleed_db."""

    def build(bleed_db, total_s=9.0):
        rng = np.random.default_rng(0)
        n = int(total_s * SAMPLE_RATE)
        a_track, b_track = np.zeros(n), np.zeros(n)
        gain = 10.0 ** (-bleed_db / 20.0)

        a_solo = _voice(rng, 2.0, 0.5)
        _place(a_track, a_solo, 0.0)
        _place(b_track, a_solo * gain, 0.0)

        b_solo = _voice(rng, 2.0, 0.5)
        _place(a_track, b_solo * gain, 3.0)
        _place(b_track, b_solo, 3.0)

        a_both, b_both = _voice(rng, 1.0, 0.5), _voice(rng, 1.0, 0.5)
        _place(a_track, a_both, 6.0)
        _place(a_track, b_both * gain, 6.0)
        _place(b_track, b_both, 6.0)
        _place(b_track, a_both * gain, 6.0)

        return {
            "A": write_wav(tmp_path / "A.wav", a_track),
            "B": write_wav(tmp_path / "B.wav", b_track),
        }

    return build


def _flags(tracks, margin_db=6.0, segments=SEGMENTS):
    out = suppress_crosstalk(segments, tracks, margin_db=margin_db)
    return dict(zip(out["tag"], out["is_crosstalk"]))


def test_bleed_is_dropped_and_real_speech_is_not(two_tracks):
    flags = _flags(two_tracks(12.0))
    assert flags["A solo, bleed"]
    assert flags["B solo, bleed"]
    assert not flags["A solo, own track"]
    assert not flags["B solo, own track"]


def test_simultaneous_speech_survives(two_tracks):
    """The failure that would matter most: suppression must not eat genuine overlap."""
    flags = _flags(two_tracks(12.0))
    assert not flags["both talking, A"]
    assert not flags["both talking, B"]


@pytest.mark.parametrize(
    "bleed_db, dropped",
    [(3.0, False), (5.0, False), (6.0, False), (7.0, True), (12.0, True)],
)
def test_margin_boundary(two_tracks, bleed_db, dropped):
    """The comparison is strictly greater, so bleed at exactly the margin stays."""
    assert _flags(two_tracks(bleed_db))["A solo, bleed"] is dropped


def test_margin_is_configurable(two_tracks):
    assert not any(_flags(two_tracks(12.0), margin_db=20.0).values())


def test_single_track_flags_nothing(two_tracks):
    tracks = two_tracks(12.0)
    out = suppress_crosstalk(SEGMENTS, {"A": tracks["A"]}, margin_db=6.0)
    assert not out["is_crosstalk"].any()


def test_empty_segments(two_tracks):
    out = suppress_crosstalk(SEGMENTS.iloc[:0], two_tracks(12.0), margin_db=6.0)
    assert len(out) == 0
    assert "is_crosstalk" in out.columns


def test_speaker_without_a_track_is_left_alone(two_tracks):
    stranger = pd.DataFrame([{"subject_id": "C", "t_start": 0.0, "t_end": 1.0, "tag": "C"}])
    out = suppress_crosstalk(stranger, two_tracks(12.0), margin_db=6.0)
    assert not out["is_crosstalk"].any()


def test_silent_track(two_tracks, tmp_path, write_wav):
    """Speech claimed on a track that recorded nothing is bleed, and the other person's
    real speech is untouched by it."""
    tracks = two_tracks(12.0)
    tracks["B"] = write_wav(tmp_path / "B.wav", np.zeros(int(9.0 * SAMPLE_RATE)))
    flags = _flags(tracks, segments=SEGMENTS.iloc[[0, 1]])
    assert flags["A solo, bleed"]
    assert not flags["A solo, own track"]
