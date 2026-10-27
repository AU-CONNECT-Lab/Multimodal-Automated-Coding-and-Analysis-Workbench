from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("opensmile")

from macaw.config.schema import ProsodySection, TextSection  # noqa: E402
from macaw.extractors.audio.prosody import standard_sets as module  # noqa: E402
from macaw.extractors.audio.prosody.standard_sets import StandardSetExtractor, functionals  # noqa: E402
from macaw.store.schema import LONG_TABLE_SCHEMA, validate_frame  # noqa: E402

RATE = 16000
F0 = "egemaps_F0semitoneFrom27.5Hz_sma3nz_amean"


def voice(seconds, hz):
    """A buzz with harmonics, close enough to a voice for a pitch tracker."""
    t = np.arange(int(seconds * RATE)) / RATE
    return sum(0.3 / k * np.sin(2 * np.pi * k * hz * t) for k in range(1, 6))


def pieces(audio, starts, ends):
    return [audio[int(a * RATE) : int(b * RATE)] for a, b in zip(starts, ends)]


def track():
    rng = np.random.default_rng(0)
    return np.concatenate([voice(2.0, 220.0), rng.normal(0, 0.05, 2 * RATE), np.zeros(RATE)])


class TestFunctionals:
    def test_a_voiced_turn_has_its_pitch(self):
        table, too_short = functionals(pieces(track(), [0.0], [2.0]), RATE, "eGeMAPSv02")
        assert table.shape == (1, 88) and too_short == 0
        assert table[F0].iloc[0] == pytest.approx(12 * np.log2(220.0 / 27.5), abs=1.0)

    def test_a_turn_without_voice_has_no_pitch_rather_than_zero(self):
        table, _ = functionals(pieces(track(), [2.2], [3.8]), RATE, "eGeMAPSv02")
        assert np.isnan(table[F0].iloc[0])
        assert table["egemaps_F1amplitudeLogRelF0_sma3nz_amean"].isna().all()
        assert table["egemaps_loudness_sma3_amean"].notna().all()
        assert table["egemaps_alphaRatioUV_sma3nz_amean"].notna().all()

    def test_a_turn_too_short_is_counted_and_left_empty(self):
        table, too_short = functionals(pieces(track(), [0.0, 1.0], [0.03, 2.0]), RATE, "eGeMAPSv02")
        assert too_short == 1
        assert table.iloc[0].isna().all() and table.iloc[1].notna().any()


def test_the_extractor_writes_one_row_per_turn_and_feature(tmp_path, monkeypatch):
    path = tmp_path / "parent.wav"
    import wave

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(track(), -1, 1) * 32767).astype("<i2").tobytes())
    prep = SimpleNamespace(t_origin_s=10.0, tracks={"parent": "parent.wav"}, track_path=lambda subject: path)
    turns = pd.DataFrame({"subject_id": ["parent", "parent", "child"], "t_start": [10.0, 12.2, 11.0], "t_end": [12.0, 13.8, 12.0]})
    monkeypatch.setattr(module, "load_prep", lambda config, rid: prep)
    monkeypatch.setattr(module, "load_turns", lambda config, rid: turns)
    recording = SimpleNamespace(
        recording_id="r1", entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None}
    )
    result = StandardSetExtractor(SimpleNamespace(prosody=ProsodySection(), text=TextSection())).run(recording, tmp_path)
    validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
    assert {spec.name for spec in result.specs} == set(result.rows.feature)
    assert len(result.rows) == 2 * 88
    first = result.rows[(result.rows.feature == F0)].sort_values("t_start")
    assert first.t_start.tolist() == [10.0, 12.2] and np.isnan(first.value.iloc[1])


def test_without_opensmile_it_says_why_it_is_not_installed(monkeypatch):
    import sys

    from macaw.exceptions import BackendNotAvailableError

    monkeypatch.setitem(sys.modules, "opensmile", None)
    with pytest.raises(BackendNotAvailableError, match="licence"):
        StandardSetExtractor(SimpleNamespace(prosody=ProsodySection())).check_available()
