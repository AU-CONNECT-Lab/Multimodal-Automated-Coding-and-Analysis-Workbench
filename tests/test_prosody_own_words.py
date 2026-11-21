from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("parselmouth")

from macaw.config.schema import ProsodySection, TextSection  # noqa: E402
from macaw.extractors.audio.content import asr  # noqa: E402
from macaw.extractors.audio.prosody import frame  # noqa: E402
from macaw.extractors.audio.prosody.frame import FrameProsodyExtractor, merge_spans, own_word_spans  # noqa: E402

RATE = 16000


def tone(seconds, hz, level):
    phase = 2 * np.pi * hz * np.arange(int(seconds * RATE)) / RATE
    return level * sum(np.sin(k * phase) / k for k in range(1, 8))


def write(path, audio):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def transcript(lines, words, monkeypatch):
    lines = pd.DataFrame(lines, columns=["subject_id", "utterance_index", "t_start", "t_end"])
    words = pd.DataFrame(words, columns=["subject_id", "utterance_index", "t_start", "t_end"])
    monkeypatch.setattr(asr, "load_transcript", lambda config, rid: lines)
    monkeypatch.setattr(asr, "load_words", lambda config, rid: words)


def test_words_close_together_join_and_far_apart_stay_apart():
    spans = merge_spans(np.array([2.0, 0.0, 1.3, 0.5]), np.array([2.5, 0.4, 1.4, 0.7]), 0.5)
    assert spans.values.tolist() == [[0.0, 0.7], [1.3, 1.4], [2.0, 2.5]]


def test_a_line_without_word_times_counts_whole(monkeypatch):
    transcript(
        [("parent", 0, 0.0, 1.0), ("parent", 1, 5.0, 6.0), ("child", 0, 2.0, 3.0)],
        [("parent", 0, 0.1, 0.3), ("parent", 0, 0.6, np.nan), ("child", 0, 2.1, 2.4)],
        monkeypatch,
    )
    spans = own_word_spans(None, "r1")
    assert spans["parent"].values.tolist() == [[0.1, 0.3], [5.0, 6.0]]
    assert spans["child"].values.tolist() == [[2.1, 2.4]]


def run_frame(tmp_path, monkeypatch, audio):
    path = tmp_path / "parent.wav"
    write(path, audio)
    prep = SimpleNamespace(t_origin_s=10.0, duration_s=audio.size / RATE, tracks={"parent": "parent.wav"}, track_path=lambda subject: path)
    monkeypatch.setattr(frame, "load_prep", lambda config, rid: prep)
    recording = SimpleNamespace(
        recording_id="r1",
        subjects=[SimpleNamespace(subject_id="parent", role="parent")],
        entities=lambda: {"group_id": "1", "session_id": None, "task": None, "run": None},
    )
    config = SimpleNamespace(prosody=ProsodySection(backend="parselmouth"), text=TextSection())
    return FrameProsodyExtractor(config).run(recording, tmp_path)


def test_a_neighbour_leaking_into_ones_microphone_is_not_ones_pitch(tmp_path, monkeypatch):
    # one second of the owner at 300 Hz, then three of a neighbour at 200 Hz, 20 dB lower
    audio = np.concatenate([tone(1.0, 300.0, 0.3), tone(3.0, 200.0, 0.03)])
    transcript([("parent", 0, 10.0, 11.0)], [("parent", 0, 10.05, 10.95)], monkeypatch)
    result = run_frame(tmp_path, monkeypatch, audio)
    rows = result.rows
    pitch = rows[rows.feature == "f0_hz"]
    assert pitch.t_start.max() < 11.0 and abs(np.median(pitch.value) - 300.0) < 10
    # the share voiced is of the frames within the words, not of the whole track
    fraction = rows[rows.feature == "voiced_fraction"].value.item()
    assert fraction > 0.8
    assert abs(rows[rows.feature == "f0_median_hz"].value.item() - 300.0) < 10


def test_no_word_of_theirs_leaves_nothing_and_says_why(tmp_path, monkeypatch):
    transcript([], [], monkeypatch)
    result = run_frame(tmp_path, monkeypatch, tone(2.0, 200.0, 0.3))
    assert result.rows.empty
    assert [warning.message for warning in result.warnings] == [
        "None of their words was transcribed, so no frame of their microphone was measured as theirs."
    ]
