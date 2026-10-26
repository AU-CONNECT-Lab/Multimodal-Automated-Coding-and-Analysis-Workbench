from __future__ import annotations

import wave

import numpy as np
import pandas as pd

from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.track_repair import choose_people
from macaw.extractors.video.pose.voice_motion import (
    HEAD_JOINTS,
    bin_edges,
    head_motion,
    match_voices,
    shift_z,
    speech_activity,
)

RNG = np.random.default_rng(7)
N_BINS = 240


def talking(n=N_BINS, share=0.3):
    return (RNG.random(n) < share).astype(float)


def moving_with(voice, coupling):
    return RNG.random(voice.size) + coupling * voice


class TestShiftZ:
    def test_a_head_that_moves_with_the_voice_stands_out(self):
        voice = talking()
        assert shift_z(voice, moving_with(voice, 1.0), 20) > 5

    def test_an_unrelated_head_does_not(self):
        assert abs(shift_z(talking(), RNG.random(N_BINS), 20)) < 3

    def test_a_silent_voice_says_nothing(self):
        assert np.isnan(shift_z(np.zeros(N_BINS), RNG.random(N_BINS), 20))


class TestMatch:
    def test_each_voice_goes_to_the_person_moving_with_it(self):
        voices = {"a": talking(), "b": talking(), "c": talking()}
        people = [moving_with(voices["c"], 1.0), moving_with(voices["a"], 1.0), moving_with(voices["b"], 1.0)]
        match = match_voices(voices, people)
        assert match.people == {"a": 1, "b": 2, "c": 0}
        assert sorted(match.clear) == ["a", "b", "c"]

    def test_a_weak_coupling_is_offered_but_not_clear(self):
        voices = {"a": talking()}
        match = match_voices(voices, [moving_with(voices["a"], 0.12), RNG.random(N_BINS)])
        assert match.clear == []

    def test_a_voice_that_never_talks_is_left_out(self):
        match = match_voices({"a": np.zeros(N_BINS)}, [RNG.random(N_BINS)])
        assert match.people == {}


def write_wav(path, samples, rate=16000):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class TestSpeechActivity:
    def test_a_neighbour_heard_faintly_does_not_count_as_talking(self, tmp_path):
        rate, t = 16000, np.arange(16000 * 4) / 16000
        voice = np.where(t < 2, np.sin(2 * np.pi * 180 * t) * 0.5, 0.0)
        noise = RNG.normal(0, 1e-4, t.size)
        write_wav(tmp_path / "a.wav", voice + noise)
        write_wav(tmp_path / "b.wav", voice * 0.05 + noise)
        activity = speech_activity({"a": tmp_path / "a.wav", "b": tmp_path / "b.wav"}, 10.0, bin_edges(10.0, 14.0))
        assert activity["a"][:4].min() > 0.9 and activity["a"][4:].max() < 0.1
        assert activity["b"].max() < 0.1


class TestHeadMotion:
    def test_speed_is_per_body_size_and_binned_on_the_recording_clock(self):
        rows = []
        for frame in range(50):
            for joint in HEAD_JOINTS + ("left_ankle",):
                y = 300.0 if joint == "left_ankle" else 100.0
                rows.append({"frame": frame, "t": 5.0 + frame / 25, "joint": joint, "x": 100.0 + 2.0 * frame, "y": y})
        speed = head_motion(pd.DataFrame(rows), bin_edges(5.0, 7.0))
        assert np.allclose(speed, 2.0 / 200.0)


def three_at_a_table(voices, fps=10, seconds=120.0):
    """Tracks 1, 2, 3 seated left to right; each head nods while the voice tied to it talks."""
    rows = []
    for frame in range(int(seconds * fps)):
        t = frame / fps
        for track, (x, voice) in enumerate(zip((100.0, 400.0, 700.0), voices), start=1):
            talking = voice[min(int(t / 0.5), voice.size - 1)] > 0
            nod = RNG.normal(0, 6.0 if talking else 0.5)
            for index, joint in enumerate(STANDARD_JOINTS):
                rows.append({"frame": frame, "t": t, "joint": joint, "x": x + (index % 5) * 10 + nod,
                             "y": 100.0 + (index // 5) * 30, "confidence": 0.9, "track_id": track})
    frame = pd.DataFrame(rows)
    frame["track_id"] = frame["track_id"].astype("Int64")
    return frame


class TestChoosePeople:
    def test_voices_place_people_whatever_the_configured_order(self):
        voices = {"a": talking(), "b": talking(), "c": talking()}
        keypoints = three_at_a_table([voices["c"], voices["a"], voices["b"]])
        choice = choose_people(keypoints, ["a", "b", "c"], voices=voices, edges=bin_edges(0.0, 120.0))
        assert choice.people == {"a": [2], "b": [3], "c": [1]}
        assert "by voice" in choice.doubt and "not yet checked" in choice.doubt

    def test_too_little_time_to_judge_places_no_one_by_voice(self):
        voices = {"a": talking(), "b": talking(), "c": talking()}
        keypoints = three_at_a_table([voices["c"], voices["a"], voices["b"]], seconds=12.0)
        choice = choose_people(keypoints, ["a", "b", "c"], voices={k: v[:24] for k, v in voices.items()}, edges=bin_edges(0.0, 12.0))
        assert choice.people == {"a": [1], "b": [2], "c": [3]}
        assert "configured subject order" in choice.doubt

    def test_without_voices_it_is_the_configured_order(self):
        keypoints = three_at_a_table([talking(), talking(), talking()])
        choice = choose_people(keypoints, ["a", "b", "c"])
        assert choice.people == {"a": [1], "b": [2], "c": [3]}
        assert "configured subject order" in choice.doubt

    def test_someone_far_away_all_along_beats_someone_close_for_a_moment(self):
        rows = []
        for frame in range(200):
            people = [(1, 100.0, 20.0), (2, 300.0, 20.0)] + ([(3, 500.0, 150.0)] if frame < 60 else [])
            for track, x, size in people:
                for index, joint in enumerate(STANDARD_JOINTS):
                    rows.append({"frame": frame, "t": frame / 10, "joint": joint, "x": x + (index % 5) * size / 5,
                                 "y": 100.0 + (index // 5) * size / 4, "confidence": 0.9, "track_id": track})
        keypoints = pd.DataFrame(rows).astype({"track_id": "Int64"})
        assert choose_people(keypoints, ["a", "b"]).people == {"a": [1], "b": [2]}
