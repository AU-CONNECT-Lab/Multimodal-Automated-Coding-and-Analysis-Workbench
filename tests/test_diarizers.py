from __future__ import annotations

import wave
from types import SimpleNamespace

import numpy as np
import pytest

from macaw.config.schema import TextSection
from macaw.extractors.audio.content import diarization as diarization_module
from macaw.extractors.audio.content.diarization import (
    NemotronBackend,
    PyannoteBackend,
    diarizer_note,
    named_diarizer,
    turns_from_activity,
)

STEP = 0.01


def activity(*slots: list[tuple[int, int, float]], frames: int = 100) -> np.ndarray:
    """Voice probabilities: each slot speaks with the given probability over [first, last) frames."""
    probs = np.zeros((frames, 8))
    for slot, spans in enumerate(slots):
        for first, last, p in spans:
            probs[first:last, slot] = p
    return probs


def spans(turns, label, exclusive):
    rows = turns[(turns["label"] == label) & (turns["exclusive"] == exclusive)]
    return [(round(a, 2), round(b, 2)) for a, b in zip(rows["t_start"], rows["t_end"])]


class TestTurnsFromActivity:
    def test_each_slot_becomes_a_voice_with_its_own_speech(self):
        turns = turns_from_activity(activity([(0, 30, 0.9)], [(40, 60, 0.8)]), STEP)
        assert spans(turns, "SPEAKER_00", False) == [(0.0, 0.3)]
        assert spans(turns, "SPEAKER_01", False) == [(0.4, 0.6)]

    def test_overlap_goes_to_the_likelier_voice_in_the_exclusive_copy(self):
        turns = turns_from_activity(activity([(0, 50, 0.9)], [(40, 80, 0.6)]), STEP)
        assert spans(turns, "SPEAKER_00", False) == [(0.0, 0.5)] and spans(turns, "SPEAKER_01", False) == [(0.4, 0.8)]
        assert spans(turns, "SPEAKER_00", True) == [(0.0, 0.5)] and spans(turns, "SPEAKER_01", True) == [(0.5, 0.8)]

    def test_only_the_voices_that_spoke_longest_are_kept(self):
        probs = activity([(0, 40, 0.9)], [(40, 45, 0.9)], [(50, 90, 0.9)])
        kept = turns_from_activity(probs, STEP, keep=2)
        assert sorted(kept["label"].unique()) == ["SPEAKER_00", "SPEAKER_02"]
        assert sorted(turns_from_activity(probs, STEP, keep=3)["label"].unique()) == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def test_a_slot_that_never_speaks_is_no_voice(self):
        turns = turns_from_activity(activity([(0, 20, 0.9)], [(0, 100, 0.3)]), STEP)
        assert list(turns["label"].unique()) == ["SPEAKER_00"]

    def test_silence_gives_no_turns(self):
        turns = turns_from_activity(np.zeros((50, 8)), STEP, keep=2)
        assert turns.empty and list(turns.columns) == ["t_start", "t_end", "label", "exclusive"]


class TestWhichDiarizer:
    @pytest.fixture(autouse=True)
    def nemotron_here(self, monkeypatch):
        monkeypatch.setattr(diarization_module, "nemotron_missing", lambda: "")

    def config(self, **text):
        return SimpleNamespace(text=TextSection(diarization="shared", **text))

    def test_nemotron_by_default_with_its_own_model(self):
        backend = named_diarizer(self.config())
        assert isinstance(backend, NemotronBackend) and backend.model == "nvidia/Nemotron-3-Diarization"

    def test_pyannote_with_its_own_model(self):
        backend = named_diarizer(self.config(diarizer="pyannote"))
        assert isinstance(backend, PyannoteBackend) and backend.model == "pyannote/speaker-diarization-community-1"

    def test_a_named_model_is_used(self):
        assert named_diarizer(self.config(diarization_model="org/other")).model == "org/other"

    def test_nemotron_gives_way_to_pyannote_where_it_cannot_be_had(self, monkeypatch):
        monkeypatch.setattr(diarization_module, "nemotron_missing", lambda: "No Nemotron here.")
        config = self.config(diarization_model="nvidia/other")
        backend = named_diarizer(config)
        assert isinstance(backend, PyannoteBackend) and backend.model == "pyannote/speaker-diarization-community-1"
        assert "No Nemotron here." in diarizer_note(config) and "set text.diarizer: pyannote" in diarizer_note(config)

    def test_pyannote_set_on_purpose_is_no_giving_way(self, monkeypatch):
        monkeypatch.setattr(diarization_module, "nemotron_missing", lambda: "No Nemotron here.")
        assert diarizer_note(self.config(diarizer="pyannote")) == ""

    def test_the_run_once_nemotron_can_be_had_is_a_different_one(self, monkeypatch):
        from macaw.extractors.audio.content.diarization import DiarizationExtractor

        extractor = DiarizationExtractor(self.config())
        assert extractor.extra_key(None) == ""
        monkeypatch.setattr(diarization_module, "nemotron_missing", lambda: "No Nemotron here.")
        assert extractor.extra_key(None) == "pyannote"


class TestNemotronAsksOnce:
    def test_the_further_voice_trial_reuses_the_first_pass(self, tmp_path):
        torch = pytest.importorskip("torch")
        path = tmp_path / "shared.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(np.zeros(16000, dtype="<i2").tobytes())
        probs = activity([(0, 40, 0.9)], [(40, 80, 0.9)], [(85, 95, 0.9)])

        class Inputs(dict):
            def to(self, *args, **kwargs):
                return self

        class Model:
            device, dtype, calls = "cpu", torch.float32, 0

            def __call__(self, **inputs):
                Model.calls += 1
                return SimpleNamespace(logits=torch.logit(torch.tensor(probs[None], dtype=torch.float32).clamp(1e-4, 1 - 1e-4)))

        class Processor:
            feature_extractor = SimpleNamespace(sampling_rate=16000, hop_length=160)

            def __call__(self, audio, sampling_rate):
                return Inputs()

        backend = NemotronBackend(device="cpu")
        backend._parts = (Processor(), Model())
        two = backend.diarize(path, n_speakers=2)
        three = backend.diarize(path, n_speakers=2, extra_speakers=1)
        assert Model.calls == 1
        assert two["label"].nunique() == 2 and three["label"].nunique() == 3
