from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import TextSection, TurnsSection
from macaw.exceptions import ConfigError
from macaw.extractors.audio.content import asr as asr_module
from macaw.extractors.audio.content import diarization as diarization_module
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, load_transcript, write_transcript
from macaw.extractors.audio.content.diarization import OTHER, speaker_names, turn_takers
from macaw.extractors.prep import PrepInfo
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store import _io
from macaw.turns.entropy import EntropyExtractor
from macaw.turns.segmentation import (
    TURN_SCHEMA, TURNS_NAME, TurnStructureExtractor, load_turns, silences, utterances_to_turns,
)
from macaw.turns.transitions import TransitionExtractor, classify_overlaps, detect_transitions

from conftest import SAMPLE_RATE


@pytest.fixture
def project(tmp_path):
    """A parent and a child on one room microphone, with room for one more voice."""
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    (raw / "003_room.wav").write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("text", {}).update(diarization="pyannote", extra_voices=1)
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SimpleNamespace(config=config, resolved=lambda *overrides: resolve_config(config, cli_overrides=list(overrides)))


NAMES = {"S0": "parent", "S1": "child", "S2": None}


def test_a_voice_left_out_speaks_as_other_only_when_extra_voices_are_allowed(project, monkeypatch):
    monkeypatch.setattr(diarization_module, "voice_names", lambda config, rid: dict(NAMES))
    assert speaker_names(project.resolved(), "003") == {**NAMES, "S2": OTHER}
    assert speaker_names(project.resolved("text.extra_voices=0"), "003") == NAMES
    assert speaker_names(project.resolved("text.diarization=per_track"), "003") == NAMES


def test_other_holds_the_floor_only_when_extra_voices_are_allowed(project):
    config = project.resolved()
    recording = build_recording("003", config)
    assert turn_takers(config, recording) == ["parent", "child", OTHER]
    assert turn_takers(project.resolved("text.extra_voices=0"), recording) == ["parent", "child"]


@pytest.mark.parametrize("role", ["other", "none"])
def test_a_role_cannot_take_the_name_of_a_voice_that_is_nobodys(project, role):
    with pytest.raises(ConfigError, match=f"{role} cannot be a role"):
        project.resolved(f"project.roles=[parent, {role}]")


def test_an_outsiders_bleed_into_an_own_microphone_is_dropped(project, monkeypatch):
    prep = PrepInfo(recording_id="003", t_origin_s=0.0, duration_s=10.0, sample_rate=SAMPLE_RATE, tracks={"parent": "parent.wav"})
    regular = pd.DataFrame(
        {"label": ["S0", "S2"], "t_start": [0.0, 5.0], "t_end": [4.0, 9.0], "exclusive": [False, False]}
    )
    monkeypatch.setattr(asr_module, "_track_clock_turns", lambda config, rid, prep, exclusive=True: regular)
    monkeypatch.setattr(diarization_module, "voice_names", lambda config, rid: {"S0": "parent", "S2": None})
    own = pd.DataFrame({"subject_id": ["parent"] * 2, "t_start": [0.5, 5.5], "t_end": [3.5, 8.0]})
    for overrides, kept in (((), [0.5]), (("text.extra_voices=0",), [0.5, 5.5])):
        config = project.resolved(*overrides)
        result = ASRExtractor(config)._drop_room_bleed(build_recording("003", config), prep, own, [])
        assert result["t_start"].tolist() == kept


TRANSCRIPT = [
    ("parent", 10.0, 12.0, "what did you do today"),
    (OTHER, 12.3, 13.0, "sorry one second"),
    ("parent", 13.2, 14.0, "go on"),
    ("child", 16.0, 17.0, "nothing"),
    (OTHER, 17.2, 19.5, "ok you can carry on now"),
    ("child", 21.0, 22.0, "fine"),
]


def test_an_outsider_ends_a_turn_and_their_talk_is_not_silence(make_transcript):
    transcript = make_transcript(TRANSCRIPT)
    turns = utterances_to_turns(transcript, TurnsSection())
    assert turns["subject_id"].tolist() == ["parent", OTHER, "parent", "child", OTHER, "child"]
    quiet = silences(transcript, min_s=TurnsSection().silence_min_s)
    assert quiet[["t_start", "t_end"]].values.tolist() == [[14.0, 16.0], [19.5, 21.0]]


def test_the_peoples_measures_never_see_other(project, make_transcript):
    config = project.resolved()
    layout = layout_for(config)
    transcript = make_transcript(TRANSCRIPT).assign(
        recording_id="003", utterance_index=0, confidence=np.nan, avg_logprob=np.nan, no_speech_prob=np.nan, language="en"
    )
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(transcript, directory / TRANSCRIPT_NAME)
    assert OTHER not in set(load_transcript(config, "003")["subject_id"])
    assert load_transcript(config, "003", others=True)["subject_id"].tolist().count(OTHER) == 2

    turns = utterances_to_turns(transcript, config.turns).assign(recording_id="003")
    directory = layout.artifacts_dir("003", TurnStructureExtractor.name)
    directory.mkdir(parents=True)
    _io.write_file(turns, directory / TURNS_NAME, TURN_SCHEMA, "turns")
    assert OTHER not in set(load_turns(config, "003")["subject_id"])
    assert len(load_turns(config, "003", others=True)) == 6


def test_other_is_a_state_of_its_own_in_transitions_and_entropy(make_turns, make_recording, make_prep):
    cfg = TurnsSection()
    config = SimpleNamespace(turns=cfg, text=TextSection(diarization="pyannote", extra_voices=1))
    turns = make_turns([(who, start, end) for who, start, end, _ in TRANSCRIPT])
    transitions = classify_overlaps(detect_transitions(turns, cfg), interruption_s=cfg.interruption_overlap_s)
    transitions["recording_id"] = "sub-003_ses-12mo"
    recording = make_recording(["parent", "child"])
    prep = make_prep(t_origin_s=10.0, duration_s=60.0)

    rows = TransitionExtractor(config)._rows(recording, turns, transitions, prep)
    counts = rows[rows.feature == "transition_count"]
    assert len(counts) == 9
    pair = counts[(counts.subject_id == "parent") & (counts.subject_id_b == OTHER)]
    assert pair["value"].tolist() == [1.0]

    rows = EntropyExtractor(config)._rows(recording, turns, transitions, prep)
    whole = rows[rows.subject_id.isna()].set_index("feature")["value"]
    assert whole["transition_entropy_norm"] == pytest.approx(whole["transition_entropy_bits"] / math.log2(9))
    assert set(rows.subject_id.dropna()) == {"parent", "child", OTHER}


def test_other_keeps_a_tier_of_its_own_in_elan_and_comes_back(project, make_transcript, tmp_path):
    pytest.importorskip("pympi")
    from macaw.exceptions import EafError
    from macaw.interop.transcript_eaf import export_transcript_eaf, read_transcript_eaf

    config = project.resolved()
    recording = build_recording("003", config)
    transcript = make_transcript(TRANSCRIPT[:2])
    path = export_transcript_eaf(transcript, transcript.iloc[:0].assign(reason="", detail=""), recording, tmp_path / "t.eaf")
    back = read_transcript_eaf(path, turn_takers(config, recording))
    assert back["subject_id"].tolist() == ["parent", OTHER]
    with pytest.raises(EafError, match="not a subject"):
        read_transcript_eaf(path, turn_takers(project.resolved("text.extra_voices=0"), recording))


def test_an_undecided_voice_is_said_to_take_turns_as_other(project, monkeypatch):
    from macaw.extractors.audio.content import speaker_id

    config = project.resolved("text.voice_review=skip")
    recording = build_recording("003", config)
    monkeypatch.setattr(diarization_module, "read_voices", lambda config, rid: {"S0": {}, "S1": {}})
    monkeypatch.setattr(speaker_id, "suggest_voices", lambda config, recording: {"people": {"S0": "child", "S1": None}})
    gate = diarization_module.SpeakerReviewGate(config)
    gate.clear_by_itself(recording)
    assert "taking turns as other" in gate.notes(recording)[0].message
