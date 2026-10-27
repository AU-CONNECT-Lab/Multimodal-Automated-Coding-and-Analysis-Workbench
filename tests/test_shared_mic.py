from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.extractors import get_extractor
from macaw.extractors import prep as prep_module
from macaw.extractors.audio.content import diarization as diarization_module
from macaw.extractors.audio.content.asr import assign_voices, split_by_voice
from macaw.extractors.audio.content.diarization import (
    VOICES_NAME,
    inside,
    solo_spans,
    turn_pieces,
    DiarizationExtractor,
    SpeakerReviewGate,
    describe_voices,
    overlapped_fraction,
)
from macaw.extractors.audio.content.speaker_id import describe_suggestion, match_voices
from macaw.extractors.prep import MediaPrepExtractor, PrepInfo
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

from conftest import SAMPLE_RATE


@pytest.fixture
def project(tmp_path):
    """One recording of a parent and a child on a single room microphone."""
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    (raw / "003_room.wav").write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("text", {})["diarization"] = "pyannote"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SimpleNamespace(config=config, resolved=lambda: resolve_config(config))


def turns(*rows, exclusive=True):
    return pd.DataFrame(
        [{"label": label, "t_start": start, "t_end": end, "exclusive": exclusive} for label, start, end in rows]
    )


def test_prep_cuts_the_shared_microphone_on_the_same_clock(project, monkeypatch, tmp_path):
    config = project.resolved()
    recording = build_recording("003", config)
    cut = []
    (tmp_path / "work").mkdir()
    extractor = MediaPrepExtractor(config)
    monkeypatch.setattr(extractor, "_probe_media", lambda recording: None)
    monkeypatch.setattr(extractor, "_apply_offsets", lambda recording, warnings: None)
    monkeypatch.setattr(prep_module, "_window", lambda recording: (0.0, 10.0))
    monkeypatch.setattr(prep_module, "_extract_track", lambda src, dest, **kwargs: cut.append((src.name, dest.name)))
    result = extractor.run(recording, tmp_path / "work")
    info = json.loads((tmp_path / "work" / "prep.json").read_text(encoding="utf-8"))
    assert cut == [("003_room.wav", "shared_003_room.wav")]
    assert info["shared_tracks"] == {"003_room.wav": "shared_003_room.wav"}
    missing = [warning for warning in result.warnings if warning.check == "prep_missing_track"]
    assert {warning.severity for warning in missing} == {"info"}


def test_overlap_is_a_share_of_the_time_anyone_talks():
    assert overlapped_fraction(turns(("A", 0.0, 4.0), ("B", 2.0, 6.0), exclusive=False)) == pytest.approx(1 / 3)


def test_each_voice_gets_clips_from_its_longest_turns(tmp_path, write_wav):
    track = write_wav(tmp_path / "room.wav", 0.1 * np.ones(12 * SAMPLE_RATE))
    found = turns(("SPEAKER_00", 100.0, 108.0), ("SPEAKER_01", 109.0, 110.0), ("SPEAKER_01", 110.5, 111.0))
    voices = describe_voices(found, track, 100.0, tmp_path)
    assert voices["SPEAKER_00"]["speech_s"] == pytest.approx(8.0)
    clip = voices["SPEAKER_00"]["clips"][0]
    assert clip["t_end"] - clip["t_start"] == pytest.approx(5.0)
    assert (tmp_path / clip["file"]).stat().st_size > 5 * SAMPLE_RATE
    assert len(voices["SPEAKER_01"]["clips"]) == 2


class TestWordsToPeople:
    TURNS = turns(("S0", 0.0, 2.0), ("S1", 2.2, 4.0), ("S2", 6.0, 7.0))
    NAMES = {"S0": "parent", "S1": "child", "S2": None}

    def test_a_word_goes_to_the_turn_it_overlaps_or_one_just_beside_it(self):
        spans = pd.DataFrame({"t_start": [0.5, 4.3, 5.0], "t_end": [0.9, 4.6, 5.3]})
        labels = assign_voices(spans, self.TURNS)["label"]
        assert labels[:2].tolist() == ["S0", "S1"] and pd.isna(labels[2])

    def test_one_segment_spoken_by_two_people_becomes_two_utterances(self):
        utterances = pd.DataFrame(
            {"utterance_index": [0, 1], "t_start": [0.0, 6.0], "t_end": [4.0, 7.0], "text": ["", ""],
             "confidence": [np.nan, np.nan], "language": ["en", "en"]}
        )
        words = pd.DataFrame(
            {"utterance_index": [0, 0, 0, 0, 1], "t_start": [0.1, 0.8, 2.5, 3.2, 6.2], "t_end": [0.6, 1.5, 3.0, 3.8, 6.8],
             "word": ["shall", "we?", "no", "thanks", "bang"], "score": [0.9, 0.8, 0.7, 0.6, 0.5]}
        )
        split, placed, unplaced = split_by_voice(utterances, words, self.TURNS, self.NAMES)
        assert split[["subject_id", "text", "utterance_index"]].values.tolist() == [
            ["parent", "shall we?", 0], ["child", "no thanks", 0]
        ]
        assert split["t_start"].tolist() == [0.1, 2.5] and split["confidence"].tolist() == pytest.approx([0.85, 0.65])
        assert unplaced == 1 and placed["subject_id"].tolist() == ["parent", "parent", "child", "child"]


def test_mouth_movement_names_the_voice_it_moves_with():
    rng = np.random.default_rng(1)
    blocks = np.arange(0, 240, 10.0)
    order = rng.permutation(np.repeat(["S0", "S1"], len(blocks) // 2))
    found = turns(*[(label, start, start + 9.0) for label, start in zip(order, blocks)])
    t = np.arange(0, 240, 0.04)
    talking = {label: talk_mask(found, label, t) for label in ("S0", "S1")}
    mouth = pd.concat([
        pd.DataFrame({"subject_id": "parent", "t_start": t, "value": talking["S1"] + 0.3 * rng.random(t.size)}),
        pd.DataFrame({"subject_id": "child", "t_start": t, "value": talking["S0"] + 0.3 * rng.random(t.size)}),
    ])
    suggestion = match_voices(found, mouth)
    assert suggestion["people"] == {"S0": "child", "S1": "parent"}
    assert suggestion["evidence"]["parent"]["rho"] > 0.5 > suggestion["evidence"]["parent"]["rival_rho"]


def test_a_voice_matched_by_elimination_is_marked_for_listening():
    suggestion = {
        "people": {"S0": "parent", "S1": "child"},
        "evidence": {"parent": {"rho": 0.5, "z": 6.0}, "child": {"rho": 0.0, "z": -0.1}},
    }
    shown = describe_suggestion(suggestion)
    assert "S0 is parent (rho 0.50, z 6.0)" in shown and "S1 is child (rho 0.00, z -0.1, weak: listen to it)" in shown


def talk_mask(found, label, t):
    own = found[found["label"] == label]
    return ((t[:, None] >= own["t_start"].to_numpy()) & (t[:, None] < own["t_end"].to_numpy())).any(axis=1).astype(float)


class TestReview:
    def voices(self, project, *labels):
        directory = layout_for(project.resolved()).artifacts_dir("003", DiarizationExtractor.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / VOICES_NAME).write_text(json.dumps({label: {"speech_s": 10.0, "clips": []} for label in labels}))

    def pending(self, project):
        config = project.resolved()
        return SpeakerReviewGate(config).pending(build_recording("003", config))

    def accept(self, project, *args):
        return CliRunner().invoke(app, ["accept-voices", "-c", str(project.config), "-r", "003", "--by", "ann", *args])

    def test_it_holds_until_every_voice_is_named(self, project):
        self.voices(project, "SPEAKER_00", "SPEAKER_01", "SPEAKER_02")
        assert "accept-voices" in self.pending(project)
        outcome = self.accept(project, "--rename", "SPEAKER_00=child,SPEAKER_01=parent")
        assert outcome.exit_code != 0 and "SPEAKER_02 not named" in str(outcome.exception)
        outcome = self.accept(project, "--rename", "SPEAKER_00=child,SPEAKER_01=parent,SPEAKER_02=none")
        assert outcome.exit_code == 0, outcome.output
        assert self.pending(project) is None
        named = json.loads(layout_for(project.resolved()).voice_owners_path("003").read_text(encoding="utf-8"))
        assert named["people"]["SPEAKER_02"] == "none" and named["by"].startswith("ann ")

    def test_a_new_diarization_with_other_voices_holds_it_again(self, project):
        self.voices(project, "SPEAKER_00", "SPEAKER_01")
        self.accept(project, "--rename", "SPEAKER_00=child,SPEAKER_01=parent")
        self.voices(project, "SPEAKER_00", "SPEAKER_01", "SPEAKER_02")
        assert self.pending(project) is not None

    def test_a_name_that_is_not_a_subject_is_refused(self, project):
        self.voices(project, "SPEAKER_00")
        assert "grandma" in str(self.accept(project, "--rename", "SPEAKER_00=grandma").exception)

    def test_without_faces_there_is_nothing_to_keep(self, project):
        self.voices(project, "SPEAKER_00")
        assert "--rename" in str(self.accept(project).exception)

    def test_own_microphones_never_wait(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["text"]["diarization"] = "per_track"
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert self.pending(project) is None


def test_with_own_microphones_diarization_reads_nothing(project, monkeypatch, tmp_path):
    data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    data["text"]["diarization"] = "per_track"
    project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
    config = project.resolved()
    monkeypatch.setattr(diarization_module, "load_prep", lambda config, rid: PrepInfo(recording_id=rid, t_origin_s=0.0, duration_s=5.0, sample_rate=16000))
    result = DiarizationExtractor(config).run(build_recording("003", config), tmp_path)
    assert result.rows.set_index("feature").value["n_shared_tracks_diarized"] == 0.0 and result.artifacts == {}


def test_voices_are_named_before_anything_is_anyones_only_on_a_shared_microphone(project):
    shared = project.resolved()
    own = shared.model_copy(update={"text": shared.text.model_copy(update={"diarization": "per_track"})})
    for name in ("text.asr", "prosody.frame"):
        assert SpeakerReviewGate.name in get_extractor(name).dependencies(shared)
        assert SpeakerReviewGate.name not in get_extractor(name).dependencies(own)
    assert get_extractor("text.review").requires == (DiarizationExtractor.name,)


def test_solo_spans_leave_out_overlap_and_unnamed_voices():
    found = turns(("S0", 0.0, 4.0), ("S1", 3.0, 6.0), ("S2", 7.0, 8.0), ("S0", 6.0, 6.5), exclusive=False)
    spans = solo_spans(found, {"S0": "parent", "S1": "child", "S2": None})
    assert spans["parent"].values.tolist() == [[0.0, 3.0], [6.0, 6.5]]
    assert spans["child"].values.tolist() == [[4.0, 6.0]]
    assert inside(np.array([2.9, 3.5, 6.2, 7.5]), spans["parent"]).tolist() == [True, False, True, False]


def test_pitch_off_a_shared_microphone_is_each_persons_own_solo_stretch(project, monkeypatch, tmp_path, write_wav):
    from macaw.extractors.audio.prosody import frame as frame_module
    from macaw.extractors.audio.prosody.frame import FrameProsodyExtractor

    t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    voice = np.where(t < 2.0, 0.3 * np.sin(2 * np.pi * 120.0 * t), 0.3 * np.sin(2 * np.pi * 300.0 * t))
    track = write_wav(tmp_path / "shared.wav", voice)
    config = project.resolved().model_copy(update={"prosody": project.resolved().prosody.model_copy(update={"backend": "builtin"})})
    spans = {"parent": pd.DataFrame({"t_start": [10.2], "t_end": [11.8]}), "child": pd.DataFrame({"t_start": [12.2], "t_end": [13.8]})}
    prep = PrepInfo(recording_id="003", t_origin_s=10.0, duration_s=4.0, sample_rate=SAMPLE_RATE)
    monkeypatch.setattr(frame_module, "load_prep", lambda config, rid: prep)
    monkeypatch.setattr(frame_module, "voice_sources", lambda config, recording, prep: {who: (track, found) for who, found in spans.items()})
    rows = FrameProsodyExtractor(config).run(build_recording("003", config), tmp_path).rows
    median = rows[rows.feature == "f0_median_hz"].set_index("subject_id").value
    assert median["parent"] == pytest.approx(120.0, rel=0.05) and median["child"] == pytest.approx(300.0, rel=0.05)
    pitch = rows[rows.feature == "f0_hz"]
    parent = pitch[pitch.subject_id == "parent"]
    assert parent.t_start.min() >= 10.2 and parent.t_start.max() < 11.8


def test_a_turn_off_a_shared_microphone_keeps_only_its_solo_stretches():
    audio = np.arange(10 * 100, dtype=float)
    spans = pd.DataFrame({"t_start": [10.0, 14.0], "t_end": [12.0, 15.0]})
    pieces, kept = turn_pieces(audio, 100.0, 10.0, np.array([11.0, 16.0]), np.array([15.0, 17.0]), spans)
    assert pieces[0].tolist() == list(range(100, 200)) + list(range(400, 500))
    assert kept == [0.5, 0.0] and pieces[1].size == 0
    whole, share = turn_pieces(audio, 100.0, 10.0, np.array([11.0]), np.array([12.0]), None)
    assert whole[0].size == 100 and share == [1.0]
