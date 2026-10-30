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
from macaw.exceptions import ConfigError
from macaw.extractors.audio.content.asr import (
    SHARED_ASR,
    VOICE_LINES_NAME,
    VOICE_PITCH_NAME,
    assign_voices,
    split_by_voice,
    voice_lines,
    voice_pitch,
    write_transcript,
)
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
from macaw.extractors.audio.content.speaker_id import by_pitch, by_words, describe_suggestion, match_voices, telling_lines, voice_hints
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


@pytest.mark.parametrize(
    "files, camera_has_sound, shared",
    [
        (["003_room.mp4"], True, {"003_room.mp4": "shared_003_room.wav"}),
        (["003_room.mp4"], False, {}),
        (["003_room.mp4", "003_parent.wav", "003_child.wav"], True, {}),
        (["003_room.mp4", "003_room.wav"], True, {"003_room.wav": "shared_003_room.wav"}),
    ],
    ids=["camera-stands-in", "silent-camera", "everyone-has-a-mic", "room-mic-first"],
)
def test_the_cameras_sound_is_the_shared_microphone_only_when_there_is_none(
    project, monkeypatch, tmp_path, files, camera_has_sound, shared
):
    raw = tmp_path / "raw" / "003"
    (raw / "003_room.wav").unlink()
    for name in files:
        (raw / name).write_bytes(b"")
    config = project.resolved()
    recording = build_recording("003", config)
    (tmp_path / "work").mkdir()

    def probe(recording):
        for item in recording.media:
            item.sample_rate = 48000.0 if item.kind == "audio" or camera_has_sound else None

    extractor = MediaPrepExtractor(config)
    monkeypatch.setattr(extractor, "_probe_media", probe)
    monkeypatch.setattr(extractor, "_apply_offsets", lambda recording, warnings: None)
    monkeypatch.setattr(prep_module, "_window", lambda recording: (0.0, 10.0))
    monkeypatch.setattr(prep_module, "_extract_track", lambda src, dest, **kwargs: None)
    result = extractor.run(recording, tmp_path / "work")
    info = json.loads((tmp_path / "work" / "prep.json").read_text(encoding="utf-8"))
    assert info["shared_tracks"] == shared
    stand_in = [warning.message for warning in result.warnings if warning.check == "prep_camera_sound"]
    assert stand_in == (["No shared audio file; the sound of 003_room.mp4 is the shared microphone."] if "003_room.mp4" in shared else [])


class TestExtraVoices:
    def test_pyannote_looks_for_the_people_and_up_to_the_extra_voices(self):
        assert diarization_module.speaker_counts(2, 0) == {"num_speakers": 2}
        assert diarization_module.speaker_counts(2, 1) == {"min_speakers": 2, "max_speakers": 3}

    def test_the_setting_reaches_the_diarizer_and_the_manifest(self, project, monkeypatch, tmp_path):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["text"]["extra_voices"] = 1
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        config = project.resolved()
        asked = []

        class Fake:
            def diarize(self, path, **counts):
                asked.append(counts)
                return turns(("SPEAKER_00", 0.0, 4.0), ("SPEAKER_01", 4.0, 8.0), ("SPEAKER_02", 8.0, 9.0))

        prep = PrepInfo(recording_id="003", t_origin_s=0.0, duration_s=10.0, sample_rate=16000, shared_tracks={"003_room.wav": "shared_003_room.wav"}, root=tmp_path)
        monkeypatch.setattr(diarization_module, "load_prep", lambda config, rid: prep)
        monkeypatch.setattr(diarization_module, "resolve_diarizer", lambda config: Fake())
        monkeypatch.setattr(diarization_module, "describe_voices", lambda turns, path, origin, workspace: {})
        result = DiarizationExtractor(config).run(build_recording("003", config), tmp_path)
        assert asked == [{"n_speakers": 2, "extra_speakers": 1}]
        assert {spec.params["extra_voices"] for spec in result.specs} == {1}


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


def three_voices():
    rng = np.random.default_rng(1)
    blocks = np.arange(0, 270, 10.0)
    order = rng.permutation(np.repeat(["S0", "S1", "S2"], len(blocks) // 3))
    found = turns(*[(label, start, start + 9.0) for label, start in zip(order, blocks)])
    t = np.arange(0, 270, 0.04)
    talking = {label: talk_mask(found, label, t) for label in ("S0", "S1", "S2")}
    return found, t, talking, rng


def test_each_persons_signal_names_the_voice_it_follows():
    found, t, talking, rng = three_voices()
    signals = {
        "parent": (t, talking["S1"] + 0.3 * rng.random(t.size), "mouth movement"),
        "child": (t, 20 * talking["S0"] + rng.random(t.size), "own microphone"),
        "sibling": (t, talking["S2"] + 0.3 * rng.random(t.size), "mouth movement"),
    }
    suggestion = match_voices(found, signals, ["parent", "child", "sibling"])
    assert suggestion["people"] == {"S0": "child", "S1": "parent", "S2": "sibling"}
    assert suggestion["evidence"]["parent"]["rho"] > 0.5 > suggestion["evidence"]["parent"]["rival_rho"]
    assert suggestion["evidence"]["child"]["source"] == "own microphone"


def test_the_one_person_and_voice_left_are_paired_by_elimination():
    found, t, talking, rng = three_voices()
    signals = {
        "parent": (t, talking["S1"] + 0.3 * rng.random(t.size), "own microphone"),
        "child": (t, talking["S0"] + 0.3 * rng.random(t.size), "own microphone"),
    }
    suggestion = match_voices(found, signals, ["parent", "child", "sibling"])
    assert suggestion["people"]["S2"] == "sibling" and suggestion["evidence"]["sibling"]["source"] == "elimination"
    assert "S2 is sibling (the one voice and person left: listen to it)" in describe_suggestion(suggestion)
    two_left = match_voices(found, {"parent": signals["parent"]}, ["parent", "child", "sibling"])
    assert two_left["people"]["S1"] == "parent" and two_left["people"]["S0"] is None is two_left["people"]["S2"]
    assert "S0 is one of child, sibling (listen to it)" in describe_suggestion(two_left)


def test_a_signal_that_barely_follows_its_voice_is_marked_for_listening():
    suggestion = {
        "people": {"S0": "parent", "S1": "child"},
        "evidence": {"parent": {"rho": 0.5, "z": 6.0, "source": "mouth movement"}, "child": {"rho": 0.0, "z": -0.1, "source": "mouth movement"}},
    }
    shown = describe_suggestion(suggestion)
    assert "S0 is parent (mouth movement, rho 0.50, z 6.0)" in shown
    assert "S1 is child (mouth movement, rho 0.00, z -0.1, weak: listen to it)" in shown
    suggestion["evidence"]["child"]["z"] = float("nan")
    assert "S1 is child (mouth movement, rho 0.00, too little of it to test: listen to it)" in describe_suggestion(suggestion)


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

    def review(self, project, mode, people, monkeypatch, evidence=None):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["text"]["voice_review"] = mode
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        from macaw.extractors.audio.content import speaker_id

        suggestion = {"people": people, "evidence": evidence or {}, "unheard": []}
        monkeypatch.setattr(speaker_id, "suggest_voices", lambda config, recording: suggestion)
        self.voices(project, *people)
        config = project.resolved()
        gate, recording = SpeakerReviewGate(config), build_recording("003", config)
        gate.clear_by_itself(recording)
        return gate, recording, layout_for(config).voice_owners_path("003")

    def test_always_waits_for_a_person_even_with_a_complete_suggestion(self, project, monkeypatch):
        gate, recording, path = self.review(project, "always", {"S0": "child", "S1": "parent"}, monkeypatch)
        assert not path.exists() and gate.pending(recording) is not None

    def test_flagged_keeps_a_complete_suggestion_and_says_so(self, project, monkeypatch):
        gate, recording, path = self.review(project, "flagged", {"S0": "child", "S1": "parent"}, monkeypatch)
        assert gate.pending(recording) is None
        assert json.loads(path.read_text(encoding="utf-8")) == {"people": {"S0": "child", "S1": "parent"}, "by": "automatic"}
        assert [(note.check, note.severity) for note in gate.notes(recording)] == [("voices_unreviewed", "info")]

    def test_flagged_waits_when_the_evidence_disagrees(self, project, monkeypatch):
        evidence = {"child": {"label": "S0", "source": "words", "n": 1, "lines": ["mommy!"], "disputed": "pitch points to S1"}}
        gate, recording, path = self.review(project, "flagged", {"S0": "child", "S1": "parent"}, monkeypatch, evidence)
        assert not path.exists() and gate.pending(recording) is not None

    def test_flagged_waits_when_a_voice_is_undecided(self, project, monkeypatch):
        gate, recording, path = self.review(project, "flagged", {"S0": "child", "S1": None}, monkeypatch)
        assert not path.exists() and gate.pending(recording) is not None

    def test_skip_leaves_an_undecided_voice_out_with_a_warning(self, project, monkeypatch):
        gate, recording, path = self.review(project, "skip", {"S0": "child", "S1": None}, monkeypatch)
        assert gate.pending(recording) is None
        assert json.loads(path.read_text(encoding="utf-8"))["people"] == {"S0": "child", "S1": "none"}
        note = gate.notes(recording)[0]
        assert note.severity == "warn" and "S1" in note.message

    def test_the_report_lists_the_voices_and_what_was_left_out_and_who_decided(self, project, monkeypatch):
        from macaw.extractors.audio.content.asr import DROPPED_NAME, DROPPED_SCHEMA, ASRExtractor
        from macaw.qc.report import write_recording_report
        from macaw.store import _io

        self.review(project, "flagged", {"S0": "child", "S1": "parent"}, monkeypatch)
        config = project.resolved()
        directory = layout_for(config).artifacts_dir("003", ASRExtractor.name)
        directory.mkdir(parents=True)
        dropped = pd.DataFrame({
            "recording_id": ["003"], "subject_id": ["parent"], "t_start": [12.0], "t_end": [13.0],
            "text": ["no no no"], "reason": ["room bleed"], "detail": ["in the voice of child"],
        })
        _io.write_file(dropped, directory / DROPPED_NAME, DROPPED_SCHEMA, "dropped utterances")
        text = write_recording_report(config, "003")[0].read_text(encoding="utf-8")
        assert "Whose voice is which" in text and ">software<" in text
        assert "S0</td>" in text and "child" in text
        assert "no no no" in text and "in the voice of child" in text and "1 as room bleed" in text

    def test_the_report_and_review_page_show_what_each_voice_says_and_its_pitch(self, project, monkeypatch):
        pytest.importorskip("dash")
        from macaw.interface.pages import review
        from macaw.qc.report import write_recording_report

        evidence = {"child": {"label": "S0", "source": "words", "n": 1, "lines": ["Mommy, look!"]}}
        self.review(project, "always", {"S0": "child", "S1": "parent"}, monkeypatch, evidence)
        config = project.resolved()
        root = layout_for(config).artifacts_dir("003", SHARED_ASR)
        root.mkdir(parents=True)
        spoken = [("S0", "Mommy, look!"), ("S0", "yes"), ("S0", "no"), ("S0", "the red one is mine"), ("S0", "ok"), ("S1", "where does it go")]
        lines = said(*spoken).assign(
            recording_id="003", utterance_index=range(len(spoken)), t_start=np.arange(len(spoken), dtype=float),
            t_end=np.arange(len(spoken)) + 0.5, n_words=2, confidence=np.nan, avg_logprob=np.nan, no_speech_prob=np.nan, language="en",
        )
        write_transcript(lines, root / VOICE_LINES_NAME)
        pitch = {"S0": {"speech_s": 5.0, "median_hz": 310.0}, "S1": {"speech_s": 4.0, "median_hz": None}}
        (root / VOICE_PITCH_NAME).write_text(json.dumps(pitch), encoding="utf-8")

        text = write_recording_report(config, "003")[0].read_text(encoding="utf-8")
        assert "310 Hz" in text and "the red one is mine" in text and "all 5 lines" in text
        assert "child (1 line(s) such as" in text and "where does it go" in text
        shown = json.dumps(review.body(project.config, "003"), default=lambda item: item.to_plotly_json())
        assert "310 Hz" in shown and "the red one is mine" in shown and "all 5 lines" in shown and "child (1 line(s) such as" in shown

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

    spans["child"] = pd.DataFrame({"t_start": [12.2], "t_end": [12.4]})
    warnings = FrameProsodyExtractor(config).run(build_recording("003", config), tmp_path).warnings
    quiet = [warning for warning in warnings if warning.check == "voiced_fraction"]
    assert [warning.subject_id for warning in quiet] == ["child"]
    assert "say little on their own" in quiet[0].message and "track may be silent" not in quiet[0].message


def test_a_turn_off_a_shared_microphone_keeps_only_its_solo_stretches():
    audio = np.arange(10 * 100, dtype=float)
    spans = pd.DataFrame({"t_start": [10.0, 14.0], "t_end": [12.0, 15.0]})
    pieces, kept = turn_pieces(audio, 100.0, 10.0, np.array([11.0, 16.0]), np.array([15.0, 17.0]), spans)
    assert pieces[0].tolist() == list(range(100, 200)) + list(range(400, 500))
    assert kept == [0.5, 0.0] and pieces[1].size == 0
    whole, share = turn_pieces(audio, 100.0, 10.0, np.array([11.0]), np.array([12.0]), None)
    assert whole[0].size == 100 and share == [1.0]


class TestMixed:
    TURNS = turns(("S0", 0.0, 4.0), ("S1", 5.0, 9.0))
    NAMES = {"S0": "parent", "S1": "child"}

    def test_words_of_someone_with_their_own_mic_are_left_to_it(self):
        utterances = pd.DataFrame({"utterance_index": [0], "t_start": [0.0], "t_end": [9.0], "text": [""], "confidence": [np.nan], "language": ["en"]})
        words = pd.DataFrame({"utterance_index": [0, 0], "t_start": [1.0, 6.0], "t_end": [1.5, 6.5], "word": ["hi", "yo"], "score": [0.9, 0.9]})
        split, _, unplaced = split_by_voice(utterances, words, self.TURNS, self.NAMES, elsewhere=frozenset({"parent"}))
        assert split["subject_id"].tolist() == ["child"] and unplaced == 0

    def test_an_own_mic_utterance_in_the_voice_of_someone_without_one_is_dropped(self, project, monkeypatch):
        from macaw.extractors.audio.content import asr as asr_module
        from macaw.extractors.audio.content.asr import ASRExtractor

        prep = PrepInfo(recording_id="003", t_origin_s=0.0, duration_s=10.0, sample_rate=SAMPLE_RATE, tracks={"parent": "parent.wav"})
        regular = turns(("S0", 0.0, 4.0), ("S1", 5.0, 9.0), ("S0", 6.0, 6.4), exclusive=False)
        monkeypatch.setattr(asr_module, "_track_clock_turns", lambda config, rid, prep, exclusive=True: regular)
        monkeypatch.setattr(diarization_module, "voice_names", lambda config, rid: self.NAMES)
        own = pd.DataFrame({"subject_id": ["parent"] * 3, "t_start": [0.5, 5.5, 6.0], "t_end": [3.5, 8.0, 6.5]})
        config = project.resolved()
        warnings = []
        extractor = ASRExtractor(config)
        kept = extractor._drop_room_bleed(build_recording("003", config), prep, own, warnings)
        # the last one also lies in the child's voice, but the owner talks over most of it
        assert kept["t_start"].tolist() == [0.5, 6.0]
        assert warnings[0].check == "asr_room_bleed" and "1 of 3" in warnings[0].message
        dropped = pd.concat(extractor._dropped)
        assert dropped[["t_start", "reason", "detail"]].values.tolist() == [[5.5, "room bleed", "in the voice of child"]]

    def test_each_person_is_measured_on_their_own_mic_or_their_solo_stretches(self, project, monkeypatch, tmp_path):
        prep = PrepInfo(
            recording_id="003", t_origin_s=0.0, duration_s=10.0, sample_rate=SAMPLE_RATE,
            tracks={"parent": "parent.wav"}, shared_tracks={"003_room.wav": "shared_003_room.wav"}, root=tmp_path,
        )
        monkeypatch.setattr(diarization_module, "read_diarization", lambda config, rid: turns(("S0", 0.0, 4.0), ("S1", 5.0, 9.0), exclusive=False))
        monkeypatch.setattr(diarization_module, "voice_names", lambda config, rid: self.NAMES)
        config = project.resolved()
        sources = diarization_module.voice_sources(config, build_recording("003", config), prep)
        assert sources["parent"] == (tmp_path / "parent.wav", None)
        assert sources["child"][0] == tmp_path / "shared_003_room.wav"
        assert sources["child"][1].values.tolist() == [[5.0, 9.0]]


def test_the_review_waits_for_faces_read_in_the_same_run_without_pulling_them_in(project):
    from macaw.execution.dag import build_dag

    config = project.resolved()
    alone = build_dag(config, ["text.asr"])
    assert "face.movement" not in alone.nodes
    assert alone.nodes["text.review"].depends_on == {DiarizationExtractor.name}
    with_faces = build_dag(config, ["text.asr", "face.movement"])
    assert "face.movement" in with_faces.nodes["text.review"].depends_on
    order = with_faces.topological_order()
    assert order.index("face.movement") < order.index("text.review") < order.index("text.asr")


def hint(says, role):
    return SimpleNamespace(says=says, role=role)


def said(*rows):
    return pd.DataFrame({"subject_id": [label for label, _ in rows], "text": [text for _, text in rows]})


ROLES = {"p1": "parent", "c1": "child"}
CHILD_SAYS = hint(r"^\s*mommy[,.!?]", "child")
PARENT_SAYS = hint(r"\bmommy's (gonna|going)\b", "parent")


class TestHints:
    def test_a_voice_goes_to_the_role_its_lines_match(self):
        lines = said(("S0", "Mommy, look!"), ("S0", "mommy."), ("S1", "Mommy's gonna help"), ("S1", "put it there"))
        found = by_words(lines, [CHILD_SAYS, PARENT_SAYS], ROLES)
        assert {who: found[who]["label"] for who in found} == {"c1": "S0", "p1": "S1"}
        assert found["c1"]["n"] == 2 and found["c1"]["lines"][0] == "Mommy, look!"

    def test_a_voice_is_shown_by_its_longest_lines_in_the_order_spoken(self):
        lines = said(("S0", "yes"), ("S0", "the red one"), ("S0", "ok"), ("S0", "where does it go"), ("S1", "a much longer line here"))
        assert telling_lines(lines.assign(t_start=range(5)), "S0", n=2) == ["the red one", "where does it go"]

    def test_a_voice_matching_two_roles_equally_says_nothing(self):
        lines = said(("S0", "Mommy, look!"), ("S0", "mommy's going to"))
        assert by_words(lines, [CHILD_SAYS, PARENT_SAYS], ROLES) == {}

    def test_pitch_orders_the_voices_that_speak_most(self):
        pitch = {
            "S0": {"speech_s": 50.0, "median_hz": 310.0}, "S1": {"speech_s": 60.0, "median_hz": 210.0},
            "S2": {"speech_s": 2.0, "median_hz": 120.0},
        }
        found = by_pitch(pitch, ["parent", "child"], ROLES)
        assert {who: found[who]["label"] for who in found} == {"p1": "S1", "c1": "S0"}

    def test_voices_too_close_in_pitch_are_not_told_apart(self):
        pitch = {"S0": {"speech_s": 50.0, "median_hz": 230.0}, "S1": {"speech_s": 60.0, "median_hz": 210.0}}
        assert by_pitch(pitch, ["parent", "child"], ROLES) == {}

    def test_hints_place_people_no_signal_placed(self):
        found = turns(("S0", 0.0, 5.0), ("S1", 6.0, 10.0))
        hints = {"c1": {"label": "S0", "source": "words", "n": 2, "lines": ["Mommy, look!"]}}
        suggestion = match_voices(found, {}, ["p1", "c1"], hints=hints)
        assert suggestion["people"] == {"S0": "c1", "S1": "p1"}
        assert suggestion["evidence"]["p1"]["source"] == "elimination"
        assert 'S0 is c1 (2 line(s) such as "Mommy, look!")' in describe_suggestion(suggestion)

    def test_a_hint_against_a_signal_is_shown_as_disputed(self):
        found, t, talking, rng = three_voices()
        signals = {"parent": (t, talking["S1"] + 0.3 * rng.random(t.size), "own microphone")}
        hints = {"parent": {"label": "S0", "source": "words", "n": 1, "lines": ["mommy's gonna"]}}
        suggestion = match_voices(found, signals, ["parent", "child", "sibling"], hints=hints)
        assert suggestion["people"]["S1"] == "parent"
        assert suggestion["evidence"]["parent"]["disputed"] == "words point to S0"
        assert "S1 is parent (own microphone, but words point to S0: listen to it)" in describe_suggestion(suggestion)

    def test_the_settings_are_read_off_what_the_shared_step_left(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["identity"] = {"voice_hints": [{"says": CHILD_SAYS.says, "role": "child"}], "pitch_order": ["parent", "child"]}
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        config = project.resolved()
        root = layout_for(config).artifacts_dir("003", SHARED_ASR)
        root.mkdir(parents=True)
        lines = said(("S1", "Mommy, look!"), ("S0", "where does it go")).assign(
            recording_id="003", utterance_index=0, t_start=0.0, t_end=1.0, n_words=2, confidence=np.nan,
            avg_logprob=np.nan, no_speech_prob=np.nan, language="en",
        )
        write_transcript(lines, root / VOICE_LINES_NAME)
        # pitch alone says the child is S0; the words win, and the pitch is kept as a doubt
        pitch = {"S0": {"speech_s": 50.0, "median_hz": 300.0}, "S1": {"speech_s": 40.0, "median_hz": 200.0}}
        (root / VOICE_PITCH_NAME).write_text(json.dumps(pitch), encoding="utf-8")
        found = voice_hints(config, build_recording("003", config))
        assert found["child"]["label"] == "S1" and found["child"]["disputed"] == "pitch points to S0"
        assert "parent" not in found

    def test_a_role_the_project_does_not_have_is_refused(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["identity"] = {"voice_hints": [{"says": "mommy", "role": "mother"}]}
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="identity names mother"):
            project.resolved()

    def test_a_pattern_python_cannot_read_is_refused(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["identity"] = {"voice_hints": [{"says": "mommy(", "role": "child"}]}
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ConfigError, match="not a pattern Python can read"):
            project.resolved()


class TestSharedASR:
    def test_it_runs_before_the_voices_are_named_and_the_transcript_after(self, project):
        shared = project.resolved()
        own = shared.model_copy(update={"text": shared.text.model_copy(update={"diarization": "per_track"})})
        assert SHARED_ASR in get_extractor("text.asr").dependencies(shared)
        assert SHARED_ASR not in get_extractor("text.asr").dependencies(own)
        assert SpeakerReviewGate.name not in get_extractor(SHARED_ASR).dependencies(shared)
        assert get_extractor(SHARED_ASR).wanted_by_default(shared) and not get_extractor(SHARED_ASR).wanted_by_default(own)

    def test_each_voices_lines_are_split_off_one_recognised_segment(self):
        utterances = pd.DataFrame({
            "utterance_index": [0], "t_start": [0.0], "t_end": [9.0], "text": ["mommy look where"], "confidence": [np.nan],
            "language": ["en"], "recording_id": ["003"], "subject_id": [None], "n_words": [3],
        })
        words = pd.DataFrame({
            "utterance_index": [0, 0, 0], "t_start": [1.0, 1.6, 6.0], "t_end": [1.5, 2.0, 6.5], "word": ["mommy", "look", "where"],
            "score": [0.9] * 3, "recording_id": ["003"] * 3, "subject_id": [None] * 3,
        })
        lines = voice_lines(utterances, words, turns(("S0", 0.0, 4.0), ("S1", 5.0, 9.0)), "003", 10.0)
        assert lines[["subject_id", "text", "t_start"]].values.tolist() == [["S0", "mommy look", 11.0], ["S1", "where", 16.0]]

    def test_each_voices_pitch_is_its_median_where_it_speaks_alone(self, project, tmp_path, write_wav):
        t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
        voice = np.where(t < 2.0, 0.3 * np.sin(2 * np.pi * 120.0 * t), 0.3 * np.sin(2 * np.pi * 300.0 * t))
        track = write_wav(tmp_path / "shared.wav", voice)
        pitch = voice_pitch(project.resolved(), track, turns(("S0", 0.2, 1.8), ("S1", 2.2, 3.8)))
        assert pitch["S0"]["median_hz"] == pytest.approx(120.0, rel=0.05)
        assert pitch["S1"]["median_hz"] == pytest.approx(300.0, rel=0.05)
        assert pitch["S1"]["speech_s"] == pytest.approx(1.6)
