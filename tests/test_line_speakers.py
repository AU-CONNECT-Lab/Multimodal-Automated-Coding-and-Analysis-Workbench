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
from macaw.extractors.audio.content import asr as asr_module
from macaw.extractors.audio.content import diarization as diarization_module
from macaw.extractors.audio.content import line_speakers as lines_module
from macaw.extractors.audio.content.asr import (
    SHARED_ASR, SHARED_UTTERANCES_NAME, SHARED_WORDS_NAME, TRANSCRIPT_SCHEMA, WORD_SCHEMA, ASRExtractor, write_transcript,
)
from macaw.extractors.audio.content.diarization import (
    CHANGED, DIARIZATION_NAME, DIARIZATION_SCHEMA, FURTHER_NAME, LINES_GATE, VOICES_NAME, DiarizationExtractor, SpeakerReviewGate,
    further_voice, read_diarization, voice_sources, with_changes,
)
from macaw.extractors.audio.content.line_speakers import (
    BY_HAND, SUGGESTED, SUGGESTION, LineSpeakersGate, changes_of, keep_suggested, split_at, spoken_lines, with_evidence, with_picks,
)
from macaw.extractors.base import PERSON, SOFTWARE
from macaw.extractors.prep import PrepInfo
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store import _io
from setup_helpers import EVERYONE, put, write_setup

from conftest import SAMPLE_RATE

ORIGIN_S = 10.0


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A parent and a child on one room microphone; the child's second line fell in the parent's voice."""
    put(tmp_path / "raw", "003", "room_mic", "a.wav")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    write_setup(config, microphones={"room_mic": {"hears": list(EVERYONE)}})
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data.setdefault("text", {})["diarization"] = "pyannote"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    resolved = resolve_config(config)
    layout = layout_for(resolved)

    prep = PrepInfo(recording_id="003", t_origin_s=ORIGIN_S, duration_s=20.0, sample_rate=16000, shared_tracks={"room_mic": "shared_room_mic.wav"}, root=tmp_path)
    for module in (lines_module, diarization_module, asr_module):
        monkeypatch.setattr(module, "load_prep", lambda config, rid: prep, raising=False)

    voices = layout.artifacts_dir("003", DiarizationExtractor.name)
    voices.mkdir(parents=True)
    spans = [("S0", 10.0, 17.5), ("S1", 18.0, 21.0)]
    turns = pd.DataFrame([
        {"track": "room_mic", "label": label, "t_start": start, "t_end": end, "exclusive": exclusive}
        for exclusive in (False, True) for label, start, end in spans
    ])
    _io.write_file(turns, voices / DIARIZATION_NAME, DIARIZATION_SCHEMA, "diarization")
    (voices / VOICES_NAME).write_text(json.dumps({"S0": {"speech_s": 7.5, "clips": []}, "S1": {"speech_s": 3.0, "clips": []}}))

    said = [(0.2, "put it there now"), (4.2, "no I want red"), (8.2, "ok here you go")]
    heard = layout.artifacts_dir("003", SHARED_ASR)
    heard.mkdir(parents=True)
    utterances = pd.DataFrame({
        "recording_id": "003", "subject_id": None, "utterance_index": range(len(said)), "t_start": [start for start, _ in said],
        "t_end": [start + 2.6 for start, _ in said], "text": [text for _, text in said], "n_words": 4, "confidence": np.nan,
        "avg_logprob": np.nan, "no_speech_prob": np.nan, "language": "en",
    })
    words = pd.DataFrame([
        {"recording_id": "003", "subject_id": None, "utterance_index": index, "t_start": start + 0.7 * k, "t_end": start + 0.7 * k + 0.5, "word": word, "score": 0.9}
        for index, (start, text) in enumerate(said) for k, word in enumerate(text.split())
    ])
    write_transcript(utterances[list(TRANSCRIPT_SCHEMA)], heard / SHARED_UTTERANCES_NAME)
    _io.write_file(words, heard / SHARED_WORDS_NAME, WORD_SCHEMA, "words")
    return SimpleNamespace(config=config, resolved=lambda: resolve_config(config), layout=layout, prep=prep)


def macaw(project, *args):
    return CliRunner().invoke(app, [*args, "-c", str(project.config), "-r", "003", "--by", "ann"])


def name_voices(project, *flags):
    outcome = macaw(project, "accept-voices", "--rename", "S0=parent,S1=child", *flags)
    assert outcome.exit_code == 0, outcome.output
    return outcome


def lines_of(project):
    config = project.resolved()
    return spoken_lines(config, build_recording("003", config))


def test_a_changed_stretch_leaves_the_voice_that_held_it():
    turns = pd.DataFrame([
        {"track": "m", "label": "S0", "t_start": 0.0, "t_end": 7.5, "exclusive": True},
        {"track": "m", "label": "S1", "t_start": 8.0, "t_end": 11.0, "exclusive": True},
    ])
    changed = with_changes(turns, [{"t_start": 4.0, "t_end": 7.0, "person": "child"}])
    assert changed[["label", "t_start", "t_end"]].values.tolist() == [
        ["S0", 0.0, 4.0], [CHANGED + "child", 4.0, 7.0], ["S0", 7.0, 7.5], ["S1", 8.0, 11.0],
    ]
    assert with_changes(turns, []) is turns


def test_lines_show_whose_voice_they_fell_in_on_the_recording_clock(project):
    name_voices(project, "--mixed")
    lines = lines_of(project)
    assert lines[["base", "person", "text"]].values.tolist() == [
        ["parent", "parent", "put it there now"], ["parent", "parent", "no I want red"], ["child", "child", "ok here you go"],
    ]
    assert lines["t_start"].round(2).tolist() == [10.2, 14.2, 18.2]


class TestEvidence:
    def lines(self):
        """Eight lines of three seconds; the fifth fell in the parent's voice and is the child's."""
        base = ["parent"] * 5 + ["child"] * 3
        starts = np.arange(8) * 4.0
        return pd.DataFrame({"t_start": starts, "t_end": starts + 3.0, "base": base, "person": base, "by": None, "why": ""})

    def mouth(self, lines, talking: list[int]) -> pd.Series:
        bins = np.arange(int(32.0 / 0.2))
        level = pd.Series(0.2, index=bins)
        for index in talking:
            level[(bins * 0.2 >= lines["t_start"][index]) & (bins * 0.2 <= lines["t_end"][index])] = 0.9
        return level

    def test_two_mouths_agreeing_against_a_line_of_two_seconds_are_sure(self):
        lines = self.lines()
        found = with_evidence(lines, {"parent": self.mouth(lines, [0, 1, 2, 3]), "child": self.mouth(lines, [4, 5, 6, 7])})
        assert found["suggested"].tolist() == [None, None, None, None, "child", None, None, None]
        assert found["sure"].tolist() == [False] * 4 + [True] + [False] * 3
        assert "the mouth of child moves and that of parent is still" in found["doubt"][4]

    def test_a_short_line_is_only_doubted(self):
        lines = self.lines()
        mouths = {"parent": self.mouth(lines, [0, 1, 2, 3]), "child": self.mouth(lines, [4, 5, 6, 7])}
        lines.loc[4, "t_end"] = lines.loc[4, "t_start"] + 1.2
        found = with_evidence(lines, mouths)
        assert found["suggested"][4] == "child" and not found["sure"][4]

    def test_one_mouth_alone_never_makes_a_line_sure(self):
        lines = self.lines()
        found = with_evidence(lines, {"parent": self.mouth(lines, [0, 1, 2, 3])})
        assert found["suggested"][4] == "child" and not found["sure"].any() and found["doubt"][4] == "the mouth of parent is still"
        found = with_evidence(lines, {"child": self.mouth(lines, [4, 5, 6, 7])})
        assert found["suggested"][4] == "child" and not found["sure"].any() and found["doubt"][4] == "the mouth of child moves"

    def test_no_face_says_nothing(self):
        found = with_evidence(self.lines(), {})
        assert found["suggested"].isna().all() and not found["sure"].any()

    def test_only_sure_suggestions_are_taken_without_a_person(self):
        lines = self.lines().assign(voice="S0", suggested=[None] * 4 + ["child"] + [None, "parent", None], sure=[False] * 4 + [True] + [False] * 3, doubt="x")
        kept = keep_suggested(lines)
        assert kept["person"].tolist() == ["parent"] * 4 + ["child"] * 4
        assert [change["by"] for change in changes_of(kept)] == [SUGGESTION]


class TestChanging:
    def test_a_person_gives_a_line_to_someone_else_and_everything_follows(self, project, tmp_path):
        name_voices(project, "--mixed")
        picks = tmp_path / "picks.csv"
        pd.DataFrame({"t_start": [14.2], "person": ["child"]}).to_csv(picks, index=False)
        outcome = macaw(project, "line-speakers", "--apply", str(picks))
        assert outcome.exit_code == 0 and "1 of 3 lines" in outcome.output, outcome.output
        kept = json.loads(project.layout.line_speakers_path("003").read_text(encoding="utf-8"))
        assert kept["route"] == BY_HAND and kept["by"].startswith("ann ")
        assert [(change["person"], change["was"], change["voice"], change["by"]) for change in kept["changes"]] == [("child", "parent", "S0", "ann")]

        config = project.resolved()
        recording = build_recording("003", config)
        assert lines_of(project)[["base", "person"]].values.tolist() == [["parent", "parent"], ["parent", "child"], ["child", "child"]]
        utterances, words = ASRExtractor(config)._from_shared_track(recording, project.prep, None, [])
        assert utterances.sort_values("t_start")[["subject_id", "text"]].values.tolist() == [
            ["parent", "put it there now"], ["child", "no I want red"], ["child", "ok here you go"],
        ]
        assert words.loc[words["word"] == "red", "subject_id"].tolist() == ["child"]
        spans = voice_sources(config, recording, project.prep)["child"][1]
        assert spans["t_start"].round(2).tolist() == [14.2, 18.0] and project.layout.line_speakers_path("003") in diarization_module.voice_inputs(config, recording, project.prep)

    def test_changes_count_only_while_the_voices_are_marked_as_mixed(self, project, tmp_path):
        name_voices(project, "--mixed")
        assert macaw(project, "line-speakers", "--at", "14.3", "--person", "child").exit_code == 0
        assert CHANGED + "child" in set(read_diarization(project.resolved(), "003", changed=True)["label"])
        name_voices(project)
        assert CHANGED + "child" not in set(read_diarization(project.resolved(), "003", changed=True)["label"])
        assert "not marked as mixed" in str(macaw(project, "line-speakers").exception)

    def test_a_line_is_split_at_the_word_being_said(self, project):
        name_voices(project, "--mixed")
        lines = split_at(lines_of(project), 15.0, "child", "ann")
        assert lines[["person", "t_start"]].round(2).values.tolist() == [["parent", 10.2], ["parent", 14.2], ["child", 14.9], ["child", 18.2]]
        assert [(change["t_start"], change["person"]) for change in changes_of(lines)] == [(pytest.approx(14.9), "child")]
        outcome = macaw(project, "line-speakers", "--at", "15.0", "--person", "child")
        assert outcome.exit_code == 0, outcome.output
        assert lines_of(project)[["person", "text"]].values.tolist() == [
            ["parent", "put it there now"], ["parent", "no"], ["child", "I want red"], ["child", "ok here you go"],
        ]
        with pytest.raises(Exception, match="no line is being said at 13"):
            split_at(lines_of(project), 13.5, "child", "ann")

    def test_a_pick_for_a_line_that_is_not_there_is_refused(self, project):
        name_voices(project, "--mixed")
        with pytest.raises(Exception, match="no line starts at 99"):
            with_picks(lines_of(project), {99.0: "child"}, "ann")

    def test_the_lines_are_listed_with_what_the_mouths_say(self, project, monkeypatch):
        name_voices(project, "--mixed")
        bins = np.arange(int(22 / 0.2))
        quiet_then = lambda start, end, level: pd.Series(np.where((bins * 0.2 >= start) & (bins * 0.2 <= end), level, 1 - level), index=bins)
        monkeypatch.setattr(lines_module, "mouths", lambda config, rid: {"parent": quiet_then(10.0, 13.5, 0.9), "child": quiet_then(14.0, 21.0, 0.9)})
        outcome = macaw(project, "line-speakers")
        assert "14.2 s parent: no I want red  [sure: child, the mouth of child moves and that of parent is still]" in outcome.output
        assert "3 lines, 0 changed, 1 doubted" in outcome.output
        assert macaw(project, "line-speakers", "--suggested").exit_code == 0
        kept = json.loads(project.layout.line_speakers_path("003").read_text(encoding="utf-8"))
        assert kept["route"] == SUGGESTED and [change["by"] for change in kept["changes"]] == [SUGGESTION]


class TestGate:
    def gate(self, project):
        config = project.resolved()
        return LineSpeakersGate(config), build_recording("003", config)

    def test_it_waits_only_where_a_person_found_the_voices_mixed(self, project):
        name_voices(project)
        gate, recording = self.gate(project)
        assert gate.pending(recording) is None and gate.decided_by(recording) is None and gate.notes(recording) == []
        name_voices(project, "--mixed")
        assert "line-speakers" in gate.pending(recording)
        assert macaw(project, "line-speakers", "--as-they-are").exit_code == 0
        assert gate.pending(recording) is None and gate.decided_by(recording) == PERSON
        assert "0 of 3 lines" in gate.notes(recording)[0].message and gate.notes(recording)[0].severity == "info"

    def test_voices_separated_again_hold_it_again(self, project):
        name_voices(project, "--mixed")
        macaw(project, "line-speakers", "--as-they-are")
        gate, recording = self.gate(project)
        path = diarization_module.diarization_path(project.resolved(), "003")
        turns = pd.read_parquet(path)
        _io.write_file(pd.concat([turns, turns.tail(1).assign(t_start=21.5, t_end=22.0)]), path, DIARIZATION_SCHEMA, "diarization")
        assert "separated again" in gate.pending(recording)

    def test_suggestions_kept_unheard_count_as_the_softwares(self, project, monkeypatch):
        name_voices(project, "--mixed")
        suggested = lines_of(project).assign(suggested=[None, "child", None], sure=[False, True, False], doubt=["", "x", ""])
        monkeypatch.setattr("macaw.extractors.audio.content.line_speakers.suggested_lines", lambda config, recording: suggested)
        assert macaw(project, "line-speakers", "--suggested").exit_code == 0
        gate, recording = self.gate(project)
        assert gate.decided_by(recording) == SOFTWARE
        note = gate.notes(recording)[0]
        assert note.severity == "warn" and "1 by mouth movement without a person listening" in note.message

    def test_the_transcript_and_the_pitch_wait_for_it_only_on_a_shared_microphone(self, project):
        shared = project.resolved()
        own = shared.model_copy(update={"text": shared.text.model_copy(update={"diarization": "per_track"})})
        for name in ("text.asr", "prosody.frame"):
            assert LINES_GATE in get_extractor(name).dependencies(shared) and LINES_GATE not in get_extractor(name).dependencies(own)
        assert LineSpeakersGate.brings_along(shared) == ("face.movement",) and LineSpeakersGate.brings_along(own) == ()


class TestFurtherVoice:
    def turns(self, *rows):
        return pd.DataFrame([{"label": label, "t_start": start, "t_end": end, "exclusive": True} for label, start, end in rows])

    def test_the_spare_slot_is_the_one_no_kept_voice_matches(self, tmp_path, write_wav):
        track = write_wav(tmp_path / "shared.wav", np.zeros(40 * SAMPLE_RATE))
        kept = self.turns(("S0", 0.0, 20.0), ("S1", 20.0, 40.0))
        trial = self.turns(("A", 8.0, 20.0), ("B", 20.0, 40.0), ("C", 0.0, 3.0), ("C", 4.0, 8.0))
        found = further_voice(kept, trial, track, 0.0, tmp_path)
        assert found["speech_s"] == 7.0 and (found["first_s"], found["last_s"]) == (0.0, 8.0) and found["held_by"] == {"S0": 7.0}
        assert len(found["spans"]) == 2 and all((tmp_path / clip["file"]).is_file() for clip in found["clips"])

    def test_a_slot_unused_or_nearly_empty_is_nobody(self, tmp_path, write_wav):
        track = write_wav(tmp_path / "shared.wav", np.zeros(40 * SAMPLE_RATE))
        kept = self.turns(("S0", 0.0, 20.0), ("S1", 20.0, 40.0))
        assert further_voice(kept, kept, track, 0.0, tmp_path) == {}
        assert further_voice(kept, self.turns(("A", 4.0, 20.0), ("B", 20.0, 40.0), ("C", 0.0, 4.0)), track, 0.0, tmp_path) == {}

    def test_one_more_voice_is_looked_for_only_when_none_is_expected(self, project, monkeypatch, tmp_path):
        asked = []

        class Fake:
            def diarize(self, path, **counts):
                asked.append(counts)
                return self_turns

        self_turns = self.turns(("SPEAKER_00", 0.0, 4.0), ("SPEAKER_01", 4.0, 8.0))
        monkeypatch.setattr(diarization_module, "resolve_diarizer", lambda config: Fake())
        monkeypatch.setattr(diarization_module, "describe_voices", lambda turns, path, origin, workspace: {})
        config = project.resolved()
        result = DiarizationExtractor(config).run(build_recording("003", config), tmp_path)
        assert asked == [{"n_speakers": 2, "extra_speakers": 0}, {"n_speakers": 2, "extra_speakers": 1}]
        assert json.loads(result.artifacts[FURTHER_NAME].read_text(encoding="utf-8")) == {}

    def found(self, project):
        further = {"speech_s": 22.0, "first_s": 10.0, "last_s": 48.0, "spans": [[10.0, 30.0], [46.0, 48.0]], "held_by": {"S0": 22.0}, "clips": []}
        (project.layout.artifacts_dir("003", DiarizationExtractor.name) / FURTHER_NAME).write_text(json.dumps(further))

    def gate(self, project, mode="always"):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["text"]["voice_review"] = mode
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        config = project.resolved()
        return SpeakerReviewGate(config), build_recording("003", config)

    def test_a_further_voice_holds_the_recording_until_a_person_heard_nobody_else(self, project):
        name_voices(project)
        self.found(project)
        gate, recording = self.gate(project)
        assert "a further voice may speak here (22 s in 2 stretch(es) between 10 s and 48 s)" in gate.pending(recording)
        assert macaw(project, "accept-voices", "--nobody-else").exit_code == 0
        assert gate.pending(recording) is None and gate.notes(recording) == []
        heard = json.loads(project.layout.further_voice_path("003").read_text(encoding="utf-8"))
        assert heard["by"].startswith("ann ") and heard["found"] == "22.0 s from 10.0 s"

    def test_another_further_voice_asks_again(self, project):
        name_voices(project)
        self.found(project)
        macaw(project, "accept-voices", "--nobody-else")
        path = project.layout.artifacts_dir("003", DiarizationExtractor.name) / FURTHER_NAME
        path.write_text(json.dumps({**json.loads(path.read_text()), "speech_s": 9.0}))
        gate, recording = self.gate(project)
        assert gate.pending(recording) is not None

    def test_a_voice_of_its_own_ends_the_question(self, project):
        name_voices(project)
        self.found(project)
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["text"]["extra_voices"] = 1
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        gate, recording = self.gate(project)
        assert gate.pending(recording) is None

    def test_without_review_it_is_only_reported(self, project):
        name_voices(project)
        self.found(project)
        gate, recording = self.gate(project, "skip")
        assert gate.pending(recording) is None
        assert [(note.check, note.severity) for note in gate.notes(recording)] == [("further_voice_unheard", "warn")]

    def test_nothing_to_dismiss_is_said(self, project):
        name_voices(project)
        assert "no further voice was found" in str(macaw(project, "accept-voices", "--nobody-else").exception)


class TestShown:
    def shown(self, project):
        from macaw.interface.pages import review

        return json.dumps(review.body(project.config, "003"), default=lambda item: item.to_plotly_json())

    def test_the_review_page_asks_about_a_further_voice_and_shows_who_heard_nobody(self, project):
        pytest.importorskip("dash")
        name_voices(project)
        TestFurtherVoice().found(project)
        shown = self.shown(project)
        assert "Someone else may speak here: 22 s in 2 stretch(es)" in shown and "/segments?recording=003" in shown
        assert "Keep it as a voice of its own" in shown and "Nobody else speaks" in shown
        macaw(project, "accept-voices", "--nobody-else")
        shown = self.shown(project)
        assert "ann heard nobody else" in shown and "Nobody else speaks" not in shown

    def test_a_voice_of_its_own_is_set_for_that_recording_alone(self, project):
        pytest.importorskip("dash")
        from macaw.interface import project_file

        path = project_file.save_for_recording(project.config, "003", {"text.extra_voices": 1})
        assert path.name == "003.yaml" and resolve_config(project.config, recording_id="003").text.extra_voices == 1
        assert project.resolved().text.extra_voices == 0

    def test_the_review_page_lists_the_lines_once_the_voices_are_marked_as_mixed(self, project, monkeypatch):
        pytest.importorskip("dash")
        from macaw.interface.pages import review

        name_voices(project)
        assert "Who speaks each line" not in self.shown(project)
        argv = review.action_command(project.config, "003", "accept-voices", "", "ann", {"S0": "parent", "S1": "child"}, mixed=True)
        assert argv[-3:] == ["--mixed", "--by", "ann"]
        name_voices(project, "--mixed")
        suggested = lines_of(project).assign(suggested=pd.Series([None, "child", None], dtype=object), sure=[False, True, False], doubt=["", "the mouth of child moves and that of parent is still", ""])
        monkeypatch.setattr("macaw.extractors.audio.content.line_speakers.suggested_lines", lambda config, recording: suggested)
        shown = self.shown(project)
        assert "Who speaks each line" in shown and "Not checked yet." in shown and "no I want red" in shown
        assert "Give the 1 sure ones to the person suggested" in shown and "the other 2 lines" in shown
        picks = review.lines_file(project.config, "003", {"14.200": "child"})
        argv = review.action_command(project.config, "003", "lines-apply", str(picks), "ann", {})
        assert CliRunner().invoke(app, argv[1:]).exit_code == 0
        assert "1 of 3 lines were given to another person" in self.shown(project)
        argv = review.action_command(project.config, "003", "lines-split", "15", "ann", {"person": "child"})
        assert argv[1] == "line-speakers" and argv[-6:] == ["--at", "15", "--person", "child", "--by", "ann"]

    def test_the_report_says_who_decided_each(self, project):
        from macaw.qc.report import write_recording_report

        name_voices(project, "--mixed")
        TestFurtherVoice().found(project)
        text = write_recording_report(project.resolved(), "003")[0].read_text(encoding="utf-8")
        assert "A further voice" in text and "nobody has listened to it" in text
        assert "Who speaks each line" in text and "the lines are not checked yet" in text
        macaw(project, "accept-voices", "--nobody-else")
        macaw(project, "line-speakers", "--at", "14.3", "--person", "child")
        text = write_recording_report(project.resolved(), "003")[0].read_text(encoding="utf-8")
        assert "a person heard nobody else" in text and "1 of 3 lines were given to another person" in text

    def test_methods_say_what_was_looked_for_and_what_was_changed(self, project):
        from macaw.qc import boilerplate

        config = project.resolved()
        assert "One voice more than the participants was also looked for" in boilerplate._further_voice(config)
        name_voices(project)
        assert boilerplate._mixed_voices(config) == ""
        name_voices(project, "--mixed")
        macaw(project, "line-speakers", "--at", "14.3", "--person", "child")
        said = boilerplate._mixed_voices(config)
        assert "In 1 recording a researcher found two participants within one voice" in said and "1 line was given to the other speaker." in said
