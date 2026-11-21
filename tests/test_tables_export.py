from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.extractors.audio.content.asr import (
    BLEED_NAME, BLEED_SCHEMA, DROPPED_NAME, DROPPED_SCHEMA, TRANSCRIPT_NAME, WORD_SCHEMA, WORDS_NAME, ASRExtractor, write_transcript,
)
from macaw.project.layout import layout_for
from macaw.store import _io
from macaw.turns.segmentation import TALK_NAME, TURN_SCHEMA, TURNS_NAME, TurnStructureExtractor
from macaw.turns.talk import TALK_RECORD_SCHEMA
from macaw.turns.transitions import TRANSITION_SCHEMA, TRANSITIONS_NAME, TransitionExtractor

runner = CliRunner()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    resolved = resolve_config(config)
    return SimpleNamespace(config=config, resolved=resolved, layout=layout_for(resolved))


def export(project, kind):
    outcome = runner.invoke(app, ["export", "-c", str(project.config), "-f", kind])
    assert outcome.exit_code == 0, (outcome.output, outcome.exception)
    return project.layout.output_dir / "exports" / kind


def frame(rows, schema):
    return pd.DataFrame(rows).reindex(columns=list(schema))


def test_the_transcript_comes_out_with_its_words_what_was_left_out_and_the_bleed_numbers(project):
    folder = project.layout.artifacts_dir("003", ASRExtractor.name)
    folder.mkdir(parents=True)
    write_transcript(pd.DataFrame({
        "recording_id": "003", "subject_id": ["parent", "child"], "utterance_index": [0, 0], "t_start": [1.0, 3.0], "t_end": [2.0, 4.0],
        "text": ["look", "doggy"], "n_words": [1, 1], "confidence": [0.9, 0.8], "avg_logprob": float("nan"),
        "no_speech_prob": float("nan"), "language": "en",
    }), folder / TRANSCRIPT_NAME)
    _io.write_file(frame([{"recording_id": "003", "subject_id": "parent", "utterance_index": 0, "t_start": 1.0, "t_end": 1.5, "word": "look", "score": 0.9}], WORD_SCHEMA),
                   folder / WORDS_NAME, WORD_SCHEMA, "words")
    _io.write_file(frame([{"recording_id": "003", "subject_id": "child", "t_start": 1.0, "t_end": 2.0, "text": "look", "reason": "crosstalk",
                           "detail": "9.0 dB louder on parent", "louder_on": "parent", "louder_by_db": 9.0}], DROPPED_SCHEMA),
                   folder / DROPPED_NAME, DROPPED_SCHEMA, "dropped utterances")
    _io.write_file(frame([
        {"recording_id": "003", "subject_id": "parent", "utterance_index": 0, "t_start": 1.0, "t_end": 2.0, "louder_on": "child", "louder_by_db": -9.0},
        {"recording_id": "003", "subject_id": "child", "utterance_index": 1, "t_start": 1.0, "t_end": 2.0, "louder_on": "parent", "louder_by_db": 9.0, "dropped_as": "crosstalk"},
    ], BLEED_SCHEMA), folder / BLEED_NAME, BLEED_SCHEMA, "bleed checks")

    out = export(project, "transcript")
    said = pd.read_csv(out / "utterances.csv")
    assert said[["subject_id", "text", "louder_by_db"]].fillna("").values.tolist() == [["parent", "look", -9.0], ["child", "doggy", ""]]
    assert said.columns[:2].tolist() == ["recording_id", "group_id"]
    assert pd.read_csv(out / "words.csv")["word"].tolist() == ["look"]
    assert pd.read_csv(out / "dropped.csv")[["reason", "louder_by_db"]].values.tolist() == [["crosstalk", 9.0]]
    dictionary = pd.read_csv(out / "utterances_dictionary.csv").set_index("column")
    assert set(said.columns) == set(dictionary.index) and dictionary.loc["louder_by_db", "unit"] == "dB"
    assert dictionary.loc["text", "step"] == "text.asr"
    assert "Description" in json.loads((out / "dropped_dictionary.json").read_text(encoding="utf-8"))["reason"]
    assert not (out / "embeddings.csv").exists()


def test_sentence_embeddings_come_out_with_the_transcript(project):
    from macaw.extractors.audio.content.features import EMBEDDINGS_NAME, TextFeatureExtractor

    folder = project.layout.artifacts_dir("003", ASRExtractor.name)
    folder.mkdir(parents=True)
    write_transcript(pd.DataFrame({
        "recording_id": "003", "subject_id": ["parent"], "utterance_index": [0], "t_start": [1.0], "t_end": [2.0], "text": ["look"],
        "n_words": [1], "confidence": [0.9], "avg_logprob": float("nan"), "no_speech_prob": float("nan"), "language": "en",
    }), folder / TRANSCRIPT_NAME)
    features = project.layout.artifacts_dir("003", TextFeatureExtractor.name)
    features.mkdir(parents=True)
    pd.DataFrame({"subject_id": ["parent"], "utterance_index": [0], "t_start": [1.0], "t_end": [2.0], "e0": [0.25], "e1": [-0.5]}).to_parquet(
        features / EMBEDDINGS_NAME, index=False
    )
    out = export(project, "transcript")
    vectors = pd.read_csv(out / "embeddings.csv")
    assert vectors[["utterance_index", "e0", "e1"]].values.tolist() == [[0, 0.25, -0.5]]
    dictionary = pd.read_csv(out / "embeddings_dictionary.csv").set_index("column")
    assert set(vectors.columns) == set(dictionary.index) and dictionary.loc["e1", "step"] == "text.features"
    assert "Dimension 1" in dictionary.loc["e1", "meaning"]
    from pathlib import Path

    own = Path(project.resolved.project.recording_config_dir) / "003.yaml"
    own.parent.mkdir(parents=True, exist_ok=True)
    own.write_text("text:\n  embedding_model: mini-lm\n", encoding="utf-8")
    dictionary = pd.read_csv(export(project, "transcript") / "embeddings_dictionary.csv").set_index("column")
    assert "embedding from mini-lm;" in dictionary.loc["e1", "meaning"]


def test_turns_transitions_and_talk_come_out_together(project):
    structure = project.layout.artifacts_dir("003", TurnStructureExtractor.name)
    structure.mkdir(parents=True)
    _io.write_file(frame([{"recording_id": "003", "subject_id": "parent", "turn_index": 0, "t_start": 1.0, "t_end": 2.0, "n_utterances": 1,
                           "n_words": 1, "text": "look", "within": False}], TURN_SCHEMA), structure / TURNS_NAME, TURN_SCHEMA, "turns")
    _io.write_file(frame([{"recording_id": "003", "subject_id": "child", "t_start": 2.5, "t_end": 2.6, "source": "voice", "kept": False,
                           "reason": "too short"}], TALK_RECORD_SCHEMA), structure / TALK_NAME, TALK_RECORD_SCHEMA, "talk")
    moves = project.layout.artifacts_dir("003", TransitionExtractor.name)
    moves.mkdir(parents=True)
    _io.write_file(frame([{"recording_id": "003", "transition_index": 0, "from_subject": "parent", "to_subject": "child", "from_turn_index": 0,
                           "to_turn_index": 0, "is_switch": True, "prev_end": 2.0, "next_start": 3.0, "fto_s": 1.0, "transition_type": "gap",
                           "within": False}], TRANSITION_SCHEMA), moves / TRANSITIONS_NAME, TRANSITION_SCHEMA, "transitions")
    out = export(project, "turns")
    assert pd.read_csv(out / "transitions.csv")["transition_type"].tolist() == ["gap"]
    assert pd.read_csv(out / "talk.csv")[["source", "kept"]].values.tolist() == [["voice", False]]
    assert pd.read_csv(out / "turns_dictionary.csv").set_index("column").loc["within", "step"] == "turns.structure"


def test_warnings_spread_their_details_over_columns(project):
    folder = project.layout.qc_dir("003")
    folder.mkdir(parents=True)
    (folder / "warnings.json").write_text(json.dumps([
        {"check": "asr_crosstalk", "recording_id": "003", "subject_id": "child", "severity": "warn", "message": "dropped",
         "t_start": None, "t_end": None, "details": {"crosstalk_margin_db": 6.0}},
        {"check": "scale", "recording_id": "003", "subject_id": None, "severity": "info", "message": "no scale", "t_start": None,
         "t_end": None, "details": {}},
    ]), encoding="utf-8")
    out = export(project, "warnings")
    found = pd.read_csv(out / "warnings.csv")
    assert found["detail_crosstalk_margin_db"].tolist()[0] == 6.0 and pd.isna(found["detail_crosstalk_margin_db"].tolist()[1])
    meaning = pd.read_csv(out / "warnings_dictionary.csv").set_index("column").loc["detail_crosstalk_margin_db", "meaning"]
    assert "asr_crosstalk" in meaning and not (out / "candidates.csv").exists()


def test_nothing_to_export_says_so(project):
    outcome = runner.invoke(app, ["export", "-c", str(project.config), "-f", "voices"])
    assert outcome.exit_code != 0 and "have not run" in str(outcome.exception)


def test_the_voices_heard_come_out_as_the_diarizer_gave_them(project):
    from macaw.extractors.audio.content.diarization import DIARIZATION_SCHEMA, diarization_path

    path = diarization_path(project.resolved, "003")
    path.parent.mkdir(parents=True)
    _io.write_file(frame([{"track": "room", "label": "S0", "t_start": 1.0, "t_end": 3.0, "exclusive": True},
                          {"track": "room", "label": "S0", "t_start": 1.0, "t_end": 3.2, "exclusive": False}], DIARIZATION_SCHEMA),
                   path, DIARIZATION_SCHEMA, "diarization")
    found = pd.read_csv(export(project, "voices") / "voices.csv")
    assert found[["label", "exclusive"]].values.tolist() == [["S0", True], ["S0", False]] and found["person"].isna().all()


def test_the_judges_answers_and_units_come_out_with_reasons_and_votes(project):
    from macaw.coding.llm.judge import SAMPLES_NAME, JudgementExtractor
    from macaw.store.annotations import build_annotations
    from macaw.store.annotations import write_annotations

    folder = project.layout.artifacts_dir("003", JudgementExtractor.name)
    folder.mkdir(parents=True)
    pd.DataFrame({
        "unit_id": ["u1"] * 3, "sample": [0, 1, 2], "label": ["praise", "praise", None], "reason": ["warm", "warm words", None],
        "problem": [None, None, "unreadable"], "given": [None, None, None], "subject_id": "parent", "t_start": 1.0, "t_end": 2.0,
    }).to_parquet(folder / SAMPLES_NAME, index=False)
    coded = build_annotations(
        recording_id="003", tier="codes", layer=JudgementExtractor.layer(), annotator="judge", subject_id="parent",
        intervals=pd.DataFrame({"t_start": [1.0], "t_end": [2.0], "label": ["praise"], "confidence": [0.67], "abstained": [False], "needs_review": [False]}),
    )
    write_annotations(coded, project.layout.step_annotations_path("003", JudgementExtractor.name))
    out = export(project, "judge")
    samples = pd.read_csv(out / "samples.csv")
    assert samples[["judge", "sample", "reason"]].fillna("").values.tolist() == [["judge", 0, "warm"], ["judge", 1, "warm words"], ["judge", 2, ""]]
    assert samples["problem"].tolist()[2] == "unreadable"
    units = pd.read_csv(out / "units.csv")
    assert units[["label", "votes"]].values.tolist() == [["praise", "praise: 2"]]
