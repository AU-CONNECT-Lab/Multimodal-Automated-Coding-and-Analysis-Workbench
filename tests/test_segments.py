from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError
from macaw.execution import runner
from macaw.execution.dag import build_dag
from macaw.extractors.audio.content.asr import (
    TRANSCRIPT_NAME, WORD_SCHEMA, WORDS_NAME, ASRExtractor, cut_transcript, write_transcript,
)
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, PrepInfo
from macaw.extractors.segments import SegmentSummaryExtractor
from macaw.coding.layer_a import LayerACodingExtractor
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.segments import (
    SUMMARY_STEP, Segment, check_recordings, clip_intervals, clip_points, read_segments, remove_segment, set_segment,
)
from macaw.store import _io
from macaw.store.longtable import read_segment_rows
from macaw.store.schema import SEGMENT_TABLE_SCHEMA, Scale, empty_frame
from macaw.turns.entropy import EntropyExtractor
from macaw.turns.segmentation import TURN_SCHEMA, TURNS_NAME, TurnStructureExtractor, silences, utterances_to_turns
from macaw.turns.transitions import TransitionExtractor, transitions_from

RID = "003"


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / RID
    raw.mkdir(parents=True)
    for who in ("parent", "child"):
        (raw / f"{RID}_{who}.wav").write_bytes(b"")
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return SimpleNamespace(
        config=lambda *overrides: resolve_config(path, cli_overrides=list(overrides)), table=tmp_path / "segments.csv",
    )


def _table(project, text):
    project.table.write_text(text, encoding="utf-8")


def test_no_table_means_no_segments(project):
    assert read_segments(project.config()) == {}


def test_segments_come_in_time_order(project):
    _table(project, "recording,segment,start_s,end_s\n003,clean_up,400,520\n003, free_play ,30,330\n\n")
    assert read_segments(project.config()) == {RID: [Segment("free_play", 30.0, 330.0), Segment("clean_up", 400.0, 520.0)]}


@pytest.mark.parametrize(
    "rows, message",
    [
        ("recording,segment,start_s\n003,a,1\n", "no end_s column"),
        ("recording,segment,start_s,end_s\n003,a,one,5\n", "line 2: start_s 'one' is not a number"),
        ("recording,segment,start_s,end_s\n003,a,5,5\n", "line 2: a has to end after it starts"),
        ("recording,segment,start_s,end_s\n003,a,-1,5\n", "a has to end after it starts, at 0 s or later"),
        ("recording,segment,start_s,end_s\n003,a,1,5\n003,a,6,9\n", "line 3: 003 has two segments named a"),
        ("recording,segment,start_s,end_s\n003,,1,5\n", "needs both a recording and a name"),
    ],
)
def test_a_bad_row_is_refused_by_its_line(project, rows, message):
    _table(project, rows)
    with pytest.raises(ConfigError, match=message):
        read_segments(project.config())


def test_setting_a_segment_adds_or_moves_it_and_keeps_peoples_own_columns(project):
    _table(project, "recording,segment,start_s,end_s,note\n003,free_play,30,330,timed from the log\n")
    config = project.config()
    set_segment(config, RID, Segment("clean_up", 400, 520.5))
    set_segment(config, RID, Segment("free_play", 31, 330))
    assert read_segments(config)[RID] == [Segment("free_play", 31.0, 330.0), Segment("clean_up", 400.0, 520.5)]
    saved = pd.read_csv(project.table, dtype=str, keep_default_na=False)
    assert saved["note"].tolist() == ["timed from the log", ""]
    remove_segment(config, RID, "free_play")
    assert read_segments(config)[RID] == [Segment("clean_up", 400.0, 520.5)]
    with pytest.raises(ConfigError, match="no segment free_play"):
        remove_segment(config, RID, "free_play")


def test_a_recording_the_project_does_not_have_is_a_typo(project):
    _table(project, "recording,segment,start_s,end_s\n003,a,1,5\nP030,a,1,5\n")
    with pytest.raises(ConfigError, match="does not have: P030"):
        check_recordings(project.config(), [RID])
    check_recordings(project.config("project.excluded_recordings=[P030]"), [RID])


def test_intervals_are_cut_at_the_edges_and_instants_kept_inside():
    frame = pd.DataFrame({"t_start": [0.0, 4.0, 9.0, 12.0], "t_end": [2.0, 11.0, 10.0, 13.0]})
    assert clip_intervals(frame, 2.0, 10.0).values.tolist() == [[4.0, 10.0], [9.0, 10.0]]
    points = pd.DataFrame({"t": [1.0, 2.0, 5.0, 10.0]})
    assert clip_points(points, 2.0, 10.0)["t"].tolist() == [2.0, 5.0]


def _utterance(subject, index, start, end, text):
    return {"recording_id": RID, "subject_id": subject, "utterance_index": index, "t_start": start, "t_end": end,
            "text": text, "n_words": len(text.split()), "confidence": 0.9, "avg_logprob": np.nan,
            "no_speech_prob": np.nan, "language": "en"}


def _words(subject, index, spans):
    return [{"recording_id": RID, "subject_id": subject, "utterance_index": index, "t_start": start, "t_end": end,
             "word": word, "score": 0.9} for word, start, end in spans]


def test_an_utterance_across_an_edge_keeps_the_words_whose_middle_is_inside():
    transcript = pd.DataFrame([
        _utterance("parent", 0, 9.0, 10.8, "a b c"),
        _utterance("parent", 1, 12.0, 13.0, "inside"),
        _utterance("child", 0, 8.0, 12.0, "one two three four"),
        _utterance("child", 1, 9.0, 10.3, "gone"),
    ])
    words = pd.DataFrame(
        _words("parent", 0, [("a", 9.0, 9.4), ("b", 9.6, 10.2), ("c", 10.4, 10.8)])
        + _words("child", 1, [("gone", 9.0, 9.8)])
    )
    cut = cut_transcript(transcript, words, 10.0, 20.0).set_index(["subject_id", "utterance_index"])
    assert cut.loc[("parent", 0), ["t_start", "text", "n_words"]].tolist() == [10.0, "c", 1]
    assert cut.loc[("parent", 1), ["t_start", "t_end", "text"]].tolist() == [12.0, 13.0, "inside"]
    # no word times, as in a corrected transcript: the words are taken as evenly spread
    assert cut.loc[("child", 0), ["t_start", "t_end", "text", "n_words"]].tolist() == [10.0, 12.0, "three four", 2]
    assert ("child", 1) not in cut.index


TRANSCRIPT = [
    ("parent", 0, 1.0, 4.0, "what did you build today"),
    ("child", 0, 4.6, 6.0, "a big tower"),
    ("parent", 1, 6.2, 9.0, "wow that sounds tall"),
    ("child", 1, 8.5, 9.5, "yeah"),
    ("child", 2, 10.5, 14.0, "it fell down and then I built it again"),
    ("parent", 2, 15.0, 15.4, "mhm"),
    ("child", 3, 16.0, 20.0, "and it was even bigger"),
]


@pytest.fixture
def saved(project):
    """What text.asr and turns.structure leave for one recording, and the analysed window 0 to 30 s."""
    config = project.config("coding.backchannel_lexicon=[mhm, yeah]")
    layout = layout_for(config)
    transcript = pd.DataFrame([_utterance(who, index, start, end, text) for who, index, start, end, text in TRANSCRIPT])
    asr = layout.artifacts_dir(RID, ASRExtractor.name)
    asr.mkdir(parents=True)
    write_transcript(transcript, asr / TRANSCRIPT_NAME)
    _io.write_file(empty_frame(WORD_SCHEMA), asr / WORDS_NAME, WORD_SCHEMA, "words")
    turns = utterances_to_turns(transcript, config.turns).assign(recording_id=RID)
    folder = layout.artifacts_dir(RID, TurnStructureExtractor.name)
    folder.mkdir(parents=True)
    _io.write_file(turns, folder / TURNS_NAME, TURN_SCHEMA, "turns")
    prep = PrepInfo(recording_id=RID, t_origin_s=0.0, duration_s=30.0, sample_rate=16000)
    folder = layout.artifacts_dir(RID, MediaPrepExtractor.name)
    folder.mkdir(parents=True)
    (folder / PREP_INFO_NAME).write_text(json.dumps(prep.to_dict()), encoding="utf-8")
    return SimpleNamespace(config=config, layout=layout, transcript=transcript, turns=turns, prep=prep,
                           recording=build_recording(RID, config))


def _whole(saved):
    """Each step's own recording-level rows, as its run makes them."""
    config, recording, transcript, turns, prep = saved.config, saved.recording, saved.transcript, saved.turns, saved.prep
    transitions = transitions_from(turns, config.turns)
    layer_a = LayerACodingExtractor(config)
    return {
        ASRExtractor: ASRExtractor(config)._rows(recording, transcript, prep),
        TurnStructureExtractor: TurnStructureExtractor(config)._rows(
            recording, turns, silences(transcript, min_s=config.turns.silence_min_s), prep
        ),
        TransitionExtractor: TransitionExtractor(config)._rows(recording, turns, transitions, prep),
        EntropyExtractor: EntropyExtractor(config)._rows(recording, turns, transitions, prep),
        LayerACodingExtractor: layer_a._rows(recording, turns, layer_a._code(turns, transitions, transcript)),
    }


def _values(rows):
    rows = rows[rows["scale"] == Scale.RECORDING.value]
    return {
        (row.feature, row.subject_id if isinstance(row.subject_id, str) else None,
         row.subject_id_b if isinstance(row.subject_id_b, str) else None): row.value
        for row in rows.itertuples()
    }


@pytest.mark.parametrize("step", [ASRExtractor, TurnStructureExtractor, TransitionExtractor, EntropyExtractor, LayerACodingExtractor])
def test_a_segment_over_the_whole_window_gives_the_steps_own_summaries(saved, step):
    whole = _values(_whole(saved)[step])
    again = _values(step(saved.config).summarise(saved.recording, 0.0, 30.0))
    assert whole and again.keys() == whole.keys()
    for key, value in whole.items():
        assert again[key] == pytest.approx(value, nan_ok=True), key


def test_a_turn_across_an_edge_counts_as_the_part_inside(saved):
    rows = _values(TurnStructureExtractor(saved.config).summarise(saved.recording, 12.0, 30.0))
    # the child's turn from 10.5 s counts from 12 s; of the two pauses after it only the one over silence_min_s counts
    assert rows[("talk_time_s", "child", None)] == pytest.approx(2.0 + 4.0)
    assert rows[("n_turns", "child", None)] == 2
    assert rows[("joint_silence_total_s", None, None)] == pytest.approx(1.0)
    assert rows[("turns_per_minute", "parent", None)] == pytest.approx(1 / (18.0 / 60.0))


def test_the_silence_before_the_first_word_inside_is_no_pause(saved):
    rows = _values(TurnStructureExtractor(saved.config).summarise(saved.recording, 14.0, 30.0))
    assert rows[("n_joint_silences", None, None)] == 0


def test_a_floor_transfer_needs_both_turns_inside(saved):
    rows = _values(TransitionExtractor(saved.config).summarise(saved.recording, 9.8, 30.0))
    # the gap from the child's "yeah" ending at 9.5 s to 10.5 s starts before the segment
    assert rows[("n_switches", None, None)] == 2
    assert rows[("transition_count", "child", "parent")] == 1


def test_codes_are_worked_out_again_inside_the_segment(saved):
    whole = _values(LayerACodingExtractor(saved.config).summarise(saved.recording, 0.0, 30.0))
    part = _values(LayerACodingExtractor(saved.config).summarise(saved.recording, 15.2, 30.0))
    assert whole[("n_backchannels_coded", "parent", None)] == 1
    # cut at 15.2 s, the parent's "mhm" is the first turn inside and nothing before it shows the floor came back
    assert part[("n_backchannels_coded", "parent", None)] == 0


def _stamp(layout, step):
    path = layout.completion_path(RID, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{step}-key", encoding="utf-8")


def test_the_summary_step_works_each_finished_step_out_again_per_segment(saved, project, tmp_path):
    _table(project, "recording,segment,start_s,end_s\n003,first,0,12\n003,second,12,30\n003,past_the_end,20,40\n")
    for step in (TurnStructureExtractor.name, TransitionExtractor.name, MediaPrepExtractor.name):
        _stamp(saved.layout, step)
    config = project.config()
    result = SegmentSummaryExtractor(config).run(build_recording(RID, config), tmp_path)

    rows = result.segment_rows
    assert list(rows.columns) == list(SEGMENT_TABLE_SCHEMA)
    assert set(rows["segment"]) == {"first", "second"}
    assert set(rows["source"]) == {TurnStructureExtractor.name, TransitionExtractor.name}
    second = rows[rows["segment"] == "second"]
    assert second["t_start"].unique().tolist() == [12.0] and second["t_end"].unique().tolist() == [30.0]
    outside = [warning for warning in result.warnings if warning.check == "segment_outside"]
    assert len(outside) == 1 and "past_the_end" in outside[0].message


def test_a_changed_segment_changes_the_summary_steps_key_alone(saved, project):
    _table(project, "recording,segment,start_s,end_s\n003,first,0,12\n")
    config = project.config()
    recording = build_recording(RID, config)
    before = runner.task_key(config, SUMMARY_STEP, recording)
    turns_before = runner.task_key(config, TurnStructureExtractor.name, recording)
    set_segment(config, RID, Segment("first", 0, 13))
    assert runner.task_key(config, SUMMARY_STEP, recording) != before
    assert runner.task_key(config, TurnStructureExtractor.name, recording) == turns_before


def test_the_summary_step_runs_after_the_steps_it_summarises_and_only_with_segments(project):
    assert SUMMARY_STEP not in build_dag(project.config()).nodes
    _table(project, "recording,segment,start_s,end_s\n003,first,0,12\n")
    dag = build_dag(project.config())
    assert {TurnStructureExtractor.name, TransitionExtractor.name, ASRExtractor.name} <= dag.dependencies(SUMMARY_STEP)


def test_segment_rows_are_written_stamped_and_an_empty_result_removes_them(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "write_manifest", lambda *a, **k: None)
    layout = layout_for(SimpleNamespace(project=SimpleNamespace(output_dir=tmp_path / "out", work_dir=None)))
    row = {column: None for column in SEGMENT_TABLE_SCHEMA} | {
        "recording_id": RID, "segment": "first", "source": "turns.structure", "scale": "recording",
        "t_start": 0.0, "t_end": 12.0, "feature": "n_turns_total", "value": 3.0,
    }

    def result(rows):
        return SimpleNamespace(rows=None, annotations=None, keypoints=None, body_model=None, segment_rows=rows,
                               specs=[], warnings=[], artifacts={})

    runner._write_result(layout, RID, SUMMARY_STEP, result(pd.DataFrame([row])), "exec-1", "cfg-1")
    written = read_segment_rows(layout.segment_rows_path(RID))
    assert written[["segment", "value", "execution_id", "config_hash"]].values.tolist() == [["first", 3.0, "exec-1", "cfg-1"]]
    runner._write_result(layout, RID, SUMMARY_STEP, result(empty_frame(SEGMENT_TABLE_SCHEMA)), "exec-2", "cfg-1")
    assert read_segment_rows(layout.segment_rows_path(RID)).empty
