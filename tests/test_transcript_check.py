from __future__ import annotations

import json
import re

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli import transcript as command
from macaw.cli.main import app
from macaw.coding.llm import backends
from macaw.exceptions import LLMBackendError
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, write_transcript
from macaw.interop.transcript_eaf import DOUBT_TYPE, export_transcript_eaf, read_transcript_eaf
from macaw.project.layout import layout_for
from macaw.qc.transcript_check import check_lines, describe, read_check, still_there, transcript_lines
from test_codebook_help import CutOff, Model
from test_transcript_correction import DROPPED, TRANSCRIPT, project  # noqa: F401

pytest.importorskip("pympi")

LONGER = pd.concat([
    TRANSCRIPT,
    TRANSCRIPT.iloc[[0]].assign(subject_id="other", t_start=5.0, t_end=5.3, text="thanks for watching please subscribe"),
], ignore_index=True)


def marks_watching(system, user):
    found = re.findall(r"^id (l\d+) \| .*: (.*)$", user, re.MULTILINE)
    return {"lines": [{"id": line_id, "kind": "misheard", "why": "a sign-off nobody said"} for line_id, text in found if "watching" in text]
            + [{"id": "l99", "why": "x"}]}


def test_the_model_sees_letters_lengths_and_context_and_marks_only_real_lines():
    lines = transcript_lines(LONGER, ["parent", "child"])
    assert lines["speaker"].tolist() == ["A", "B", "other"] and lines["id"].tolist() == ["l1", "l2", "l3"]
    model = Model(marks_watching)
    marked = check_lines(model, lines, batch_size=2)
    assert len(model.requests) == 2
    system, user = model.requests[1]
    assert "The speaker other is someone besides the people recorded" in system and "young child's speech" in system
    assert '"kind": "misheard" or "cut"' in system
    assert "Earlier in the conversation (context only, do not check):\nspeaker A | 1.5 s, 3 words: look at that" in user
    assert "id l3 | speaker other | 0.3 s, 5 words: thanks for watching please subscribe" in user
    assert marked[["subject_id", "text", "kind", "why"]].values.tolist() == [
        ["other", "thanks for watching please subscribe", "misheard", "a sign-off nobody said"]
    ]


def test_a_piece_cut_off_is_kept_apart_and_a_made_up_kind_counts_as_misheard():
    lines = transcript_lines(LONGER, ["parent", "child"])
    model = Model(lambda system, user: {"lines": [
        {"id": "l1", "kind": "Cut", "why": "a piece of the next line"}, {"id": "l3", "kind": "sign-off", "why": "nobody said it"},
    ]})
    marked = check_lines(model, lines, batch_size=10)
    assert marked["kind"].tolist() == ["cut", "misheard"]
    found = {"model": "tiny", "n_lines": 3, "lines": marked.to_dict("records")}
    assert describe(found) == "2 of 3 line(s) marked by tiny: 1 possibly misheard, 1 cut off"
    assert describe({"model": "tiny", "n_lines": 3, "lines": []}) == "0 of 3 line(s) marked by tiny"


def test_an_answer_cut_off_is_refused_so_lines_it_never_reached_do_not_pass_as_checked():
    lines = transcript_lines(LONGER, ["parent", "child"])
    with pytest.raises(LLMBackendError, match="cut off at coding.llm.max_tokens"):
        check_lines(CutOff(marks_watching, ""), lines, batch_size=2)


def test_the_command_keeps_the_marks_and_the_export_puts_them_beside_the_transcript(project, monkeypatch, tmp_path):  # noqa: F811
    layout = layout_for(project.resolved)
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(LONGER, directory / TRANSCRIPT_NAME)
    monkeypatch.setattr(backends, "resolve_llm_backend", lambda cfg: Model(marks_watching))
    result = CliRunner().invoke(app, ["transcript", "check", "-c", str(project.config), "-r", "003"])
    assert result.exit_code == 0, result.output
    assert "1 of 3 line(s) marked by tiny@abc: 1 possibly misheard" in result.output
    assert "5.0 s other: thanks for watching please subscribe  (possibly misheard: a sign-off nobody said)" in result.output
    found = read_check(layout.transcript_check_path("003"))
    assert found["n_lines"] == 3 and found["model"] == "tiny@abc"

    exported = tmp_path / "003.eaf"
    result = CliRunner().invoke(app, ["transcript", "export", "-c", str(project.config), "-r", "003", "--out", str(exported)])
    assert result.exit_code == 0 and "1 line(s) the model marked" in result.output
    from pympi.Elan import Eaf

    eaf = Eaf(str(exported))
    doubtful = [name for name in eaf.get_tier_names() if eaf.tiers[name][2].get("LINGUISTIC_TYPE_REF") == DOUBT_TYPE]
    assert doubtful == ["other_doubtful"]
    assert eaf.get_annotation_data_for_tier("other_doubtful")[0][2] == "possibly misheard: a sign-off nobody said"
    # the marks are never read back as transcript
    assert len(read_transcript_eaf(exported, ["parent", "child", "other"])) == 3

    # a line corrected since is no longer shown as marked
    corrected = LONGER.assign(text=LONGER["text"].replace({"thanks for watching please subscribe": "thanks for coming"}))
    marked, gone = still_there(found, corrected)
    assert marked.empty and gone == 1 and describe(found, corrected).endswith("1 of them changed since")


def test_without_marks_the_export_has_no_doubtful_tier(project, tmp_path):  # noqa: F811
    path = export_transcript_eaf(TRANSCRIPT, DROPPED, project.recording, tmp_path / "t.eaf")
    from pympi.Elan import Eaf

    assert DOUBT_TYPE not in Eaf(str(path)).linguistic_types
    assert still_there({}, TRANSCRIPT)[1] == 0 and json.dumps(read_check(tmp_path / "none.json")) == "{}"
    assert command.check.__doc__.startswith("Have the model in coding.llm mark lines")


def test_the_qc_report_says_how_many_lines_a_model_marked(project):  # noqa: F811
    from macaw.qc.checks import RecordingResults
    from macaw.qc.report import _decisions

    layout = layout_for(project.resolved)
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(LONGER, directory / TRANSCRIPT_NAME)
    path = layout.transcript_check_path("003")
    path.parent.mkdir(parents=True, exist_ok=True)
    marked = [{"subject_id": "other", "t_start": 5.0, "t_end": 5.3, "text": "thanks for watching please subscribe", "kind": "misheard", "why": "a sign-off"}]
    path.write_text(json.dumps({"model": "tiny", "n_lines": 3, "lines": marked}), encoding="utf-8")
    rows = _decisions(project.resolved, layout, RecordingResults(layout, "003"), "003", [], [], [], [], [])
    row = next(row for row in rows if row["what"] == "Transcript lines a model marked")
    assert row["decided"].startswith("1 of 3 line(s) marked by tiny: 1 possibly misheard; macaw transcript export -r 003")
