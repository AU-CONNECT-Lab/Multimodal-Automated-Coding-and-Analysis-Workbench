from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import EafError
from macaw.extractors.audio.content.asr import CORRECTED_CHECK, TRANSCRIPT_NAME, ASRExtractor, write_transcript
from macaw.extractors.prep import PrepInfo
from macaw.interop.transcript_eaf import export_transcript_eaf, read_transcript_eaf
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.boilerplate import transcripts_corrected

pytest.importorskip("pympi")


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    resolved = resolve_config(config)
    return SimpleNamespace(config=config, resolved=resolved, recording=build_recording("003", resolved))


TRANSCRIPT = pd.DataFrame(
    {
        "recording_id": "003", "subject_id": ["parent", "child"], "utterance_index": [0, 0],
        "t_start": [1.0, 3.0], "t_end": [2.5, 4.0], "text": ["look at that", "doggy"],
        "n_words": [3, 1], "confidence": [0.9, 0.8], "avg_logprob": float("nan"), "no_speech_prob": float("nan"), "language": "en",
    }
)
DROPPED = pd.DataFrame(
    {"subject_id": ["child"], "t_start": [1.0], "t_end": [2.5], "text": ["look at that"], "reason": ["crosstalk"], "detail": ["9.0 dB louder on parent"]}
)


def test_the_transcript_comes_back_and_dropped_tiers_stay_out(project, tmp_path):
    path = export_transcript_eaf(TRANSCRIPT, DROPPED, project.recording, tmp_path / "t.eaf")
    back = read_transcript_eaf(path, ["parent", "child"])
    assert back.values.tolist() == [["parent", 1.0, 2.5, "look at that"], ["child", 3.0, 4.0, "doggy"]]


def test_words_no_voice_took_have_a_dropped_tier_of_nobody(project, tmp_path):
    from macaw.interop.eaf import _elan

    nobody = pd.concat([DROPPED, pd.DataFrame({"subject_id": [None], "t_start": [5.0], "t_end": [5.5], "text": ["hm"], "reason": ["no voice"], "detail": ["in none of the voices"]})])
    path = export_transcript_eaf(TRANSCRIPT, nobody, project.recording, tmp_path / "t.eaf")
    eaf = _elan()(str(path))
    assert [value for _, _, value, *_ in eaf.get_annotation_data_for_tier("dropped")] == ["hm"]
    assert [value for _, _, value, *_ in eaf.get_annotation_data_for_tier("why dropped")] == ["no voice: in none of the voices"]
    assert len(read_transcript_eaf(path, ["parent", "child"])) == 2


def test_words_moved_back_from_a_dropped_tier_come_in_as_they_were_said(project, tmp_path):
    from macaw.interop.eaf import _elan

    path = export_transcript_eaf(TRANSCRIPT, DROPPED, project.recording, tmp_path / "t.eaf")
    eaf = _elan()(str(path))
    (start, end, words, *_), = eaf.get_annotation_data_for_tier("child_dropped")
    eaf.remove_annotation("child_dropped", start)
    eaf.add_annotation("child_transcript", start, end, words)
    eaf.to_file(str(path))
    back = read_transcript_eaf(path, ["parent", "child"])
    assert back[back["subject_id"] == "child"]["text"].tolist() == ["look at that", "doggy"]


def test_a_line_of_no_length_goes_out_and_comes_back(project, tmp_path):
    point = TRANSCRIPT.assign(t_end=[2.5, 3.0])
    back = read_transcript_eaf(export_transcript_eaf(point, DROPPED.iloc[0:0], project.recording, tmp_path / "t.eaf"), ["parent", "child"])
    assert back["text"].tolist() == ["look at that", "doggy"] and back["t_end"].tolist() == [2.5, 3.001]


def test_a_tier_of_someone_who_is_not_a_subject_is_refused(project, tmp_path):
    path = export_transcript_eaf(TRANSCRIPT, DROPPED.iloc[:0], project.recording, tmp_path / "t.eaf")
    with pytest.raises(EafError, match="not a subject"):
        read_transcript_eaf(path, ["parent"])


def test_text_asr_reads_the_correction_instead_of_recognising(project, tmp_path):
    layout = layout_for(project.resolved)
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(TRANSCRIPT, directory / TRANSCRIPT_NAME)
    exported = tmp_path / "003.eaf"
    runner = CliRunner()
    outcome = runner.invoke(app, ["transcript", "export", "-c", str(project.config), "-r", "003", "--out", str(exported)])
    assert outcome.exit_code == 0, outcome.output
    outcome = runner.invoke(app, ["transcript", "import", str(exported), "-c", str(project.config), "-r", "003", "--by", "ann"])
    assert outcome.exit_code == 0, outcome.output
    assert "kept 2 utterances" in outcome.output

    extractor = ASRExtractor(project.resolved)
    prep = PrepInfo(recording_id="003", t_origin_s=0.5, duration_s=10.0, sample_rate=16000)
    warnings = []
    utterances, words = extractor._from_correction(project.recording, prep, warnings)
    # back on the prepared-track clock, which run() shifts forward again
    assert utterances["t_start"].tolist() == [0.5, 2.5] and words.empty
    assert "ann corrected" in warnings[0].message

    outcome = runner.invoke(app, ["transcript", "reset", "-c", str(project.config), "-r", "003"])
    assert outcome.exit_code == 0 and not layout.corrected_transcript_path("003").exists()


def test_a_correction_is_cut_to_the_analysed_part_and_what_lies_outside_is_kept_as_dropped(project, tmp_path):
    """The analysed part was narrowed after the correction came in."""
    layout = layout_for(project.resolved)
    lines = pd.concat([TRANSCRIPT, TRANSCRIPT.iloc[[0]].assign(utterance_index=1, t_start=20.0, t_end=21.0, text="bye now")], ignore_index=True)
    path = export_transcript_eaf(lines, DROPPED.iloc[:0], project.recording, tmp_path / "t.eaf")
    corrected = layout.corrected_transcript_path("003")
    corrected.parent.mkdir(parents=True, exist_ok=True)
    corrected.write_bytes(path.read_bytes())
    extractor = ASRExtractor(project.resolved)
    warnings = []
    # the analysed part runs from 2 s to 5 s of the recording
    utterances, _ = extractor._from_correction(project.recording, PrepInfo(recording_id="003", t_origin_s=2.0, duration_s=3.0, sample_rate=16000), warnings)
    assert utterances[["subject_id", "t_start", "t_end", "text"]].values.tolist() == [["parent", 0.0, 0.5, "that"], ["child", 1.0, 2.0, "doggy"]]
    dropped = pd.concat(extractor._dropped)
    assert dropped["text"].tolist() == ["bye now"] and dropped["reason"].tolist() == ["outside the analysed part"]
    assert "1 utterance(s) of the corrected transcript lie outside" in warnings[0].message


def test_without_a_correction_the_cache_key_reads_only_files_that_exist(project, monkeypatch):
    from macaw.extractors.audio.content import asr as asr_module

    monkeypatch.setattr(asr_module, "load_prep", lambda config, rid: None)
    monkeypatch.setattr(asr_module, "voice_inputs", lambda config, recording, prep: [])
    extractor = ASRExtractor(project.resolved)
    assert extractor.inputs_for(project.recording) == []
    corrected = layout_for(project.resolved).corrected_transcript_path("003")
    corrected.parent.mkdir(parents=True, exist_ok=True)
    corrected.write_text("<ANNOTATION_DOCUMENT/>", encoding="utf-8")
    assert extractor.inputs_for(project.recording) == [corrected]


def test_the_methods_count_recordings_whose_last_run_read_a_correction(project):
    path = layout_for(project.resolved).warnings_path("003", ASRExtractor.name)
    assert transcripts_corrected(project.resolved) == (0, 0)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([{"check": "asr_confidence"}]), encoding="utf-8")
    assert transcripts_corrected(project.resolved) == (0, 1)
    path.write_text(json.dumps([{"check": CORRECTED_CHECK}]), encoding="utf-8")
    assert transcripts_corrected(project.resolved) == (1, 1)


def test_exporting_again_keeps_the_file_a_person_may_have_corrected(project, tmp_path):
    layout = layout_for(project.resolved)
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(TRANSCRIPT, directory / TRANSCRIPT_NAME)
    out = tmp_path / "t.eaf"
    runner = CliRunner()
    assert runner.invoke(app, ["transcript", "export", "-c", str(project.config), "-r", "003", "--out", str(out)]).exit_code == 0
    out.write_text(out.read_text(encoding="utf-8").replace("doggy", "a doggy"), encoding="utf-8")
    again = runner.invoke(app, ["transcript", "export", "-c", str(project.config), "-r", "003", "--out", str(out)])
    kept = [path for path in tmp_path.glob("t.*.eaf")]
    assert again.exit_code == 0 and len(kept) == 1 and "a doggy" in kept[0].read_text(encoding="utf-8")
    assert f"the earlier one kept as {kept[0].name}" in again.output
