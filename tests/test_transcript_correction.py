from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import EafError
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, write_transcript
from macaw.extractors.prep import PrepInfo
from macaw.interop.transcript_eaf import export_transcript_eaf, read_transcript_eaf
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

pytest.importorskip("pympi")


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
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
