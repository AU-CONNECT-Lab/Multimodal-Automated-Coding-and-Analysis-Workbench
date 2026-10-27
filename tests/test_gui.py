from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dash")

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.extractors.video.pose.keypoints import PoseExtractor
from macaw.interface import cli_bridge
from macaw.interface.app import create_app
from macaw.interface.files import file_url, path_from_url
from macaw.interface.pages import results, review, run
from macaw.project.layout import layout_for
from macaw.qc.downsample import min_max_downsample
from macaw.store.longtable import build_long_rows
from macaw.store.schema import Scale


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav", "003_room.mp4"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return config


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json() if hasattr(component, "to_plotly_json") else component, default=lambda item: item.to_plotly_json())


def test_the_command_shown_is_one_a_terminal_takes(project):
    argv = cli_bridge.build_command(["run"], config_path=project, recordings=["003"], extractors=["text.asr", "turns.structure"])
    assert argv == ["macaw", "run", "-c", str(project), "-r", "003", "-e", "text.asr", "-e", "turns.structure"]
    every = run.run_command(project, ["003"], ["text.asr"], ["003"])
    assert "-r" not in every


def test_a_short_command_runs_with_the_guis_own_interpreter():
    code, text = cli_bridge.run_now(["macaw", "--help"])
    assert code == 0 and "Usage" in text


def test_only_files_inside_the_project_are_served(project, tmp_path):
    config = resolve_config(project)
    inside = layout_for(config).output_dir / "reports" / "qc_003.html"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_text("<p>page</p>", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("no", encoding="utf-8")
    assert path_from_url(file_url(inside)[len("/files/"):], config) == inside.resolve()
    assert path_from_url(file_url(outside)[len("/files/"):], config) is None
    client = create_app(project).server.test_client()
    assert client.get(file_url(inside)).status_code == 200
    assert client.get(file_url(outside)).status_code == 404


def test_the_review_page_offers_to_keep_a_camera_in_doubt(project):
    config = resolve_config(project)
    owners = layout_for(config).artifacts_dir("003", PoseExtractor.name) / "owners"
    owners.parent.mkdir(parents=True, exist_ok=True)
    owners.write_text(json.dumps({"003_room.mp4": {"people": {"parent": [1], "child": [2]}, "chosen_by": {}, "doubt": "who is who", "spans": []}}))
    shown = text_of(review.body(project, "003"))
    assert "Who is on camera" in shown and "accept-owner" in shown and "Which file is whose" in shown
    argv = review.action_command(project, "003", "rename-owner", "003_room.mp4", "ann", {}, {"003_room.mp4": "parent=child,child=parent"})
    assert argv[1] == "accept-owner" and argv[argv.index("--rename") + 1] == "parent=child,child=parent"
    voices = review.action_command(project, "003", "accept-voices", "", "ann", {"S1": "child", "S0": "none"}, {})
    assert voices[voices.index("--rename") + 1] == "S0=none,S1=child"


def test_results_show_one_number_per_person_or_a_curve(project):
    config = resolve_config(project)
    directory = layout_for(config).long_table_path("003")
    directory.mkdir(parents=True)
    frame = build_long_rows(
        values=pd.DataFrame({"t_start": np.arange(5.0), "f0_hz": [100.0, 110, np.nan, 120, 130]}),
        scale=Scale.FRAME, recording_id="003", source="prosody.frame", subject_id="parent",
    )
    frame.to_parquet(directory / "prosody.frame.parquet", index=False)
    assert results.sources(project, "003") == ["prosody.frame"]
    graph = results.view(project, "003", "prosody.frame", "f0_hz")
    assert graph.figure.data[0].name == "parent"


def test_min_max_keeps_the_extremes_of_every_run():
    t = np.arange(10_000.0)
    y = np.sin(t / 50.0)
    y[5000] = 9.0
    small_t, small_y = min_max_downsample(t, y, 200)
    assert small_t.size <= 200 and 9.0 in small_y and np.all(np.diff(small_t) > 0)
    assert min_max_downsample(t[:50], y[:50], 200)[0].size == 50


def test_the_review_page_lists_uncertain_speech_acts_and_saves_a_pick(project):
    from macaw.coding.codebook import Codebook, CodeDefinition, write_codebook
    from macaw.coding.review_queue import REVIEWED_NAME
    from macaw.store.annotations import write_annotations
    from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame

    write_codebook(
        Codebook("acts", "utterance", [CodeDefinition("ask", "asks", criteria=["?"]), CodeDefinition("tell", "tells", criteria=["."])]),
        project.parent / "codebook.yaml",
    )
    project.write_text(project.read_text(encoding="utf-8") + "\ncoding:\n  codebook: codebook.yaml\n", encoding="utf-8")
    config = resolve_config(project)
    rows = pd.DataFrame({
        "recording_id": "003", "group_id": "003", "session_id": None, "task": None, "run": None,
        "subject_id": ["parent", "child"], "tier": "acts", "t_start": [1.0, 2.0], "t_end": [1.5, 2.5],
        "label": ["ask", "tell"], "annotator": "llm:ollama/x", "layer": "B", "confidence": [0.6, 1.0],
        "abstained": False, "needs_review": [True, False], "execution_id": "e", "config_hash": "h",
    })
    write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(config).annotations_path("003"))
    samples = layout_for(config).artifacts_dir("003", "coding.judge") / "samples.parquet"
    samples.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": ["u0"] * 5, "sample": range(5), "label": ["ask"] * 3 + ["tell"] * 2,
                  "subject_id": "parent", "t_start": 1.0, "t_end": 1.5}).to_parquet(samples)

    shown = text_of(review.body(project, "003"))
    assert "Speech acts" in shown and "ask 3, tell 2" in shown and "apply-review" in shown and "act-pick" in shown

    path = review.review_file(project, "003", "ann", {"parent|1.000|acts": "tell"})
    code, text = cli_bridge.run_now(["macaw", "review", "-c", str(project), "--apply", str(path)])
    assert code == 0, text
    reviewed = pd.read_parquet(layout_for(config).annotations_path("003") / REVIEWED_NAME)
    assert reviewed[["annotator", "label", "t_start"]].values.tolist() == [["ann", "tell", 1.0]]
    assert "No coded unit is waiting" in text_of(review.body(project, "003"))
