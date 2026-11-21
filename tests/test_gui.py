from __future__ import annotations

import shutil

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import pandas as pd
import pytest

pytest.importorskip("dash")
pytest.importorskip("ruamel.yaml")

from dash import no_update

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import MacawError
from macaw.execution.runner import task_id_for
from macaw.extractors.video.pose.keypoints import PoseExtractor
from macaw.interface import cli_bridge, project_file, ui
from macaw.interface.app import build_layout, create_app, page_for
from macaw.interface.files import file_url, path_from_url
from macaw.interface.pages import agreement, overview, results, review, run
from macaw.project import status
from macaw.project.layout import layout_for
from macaw.qc.downsample import min_max_downsample
from macaw.store.longtable import build_long_rows
from macaw.store.schema import Scale
from setup_helpers import camera as setup_camera, mics, put, write_setup as write_devices


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for device, name in (("parent_mic", "a.wav"), ("child_mic", "a.wav"), ("room", "a.mp4")):
        put(raw, "003", device, name)
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out", roles=("parent", "child"))
    write_devices(config, cameras={"room": setup_camera()}, microphones=mics())
    return config


def text_of(component) -> str:
    return json.dumps(component.to_plotly_json() if hasattr(component, "to_plotly_json") else component, default=lambda item: item.to_plotly_json())


def test_the_command_shown_is_one_a_terminal_takes(project):
    argv = cli_bridge.build_command(["run"], config_path=project, recordings=["003"], extractors=["text.asr", "turns.structure"])
    assert argv == ["macaw", "run", "-c", str(project), "-r", "003", "-e", "text.asr", "-e", "turns.structure"]
    every = run.run_command(project, ["003"], ["text.asr"], ["003"])
    assert "-r" not in every
    plain = run.run_command(project, ["003"], ["text.asr"], ["003"], ["text.asr"])
    assert plain == ["macaw", "run", "-c", str(project)]


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
    owners.write_text(json.dumps({"room": {"people": {"parent": [1], "child": [2]}, "chosen_by": {}, "doubt": "who is who", "spans": []}}))
    shown = text_of(review.body(project, "003"))
    assert "Who is on camera" in shown and "accept-owner" in shown and "Which file is whose" not in shown
    argv = review.action_command(project, "003", "accept-owner", "room", "ann", {})
    assert argv[1] == "accept-owner" and argv[argv.index("--camera") + 1] == "room"
    assert "names-person" in shown and "name-people" in shown and "/setup?" not in shown
    voices = review.action_command(project, "003", "accept-voices", "", "ann", {"S1": "child", "S0": "none"})
    assert voices[voices.index("--rename") + 1] == "S0=none,S1=child"


def test_the_review_page_measures_a_yardstick_and_saves_it(project):
    shown = text_of(review.body(project, "003"))
    assert "Pixels to centimetres" in shown and "set-scale" in shown and "No scale yet" in shown
    assert "holds only at the object's distance from the camera" in shown
    measured = {"video": "room", "at": 4.5, "points": [[0, 0], [30, 40]], "length": 10, "object": "table edge", "person": "child"}
    argv = review.action_command(project, "003", "set-scale", "", "ann", {}, measured)
    assert argv[1] == "set-scale" and argv[argv.index("-r") + 1] == "003" and argv[argv.index("--points") + 1] == "0,0,30,40"
    assert argv[argv.index("--length") + 1] == "10" and argv[argv.index("--object") + 1] == "table edge"
    assert argv[argv.index("--person") + 1] == "child"
    assert review.action_command(project, "003", "clear-scale", "", "ann", {})[-1] == "--clear"
    everyone = review.action_command(project, "003", "set-scale", "", "ann", {}, {**measured, "person": review.EVERYONE})
    assert "--person" not in everyone


def test_the_review_page_labels_where_people_look_and_reports_agreement(project):
    from macaw.project.gaze_labels import GazeLabel, write_labels

    shown = text_of(review.body(project, "003"))
    assert "Where people look" in shown
    assert "gaze-random" in shown
    assert "label-gaze" in shown
    argv = review.action_command(project, "003", "label-gaze", "child", "ann", {}, gaze={"t": 12.5, "looks_at": "parent"})
    assert argv[1] == "label-gaze" and argv[argv.index("--at") + 1] == "12.5" and argv[argv.index("--person") + 1] == "child"
    assert argv[argv.index("--looks-at") + 1] == "parent" and argv[argv.index("--by") + 1] == "ann"
    write_labels([GazeLabel(12.5, "child", "parent"), GazeLabel(13.0, "child", "parent")], layout_for(resolve_config(project)).gaze_labels_path("003"))
    assert "Looks at" in text_of(review.body(project, "003")) and "Looks at" not in shown


def test_the_review_page_lists_coding_brought_in_and_how_to_compare_it(project, tmp_path):
    before = text_of(review.body(project, "003"))
    assert "No coding imported for this recording" in before and '"kind": "coding"' in before
    table = tmp_path / "coding.csv"
    table.write_text("tier,start,end,code,person\nspeech,1.0,2.5,question,child\nspeech,3,4,answer,child\n", encoding="utf-8")
    code, text = cli_bridge.run_now(["macaw", "import-codes", str(table), "-c", str(project), "-r", "003", "--coder", "ann"], cwd=project.parent)
    assert code == 0, text
    shown = text_of(review.body(project, "003"))
    assert "Your coding" in shown and "agreement" in shown and "--units" in shown and "speech" in shown


def test_the_agreement_page_compares_coders_across_the_recordings(project, tmp_path):
    config = resolve_config(project)
    assert agreement.coded_tiers(config) == {}
    table = tmp_path / "coding.csv"
    for coder, second in (("ann", "answer"), ("bob", "question")):
        table.write_text(f"tier,start,end,code,person\nspeech,1.0,2.5,question,child\nspeech,3,4,{second},child\n", encoding="utf-8")
        code, text = cli_bridge.run_now(["macaw", "import-codes", str(table), "-c", str(project), "-r", "003", "--coder", coder], cwd=project.parent)
        assert code == 0, text
    assert agreement.coded_tiers(config) == {"speech": ["ann", "bob"]}
    assert '"/agreement?tier=speech&units=ann"' in text_of(review.body(project, "003"))

    app = create_app(project)
    fill = callback(app, "..agreement-tier.options...agreement-tier.value...agreement-segment.options...agreement-recordings.options...agreement-message.children..")
    assert fill("/agreement", "?tier=speech&units=bob", None)[1] == "speech"
    assert callback(app, "..agreement-coder.options...agreement-coder.value..")("speech", "?tier=speech&units=bob", None) == (["ann", "bob"], "bob")

    argv = agreement.agreement_command(project, "speech", "units", "bob", None, None, None, False, "compared")
    assert argv[1] == "agreement" and argv[argv.index("--units") + 1] == "bob" and argv[-2:] == ["--out", "compared"]
    grid = agreement.agreement_command(project, "speech", "grid", "bob", 0.5, "free_play", ["003"], True, None)
    assert grid[grid.index("--grid-s") + 1] == "0.5" and "--units" not in grid and grid[grid.index("--segment") + 1] == "free_play"
    assert "--samples" in grid and grid[grid.index("-r") + 1] == "003" and grid[-1] == agreement.FOLDER
    assert agreement.missing_choice("speech", "grid", "bob", None) and not agreement.missing_choice("speech", "exact", None, None)

    shown = text_of(agreement.compare(project, argv, "compared"))
    assert (project.parent / "compared" / "units.csv").is_file()
    assert "Each pair of coders" in shown and "Each code, each pair of coders" in shown and "0.000" in shown
    broken = text_of(agreement.compare(project, agreement.agreement_command(project, "speech", "units", "nobody", None, "late", None, False, "x"), "x"))
    assert "failed" in broken


def test_the_agreement_page_asks_the_model_for_a_clearer_codebook(project, monkeypatch):
    from macaw.coding.codebook import Codebook, CodeDefinition, write_codebook

    app = create_app(project)
    refine = callback(app, "agreement-refine-result.children")
    assert "Set the codebook" in text_of(refine(1, "x.yaml", None, None))
    write_codebook(Codebook("speech", "utterance", [CodeDefinition("question", "Asks.")]), project.parent / "codebooks" / "speech.yaml")
    project.write_text(project.read_text(encoding="utf-8") + "\ncoding:\n  codebook: codebooks/speech.yaml\n", encoding="utf-8")
    assert agreement.refined_name(project) == "codebooks/speech_refined.yaml"
    assert "agreement-refine-out" in text_of(create_app(project).layout())
    monkeypatch.setattr(agreement, "coded_tiers", lambda config: {"speech": ["ann", "llm:ollama/qwen", "bob"]})
    assert callback(app, "agreement-refine-coder.options")("/agreement") == ["ann", "bob"]
    ran = []
    monkeypatch.setattr(cli_bridge, "run_now", lambda argv, **kwargs: ran.append(argv) or (0, "question: revised"))
    shown = text_of(refine(1, " codebooks/speech_refined.yaml ", "bob", ["003"]))
    argv = ran[0]
    assert argv[1:3] == ["codebook", "refine"] and argv[argv.index("--coder") + 1] == "bob" and argv[argv.index("-r") + 1] == "003"
    assert "codebooks/speech_refined.yaml" in argv and "question: revised" in shown
    assert "Give a file name" in text_of(refine(1, " ", None, None))


def test_a_calibration_clip_is_offered_and_measured_for_every_recording_that_names_it(project):
    from macaw.exceptions import ExternalToolError
    from macaw.media.ffmpeg import run_ffmpeg

    folder = project.parent / "calibration"
    folder.mkdir()
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2", "-pix_fmt", "yuv420p", str(folder / "lab.mp4")])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    shown = text_of(review.body(project, "003"))
    assert "calibration clip lab.mp4" in shown and "scale: {calibration: name}" in shown
    measured = {"video": review.CLIP + "lab.mp4", "at": 1.0, "points": [[0, 0], [0, 50]], "length": 10, "object": None, "person": None}
    argv = review.action_command(project, "003", "set-scale", "", "ann", {}, measured)
    assert argv[argv.index("--calibration") + 1] == "lab.mp4" and "-r" not in argv
    assert review.action_command(project, "003", "clear-scale", review.CLIP + "lab.mp4", "ann", {})[-2:] == ["lab.mp4", "--clear"]
    uri, problem = review.frame_uri(project, "003", review.CLIP + "lab.mp4", 1.0)
    assert problem is None and uri.startswith("data:image/jpeg;base64,")


def test_two_clicks_mark_the_ends_and_a_third_starts_again():
    click = lambda x, y: {"points": [{"x": x, "y": y}]}
    first = review.clicked([], click(1, 2))
    both = review.clicked(first, click(3, 4))
    assert both == [[1.0, 2.0], [3.0, 4.0]] and review.clicked(both, click(5, 6)) == [[5.0, 6.0]]
    figure = review.scale_figure("data:image/jpeg;base64,AAAA", both)
    assert figure.data[0].type == "image" and list(figure.data[1].x) == [1.0, 3.0]


def test_the_frame_is_read_from_the_video_the_people_are_measured_on(project):
    from macaw.exceptions import ExternalToolError
    from macaw.media.ffmpeg import run_ffmpeg

    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2", "-pix_fmt", "yuv420p", str(video)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    uri, problem = review.frame_uri(project, "003", "room", 1.0)
    assert uri is None and "prep.video" in problem
    with open(project, "a", encoding="utf-8") as handle:
        handle.write("\nvideo:\n  normalise: false\n")
    uri, problem = review.frame_uri(project, "003", "room", 1.0)
    assert problem is None and uri.startswith("data:image/jpeg;base64,")


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


def test_results_give_metres_in_the_projects_unit_and_other_measures_as_stored(project):
    from macaw.project.manifest import FeatureManifest, FeatureSpec, write_manifest

    config = resolve_config(project)
    layout = layout_for(config)
    directory = layout.long_table_path("003")
    directory.mkdir(parents=True)
    curves = build_long_rows(
        values=pd.DataFrame({"t_start": np.arange(3.0), "wrist_speed_3d": [0.5, 1.0, 0.25], "wrist_speed": [10.0, 20, 30]}),
        scale=Scale.FRAME, recording_id="003", source="pose.kinematics3d", subject_id="parent",
    )
    whole = build_long_rows(
        values=pd.DataFrame({"t_start": [0.0], "interpersonal_distance_3d_mean": [1.0]}),
        scale=Scale.RECORDING, recording_id="003", source="pose.distance3d", subject_id="parent",
    )
    curves.to_parquet(directory / "pose.kinematics3d.parquet", index=False)
    whole.to_parquet(directory / "pose.distance3d.parquet", index=False)
    manifest = FeatureManifest()
    for name, unit, scale in (("wrist_speed_3d", "m/s", Scale.FRAME), ("wrist_speed", "px/s", Scale.FRAME), ("interpersonal_distance_3d_mean", "m", Scale.RECORDING)):
        manifest.add(FeatureSpec(name=name, description="", unit=unit, scale=scale, group="pose", extractor="pose.kinematics3d", extractor_version="0.1.0"))
    write_manifest(manifest, layout.manifest_path)
    graph = results.view(project, "003", "pose.kinematics3d", "wrist_speed_3d")
    assert list(graph.figure.data[0].y) == pytest.approx([50.0, 100.0, 25.0]) and graph.figure.layout.yaxis.title.text == "wrist_speed_3d (cm/s)"
    graph = results.view(project, "003", "pose.kinematics3d", "wrist_speed")
    assert list(graph.figure.data[0].y) == pytest.approx([10.0, 20.0, 30.0]) and graph.figure.layout.yaxis.title.text == "wrist_speed (px/s)"
    project.write_text(project.read_text(encoding="utf-8").replace("units: cm", "units: in"), encoding="utf-8")
    shown = text_of(results.view(project, "003", "pose.distance3d", "interpersonal_distance_3d_mean"))
    assert "interpersonal_distance_3d_mean (in)" in shown and "39.37" in shown


def test_min_max_keeps_the_extremes_of_every_run():
    t = np.arange(10_000.0)
    y = np.sin(t / 50.0)
    y[5000] = 9.0
    small_t, small_y = min_max_downsample(t, y, 200)
    assert small_t.size <= 200 and 9.0 in small_y and np.all(np.diff(small_t) > 0)
    assert min_max_downsample(t[:50], y[:50], 200)[0].size == 50


def test_the_review_page_lists_uncertain_speech_acts_and_saves_a_pick(project):
    from macaw.coding.codebook import Codebook, CodeDefinition, write_codebook
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
    write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(config).step_annotations_path("003", "coding.judge"))
    samples = layout_for(config).artifacts_dir("003", "coding.judge") / "samples.parquet"
    samples.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"unit_id": ["u0"] * 5, "sample": range(5), "label": ["ask"] * 3 + ["tell"] * 2,
                  "subject_id": "parent", "t_start": 1.0, "t_end": 1.5}).to_parquet(samples)

    shown = text_of(review.body(project, "003"))
    assert "Speech acts" in shown and "ask 3, tell 2" in shown and "apply-review" in shown and "act-pick" in shown
    # the child's unit the model was sure of waits too, as a random check
    assert "random check" in shown

    path = review.review_file(project, "003", "ann", {"parent|1.000|acts": "tell", "child|2.000|acts": "tell"})
    code, text = cli_bridge.run_now(["macaw", "review", "-c", str(project), "--apply", str(path)])
    assert code == 0, text
    reviewed = pd.read_parquet(layout_for(config).reviewed_codes_path("003"))
    assert reviewed[["annotator", "label", "t_start"]].values.tolist() == [["ann", "tell", 1.0], ["ann", "tell", 2.0]]
    assert "No coded unit is waiting" in text_of(review.body(project, "003"))


def test_every_step_is_listed_with_how_a_run_would_find_it(project):
    order, ran, counts, needs = run.step_states(project, ["003"])
    assert ran and set(ran) <= set(order) and "prep.media" in order
    assert counts["prep.media"] == {"held": 1} and "align.review" in needs["prep.media"]
    assert all(set(counted) <= {"new", "held"} for counted in counts.values())
    project.write_text(project.read_text(encoding="utf-8") + "\nalign:\n  method: none\n", encoding="utf-8")
    assert run.step_states(project, ["003"])[2]["prep.media"] == {"new": 1}


def test_pages_open_by_their_address():
    assert [page_for(path) for path in ("/review", "/settings", "/cameras", "/run", "/agreement", "/setup", "/", "/nowhere")] == [
        "review", "settings", "overview", "run", "agreement", "setup", "overview", "overview",
    ]


def test_the_recording_picker_shows_only_on_pages_of_one_recording(project):
    app = create_app(project)
    show = callback(app, "..page-overview.className")
    shown = {page: show(f"/{page}")[-1] for page in ("overview", "setup", "run", "review", "agreement", "segments", "results", "settings")}
    assert [page for page, style in shown.items() if style is None] == ["setup", "review", "segments", "results"]
    assert '"id": "recording"' in text_of(build_layout(project)) and '"recording-box"' in text_of(build_layout(project))
    hand_over = callback(app, "..setup-recording-shown.data")
    assert hand_over("003", "/review", *[None] * 4) == [no_update, "003", no_update, no_update]
    assert hand_over("003", "/review", None, "003", None, None) == [no_update] * 4
    assert hand_over("003", "/run", *[None] * 4) == [no_update] * 4


def test_each_step_has_a_run_button_of_its_own(project):
    order, _, counts, needs = run.step_states(project, ["003"])
    shown = text_of(run._step_label("prep.media", counts["prep.media"], needs["prep.media"], 1))
    assert '"type": "run-one"' in shown and '"step": "prep.media"' in shown
    assert run.run_command(project, ["003"], ["prep.media"], ["003"], ["prep.media", "text.asr"])[-2:] == ["-e", "prep.media"]


def test_the_overview_counts_what_there_is_and_what_waits(project):
    found = overview.standing(project)
    assert found.recordings == ["003"] and "prep.media" in found.steps
    assert found.grid["003", "prep.media"] == "held" and not found.n_prepared
    tiles = text_of(overview.tiles(resolve_config(project), found))
    assert "parent, child" in tiles and "known once prep.media has run" in tiles and "0 of" in tiles
    assert "Waits for you" in tiles and "1 check, 0 coded units" in tiles and '"/review"' in tiles and "nothing failed" in tiles
    grid = text_of(overview.grid(found))
    assert "cell held" in grid and '"data-label": "Sound and timing"' in grid and '"data-state": "waiting for a person"' in grid
    assert found.checks["003", "align.review"] == "waiting" and "cell check waiting" in grid and "Check: file offsets" in grid
    assert found.columns.index("align.review") < found.columns.index("prep.media")


def test_the_overview_grid_is_written_as_a_table(project, tmp_path):
    table = pd.read_csv(status.write_status(project, tmp_path / "status.csv"), dtype=str)
    found = overview.standing(project)
    assert table.columns.tolist() == ["recording_id", *found.columns]
    row = table.set_index("recording_id").loc["003"]
    assert row["prep.media"] == "held" and row["align.review"] == "waiting"
    said = pd.read_csv(tmp_path / "status_dictionary.csv").set_index("column")
    assert set(said.index) == set(table.columns)
    assert said.loc["align.review", "meaning"].startswith("Where the check by a person") and "failed" in said.loc["prep.media", "meaning"]


def test_the_status_table_of_a_run_into_another_folder_tells_that_folder(project, tmp_path, monkeypatch):
    elsewhere, seen, real = resolve_config(project, output_dir=tmp_path / "out2"), [], status.task_states
    monkeypatch.setattr(status, "task_states", lambda config, dag, recordings: seen.append(config.project.output_dir) or real(config, dag, recordings))
    status.write_status(elsewhere, tmp_path / "status.csv")
    assert [Path(found) for found in seen] == [tmp_path / "out2"]


def test_a_failed_step_shows_on_its_own_tile_and_row(project, monkeypatch):
    monkeypatch.setattr(status, "failed_last_time", lambda config_path: {task_id_for("003", "prep.media")})
    found = overview.standing(project)
    assert found.grid["003", "prep.media"] == "failed"
    failed = text_of(overview.tiles(resolve_config(project), found)[2])
    assert "Failed last time" in failed and "1 step, 1 recording" in failed and '"/run"' in failed and "tile bad" in failed
    assert overview.row_summary(found, "003").startswith("1 failed")


def test_a_square_leads_to_what_the_step_produced_waits_on_or_runs(project):
    assert overview.cell_link("003", "turns.structure", "current") == ("/results?recording=003&step=turns.structure", "Results")
    assert overview.cell_link("003", "prep.media", "held") == ("/review?recording=003", "Review")
    assert [overview.cell_link("003", "text.asr", state)[0] for state in ("new", "stale", "failed")] == ["/run?recording=003&step=text.asr"] * 3
    grid = text_of(overview.grid(overview.standing(project)))
    assert '"href": "/review?recording=003"' in grid and '"data-page": "Review"' in grid
    assert ui.asked("?recording=a%20b&step=text.asr") == {"recording": "a b", "step": "text.asr"} and ui.asked(None) == {}


def callback(app, output: str):
    # an output that another callback also sets carries a suffix of its own
    found = [key for key in app.callback_map if key == output or key.split("@")[0] == output.split("@")[0] or key.startswith(output)]
    return app.callback_map[found[0]]["callback"].__wrapped__


def review_update(app):
    return callback(app, "..review-body.children...review-message.children...recording.options")


# the states after the recording, the clicks, the uploads and the split: name, voices, acts, gaze, scale, camera, names, note, mixed, lines, picker
def review_states(by="", cameras=(), marks=None, frame=None, tags=None, text=None, corners=None, times=()):
    return [by, [], [], [], [], [], [], [], *([[]] * 6), list(cameras), [marks] if marks else [], [frame] if frame else [],
            [corners] if corners else [], [tags] if tags is not None else [], [text] if text is not None else [], [], [], [], [], list(times), [], [], [], []]


def test_a_link_picks_the_recording_and_step_on_the_page_it_opens(project, monkeypatch):
    app = create_app(project)
    linked = callback(app, "recording.value")
    assert linked("/review", "?recording=003") == "003" and linked("/results", "?recording=003&step=x") == "003"
    assert linked("/segments", "?recording=003") == "003" and linked("/setup", "?recording=003") == "003"
    for page, search in (("/review", "?recording=nowhere"), ("/run", "?recording=003"), ("/overview", "?recording=003")):
        assert linked(page, search) is no_update
    assert results.chosen_source(["text.asr", "turns.structure"], "turns.structure", "text.asr") == "turns.structure"
    assert results.chosen_source(["text.asr", "turns.structure"], "prep.video", "turns.structure") == "turns.structure"
    assert results.chosen_source(["text.asr"], None, None) == "text.asr" and results.chosen_source([], "x", None) is None
    assert results.out_of_date([], []) == []
    late = text_of(results.out_of_date(["pose.keypoints", "face.au"], ["names.json"]))
    assert "out of date" in late and "2 steps ran and 1 decision changed since" in late and "names.json" in late

    fill = callback(app, "..run-recordings.options...run-recordings.value...run-steps.options...run-steps.value...run-more.options...run-more.value...run-more-summary.children...run-summary.children..")
    monkeypatch.setattr(run, "ctx", SimpleNamespace(triggered_id="url"))
    linked = fill("/run", "?recording=003&step=prep.media", None, ["003"], ["text.asr", "turns.structure"], [])
    assert linked[1] == ["003"] and linked[3] == ["prep.media"] and linked[5] == []
    monkeypatch.setattr(run, "ctx", SimpleNamespace(triggered_id="run-state"))
    kept = fill("/run", "?recording=003&step=prep.media", "done", ["003"], ["text.asr"], [])
    assert kept[3] == ["text.asr"]


def test_the_grid_groups_steps_by_lane_and_folds_the_recordings_with_nothing_left():
    steps = ["text.asr", "prep.media", "pose.detect", "my.plugin"]
    needs = {"prep.media": [], "text.asr": ["prep.media"], "pose.detect": ["prep.media"], "my.plugin": ["text.asr"]}
    grid = {(rid, step): "current" for rid in "abc" for step in steps}
    grid["b", "text.asr"], grid["c", "pose.detect"] = "failed", "stale"
    found = overview.Standing(["a", "b", "c"], steps, grid, columns=steps, needs=needs)
    assert overview.lanes(found) == [("Prepare", ["prep.media"]), ("Speech", ["text.asr"]), ("Body", ["pose.detect"]), ("Other", ["my.plugin"])]
    assert overview.lane_of(overview.SPEECH_ACTS) == "Coding"
    assert [overview.row_summary(found, rid) for rid in "abc"] == ["done", "1 failed", "1 to run"]
    shown = text_of(overview.grid(found))
    assert "1 more recording, everything done" in shown and "done closed" in shown and shown.index('"b"') < shown.index('"a"')
    assert '"Other"' in shown and "Checks by a person" not in shown and '"--cols": 4' in shown
    everything = overview.Standing(["a"], steps, {("a", step): "current" for step in steps}, columns=steps, needs=needs)
    assert "fold-row" not in text_of(overview.grid(everything))


def test_the_folder_button_is_only_offered_on_the_machine_that_has_the_folder(project):
    assert "app-open-folder" in text_of(build_layout(project.resolve()))
    assert "app-open-folder" not in text_of(build_layout(project.resolve(), local=False))


def props_of(component, identity):
    found = []

    def walk(node):
        if hasattr(node, "to_plotly_json"):
            node = node.to_plotly_json()
        if isinstance(node, dict):
            if node.get("props", {}).get("id") == identity:
                found.append(node["props"])
            for value in node.get("props", {}).values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(component)
    assert len(found) == 1
    return found[0]


def test_your_name_is_taken_as_typed_without_leaving_the_box(project):
    """A file dropped on Import your coding does not move the focus, so a name waiting for it would be lost."""
    assert not props_of(review.layout(project), "review-by").get("debounce")


def test_your_name_starts_blank_is_kept_in_the_browser_and_is_needed_for_a_decision(project, monkeypatch):
    name = props_of(review.layout(project), "review-by")
    assert name["value"] == "" and name["persistence"] is True and name["persistence_type"] == "local"
    update = review_update(create_app(project))
    pressed = {"type": review.ACT, "action": "accept-offset", "target": "mic"}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=pressed, triggered=[{"value": 1}]))
    body, said, _ = update("003", [1], [], [], [], *review_states("  "))
    assert body is no_update and "Type your name first" in text_of(said)
    dropped = {"type": review.UPLOAD, "kind": review.CODING_UPLOAD}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=dropped, triggered=[{"value": "data:,"}],
                                                       inputs_list=[None, None, [{"id": dropped}]]))
    body, said, _ = update("003", [], ["data:,"], [], ["a.csv"], *review_states(""))
    assert body is no_update and "Type your name first" in text_of(said)


def test_speech_acts_count_what_is_decided_and_save_from_the_top_too(project):
    decided = callback(create_app(project), "acts-decided.children")
    assert decided(["ask", None, "tell"]) == "2 of 3 decided" == review.decided_text(2, 3)


def test_a_message_shows_its_backticked_command_as_code():
    shown = text_of(ui.with_code("no offset table yet: run `macaw align -r 001`, or set align.method"))
    assert '"Code"' in shown and "macaw align -r 001" in shown
    assert ui.with_code("an odd ` stays") == "an odd ` stays"


def test_short_tables_keep_to_their_content_and_long_pages_link_their_sections(project):
    assert ui.table(["a"], [[1]], fit=True).className == "flat fit" and ui.table(["a"], [[1]]).className == "flat"
    from macaw.interface.pages import settings, setup

    page = text_of(settings.layout(project))
    assert '"href": "#settings-project"' in page and '"href": "#settings-file"' in page and '"id": "settings-checks-by-a-person"' in page
    assert props_of(settings.layout(project), {"type": settings.WRAP, "key": "project.raw_dir"})["className"] == "wide"
    app = create_app(project)
    preview = callback(app, next(key for key in app.callback_map if "settings-changes.children" in key))
    ids = [{"type": settings.SETTING, "key": key} for key in ("project.raw_dir", "project.name")]
    values = [project_file.to_form(key, project_file.file_values(project)[key]) for key in ("project.raw_dir", "project.name")]
    assert preview(values[:1] + ["other"], ids)[1] == ["wide", "changed"]
    assert '"href": "#setup-room"' in text_of(setup.layout(project))


def test_the_settings_hints_stay_short():
    hints = project_file.hints()
    long = {key: hints[key] for _, _, keys in project_file.FORM for key in keys if len(hints.get(key, "")) > 90}
    assert not long


def test_the_run_page_says_beside_its_title_when_a_run_from_here_goes(project, monkeypatch):
    assert run.live(project) is None
    monkeypatch.setitem(run._CURRENT, str(project), SimpleNamespace())
    monkeypatch.setattr(cli_bridge, "exit_code", lambda current: None)
    monkeypatch.setattr(cli_bridge, "poll_status", lambda config_path: ("e", {"done": 3, "failed": 0}))
    assert "running" in text_of(run.live(project)) and "3 done" in text_of(run.live(project))
    monkeypatch.setattr(cli_bridge, "exit_code", lambda current: 1)
    assert "stopped, exit 1" in text_of(run.live(project))


def test_the_3d_page_has_its_tab_only_where_two_cameras_film(project):
    from macaw.interface import app as workbench
    from macaw.interface.pages import cameras, setup

    assert cameras not in workbench.shown_tabs(resolve_config(project)) and "nothing to set here" in text_of(cameras.layout(project))
    put(project.parent / "raw", "003", "side", "a.mp4")
    write_devices(project, cameras={"room": setup_camera(), "side": setup_camera(uses=("body",))}, microphones=mics())
    assert cameras in workbench.shown_tabs(resolve_config(project)) and "Board" in text_of(cameras.layout(project))
    assert "Cameras for three dimensions" not in text_of(setup.layout(project))


def calibrated(project, notes=(), together=40):
    from macaw.extractors.video.pose.board import describe_board
    from macaw.extractors.video.pose.cameras import CameraSetup, write_setup
    from test_cameras import look_at

    config = resolve_config(project)
    folder = config.project.calibration_dir / "lab"
    for name in ("room", "side"):
        (folder / name).mkdir(parents=True)
        (folder / name / "clip.mp4").write_bytes(b"")
    target = [0.0, 0.0, 2000.0]
    cameras = [look_at("room", [0, 0, 0], target), look_at("side", [1500, 0, 500], target)]
    metadata = {
        "made_by": "macaw calibrate-cameras", "made_at": "2026-10-31T10:00:00", "board": describe_board(config.pose3d.board.in_mm(config.project.units)),
        "error_px": {"room": 0.4, "side": 2.6}, "lens_error_px": {"room": 0.3, "side": 0.5}, "views": {"room": 120, "side": 98},
        "together": {"room and side": together}, "notes": list(notes),
    }
    return write_setup(CameraSetup(cameras, "mm", metadata), layout_for(config).camera_setup_path("lab"))


def test_a_setup_shows_each_camera_and_what_to_redo(project):
    from macaw.interface.pages import cameras

    config = resolve_config(project)
    assert cameras.setups(config) == [] and "No calibration yet" in text_of(cameras.setup_view(project, None))
    for name in ("room", "side"):
        (config.project.calibration_dir / "lab" / name).mkdir(parents=True)
        (config.project.calibration_dir / "lab" / name / "clip.mp4").write_bytes(b"")
    assert cameras.setups(config) == ["lab"]
    waiting = text_of(cameras.setup_view(project, "lab"))
    assert "not found yet" in waiting and "calibrate-cameras" in waiting
    shutil.rmtree(config.project.calibration_dir / "lab")

    calibrated(project, notes=["side.mp4 has no sound"], together=8)
    shown = text_of(cameras.setup_view(project, "lab"))
    assert "Found on 2026-10-31 10:00 from" in shown and "side.mp4 has no sound" in shown
    assert "up to 2.6 px" in shown and "8 (few: film more where both see it)" in shown
    assert "Across, ahead, up of the first camera (cm)" in shown and "150, 50, 0" in shown and "used by this project" not in shown
    project.write_text(project.read_text(encoding="utf-8") + "\npose3d:\n  board:\n    square: 7.45\n", encoding="utf-8")
    assert "pose3d.board differs now" in text_of(cameras.setup_view(project, "lab"))


def test_the_cameras_are_drawn_where_they_stand(project):
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.interface.pages import cameras

    setup = read_setup(calibrated(project))
    figure = cameras.cameras_figure(setup, "lab", "cm")
    shown = next(trace for trace in figure.data if trace.name == "cameras")
    assert list(shown.text) == ["room", "side"] and list(shown.x) == pytest.approx([0.0, 150.0]) and list(shown.y) == pytest.approx([0.0, 50.0])
    assert figure.layout.scene.xaxis.title.text == "across (cm)"
    inches = next(trace for trace in cameras.cameras_figure(setup, "lab", "in").data if trace.name == "cameras")
    assert list(inches.x) == pytest.approx([0.0, 150 / 2.54])


def test_the_view_follows_the_first_camera_whatever_axes_a_file_uses(project):
    from scipy.spatial.transform import Rotation

    from macaw.extractors.video.pose.cameras import Camera, CameraSetup, read_setup
    from macaw.interface.pages.results import in_first_camera

    setup = read_setup(calibrated(project))
    # the same cameras in a room whose z points up, as some calibration tools write it
    turn = Rotation.from_euler("x", -90, degrees=True)
    upright = CameraSetup([
        Camera(camera.name, camera.size, camera.matrix, camera.distortions, (Rotation.from_rotvec(camera.rotation) * turn.inv()).as_rotvec(), camera.translation)
        for camera in setup.cameras
    ], setup.units)
    points = np.random.default_rng(5).uniform(-1, 1, (20, 3)) + [0.0, 0.0, 2.0]
    assert np.allclose(in_first_camera(upright, turn.apply(points)), in_first_camera(setup, points))


def test_the_3d_view_opens_where_most_points_were_placed():
    points = pd.DataFrame({"t": [0.0, 1.0, 1.0, 1.0, 2.0, 2.0], "joint": list("abcdef")})
    assert results.fullest_moment(points) == 1.0


def test_the_3d_plays_beside_the_recording_once_it_has_a_copy_and_opens_at_the_fullest_moment(project, monkeypatch):
    import json

    points = pd.DataFrame({"t": [0.0, 1.0, 1.0, 2.5], "joint": list("abcd")})
    copy, strip = results.strip_3d(project, "dyad01", points)
    shown = json.loads(strip.to_plotly_json()["props"]["data-strip"])
    assert shown["start"] == 1.0 and shown["length"] >= 2.5 and shown["timeStore"] == results.TIME_3D and shown["video"] is None
    assert "Make a copy to play here" in text_of(copy)
    from macaw.project import previews

    preview = layout_for(resolve_config(project)).preview_path("dyad01")
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"mp4")
    monkeypatch.setattr(previews, "current_recording_preview", lambda config, rid: preview)
    player, strip = results.strip_3d(project, "dyad01", points)
    assert player.id == results.PLAYER_3D and json.loads(strip.to_plotly_json()["props"]["data-strip"])["video"] == results.PLAYER_3D


def test_choosing_a_setup_writes_it_into_the_project(project):
    from macaw.interface.pages import cameras

    calibrated(project)
    cameras.use_setup(project, "lab")
    assert resolve_config(project).pose3d.calibration == "lab" and "# short project name" in project.read_text(encoding="utf-8")
    assert "used by this project" in text_of(cameras.setup_view(project, "lab"))


def test_the_board_says_which_paper_it_fits(project):
    from macaw.interface.pages import cameras

    shown = text_of(cameras.board_view(project))
    assert "fits A2, A1." in shown and "--paper" in shown
    drawn = resolve_config(project).project.calibration_dir / "board_A2.png"
    drawn.parent.mkdir(parents=True)
    drawn.write_bytes(b"png")
    assert path_from_url(file_url(drawn).removeprefix("/files/"), resolve_config(project)) == drawn.resolve()


def test_a_setup_drawing_and_a_network_path_are_served(project):
    from macaw.interface.files import posix_of, url_path

    drawing = resolve_config(project).project.setups_dir / "main.png"
    drawing.write_bytes(b"png")
    assert path_from_url(file_url(drawing).removeprefix("/files/"), resolve_config(project)) == drawing.resolve()
    for posix in ("//labserver/share/out/a.mp4", "C:/data/a.mp4", "/home/lab/a.mp4"):
        assert posix_of(url_path(posix)) == posix


def test_a_recording_s_settings_that_do_not_read_are_not_left_on_disk(project):
    from macaw.interface import project_file

    path = project_file.save_for_recording(project, "003", {"segment.task_start_s": 1.0})
    before = path.read_text(encoding="utf-8")
    with pytest.raises(MacawError):
        project_file.save_for_recording(project, "003", {"segment.task_start_s": "soon"})
    assert path.read_text(encoding="utf-8") == before


def test_the_whole_workbench_lays_out(project):
    shown = text_of(build_layout(project.resolve()))
    assert all(name in shown for name in ("overview-grid", "run-steps", "review-body", "results-view", "settings-yaml", "theme-choice", "style-choice"))


def _css_blocks():
    import re

    text = (Path(ui.__file__).parent / "assets" / "macaw.css").read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    blocks = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text):
        blocks.setdefault(" ".join(selector.split()), []).append(dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body)))
    return text, blocks


def test_each_style_sets_only_tokens_and_the_dark_palette_covers_its_colours():
    from macaw.interface.pages.settings import STYLES

    _, blocks = _css_blocks()
    default = blocks[":root"][0]
    dark = [blocks[':root:not([data-theme="light"])'][0], blocks[':root[data-theme="dark"]'][0]]
    assert dark[0] == dark[1]
    assert STYLES[0]["value"] == "lab"
    for style in [entry["value"] for entry in STYLES[1:]]:
        tokens = blocks[f':root[data-style="{style}"]'][0]
        assert set(tokens) <= set(default), style
        coloured = {name for name, value in tokens.items() if "#" in value}
        assert coloured and coloured <= set(dark[0]), (style, coloured - set(dark[0]))


def test_every_font_ships_with_its_licence():
    import re

    text, _ = _css_blocks()
    fonts = Path(ui.__file__).parent / "assets" / "fonts"
    files = re.findall(r'url\("fonts/([^"]+)"\)', text)
    assert len(files) == 6
    for name in files:
        family = re.sub(r"(-\d00)?\.woff2$", "", name)
        assert (fonts / name).is_file() and "SIL Open Font License" in (fonts / f"{family}-LICENSE.txt").read_text(encoding="utf-8")


def test_a_file_that_does_not_check_still_opens_the_settings(project):
    project.write_text(project.read_text(encoding="utf-8") + "\npose:\n  review: sometimes\n", encoding="utf-8")
    shown = text_of(build_layout(project.resolve()))
    assert "settings-yaml" in shown and "pose.review" in shown and "run-steps" not in shown


def test_a_form_edit_keeps_the_files_comments(project):
    text = project.read_text(encoding="utf-8")
    edited = project_file.with_values(
        text, {"execution.n_workers": 2, "pose.backend": "yolo", "project.n_subjects": 3, "project.roles": ["parent", "child", "sibling"]}
    )
    assert "# short project name" in edited and "n_workers: 2" in edited and "roles: [parent, child, sibling]" in edited
    assert project_file.check(project, edited).pose.backend == "yolo"
    with pytest.raises(MacawError, match="setup main names child, parent"):
        project_file.check(project, project_file.with_values(text, {"project.roles": ["a", "b"]}))


def test_a_yes_or_no_setting_is_asked_as_one_and_written_back(project):
    assert project_file.control("video.normalise") == {"kind": "yes_no", "optional": False}
    # null keeps the model's own default, so the answer may be left empty
    assert project_file.control("coding.llm.think") == {"kind": "yes_no", "optional": True}
    form = {key: project_file.to_form(key, value) for key, value in project_file.file_values(project).items()}
    form["video.normalise"] = False
    assert project_file.changes(project, form) == {"video.normalise": (True, False)}
    edited = project_file.with_values(project.read_text(encoding="utf-8"), {"video.normalise": False})
    assert project_file.check(project, edited).video.normalise is False
    with pytest.raises(MacawError, match="video.normalise"):
        project_file.check(project, project_file.with_values(edited, {"video.fps": 30.0}))


def test_changes_compare_the_form_with_the_file(project):
    form = {key: project_file.to_form(key, value) for key, value in project_file.file_values(project).items()}
    assert project_file.changes(project, form) == {}
    form["execution.n_workers"] = 2
    form["text.language"] = ""
    assert project_file.changes(project, form) == {"execution.n_workers": (4, 2)}


def test_a_setting_that_does_not_check_is_refused_and_leaves_nothing_behind(project):
    before = sorted(project.parent.iterdir())
    with pytest.raises(MacawError, match="pose.review"):
        project_file.check(project, project_file.with_values(project.read_text(encoding="utf-8"), {"pose.review": "sometimes"}))
    assert sorted(project.parent.iterdir()) == before


def test_a_change_names_the_steps_whose_results_go_out_of_date(project):
    after = project_file.check(project, project_file.with_values(project.read_text(encoding="utf-8"), {"turns.max_gap_s": 3.0}))
    names = project_file.runs_again(resolve_config(project), after)
    assert "turns.structure" in names and "turns.transitions" in names and "prep.media" not in names


def test_saving_keeps_the_files_line_endings(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_bytes(b"a: 1\r\n")
    project_file.save(path, "a: 2\n")
    assert path.read_bytes() == b"a: 2\r\n"


def test_the_form_hints_come_from_the_packaged_comments():
    hints = project_file.hints()
    assert "ISO code" in hints["text.language"] and "report titles" in hints["project.name"]
    assert project_file.control("align.review") == {"kind": "choice", "options": ["always", "flagged", "skip"], "optional": False}


def test_the_latest_run_is_named_by_when_it_began(project, monkeypatch):
    assert cli_bridge.latest_run(project) == "Nothing has run yet."
    monkeypatch.setattr(cli_bridge, "poll_status", lambda config_path: ("20261031-214220-e17d5e", {"done": 36, "failed": 0}))
    assert cli_bridge.latest_run(project) == "Latest run 2026-10-31 21:42: 36 done"


def test_the_segments_page_plays_a_copy_and_saves_what_is_marked(project):
    from macaw.exceptions import ExternalToolError
    from macaw.interface.pages import segments
    from macaw.media.ffmpeg import run_ffmpeg
    from macaw.media.probe import probe
    from macaw.project.previews import make_recording_preview

    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2", "-f", "lavfi", "-i", "sine=duration=2",
                    "-pix_fmt", "yuv420p", "-shortest", str(video)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    assert "Make a copy to play here" in text_of(segments.body(project, "003"))
    copy = probe(make_recording_preview(resolve_config(project, recording_id="003"), "003"))
    assert copy.video_streams() and copy.audio_streams()
    assert segments.PLAYER in text_of(segments.body(project, "003"))

    argv = segments.segment_command(project, "003", "free_play", 0.5, 1.5)
    assert argv[:2] == ["macaw", "set-segment"] and argv[-4:] == ["--start", "0.5", "--end", "1.5"]
    code, text = cli_bridge.run_now(argv, cwd=project.parent)
    assert code == 0, text
    assert "free_play" in text_of(segments.body(project, "003"))
    code, text = cli_bridge.run_now(segments.segment_command(project, "003", "free_play"), cwd=project.parent)
    assert code == 0, text
    assert "No segments yet" in text_of(segments.body(project, "003"))


def test_the_analysed_part_is_written_to_the_recordings_own_file(project):
    from macaw.interface.pages import segments

    path = segments.set_window(project, "003", 1.5, 20.0)
    assert path.name == "003.yaml"
    own = resolve_config(project, recording_id="003")
    assert (own.segment.task_start_s, own.segment.task_end_s) == (1.5, 20.0)
    assert "Analysed part: 1.5 s to 20 s." in text_of(segments.body(project, "003"))


def test_a_long_recordings_times_reach_the_command_in_full(project):
    from macaw.interface.pages import segments

    argv = segments.segment_command(project, "003", "late", 10800.25, 12345.678)
    assert argv[-4:] == ["--start", "10800.25", "--end", "12345.678"]
    segments.set_window(project, "003", 10800.25, 12345.678)
    assert "Analysed part: 10800.25 s to 12345.678 s." in text_of(segments.body(project, "003"))


def test_a_segment_is_set_only_on_a_recording_the_project_has(project):
    from macaw.cli.segments import set_segment_command

    with pytest.raises(MacawError, match="not one of this project's recordings"):
        set_segment_command(config=project, recording="004", name="a", start=1.0, end=2.0, clear=False)
    with pytest.raises(MacawError, match="give --start and --end"):
        set_segment_command(config=project, recording="003", name="a", start=1.0, end=None, clear=False)


def test_the_overview_asks_for_the_setup_first_until_it_is_saved(project):
    assert overview.setup_message(overview.standing(project)) is None
    (project.parent / "setups" / "main.yaml").unlink()
    first = text_of(overview.setup_message(overview.standing(project)))
    assert "Describe the setup" in first and '"/setup"' in first
    assert "not saved yet" in text_of(run.layout(project))


def test_review_steps_through_the_recordings_and_lists_what_waits_and_the_tags(project):
    options = [{"label": "a", "value": "a"}, {"label": "b", "value": "b"}, {"label": "c", "value": "c"}]
    assert review.stepped(options, "b", 1) == "c" and review.stepped(options, "c", 1) == "c"
    assert review.stepped(options, "a", -1) == "a" and review.stepped(options, "gone", 1) == "a" and review.stepped([], None, 1) is None
    from macaw.project.review_notes import add_note

    add_note(layout_for(resolve_config(project)).review_notes_path("003"), ["rerun"], "pose again", "ann")
    assert review.recording_label(resolve_config(project), "003", 2) == "003 · 2 waiting · rerun"


def test_a_note_is_kept_with_its_tags_and_can_be_taken_back(project, monkeypatch):
    from macaw.cli.main import app as cli
    from typer.testing import CliRunner

    monkeypatch.setattr(review.cli_bridge, "run_now", lambda argv, cwd=None, **_: (0, CliRunner().invoke(cli, argv[1:]).output))
    update = review_update(create_app(project))
    pressed = {"type": review.ACT, "action": "add-note", "target": ""}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=pressed, triggered=[{"value": 1}]))
    _, said, _ = update("003", [1], [], [], [], *review_states("ann", tags=[], text=" "))
    assert "Tick a tag or write the note first" in text_of(said)
    body, said, _ = update("003", [1], [], [], [], *review_states("ann", tags=["trim the start"], text="cut the first minute"))
    assert "kept in" in text_of(said) and "cut the first minute" in text_of(body) and "trim the start" in text_of(body)
    from macaw.project.review_notes import read_notes

    note = read_notes(layout_for(resolve_config(project)).review_notes_path("003"))[0]
    assert note["tags"] == ["trim the start"] and note["by"] == "ann"
    with pytest.raises(Exception, match="not in qc.note_tags"):
        CliRunner().invoke(cli, ["note", "-c", str(project), "-r", "003", "--tag", "nonsense"], catch_exceptions=False)
    removed = {"type": review.ACT, "action": "remove-note", "target": note["id"]}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=removed, triggered=[{"value": 1}]))
    body, said, _ = update("003", [1], [], [], [], *review_states("ann"))
    assert "taken back" in text_of(said) and "cut the first minute" not in text_of(body)


def test_a_person_clicks_each_person_on_a_frame_and_the_names_are_saved(project, monkeypatch):
    from macaw.cli.main import app as cli
    from macaw.interface.pages import naming
    from macaw.media.ffmpeg import run_ffmpeg
    from typer.testing import CliRunner

    # a camcorder's wide picture stored in narrower pixels: the frame is shown as wide as it is seen
    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=gray:s=480x360:r=10:d=3", "-vf", "setsar=4/3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    shown, problem = naming.frame_of(project, "003", "room", 1.0)
    assert problem is None and (shown["width"], shown["height"]) == (640, 360) and shown["uri"].startswith("data:image/jpeg")
    assert shown["detected"] is None
    assert "No frame at 9 s" in naming.frame_of(project, "003", "room", 9.0)[1]

    people = ["parent", "child"]
    marks = naming.marked({}, "parent", {"points": [{"x": 160, "y": 180}]}, shown)
    assert marks == {"parent": [0.25, 0.5]}
    marks = naming.marked(marks, "child", {"points": [{"x": 700, "y": 90}]}, shown)
    assert marks["child"] == [1.0, 0.25]
    figure = naming.marks_figure(shown, marks, people)
    assert len(figure.data) == 3 and figure.data[1].marker.color != figure.data[2].marker.color

    monkeypatch.setattr(review.cli_bridge, "run_now", lambda argv, cwd=None, **_: (0, CliRunner().invoke(cli, argv[1:]).output))
    update = review_update(create_app(project))
    saving = {"type": review.ACT, "action": "name-people", "target": ""}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=saving, triggered=[{"value": 1}]))
    _, said, _ = update("003", [1], [], [], [], *review_states(" ", ["room"], marks, shown))
    assert "Type your name first" in text_of(said)
    body, said, _ = update("003", [1], [], [], [], *review_states("ann", ["room"], marks, shown))
    assert "on room, child at 1 s, parent at 1 s" in text_of(said)
    listed = text_of(naming.saved(resolve_config(project), "003", "room", people))
    assert "25% across, 50% down" in listed and "100% across, 25% down" in listed
    marked = text_of(naming.saved(resolve_config(project), "003", "room", people, (2.0, 3.0)))
    assert "no, outside the analysed part" in marked
    taking = {"type": review.ACT, "action": "take-back-click", "target": "1|child"}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=taking, triggered=[{"value": 1}]))
    body, said, _ = update("003", [1], [], [], [], *review_states("", ["room"]))
    assert "took back 1 click on room at 1 s" in text_of(said)
    left = text_of(naming.saved(resolve_config(project), "003", "room", people))
    assert "25% across, 50% down" in left and "100% across, 25% down" not in left and "take-back-click" in text_of(body)
    clearing = {"type": review.ACT, "action": "clear-names", "target": ""}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=clearing, triggered=[{"value": 1}]))
    _, said, _ = update("003", [1], [], [], [], *review_states("", ["room"]))
    assert "no clicks on room now" in text_of(said)
    assert "Nobody is named on room yet" in text_of(naming.saved(resolve_config(project), "003", "room", people))


def test_a_person_marks_an_area_to_ignore_by_two_corners_and_takes_it_back(project, monkeypatch):
    from macaw.cli.main import app as cli
    from macaw.interface.pages import naming
    from typer.testing import CliRunner

    shown = {"uri": "data:image/jpeg;base64,", "width": 640, "height": 360, "rid": "003", "camera": "room", "at": 1.0, "detected": None, "areas": []}
    corners = naming.cornered([], {"points": [{"x": 640, "y": 180}]}, shown)
    assert corners == [[1.0, 0.5]]
    corners = naming.cornered(corners, {"points": [{"x": 480, "y": 0}]}, shown)
    assert corners == [[1.0, 0.5], [0.75, 0.0]] and naming.area_of(corners) == (0.75, 0.0, 1.0, 0.5)
    assert naming.cornered(corners, {"points": [{"x": 0, "y": 0}]}, shown) == [[0.0, 0.0]]
    figure = naming.marks_figure(shown, {}, ["parent", "child"], corners)
    assert len(figure.layout.shapes) == 1 and (figure.layout.shapes[0].x0, figure.layout.shapes[0].x1) == (480.0, 640.0)
    assert naming.areas_kept(resolve_config(project), "room") is None

    monkeypatch.setattr(review.cli_bridge, "run_now", lambda argv, cwd=None, **_: (0, CliRunner().invoke(cli, argv[1:]).output))
    update = review_update(create_app(project))
    keeping = {"type": review.ACT, "action": "ignore-area", "target": ""}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=keeping, triggered=[{"value": 1}]))
    _, said, _ = update("003", [1], [], [], [], *review_states("", ["room"], None, shown, corners=corners[:1]))
    assert "click two opposite corners" in text_of(said)
    body, said, _ = update("003", [1], [], [], [], *review_states("", ["room"], None, shown, corners=corners))
    assert "room in setup main: 1: 75% to 100% across, 0% to 50% down" in text_of(said)
    assert resolve_config(project).setup.ignore == {"room": [(0.75, 0.0, 1.0, 0.5)]}
    assert "75% to 100% across, 0% to 50% down" in text_of(body) and "take-back-area" in text_of(body) and "Ignore this area" in text_of(body)
    kept = naming.marks_figure({**shown, "areas": [[0.75, 0.0, 1.0, 0.5]]}, {}, ["parent", "child"])
    assert len(kept.layout.shapes) == 1 and kept.layout.shapes[0].line.dash == "dash"
    taking = {"type": review.ACT, "action": "take-back-area", "target": "1"}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=taking, triggered=[{"value": 1}]))
    body, said, _ = update("003", [1], [], [], [], *review_states("", ["room"]))
    assert "no area to ignore" in text_of(said) and resolve_config(project).setup.ignore == {} and "take-back-area" not in text_of(body)


def test_a_click_shows_whom_it_lands_on_among_the_people_found():
    from macaw.interface.pages import naming

    joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    detected = [
        {"track_id": track, "frame": 5, "t": 0.5, "joint": joint, "x": x0 + dx, "y": 100 + dy}
        for track, x0 in ((1, 100), (2, 400)) for joint, (dx, dy) in zip(joints, [(0, 0), (60, 0), (0, 120), (60, 120)])
    ]
    frame = {"width": 640, "height": 360, "detected": detected}
    assert naming.landed(detected, [130 / 640, 160 / 360], frame) == (1, "")
    assert naming.landed(detected, [430 / 640, 160 / 360], frame) == (2, "")
    assert naming.landed(detected, [300 / 640, 20 / 360], frame) == (None, "nobody was found there")
    assert "parent: nobody was found there" in naming.where_they_landed(frame, {"parent": [300 / 640, 20 / 360]})


def placed_project(project):
    from setup_helpers import write_setup as write_devices

    path = write_devices(project, cameras={"room": setup_camera()}, microphones=mics())
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["room"] = {
        "people": {"parent": {"at": [100, 200], "eyes": 115, "facing": 90}, "child": {"at": [200, 200], "eyes": 70, "facing": 270}},
        "cameras": {"room": {"at": [150, 0], "height": 120, "facing": 0, "lens": 70}},
        "things": {"table": {"at": [150, 200], "height": 72, "size": [60, 50]}},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_the_setup_page_shows_the_editor_and_what_each_recording_holds(project):
    from macaw.interface.pages import setup

    shown = text_of(setup.layout(project))
    assert "The setup" in shown and "room-main" in shown and "room-top" in shown and "Save the setup" in shown
    (project.parent / "raw" / "003" / "stray.wav").write_bytes(b"")
    held = text_of(setup.holdings(project))
    assert "room: a.mp4" in held and "parent_mic: a.wav" in held and "stray.wav lies outside the device folders" in held


def test_the_editor_holds_every_person_device_and_thing_placed_or_not(project):
    from macaw.interface.pages import room

    placed_project(project)
    model = room.model_of(resolve_config(project), "main")
    assert list(model["people"]) == ["parent", "child"] and list(model["cameras"]) == ["room"]
    assert model["cameras"]["room"]["face"] == ["parent", "child"] and model["cameras"]["room"]["x"] == 150
    assert model["microphones"]["parent_mic"] == {"x": None, "y": None, "height": None, "hears": ["parent"]}
    assert model["things"]["table"] == {"x": 150, "y": 200, "height": 72, "width": 60, "depth": 50}
    assert room.room_of(model)["microphones"] == {"parent_mic": {}, "child_mic": {}} and "size" not in room.room_of(model)
    walled = room.moved(model, room.WALLS, "room", {"width": 400, "length": 350})
    assert room.room_of(walled)["size"] == [400, 350]
    listed = text_of(room.item_list(model, {"kind": "cameras", "name": "room"}))
    assert "not placed" in listed and "room-item chosen" in listed and "Set the room's size first" in listed and "room-alert" not in listed
    assert "Set the room's size first" not in text_of(room.item_list(walled, None))
    panel = text_of(room.panel(model, {"kind": "cameras", "name": "room"}))
    assert "Whom it records" in panel and "Lens angle" in panel and "Remove this camera" in panel and "Take out of the room" in panel
    assert "What it sees" in panel and "A pixel spans 1.03" in panel and "Nothing chosen" in text_of(room.panel(model, None))
    seen = room.sight(*room.loose_setup(model), "room")
    assert [row["person"] for row in seen["rows"]] == ["child", "parent"] and seen["spread"][:2] == ("child", "parent")
    assert round(seen["rows"][1]["distance"]) == 206 and round(seen["rows"][1]["angle"]) == 76 and seen["rows"][1]["inside"]
    # turned away, the camera sees nobody, and the list marks it
    model["cameras"]["room"]["facing"] = 180
    assert room.troubles(room.sight(*room.loose_setup(model), "room")) == ["child is outside the picture", "parent is outside the picture"]
    assert "room-alert" in text_of(room.item_list(model, None))


def test_editing_changes_the_model_and_says_why_not(project):
    from macaw.interface.pages import room

    placed_project(project)
    model = room.model_of(resolve_config(project), "main")
    moved = room.changed_by(project, model, "room-moved", None, {"kind": "people", "name": "parent", "fields": {"x": 82, "y": 250}})[0]
    assert (moved["people"]["parent"]["x"], moved["people"]["parent"]["y"]) == (82, 250)
    turned = room.changed_by(project, model, {"type": room.FIELD, "kind": "people", "name": "parent", "field": "facing"}, 370)[0]
    assert turned["people"]["parent"]["facing"] == 10
    off = room.changed_by(project, model, {"type": room.USE, "kind": "cameras", "name": "room", "use": "body", "role": "parent"}, [])[0]
    assert off["cameras"]["room"]["body"] == ["child"] and room.own_colour(off, "cameras", "room") is None
    # a device the setup gives to one person alone is drawn in their colour
    alone = room.used(room.used(off, "cameras", "room", "face", "parent", False), "cameras", "room", "gaze", "parent", False)
    assert room.own_colour(alone, "cameras", "room") == alone["colours"]["child"]
    assert room.own_colour(model, "microphones", "parent_mic") == model["colours"]["parent"]
    gone = room.changed_by(project, model, "room-moved", None, {"op": "unplace", "kind": "people", "name": "child"})[0]
    assert gone["people"]["child"]["x"] is None and gone["people"]["child"]["eyes"] is None
    back = room.changed_by(project, gone, {"type": room.ACT, "kind": "people", "name": "child", "act": "place"}, 1)[0]
    assert back["people"]["child"]["x"] is not None
    renamed, chosen, _ = room.changed_by(project, model, {"type": room.RENAME, "kind": "cameras", "name": "room"}, "wide")
    assert "wide" in renamed["cameras"] and "room" not in renamed["cameras"] and chosen == {"kind": "cameras", "name": "wide"}
    picked = room.changed_by(project, model, {"type": room.PICK, "kind": "things", "name": "table"}, 1)
    assert picked[0] is no_update and picked[1] == {"kind": "things", "name": "table"}
    assert room.changed_by(project, model, {"type": room.PICK, "kind": "things", "name": "table"}, None) == (no_update, no_update, no_update)
    toy, chosen, _ = room.changed_by(project, model, "room-add", 1, None, "things", "toy")
    # what is added is chosen and waits to be pinned where a click puts it
    assert chosen == {"kind": "things", "name": "toy"} and toy["things"]["toy"]["x"] is None
    assert "Click in the plan" in text_of(room.panel(toy, chosen))
    for kind, name, reason in (("things", "table", "already in this setup"), ("cameras", "parent", "already in this setup"),
                               ("things", "two words", "letters, digits"), ("things", "camera", "names something else")):
        with pytest.raises(MacawError, match=reason):
            room.changed_by(project, model, "room-add", 1, None, kind, name)
    with pytest.raises(MacawError, match="under Settings"):
        room.changed_by(project, model, {"type": room.ACT, "kind": "people", "name": "child", "act": "remove"}, 1)
    removed = room.changed_by(project, model, {"type": room.ACT, "kind": "things", "name": "table", "act": "remove"}, 1)
    assert "table" not in removed[0]["things"] and removed[1] is None


def test_a_camera_ticked_as_recording_depth_is_saved_so(project):
    from macaw.interface.pages import room

    placed_project(project)
    model = room.model_of(resolve_config(project), "main")
    assert model["cameras"]["room"]["depth"] is False and "Records depth" in text_of(room.panel(model, {"kind": "cameras", "name": "room"}))
    ticked = room.changed_by(project, model, {"type": room.USE, "kind": "cameras", "name": "room", "use": room.DEPTH, "role": room.DEPTH}, [room.DEPTH])[0]
    assert ticked["cameras"]["room"]["depth"] is True and ticked["cameras"]["room"]["body"] == model["cameras"]["room"]["body"]
    path = room.save(project, "main", ticked)
    assert "depth: true" in path.read_text(encoding="utf-8")
    assert resolve_config(project).setup.cameras["room"].depth and room.model_of(resolve_config(project), "main")["cameras"]["room"]["depth"]


def test_saving_the_setup_keeps_a_camera_s_areas_to_ignore_under_a_new_name_too(project):
    from macaw.cli.main import app as cli
    from macaw.interface.pages import room
    from typer.testing import CliRunner

    placed_project(project)
    assert CliRunner().invoke(cli, ["ignore-area", "-c", str(project), "--camera", "room", "--area", "0.75,0,1,0.5"]).exit_code == 0
    model = room.model_of(resolve_config(project), "main")
    room.save(project, "main", model)
    assert resolve_config(project).setup.ignore == {"room": [(0.75, 0.0, 1.0, 0.5)]}
    renamed, _ = room.renamed(model, "cameras", "room", "wide")
    room.save(project, "main", renamed)
    assert resolve_config(project).setup.ignore == {"wide": [(0.75, 0.0, 1.0, 0.5)]}
    room.save(project, "main", room.removed(renamed, "cameras", "wide"))
    assert resolve_config(project).setup.ignore == {}


def test_saving_the_setup_keeps_an_area_the_review_page_marked_after_the_setup_page_was_opened(project):
    from macaw.cli.main import app as cli
    from macaw.interface.pages import room
    from typer.testing import CliRunner

    placed_project(project)
    opened = room.model_of(resolve_config(project), "main")
    assert CliRunner().invoke(cli, ["ignore-area", "-c", str(project), "--camera", "room", "--area", "0.75,0,1,0.5"]).exit_code == 0
    room.save(project, "main", opened)
    assert resolve_config(project).setup.ignore == {"room": [(0.75, 0.0, 1.0, 0.5)]}


def test_adding_a_person_adds_a_role_to_the_project(project):
    from macaw.interface.pages import room

    model = room.model_of(resolve_config(project), "main")
    after, chosen, said = room.changed_by(project, model, "room-add", 1, None, "people", "sibling")
    config = resolve_config(project)
    assert config.project.roles == ["parent", "child", "sibling"] and config.project.n_subjects == 3
    assert after["roles"][-1] == "sibling" and "sibling" in after["colours"] and "every recording" in text_of(said)
    with pytest.raises(MacawError, match="names something else"):
        room.changed_by(project, after, "room-add", 1, None, "people", "other")


def test_the_setup_is_saved_whole_or_refused_with_a_reason(project):
    from macaw.interface.pages import room

    placed_project(project)
    model = room.model_of(resolve_config(project), "main")
    model["people"]["child"]["eyes"] = 80.5
    model, _ = room.added(model, "things", "toy")
    model["things"]["toy"].update(x=150, y=190, height=75)
    path = room.save(project, "main", model)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["room"]["people"]["child"]["eyes"] == 80.5 and saved["room"]["things"]["toy"] == {"at": [150, 190], "height": 75}
    assert saved["microphones"]["parent_mic"] == {"hears": ["parent"]} and saved["cameras"]["room"]["face"] == ["parent", "child"]
    assert room.comparable(room.model_of(resolve_config(project), "main")) == room.comparable(model)
    assert room.save(project, "home", model) == project.parent / "setups" / "home.yaml"
    for change, reason in (
        (lambda m: m["microphones"]["parent_mic"].update(hears=[]), "carries nobody's voice"),
        (lambda m: m["microphones"]["child_mic"].update(hears=["parent"]), "both carry parent's voice alone"),
        (lambda m: [m["cameras"]["room"].update({use: []}) for use in ("body", "face", "gaze")], "read for nobody"),
    ):
        broken = room.model_of(resolve_config(project), "main")
        change(broken)
        with pytest.raises(MacawError, match=reason):
            room.save(project, "main", broken)
    with pytest.raises(MacawError, match="letters, digits"):
        room.save(project, "home visits", model)


def test_the_editors_buttons_reach_the_saved_file(project):
    from macaw.interface.pages import room

    placed_project(project)
    app = create_app(project)
    pressed = lambda button: callback(app, next(key for key, value in app.callback_map.items() if any(item["id"] == button for item in value["inputs"])))
    keep, draw = pressed("room-save"), pressed("room-draw")
    frame = callback(app, next(key for key in app.callback_map if "room-frame.data" in key))
    model = room.model_of(resolve_config(project), "main")
    model["people"]["parent"]["x"] = 90
    message, saved, files = keep(1, model, "main")
    assert "Saved setups/main.yaml" in text_of(message) and saved == room.comparable(model) and "room: a.mp4" in text_of(files)
    assert resolve_config(project).setup.room.people["parent"].at == (90, 200)
    model["people"]["parent"]["x"] = 95
    assert "save it first" in text_of(draw(1, model, saved, "main"))
    # looking through a camera chooses it, so its numbers are beside the view
    assert frame("camera", "room", None, 0, model)[2] == {"kind": "cameras", "name": "room"}
    assert frame("free", "room", "003", 0, model) == (None, None, no_update)
    # a camera without a height is drawn from a stand-in height, which the page says
    model["cameras"]["room"]["height"] = None
    assert "Give room's height first" in text_of(frame("camera", "room", "003", 0, model)[1])


def test_the_page_loads_the_room_editor_through_an_import_map(project):
    index = create_app(project).index_string
    assert '<script type="importmap">' in index and "room/three/three.module.min.js" in index
    assert 'type="module" src="/assets/room/editor.js?m=' in index and index.index("importmap") < index.index("editor.js")


def test_a_page_opens_in_the_look_chosen_before(project):
    import subprocess

    from macaw.interface.pages import settings

    app = create_app(project)
    index = app.index_string
    assert index.index("macaw.style") < index.index("{%css%}") and index.index("macaw.theme") < index.index("{%css%}")
    # what the settings' callbacks store is what the head reads
    stored = [str(item) for item in app._inline_scripts]
    assert any("localStorage.setItem('macaw.style'" in script for script in stored)
    assert any("localStorage.setItem('macaw.theme'" in script for script in stored)
    node = shutil.which("node")
    if node is None:
        return
    script = settings.early_look().removeprefix("<script>").removesuffix("</script>")
    for stored_values, expected in (({"macaw.style": "blueprint", "macaw.theme": "dark"}, '{"style":"blueprint","theme":"dark"}'),
                                    ({"macaw.style": "lab", "macaw.theme": "system"}, "{}"), ({"macaw.style": "<x>"}, "{}")):
        program = (
            f"const kept = {json.dumps(stored_values)}; globalThis.localStorage = {{ getItem: (key) => kept[key] ?? null }};"
            " globalThis.document = { documentElement: { dataset: {} } };"
            f" {script} console.log(JSON.stringify(document.documentElement.dataset));"
        )
        shown = subprocess.run([node, "-e", program], capture_output=True, text=True, check=True).stdout.strip()
        assert shown == expected


def test_changing_the_unit_converts_the_lengths_already_typed(project, monkeypatch):
    from macaw.interface.pages import settings

    placed_project(project)
    recordings = project.parent / "recordings"
    recordings.mkdir(exist_ok=True)
    (recordings / "003.yaml").write_text("scale:\n  per_px: 0.25  # measured by hand\n", encoding="utf-8")
    project.write_text(
        project.read_text(encoding="utf-8") + "\nkinematics:\n  gestures:\n    still_speed_3d: 2.54\n  measures:\n"
        "    - {name: near, person: parent, joints: [nose], to_person: child, to_joints: [nose], below_3d: 76.2}\n"
        "    - {name: lean, person: parent, joints: [nose], to_person: child, to_joints: [nose], method: offset_toward, per_body_size: true, below_3d: 0.5}\n",
        encoding="utf-8",
    )
    app = create_app(project)
    save = callback(app, next(key for key, value in app.callback_map.items() if any(item["id"] == "settings-yaml-save" for item in value["inputs"])))
    monkeypatch.setattr(settings, "ctx", SimpleNamespace(triggered_id="settings-yaml-save"))
    text = project.read_text(encoding="utf-8").replace("units: cm", "units: in")
    _, _, message = save(None, None, 1, [], [], text)
    assert "converted from cm to in" in text_of(message)
    config = resolve_config(project)
    assert config.project.units == "in" and config.kinematics.gestures.still_speed_3d == 1
    assert [measure.below_3d for measure in config.kinematics.measures] == [30, 0.5]
    assert config.setup.room.people["parent"].at == (pytest.approx(39.3701), pytest.approx(78.7402))
    assert config.setup.room.things["table"].size == (pytest.approx(23.622), pytest.approx(19.685))
    own = (recordings / "003.yaml").read_text(encoding="utf-8")
    assert "per_px: 0.0984" in own and "# measured by hand" in own

    # a file that no longer reads is mended in the text box and saved
    mended = project.read_text(encoding="utf-8")
    project.write_text(mended + "\nnot_a_section: 1\n", encoding="utf-8")
    _, _, message = save(None, None, 1, [], [], mended)
    assert project.read_text(encoding="utf-8") == mended and "Saved to project.yaml" in text_of(message)


def test_the_transcript_card_lists_the_lines_a_model_marked(project):
    from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, write_transcript
    from test_transcript_correction import TRANSCRIPT

    layout = layout_for(resolve_config(project))
    directory = layout.artifacts_dir("003", ASRExtractor.name)
    directory.mkdir(parents=True)
    write_transcript(TRANSCRIPT, directory / TRANSCRIPT_NAME)
    shown = text_of(review.body(project, "003"))
    assert "Export to ELAN" in shown and "doubtful lines" not in shown
    # nothing to go back from until a correction is imported, and going back is asked again before it deletes one
    assert "Back to the recogniser" not in shown
    with pytest.raises(ValueError, match="unknown action"):
        review.action_command(project, "003", "transcript-reset", "", "", {})
    assert review.action_command(project, "003", "transcript-reset-sure", "", "", {})[1:3] == ["transcript", "reset"]
    project.write_text(project.read_text(encoding="utf-8") + "\ncoding:\n  llm:\n    model: tiny\n", encoding="utf-8")
    path = layout.transcript_check_path("003")
    path.parent.mkdir(parents=True, exist_ok=True)
    marked = [{"subject_id": "child", "t_start": 3.0, "t_end": 4.0, "text": "doggy", "kind": "cut", "why": "a word out of place"}]
    path.write_text(json.dumps({"model": "tiny", "n_lines": 2, "lines": marked}), encoding="utf-8")
    shown = text_of(review.body(project, "003"))
    assert "Ask the model for doubtful lines" in shown and "1 of 2 line(s) marked by tiny: 1 cut off" in shown and "a word out of place" in shown
    assert review.action_command(project, "003", "transcript-check", "", "", {})[1:3] == ["transcript", "check"]


def test_marking_the_analysed_part_replaces_a_length_kept_from_the_end(project):
    from macaw.interface import project_file
    from macaw.interface.pages import segments

    project.write_text(project_file.with_values(project.read_text(encoding="utf-8"), {"segment.keep_last_s": 300.0}), encoding="utf-8")
    segments.set_window(project, "003", 1.5, 20.0)
    own = resolve_config(project, recording_id="003")
    assert (own.segment.task_start_s, own.segment.keep_last_s) == (1.5, None)


def test_the_naming_frame_follows_where_the_player_stands(project, monkeypatch):
    from macaw.interface.pages import naming
    from macaw.media.ffmpeg import run_ffmpeg

    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=gray:s=320x240:r=10:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    shown = text_of(naming.editor(resolve_config(project, recording_id="003"), "003", "room", ["parent", "child"]))
    assert "on-frame" in shown and "names-timeline" not in shown and "names-at" not in shown
    frame = callback(create_app(project), "..names-graph.figure")
    options = [{"value": "parent"}, {"value": "child"}, {"value": naming.AREA}]
    states = ("003", "parent", options, {}, None, None, [])
    monkeypatch.setattr(naming, "ctx", SimpleNamespace(triggered_id={"type": "review-time", "key": "003"}))
    figure, marks, found, note, corners = frame([{"camera": "room", "own": 4.5, "t": 4.5}], None, *states)
    assert found["at"] == 4.5 and found["camera"] == "room" and "Click on each person" in text_of(note) and corners == []
    assert frame([], None, *states) == (no_update,) * 5
    *_, note, _ = frame([{"camera": "room", "own": 1.25, "t": 1.25}], None, *states[:-2], [2.0, 5.0], [])
    assert "1.25 s is outside the analysed part of this file (2.0 to 5.0 s)" in text_of(note)
    # with an area chosen under Click on, two clicks are its corners and name nobody
    monkeypatch.setattr(naming, "ctx", SimpleNamespace(triggered_id=naming.GRAPH))
    area = ("003", naming.AREA, options, {}, found, None)
    figure, marks, _, note, corners = frame([], {"points": [{"x": 240, "y": 0}]}, *area, [])
    assert corners == [[0.75, 0.0]] and marks is no_update and "opposite corner" in text_of(note)
    figure, _, _, note, corners = frame([], {"points": [{"x": 320, "y": 120}]}, *area, corners)
    assert corners == [[0.75, 0.0], [1.0, 0.5]] and "Ignore this area" in text_of(note) and len(figure.layout.shapes) == 1
    figure, marks, *_, corners = frame([], {"points": [{"x": 80, "y": 120}]}, "003", "parent", options, {}, found, None, corners)
    assert marks == {"parent": [0.25, 0.5]} and corners is no_update and len(figure.layout.shapes) == 1


def test_the_first_moment_shown_is_inside_the_analysed_part():
    from macaw.interface.pages import naming

    assert naming.first_moment((25.0, 300.0), 400.0) == 80.0
    assert naming.first_moment(None, 400.0) == naming.FIRST_AT_S and naming.first_moment(None, 6.0) == 3.0
    assert naming.outside(30.0, (25.0, 300.0)) is None and naming.outside(10.0, None) is None
    assert "10 s is outside the analysed part" in naming.outside(10.0, (25.0, 300.0))


def test_the_workflow_draws_each_step_after_what_it_needs_and_links_to_it():
    steps = ["prep.media", "align.review", "pose.detect", "text.asr"]
    grid = {("a", step): state for step, state in (("prep.media", "current"), ("pose.detect", "failed"), ("text.asr", "stale"))}
    found = overview.Standing(["a"], ["prep.media", "pose.detect", "text.asr"], grid, columns=steps, checks={("a", "align.review"): "waiting"},
                              needs={"align.review": [], "prep.media": ["align.review"], "pose.detect": ["prep.media"], "text.asr": ["prep.media"]})
    flow = overview.workflow_data(found)
    nodes = {node["id"]: node for node in flow["nodes"]}
    assert flow["lanes"] == ["Prepare", "Speech", "Body"]
    assert [nodes[name]["depth"] for name in steps] == [1, 0, 2, 2]
    assert [nodes[name]["state"] for name in ("prep.media", "pose.detect", "text.asr")] == ["current", "failed", "stale"]
    assert nodes["align.review"]["check"] and nodes["align.review"]["state"] == "waiting" and nodes["align.review"]["href"] == "/review"
    assert nodes["align.review"]["issue"] == [1, "waiting"] and nodes["pose.detect"]["issue"] == [1, "failed"]
    assert nodes["pose.detect"]["href"] == "/run?step=pose.detect" and nodes["pose.detect"]["label"] == "Find people"
    assert sorted(flow["edges"]) == [["align.review", "prep.media"], ["prep.media", "pose.detect"], ["prep.media", "text.asr"]]


def test_every_step_has_a_short_name_for_under_the_grid():
    from macaw.interface import step_labels

    assert set(step_labels.LABELS) <= set(step_labels.SHORT) and max(map(len, step_labels.SHORT.values())) <= 16
    assert step_labels.short("coding.judge.v2") == "Acts, v2" and step_labels.short("my.plugin") == "my.plugin"
    assert overview.column_short(overview.SPEECH_ACTS) == "Acts checked"


def test_the_workflow_leaves_out_a_need_reached_through_another():
    needs = {"a": [], "b": ["a"], "c": ["a", "b"]}
    assert overview.direct_needs(["a", "b", "c"], needs) == {"a": [], "b": ["a"], "c": ["b"]}


def strip_config(component) -> dict:
    found = []

    def walk(item):
        if isinstance(item, dict):
            props = item.get("props", {})
            if "time-strip" in str(props.get("className", "")):
                found.append(json.loads(props["data-strip"]))
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(json.loads(text_of(component)))
    return found[0] if found else {}


def test_segments_are_made_moved_renamed_and_removed_on_the_strip(project):
    from macaw.interface.pages import segments

    shown = strip_config(segments.body(project, "003"))
    assert shown["edit"] and shown["segments"] == [] and shown["editStore"] == segments.EDITS
    assert segments.apply_edit(project, "003", {"op": "set", "name": "free_play", "old": None, "start": 1.5, "end": 4.0}) == "saved"
    assert segments.apply_edit(project, "003", {"op": "set", "name": "warm_up", "old": None, "start": 0.5, "end": 1.5}) == "saved"
    with pytest.raises(MacawError, match="named warm_up already"):
        segments.apply_edit(project, "003", {"op": "set", "name": "warm_up", "old": "free_play", "start": 1.5, "end": 4.0})
    segments.apply_edit(project, "003", {"op": "set", "name": "play", "old": "free_play", "start": 2.0, "end": 4.5})
    config = resolve_config(project)
    assert [(s["name"], s["a"], s["b"], s["done"]) for s in segments.strip_segments(config, "003")] == [("warm_up", 0.5, 1.5, False), ("play", 2.0, 4.5, False)]
    assert "removed" in segments.apply_edit(project, "003", {"op": "remove", "name": "warm_up"})
    said = segments.apply_edit(project, "003", {"op": "window", "start": 1.0, "end": 5.0})
    assert "runs this recording again" in said or "starts this recording again" in said
    own = resolve_config(project, recording_id="003")
    assert segments.window_of(own.segment, 9.0) == [1.0, 5.0] and strip_config(segments.body(project, "003"))["window"] == [1.0, 5.0]


def test_the_strip_reads_the_played_file_on_the_recording_s_clock_where_it_started_later(project, monkeypatch):
    from macaw.interface.pages import segments

    # the file played began 5 s after the one the recording's clock was set by
    monkeypatch.setattr(segments, "played_clock", lambda config, rid: (5.0, 1.0))
    segments.apply_edit(project, "003", {"op": "set", "name": "play", "old": None, "start": 12.0, "end": 20.0})
    shown = strip_config(segments.body(project, "003"))
    assert (shown["offset"], shown["scale"]) == (5.0, 1.0) and (shown["segments"][0]["a"], shown["segments"][0]["b"]) == (7.0, 15.0)
    edited = callback(create_app(project), "..segments-table.children")
    _, shown, _ = edited({"op": "set", "name": "late", "old": None, "start": 1.0, "end": 2.0, "n": 1}, "003")
    late = next(segment for segment in segments.segments_of(resolve_config(project), "003") if segment.name == "late")
    assert (late.start_s, late.end_s) == (6.0, 7.0) and {"name": "late", "a": 1.0, "b": 2.0}.items() <= next(
        segment for segment in shown["segments"] if segment["name"] == "late"
    ).items()


def test_the_edits_callback_says_why_a_change_was_not_saved(project):
    from macaw.interface.pages import segments

    edited = callback(create_app(project), "..segments-table.children")
    table, shown, message = edited({"op": "set", "name": "x", "old": None, "start": 3.0, "end": 2.0, "n": 1}, "003")
    assert shown["said"] == "not saved" and shown["segments"] == [] and "has to end after it starts" in text_of(message)
    table, shown, message = edited({"op": "set", "name": "x", "old": None, "start": 1.0, "end": 2.0, "n": 2}, "003")
    assert message is None and shown["segments"][0]["name"] == "x" and "x" in text_of(table)


def test_the_loudness_of_a_copy_is_worked_out_once(tmp_path):
    from macaw.exceptions import ExternalToolError
    from macaw.media.ffmpeg import run_ffmpeg
    from macaw.project.previews import LOUDNESS_STEP_S, loudness

    copy = tmp_path / "copy.mp4"
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-filter_complex", "[1]atrim=duration=1[q];[0][q]concat=n=2:v=0:a=1", str(copy)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    found = loudness(copy)
    values = found["values"]
    assert found["step"] == LOUDNESS_STEP_S and 19 <= len(values) <= 21
    assert min(values[:9]) > 0.9 and max(values[11:]) == 0.0
    assert copy.with_suffix(".loudness.npy").is_file() and loudness(copy) == found


def test_the_review_player_strip_holds_the_page_time(project):
    from macaw.media.ffmpeg import run_ffmpeg

    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "color=gray:s=320x240:r=10:d=6", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)])
    shown = strip_config(review.body(project, "003"))
    assert shown["camera"] == "room" and shown["timeStore"] == {"type": review.TIME, "key": "003"}
    assert shown["ticks"] == "#review-body .seek" and shown["video"] is None
    page = text_of(review.body(project, "003"))
    assert "the camera playing" in page and "scale-at-wrap" in page and "gaze-random" in page and '"type": "gaze-video"' not in page
    # timeline.js pins the card small by its slot
    assert "player-slot" in page


def test_a_scale_on_the_camera_playing_is_measured_where_the_player_stands(project, monkeypatch):
    app = create_app(project)
    update = review_update(app)
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id={"type": review.ACT, "action": "set-scale", "target": ""}, triggered=[{"value": 1}]))
    commands = []
    monkeypatch.setattr(review.cli_bridge, "run_now", lambda argv, **_: (commands.append(argv), (0, "ok"))[1])
    states = review_states("ann", ["room"], times=[{"camera": "room", "own": 7.5, "t": 8.25}])
    states[8:14] = [[review.PLAYING], [0], [[[1, 2], [11, 2]]], [30], ["ruler"], ["everyone"]]
    update("003", [1], [], [], [], *states)
    argv = commands[0]
    assert argv[argv.index("--camera") + 1] == "room" and argv[argv.index("--at") + 1] == "8.25"


def test_the_agreement_result_puts_the_overall_figures_first():
    report = {
        "n_units": 412, "n_pairable": 398, "n_annotators": 3, "coverage": 0.941, "alpha": 0.713, "abstention_rate": {},
        "pairwise_kappa": [{"annotator_a": "ann", "annotator_b": "bo", "n_units": 371, "observed_agreement": 0.836, "kappa": 0.768}],
        "per_code_pairs": [], "per_code": [{"code": "question", "n_units": 398, "n_ratings": 1140, "alpha": 0.801}], "warnings": [],
    }
    shown = text_of(agreement.result_view(report, Path("agreement"), "printed"))
    assert shown.index("headline") < shown.index("result-grid") and "94%" in shown and "0.713" in shown
    page = text_of(agreement.layout(Path("project.yaml")))
    assert "page-split" in page and "choices" in page and "segmented" not in page


def test_a_person_starts_a_thing_where_they_clicked_and_the_button_says_what_is_missing(project, monkeypatch):
    from macaw.cli.main import app as cli
    from macaw.extractors.video.objects.decisions import STARTS, read_decisions
    from typer.testing import CliRunner

    data = yaml.safe_load(project.read_text(encoding="utf-8"))
    data["objects"] = {"names": ["ball"]}
    project.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(review.cli_bridge, "run_now", lambda argv, cwd=None, **_: (0, CliRunner().invoke(cli, argv[1:]).output))
    update = review_update(create_app(project))
    pressed = {"type": review.ACT, "action": "object-start", "target": ""}
    monkeypatch.setattr(review, "ctx", SimpleNamespace(triggered_id=pressed, triggered=[{"value": 1}]))
    states = review_states("ann")
    shown = {"camera": "room", "t": 2.0}
    _, said, _ = update("003", [1], [], [], [], *states[:-4], ["ball"], [], [shown], [])
    assert "Click on the thing in the frame first" in text_of(said)
    body, said, _ = update("003", [1], [], [], [], *states[:-4], ["ball"], [[120.0, 80.5]], [shown], [])
    assert "follows it from there" in text_of(said) and "--point 120,80.5" in text_of(said).replace('", "', " ")
    point = read_decisions(layout_for(resolve_config(project)).objects_path("003"))[STARTS]["room"]["ball"][0]
    assert (point["t"], point["x"], point["y"], point["by"].split()[0]) == (2.0, 120.0, 80.5, "ann")
    assert "Things followed" in text_of(body)


def test_the_3d_card_reads_the_recording_s_own_calibration(project, monkeypatch):
    from macaw.extractors.video.pose import triangulate

    folder = project.parent / "recordings"
    folder.mkdir(exist_ok=True)
    (folder / "003.yaml").write_text("pose3d:\n  calibration: other_day\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(triangulate, "points_file", lambda config, rid: seen.append(config.pose3d.calibration) or project.parent / "none.parquet")
    assert results.points_3d(project, "003") is None and seen == ["other_day"]


def test_a_command_shown_pastes_as_the_same_arguments():
    import shlex
    import sys

    argv = ["macaw", "note", "--text", 'said "stop" for $5', ""]
    shown = ui.command_text(argv)
    if sys.platform == "win32":
        assert shown == 'macaw note --text "said \\"stop\\" for $5" ""'
    else:
        assert shlex.split(shown) == argv
