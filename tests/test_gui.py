from __future__ import annotations

import shutil

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
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
    assert "/setup?recording=003&amp;camera=room" in shown or "/setup?recording=003&camera=room" in shown
    voices = review.action_command(project, "003", "accept-voices", "", "ann", {"S1": "child", "S0": "none"})
    assert voices[voices.index("--rename") + 1] == "S0=none,S1=child"


def test_the_review_page_measures_a_yardstick_and_saves_it(project):
    shown = text_of(review.body(project, "003"))
    assert "Pixels to centimetres" in shown and "set-scale" in shown and "No scale yet" in shown
    assert "right only at the distance from the camera" in shown
    measured = {"video": "room", "at": 4.5, "points": [[0, 0], [30, 40]], "length": 10, "object": "table edge", "person": "child"}
    argv = review.action_command(project, "003", "set-scale", "", "ann", {}, measured)
    assert argv[1] == "set-scale" and argv[argv.index("-r") + 1] == "003" and argv[argv.index("--points") + 1] == "0,0,30,40"
    assert argv[argv.index("--length-cm") + 1] == "10" and argv[argv.index("--object") + 1] == "table edge"
    assert argv[argv.index("--person") + 1] == "child"
    assert review.action_command(project, "003", "clear-scale", "", "ann", {})[-1] == "--clear"
    everyone = review.action_command(project, "003", "set-scale", "", "ann", {}, {**measured, "person": review.EVERYONE})
    assert "--person" not in everyone


def test_the_review_page_labels_where_people_look_and_reports_agreement(project):
    from macaw.project.gaze_labels import GazeLabel, write_labels

    shown = text_of(review.body(project, "003"))
    assert "Where people look" in shown and "at random" in shown and "label-gaze" in shown
    argv = review.action_command(project, "003", "label-gaze", "child", "ann", {}, gaze={"t": 12.5, "looks_at": "parent"})
    assert argv[1] == "label-gaze" and argv[argv.index("--at") + 1] == "12.5" and argv[argv.index("--person") + 1] == "child"
    assert argv[argv.index("--looks-at") + 1] == "parent" and argv[argv.index("--by") + 1] == "ann"
    write_labels([GazeLabel(12.5, "child", "parent"), GazeLabel(13.0, "child", "parent")], layout_for(resolve_config(project)).gaze_labels_path("003"))
    assert "Looks at" in text_of(review.body(project, "003")) and "Looks at" not in shown


def test_the_review_page_lists_coding_brought_in_and_how_to_compare_it(project, tmp_path):
    assert "Your coding" not in text_of(review.body(project, "003"))
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
    write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(config).annotations_path("003"))
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
        "review", "settings", "cameras", "run", "agreement", "setup", "overview", "overview",
    ]


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
    items = text_of(overview.next_up(found))
    assert "wait for a person" in items and '"/review"' in items
    grid = text_of(overview.grid(found))
    assert "cell held" in grid and "003, prep.media: waiting for a person" in grid


def test_a_failed_step_comes_first_in_next_up(project, monkeypatch):
    monkeypatch.setattr(cli_bridge, "failed_last_time", lambda config_path: {task_id_for("003", "prep.media")})
    found = overview.standing(project)
    assert found.grid["003", "prep.media"] == "failed"
    first = text_of(overview.next_up(found)[0])
    assert "prep.media failed" in first and "for 003" in first and '"/run"' in first


def test_a_square_leads_to_what_the_step_produced_waits_on_or_runs(project):
    assert overview.cell_link("003", "turns.structure", "current") == ("/results?recording=003&step=turns.structure", "Results")
    assert overview.cell_link("003", "prep.media", "held") == ("/review?recording=003", "Review")
    assert [overview.cell_link("003", "text.asr", state)[0] for state in ("new", "stale", "failed")] == ["/run?recording=003&step=text.asr"] * 3
    grid = text_of(overview.grid(overview.standing(project)))
    assert '"href": "/review?recording=003"' in grid and "Opens Review." in grid
    assert ui.asked("?recording=a%20b&step=text.asr") == {"recording": "a b", "step": "text.asr"} and ui.asked(None) == {}


def callback(app, output: str):
    return app.callback_map[output]["callback"].__wrapped__


def test_a_link_picks_the_recording_and_step_on_the_page_it_opens(project, monkeypatch):
    app = create_app(project)
    assert callback(app, "review-recording.value")("/review", "?recording=003") == "003"
    assert callback(app, "results-recording.value")("/results", "?recording=003&step=x") == "003"
    for page, search in (("/review", "?recording=nowhere"), ("/run", "?recording=003")):
        assert callback(app, "review-recording.value")(page, search) is no_update
    assert results.chosen_source(["text.asr", "turns.structure"], "turns.structure", "text.asr") == "turns.structure"
    assert results.chosen_source(["text.asr", "turns.structure"], "prep.video", "turns.structure") == "turns.structure"
    assert results.chosen_source(["text.asr"], None, None) == "text.asr" and results.chosen_source([], "x", None) is None

    fill = callback(app, "..run-recordings.options...run-recordings.value...run-steps.options...run-steps.value...run-more.options...run-more.value...run-more-summary.children...run-summary.children..")
    monkeypatch.setattr(run, "ctx", SimpleNamespace(triggered_id="url"))
    linked = fill("/run", "?recording=003&step=prep.media", None, ["003"], ["text.asr", "turns.structure"], [])
    assert linked[1] == ["003"] and linked[3] == ["prep.media"] and linked[5] == []
    monkeypatch.setattr(run, "ctx", SimpleNamespace(triggered_id="run-state"))
    kept = fill("/run", "?recording=003&step=prep.media", "done", ["003"], ["text.asr"], [])
    assert kept[3] == ["text.asr"]


def test_steps_left_to_run_are_listed_after_the_rest_are_counted():
    steps = [f"step{k}" for k in range(9)]
    found = overview.Standing(["a", "b"], steps, {(rid, step): "current" if step == "step0" else "stale" for rid in "ab" for step in steps})
    items = [text_of(item) for item in overview.next_up(found)]
    assert len(items) == overview.NEXT_STEPS + 1 and "2 to run again" in items[0] and "and 2 more steps" in items[-1]
    fresh = overview.Standing(["a"], steps, {("a", step): "new" for step in steps})
    assert ["Nothing has run yet" in text_of(item) for item in overview.next_up(fresh)] == [True]


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


def nav_style(project, name):
    return props_of(build_layout(project.resolve()), f"nav-{name}").get("style")


def test_your_name_is_taken_as_typed_without_leaving_the_box(project):
    """A file dropped on Import your coding does not move the focus, so a name waiting for it would be lost."""
    assert not props_of(review.layout(project), "review-by").get("debounce")


def test_the_cameras_tab_shows_only_where_two_cameras_film(project):
    assert nav_style(project, "cameras") == {"display": "none"}
    put(project.parent / "raw", "003", "side", "a.mp4")
    write_devices(project, cameras={"room": setup_camera(), "side": setup_camera(uses=("body",))}, microphones=mics())
    assert nav_style(project, "cameras") is None


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
        "made_by": "macaw calibrate-cameras", "made_at": "2026-10-31T10:00:00", "board": describe_board(config.pose3d.board),
        "error_px": {"room": 0.4, "side": 2.6}, "lens_error_px": {"room": 0.3, "side": 0.5}, "views": {"room": 120, "side": 98},
        "together": {"room and side": together}, "notes": list(notes),
    }
    return write_setup(CameraSetup(cameras, "mm", metadata), layout_for(config).camera_setup_path("lab"))


def test_a_setup_shows_each_camera_and_what_to_redo(project):
    from macaw.interface.pages import cameras

    config = resolve_config(project)
    assert cameras.setups(config) == [] and "No setup yet" in text_of(cameras.setup_view(project, None))
    for name in ("room", "side"):
        (config.project.calibration_dir / "lab" / name).mkdir(parents=True)
        (config.project.calibration_dir / "lab" / name / "clip.mp4").write_bytes(b"")
    assert cameras.setups(config) == ["lab"]
    waiting = text_of(cameras.setup_view(project, "lab"))
    assert "not found yet" in waiting and "calibrate-cameras" in waiting
    shutil.rmtree(config.project.calibration_dir / "lab")

    calibrated(project, notes=["side.mp4 has no sound"], together=8)
    shown = text_of(cameras.setup_view(project, "lab"))
    assert "Found on 2026-10-31T10:00:00" in shown and "side.mp4 has no sound" in shown
    assert "up to 2.6 px" in shown and "8 (few: film more where both see it)" in shown
    assert "1.50, 0.50, 0.00" in shown and "used by this project" not in shown
    project.write_text(project.read_text(encoding="utf-8") + "\npose3d:\n  board:\n    square_mm: 74.5\n", encoding="utf-8")
    assert "pose3d.board differs now" in text_of(cameras.setup_view(project, "lab"))


def test_the_cameras_are_drawn_where_they_stand(project):
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.interface.pages import cameras

    figure = cameras.cameras_figure(read_setup(calibrated(project)), "lab")
    shown = next(trace for trace in figure.data if trace.name == "cameras")
    assert list(shown.text) == ["room", "side"] and list(shown.x) == pytest.approx([0.0, 1.5]) and list(shown.y) == pytest.approx([0.0, 0.5])


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


def test_the_whole_workbench_lays_out(project):
    shown = text_of(build_layout(project.resolve()))
    assert all(name in shown for name in ("overview-grid", "run-steps", "review-body", "results-view", "settings-yaml", "theme-choice"))


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
    assert overview.latest_run(project) == "Nothing has run yet."
    monkeypatch.setattr(cli_bridge, "poll_status", lambda config_path: ("20261031-214220-e17d5e", {"done": 36, "failed": 0}))
    assert overview.latest_run(project) == "Latest run 2026-10-31 21:42: 36 done"


def test_the_segments_page_plays_a_copy_and_saves_what_is_marked(project):
    from macaw.exceptions import ExternalToolError
    from macaw.interface.pages import segments
    from macaw.media.ffmpeg import make_preview, run_ffmpeg
    from macaw.media.probe import probe

    video = project.parent / "raw" / "003" / "room" / "a.mp4"
    try:
        run_ffmpeg(["-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=2", "-f", "lavfi", "-i", "sine=duration=2",
                    "-pix_fmt", "yuv420p", "-shortest", str(video)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    assert "Make a copy to play here" in text_of(segments.body(project, "003"))
    config = resolve_config(project)
    copy = probe(make_preview(segments.played(config, "003"), layout_for(config).preview_path("003")))
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


def test_the_setup_page_shows_the_saved_devices_and_what_each_recording_holds(project):
    from macaw.interface.pages import setup

    config = resolve_config(project)
    rows = setup.rows_of(config, "main")
    assert [row["name"] for row in rows["cameras"]] == ["room"] and rows["cameras"][0]["face"] == ["parent", "child"]
    assert [row["name"] for row in rows["microphones"]] == ["parent_mic", "child_mic"]
    shown = text_of(setup.form(config, "main", rows))
    assert "Saved as setups/main.yaml" in shown and "Add a camera" in shown and "Voices" in shown
    assert "not saved yet" in text_of(setup.form(config, "home", setup.rows_of(config, "home")))
    (project.parent / "raw" / "003" / "stray.wav").write_bytes(b"")
    held = text_of(setup.holdings(project))
    assert "room: a.mp4" in held and "parent_mic: a.wav" in held and "stray.wav lies outside the device folders" in held


def test_the_setup_page_saves_what_the_form_says_or_says_why_not(project):
    from macaw.interface.pages import setup

    rows = {"cameras": [{"name": "front", "body": ["child"], "face": ["child"], "gaze": [], "hears": []}],
            "microphones": [{"name": "room_mic", "hears": ["parent", "child"]}]}
    path = setup.save(project, "home", rows)
    assert path == project.parent / "setups" / "home.yaml"
    assert setup.rows_of(resolve_config(project), "home") == rows
    for broken, reason in (
        ({**rows, "cameras": [{**rows["cameras"][0], "name": ""}]}, "needs the name of its folder"),
        ({**rows, "microphones": [{"name": "front", "hears": ["parent"]}]}, "two devices are called front"),
        ({**rows, "microphones": [{"name": "mic", "hears": ["teacher"]}]}, "names teacher"),
    ):
        with pytest.raises(MacawError, match=reason):
            setup.save(project, "home", broken)
    with pytest.raises(MacawError, match="letters, digits"):
        setup.save(project, "home visits", rows)
    named = setup.rows_from(["front"], [{"kind": "cameras", "index": 0}], [["child"], []], [
        {"kind": "cameras", "index": 0, "use": "body"}, {"kind": "cameras", "index": 0, "use": "hears"}])
    assert named == {"cameras": [{"name": "front", "body": ["child"], "face": [], "gaze": [], "hears": []}], "microphones": []}


def test_the_overview_asks_for_the_setup_first_until_it_is_saved(project):
    assert "Describe the setup" not in text_of(overview.next_up(overview.standing(project)))
    (project.parent / "setups" / "main.yaml").unlink()
    first = text_of(overview.next_up(overview.standing(project))[0])
    assert "Describe the setup" in first and '"/setup"' in first
    assert "not saved yet" in text_of(run.layout(project))


def test_the_setup_pages_buttons_add_remove_and_save_rows(project, monkeypatch):
    from macaw.interface.pages import setup

    app = create_app(project)
    edit = callback(app, "setup-form.children")
    keep = callback(app, next(key for key in app.callback_map if "setup-files.children" in key))

    def pressed(identity, value=1):
        monkeypatch.setattr(setup, "ctx", SimpleNamespace(triggered_id=identity, triggered=[{"value": value}]))

    names, name_ids = ["room", "parent_mic"], [{"type": setup.NAME, "kind": "cameras", "index": 0}, {"type": setup.NAME, "kind": "microphones", "index": 0}]
    uses = [["parent", "child"], ["child"], [], [], ["parent"]]
    use_ids = [{"type": setup.USE, "kind": "cameras", "index": 0, "use": use} for use in ("body", "face", "gaze", "hears")]
    use_ids.append({"type": setup.USE, "kind": "microphones", "index": 0, "use": "hears"})
    state = (names, name_ids, uses, use_ids)

    pressed("setup-pick")
    assert text_of(edit("main", [None, None], [None], *state)).count("folder name") == 3
    pressed({"type": setup.ADD, "kind": "cameras"})
    assert text_of(edit("main", [1, None], [None], *state)).count("folder name") == 3
    pressed({"type": setup.REMOVE, "kind": "microphones", "index": 0})
    removed = text_of(edit("main", [None, None], [None, 1], *state))
    assert "No microphone yet" in removed and '"value": "room"' in removed
    pressed({"type": setup.REMOVE, "kind": "cameras", "index": 0}, value=None)
    assert edit("main", [None, None], [None, None], *state) is no_update

    message, files, _ = keep(1, "main", *state)
    assert "Saved main.yaml" in text_of(message)
    assert setup.rows_of(resolve_config(project), "main")["microphones"] == [{"name": "parent_mic", "hears": ["parent"]}]
    assert "room: a.mp4" in text_of(files)
    refused, _, _ = keep(1, "main", ["room", "room"], name_ids, uses, use_ids)
    assert "two devices are called room" in text_of(refused)
