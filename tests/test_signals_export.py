from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")

from test_signals_align import TRUE_OFFSET, aligned_project, run_steps  # noqa: E402,F401  (the fixture is used by name)


def test_marks_of_one_name_are_kept_apart_moved_and_removed(tmp_path):
    from macaw.exceptions import ConfigError
    from macaw.signals.marks import Mark, marks_of, remove_mark, set_mark

    config = SimpleNamespace(project=SimpleNamespace(marks=tmp_path / "marks.csv"))
    set_mark(config, "003", Mark("touched_cap", 40.0, 41.5))
    set_mark(config, "003", Mark("touched_cap", 10.0, 12.0, "reached up"))
    set_mark(config, "003", Mark("touched_cap", 40.0, 43.0), old=Mark("touched_cap", 40.0, 41.5))
    assert [(m.start_s, m.end_s, m.note) for m in marks_of(config, "003")] == [(10.0, 12.0, "reached up"), (40.0, 43.0, "")]
    remove_mark(config, "003", Mark("touched_cap", 10.0, 12.0))
    assert [(m.name, m.start_s) for m in marks_of(config, "003")] == [("touched_cap", 40.0)]
    with pytest.raises(ConfigError, match="has to end after it starts"):
        set_mark(config, "003", Mark("x", 5.0, 4.0))
    with pytest.raises(ConfigError, match="changed elsewhere"):
        set_mark(config, "003", Mark("x", 1.0, 2.0), old=Mark("y", 1.0, 2.0))
    # a table written by hand needs no note column
    (tmp_path / "marks.csv").write_text("recording,mark,start_s,end_s\n004,cried,3,9\n", encoding="utf-8")
    assert marks_of(config, "004") == [Mark("cried", 3.0, 9.0, "")]


def test_frame_measures_are_averaged_over_each_sample_and_empty_beyond_the_video():
    from macaw.signals.export import measures_on_samples

    frames = np.arange(0.0, 10.0, 0.04)
    rows = pd.concat([
        pd.DataFrame({"subject_id": "child", "subject_id_b": None, "feature": "speed", "t_start": frames, "value": frames}),
        pd.DataFrame({"subject_id": "parent", "subject_id_b": "child", "feature": "distance", "t_start": frames, "value": 2.0}),
    ], ignore_index=True)
    samples = np.arange(-2.0, 12.0, 0.1)
    found = measures_on_samples(rows, samples)
    assert list(found.columns) == ["child.speed", "parent-child.distance"]
    inside = (samples > 0.1) & (samples < 9.9)
    assert np.allclose(found["child.speed"][inside], samples[inside], atol=0.03)
    assert found["child.speed"][samples < -0.1].isna().all() and found["parent-child.distance"][samples > 10.1].isna().all()


def test_events_come_from_tiers_marks_and_segments_and_land_on_each_file_s_clock():
    from macaw.exceptions import ConfigError
    from macaw.project.segments import Segment
    from macaw.signals.export import bids_entities, event_spans, events_name, events_template, on_file
    from macaw.signals.marks import Mark

    coded = pd.DataFrame({
        "tier": ["speech_act"] * 3 + ["mood"], "annotator": ["ann", "ann", "llm", "ann"], "subject_id": ["child", None, "child", None],
        "t_start": [1.0, 5.0, 1.0, 0.0], "t_end": [2.0, 6.0, 2.0, 9.0], "label": ["question", "praise", "question", ""], "abstained": False,
    })
    config = SimpleNamespace(signals=SimpleNamespace(events=["speech_act"]))
    with pytest.raises(ConfigError, match="coded by ann, llm; name one"):
        event_spans(config, "003", coded, [], [])
    config.signals.events = ["speech_act/ann", "touched_cap", "free_play", "mood", "nothing"]
    spans, absent = event_spans(config, "003", coded, [Mark("touched_cap", 3.0, 4.0)], [Segment("free_play", 10.0, 70.0)])
    assert spans.values.tolist() == [[1.0, 2.0, "child.question"], [5.0, 6.0, "praise"], [3.0, 4.0, "touched_cap"], [10.0, 70.0, "free_play"]]
    assert absent == ["nothing"]   # mood has only an empty code, so names something but gives no event
    # recording time = -10 + own time x 1.001; the file holds own times 0 to 60
    found, outside = on_file(spans, (-10.0, 1.001), 0.0, 60.0)
    assert outside == 0 and found["trial_type"].tolist() == ["child.question", "touched_cap", "praise", "free_play"]
    assert found["onset"].tolist() == pytest.approx([11.0 / 1.001, 13.0 / 1.001, 15.0 / 1.001, 20.0 / 1.001])
    assert found["duration"].tolist() == pytest.approx([1.0 / 1.001, 1.0 / 1.001, 1.0 / 1.001, 60.0 - 20.0 / 1.001])
    found, outside = on_file(spans, (-58.0, 1.0), 0.0, 60.0)
    assert outside == 3 and found.values.tolist() == [[59.0, 1.0, "child.question"]]
    entities = bids_entities(Path("sub-c003_ses-2_task-play_run-1_desc-preproc_nirs.snirf"))
    assert entities == {"sub": "c003", "ses": "2", "task": "play", "run": "1"}
    assert events_name(entities) == "sub-c003_ses-2_task-play_run-1_events.tsv"
    assert events_template(entities) == "sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"
    with pytest.raises(ConfigError, match="no task-"):
        bids_entities(Path("sub-01_nirs.snirf"))


def test_nirspipe_gets_events_and_bad_spans_and_each_file_its_samples_with_the_measures(aligned_project):
    from typer.testing import CliRunner

    from macaw.cli.main import app
    from macaw.signals.marks import Mark, set_mark

    project, config, layout = aligned_project
    with project.open("a", encoding="utf-8") as handle:
        handle.write("signals:\n  events: [touched_cap, free_play]\n  bad_marks: [touched_cap]\n")
    (project.parent / "segments.csv").write_text("recording,segment,start_s,end_s\n003,free_play,30,90\n", encoding="utf-8")
    from macaw.config.loader import resolve_config

    config = resolve_config(project)
    set_mark(config, "003", Mark("touched_cap", 20.0, 22.0))
    run_steps(config, layout)
    runner = CliRunner()
    said = runner.invoke(app, ["export", "-c", str(project), "-f", "nirspipe"])
    assert said.exit_code != 0 and "waits for a person" in str(said.exception)
    runner.invoke(app, ["accept-signal-offset", "-c", str(project), "-r", "003", "--by", "tester"])
    runner.invoke(app, ["set-signal-offset", "-c", str(project), "-r", "003", "--person", "parent", "--offset", "-140", "--by", "tester"])

    said = runner.invoke(app, ["export", "-c", str(project), "-f", "nirspipe"])
    assert said.exit_code == 0, said.output
    folder = layout.output_dir / "exports" / "nirspipe"
    child = pd.read_csv(folder / "events" / "sub-c003_task-play_events.tsv", sep="\t")
    assert child["trial_type"].tolist() == ["touched_cap", "free_play"]
    assert child["onset"].tolist() == pytest.approx([20.0 - TRUE_OFFSET, 30.0 - TRUE_OFFSET], abs=0.15)
    parent = pd.read_csv(folder / "events" / "sub-p003_task-play_run-1_events.tsv", sep="\t")
    assert parent["onset"].tolist() == pytest.approx([160.0, 170.0], abs=1e-3)
    bad = pd.read_csv(folder / "bad_spans.tsv", sep="\t", dtype={"run": str})
    assert list(bad.columns) == ["participant_id", "session", "task", "acq", "run", "onset", "duration"]
    assert bad["participant_id"].tolist() == ["sub-c003", "sub-p003"] and bad["run"].fillna("").tolist() == ["", "1"]
    assert bad["duration"].tolist() == pytest.approx([2.0, 2.0], abs=1e-3)
    assert "--events-path" in said.output and "sub-{sub}_task-{task}_events.tsv" in said.output and "--bad-spans" in said.output

    said = runner.invoke(app, ["export", "-c", str(project), "-f", "signals"])
    assert said.exit_code == 0, said.output
    table = pd.read_csv(layout.output_dir / "exports" / "signals" / "003_child_raw.csv")
    assert list(table.columns[:3]) == ["time", "recording_time", "S1_D1 hbo"] and "child.face_yaw_deg" in table.columns
    assert np.allclose(table["recording_time"] - table["time"], table["recording_time"].iloc[0] - table["time"].iloc[0])
    seen = table[table["child.face_yaw_deg"].notna()]
    assert seen["recording_time"].min() == pytest.approx(0.0, abs=0.1) and seen["recording_time"].max() < 300.1
    dictionary = pd.read_csv(layout.output_dir / "exports" / "signals" / "003_child_raw_dictionary.csv")
    assert "mean of the video frames" in dictionary.set_index("column").at["child.face_yaw_deg", "aggregation"]


def test_the_signals_page_draws_each_file_under_the_video_and_saves_marks(aligned_project, monkeypatch):
    from test_gui import callback, strip_config, text_of

    from macaw.config.loader import resolve_config
    from macaw.interface.app import create_app, shown_tabs
    from macaw.interface.pages import overview, signals
    from macaw.signals.marks import marks_of

    project, config, layout = aligned_project
    run_steps(config, layout)
    assert signals in shown_tabs(config) and overview.check_page("signals.review") == ("/signals", "Signals")
    assert "No signal files for 004" in text_of(signals.body(project, "004"))
    page = text_of(signals.body(project, "003"))
    assert "Where each file sits" in page and "not looked at yet" in page and "not aligned" in page
    shown = strip_config(signals.strip(project, "003", ["S1_D1"], [], []))
    kinds = [(lane["kind"], lane["name"]) for lane in shown["lanes"]]
    assert ("stack", "child raw") in kinds and ("stack", "child preproc") in kinds and ("lines", "child turning") in kinds
    raw = next(lane for lane in shown["lanes"] if lane["name"] == "child raw")
    assert [row["name"] for row in raw["rows"]] == ["S1_D1"] and raw["rows"][0]["series"][0]["colour"] == signals.HAEMOGLOBIN_COLOURS["hbo"]
    assert "1 of 1 pairs" in raw["title"] and "events as lines" in raw["title"]
    assert shown["noun"] == "mark" and shown["notes"] and not shown["editWindow"]
    turning = next(lane for lane in shown["lanes"] if lane["name"] == "child turning")
    assert [series["name"] for series in turning["series"]] == ["head on the video", "cap"]
    # each person's files share their events, so one lane shows them, where the offset puts them
    assert [lane["name"] for lane in shown["lanes"] if lane["kind"] == "bars"] == ["child events", "parent events"]
    assert shown["names"] == ["start"]

    ran = []
    monkeypatch.setattr(signals.cli_bridge, "run_now", lambda argv, cwd=None: (ran.append(argv), run_steps(config, layout), (0, ""))[-1])
    edited = callback(create_app(project), "..signals-shown.data")
    played = shown["offset"]
    edited({"op": "set", "name": "cried", "start": 1.0, "end": 3.0, "note": "loud", "was": None, "n": 1}, "003", [], [], [], [])
    assert [(m.name, m.start_s - played, m.note) for m in marks_of(resolve_config(project), "003")] == [("cried", 1.0, "loud")]
    shown_after, message, *_ = edited({"op": "set", "name": "cried", "start": 1.0, "end": 4.0, "note": "loud",
                                       "was": {"name": "cried", "a": 1.0, "b": 3.0}, "n": 2}, "003", [], [], [], [])
    assert message is None and [(s["a"], s["b"]) for s in shown_after["segments"]] == [(1.0, 4.0)]
    shown_after, message, *_ = edited({"op": "remove", "name": "cried", "start": 1.0, "end": 4.0, "n": 3}, "003", [], [], [], [])
    assert shown_after["segments"] == [] and marks_of(resolve_config(project), "003") == [] and not ran
    # a mark named as the files' event places them by it at once
    _, message, offsets, _ = edited({"op": "set", "name": "Start", "start": 5.0, "end": 6.0, "note": "", "was": None, "n": 4},
                                    "003", [], [], [], [])
    assert ran and ran[0][1:2] == ["run"] and "signals.align" in ran[0] and message is None
    assert "from 1 marked event(s)" in text_of(offsets)
    keep = signals.action_command(project, "003", "keep", "child_raw,child_preproc", "ann")
    assert keep[1:2] == ["accept-signal-offset"] and keep[-6:-2] == ["--file", "child_raw", "--file", "child_preproc"]
    assert signals.action_command(project, "003", "set", "parent_raw", "ann", -140.5)[-6:] == ["--file", "parent_raw", "--offset", "-140.5", "--by", "ann"]


def test_the_signals_tab_shows_only_where_a_project_lists_signals(tmp_path):
    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.interface.app import shown_tabs
    from macaw.interface.pages import signals

    project = tmp_path / "project.yaml"
    (tmp_path / "raw").mkdir()
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    assert signals not in shown_tabs(resolve_config(project))


def test_each_file_stacks_its_pairs_as_rows_on_one_scale_with_its_events():
    from macaw.interface.pages.signals import DEFAULT_ROWS, HAEMOGLOBIN_COLOURS, channel_rows
    from macaw.signals.lanes import DIFFERENCE, OWN, RELATIVE, lines, stack, unpack

    names = [f"S{k}_D{k} {wave}" for k in range(1, 23) for wave in (760, 850)]
    rows, mode, total = channel_rows(names, ["raw"] * len(names), [])
    assert mode == RELATIVE and total == 22 and [row for row, _ in rows] == [f"S{k}_D{k}" for k in range(1, DEFAULT_ROWS + 1)]
    assert [label for _, label, _ in rows[0][1]] == ["760 nm", "850 nm"]
    assert [row for row, _ in channel_rows(names, ["raw"] * len(names), ["S9_D9", "S5_D5", "S99_D1"])[0]] == ["S5_D5", "S9_D9"]
    assert channel_rows(names, ["od"] * len(names), [])[1] == DIFFERENCE
    haemoglobin, mode, _ = channel_rows(["S1_D1 hbo", "S1_D1 hbr"], ["hbo", "hbr"], [])
    assert mode == DIFFERENCE and [colour for _, _, colour in haemoglobin[0][1]] == [HAEMOGLOBIN_COLOURS["hbo"], HAEMOGLOBIN_COLOURS["hbr"]]
    table, mode, _ = channel_rows(["ECG", "Heart rate"], ["value", "value"], [])
    assert mode == OWN and [row for row, _ in table] == ["ECG", "Heart rate"]

    # raw light whose level differs fourfold between channels fills one scale once each is a % change from its mean
    times = np.arange(0.0, 20.0, 0.1)
    wave = 0.01 * np.sin(times)
    drawn = stack([("S1_D1", [("760 nm", "#000", times, 0.34 * (1 + wave))]), ("S2_D1", [("760 nm", "#000", times, 0.09 * (1 + wave))])],
                  20.0, mode=RELATIVE, events=[5.0, 25.0])
    assert drawn["unit"] == "%" and drawn["events"] == [5.0] and drawn["scale"] == pytest.approx(1.0, abs=0.15)
    first, second = (unpack(row["series"][0]["packed"]) for row in drawn["rows"])
    assert np.nanmax(np.abs(first - second)) < 1e-3
    # haemoglobin in molar units keeps its shape: values come in units of the scale
    molar = stack([("S1_D1", [("HbO", "#000", times, 3e-7 * np.sin(times))])], 20.0)
    assert molar["scale"] == pytest.approx(3e-7, rel=0.1) and np.nanmax(np.abs(unpack(molar["rows"][0]["series"][0]["packed"]))) > 0.9
    own = stack([("ECG", [("ECG", "#000", times, 500 + 40 * np.sin(times))])], 20.0, mode=OWN)
    assert own["own"] and np.nanmax(np.abs(unpack(own["rows"][0]["series"][0]["packed"]))) < 1.05
    fixed = lines([("a", times, np.sin(times) * 9)], 20.0, bounds=(-3.0, 3.0), unit=" SD")
    assert (fixed["low"], fixed["high"], fixed["unit"]) == (-3.0, 3.0, " SD")
    assert lines([("a", times, np.full(times.size, np.nan))], 20.0) is None
    assert stack([("a", [("a", "#000", times, np.full(times.size, np.nan))])], 20.0) is None
    # a series of one value a second over a 20 s lane fills every point it is drawn with, so its line is joined
    slow = np.arange(0.0, 20.0, 1.0)
    sparse = lines([("coherence", slow, np.full(slow.size, 0.3))], 20.0, bounds=(0.0, 1.0))
    assert len(sparse["series"][0]["values"]) == slow.size and None not in sparse["series"][0]["values"]
    # points travel as 16-bit integers with gaps kept, fine enough for a row a few pixels high
    packed = unpack(stack([("a", [("a", "#000", times, np.where(times < 10, np.sin(times), np.nan))])], 20.0)["rows"][0]["series"][0]["packed"])
    assert np.isnan(packed[-1]) and np.nanmax(np.abs(packed)) == pytest.approx(1.0, abs=0.1)
