from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("h5py")

TRUE_OFFSET = -150.3   # the file started 150.3 s before the recording's clock


def turning(times: np.ndarray, turns: list[tuple[float, float, float]]) -> np.ndarray:
    """Head yaw in degrees at times, from (start, length, degrees) turns, each a smooth step."""
    yaw = np.zeros_like(times, dtype=float)
    for start, length, degrees in turns:
        x = np.clip((times - start) / length, 0.0, 1.0)
        yaw += degrees * (1.0 - np.cos(np.pi * x)) / 2.0
    return yaw


def synthetic(*, offset=TRUE_OFFSET, scale=1.0, video_s=300.0, imu_s=1500.0, seed=1, turn_deg=30.0):
    """Face yaw rows of a video on the recording clock, and the IMU of a cap that turns with the head on its own clock."""
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(-400.0, 1600.0, 500))
    turns = [(start, rng.uniform(0.3, 1.5), rng.normal(0.0, turn_deg)) for start in starts]
    frames = np.arange(0.0, video_s, 1 / 25)
    yaw = turning(frames, turns) + rng.normal(0.0, 0.5, frames.size)
    yaw[(frames > 100) & (frames < 120)] = np.nan   # the face lost for a while
    rows = pd.DataFrame({"feature": "face_yaw_deg", "t_start": frames, "value": yaw})
    own = np.arange(0.0, imu_s, 1 / 50)
    turned = np.gradient(turning(offset + own * scale, turns), own)
    aux = pd.concat([
        pd.DataFrame({"aux": "GYRO_X_1", "time": own, "value": rng.normal(0.7, 2.0, own.size)}),
        pd.DataFrame({"aux": "GYRO_Y_1", "time": own, "value": rng.normal(-0.3, 2.0, own.size)}),
        pd.DataFrame({"aux": "GYRO_Z_1", "time": own, "value": turned + rng.normal(1.1, 2.0, own.size)}),
        pd.DataFrame({"aux": "ACCEL_X_1", "time": own, "value": rng.normal(9.8, 0.1, own.size)}),
    ], ignore_index=True)
    return rows, aux


def test_the_cap_imu_finds_where_its_file_sits_on_the_recording_clock():
    from macaw.media.drift import drift_is_real
    from macaw.signals.imu import AGREEMENT_S, FACE_ANGLES, gyro_speed, head_speed, match

    rows, aux = synthetic()
    found = match(*head_speed(rows, FACE_ANGLES), *gyro_speed(aux, "gyro"))
    assert found.offset_s == pytest.approx(TRUE_OFFSET, abs=0.1)
    assert found.confidence > 0.3 and found.correlation > 0.5 and found.agreement == 1.0
    assert not drift_is_real(found.drift_ppm, found.drift_r2, found.overlap_s, AGREEMENT_S)
    # a window around the start times' hint keeps the search near it
    near = match(*head_speed(rows, FACE_ANGLES), *gyro_speed(aux, "gyro"), hint_s=TRUE_OFFSET + 4.0, window_s=10.0)
    assert near.offset_s == pytest.approx(TRUE_OFFSET, abs=0.1)
    assert gyro_speed(aux, "no such sensor") is None


def test_a_file_with_several_gyroscopes_asks_which_one_is_on_the_head():
    import re

    from macaw.exceptions import ConfigError
    from macaw.signals.imu import gyro_sensors, gyro_speed, sensor_pattern

    own = np.arange(0.0, 10.0, 0.01)
    rng = np.random.default_rng(2)
    aux = pd.concat([pd.DataFrame({"aux": f"gyroscope_{k}_{axis}", "time": own, "value": rng.normal(0.0, k, own.size)})
                     for k in (1, 2) for axis in "xyz"], ignore_index=True)
    with pytest.raises(ConfigError, match=re.escape("'^gyroscope_2_[xyz]$'")):
        gyro_speed(aux, "gyro")
    _, speed = gyro_speed(aux, "^gyroscope_1_[xyz]$")
    assert np.allclose(speed, gyro_speed(aux[aux["aux"].str.startswith("gyroscope_1")], "gyro")[1])
    for names, count in ((["GYRO_X_1", "GYRO_Y_1", "GYRO_Z_1", "GYRO_X_2", "GYRO_Y_2", "GYRO_Z_2"], 2),
                         (["gyro-x", "gyro-y", "gyro-z"], 1), (["GyroX", "GyroY", "GyroZ"], 1), (["gyro_1", "gyro_2", "gyro_3"], 1)):
        sensors = gyro_sensors(names)
        assert len(sensors) == count
        for axes in sensors.values():
            assert [name for name in names if re.search(sensor_pattern(axes), name, re.IGNORECASE)] == axes


def test_a_face_tracker_jittering_by_degrees_each_frame_still_finds_the_offset():
    from macaw.signals.imu import FACE_ANGLES, gyro_speed, head_speed, match

    # turns of a few degrees, smaller than a frame-to-frame slope of the jitter
    rows, aux = synthetic(seed=4, turn_deg=8.0)
    rows["value"] += np.random.default_rng(5).normal(0.0, 3.0, len(rows))
    found = match(*head_speed(rows, FACE_ANGLES), *gyro_speed(aux, "gyro"))
    assert found.offset_s == pytest.approx(TRUE_OFFSET, abs=0.2)
    times, speed = head_speed(rows, FACE_ANGLES)
    assert np.isnan(speed[(times > 101) & (times < 119)]).all() and np.isfinite(speed[times < 99]).all()


def test_a_clock_running_fast_is_fitted_and_maps_the_file_onto_the_recording():
    from macaw.media.align import AlignResult
    from macaw.signals.align import IMU, drift_corrected, signal_clock
    from macaw.signals.imu import FACE_ANGLES, gyro_speed, head_speed, match

    scale = 1 / (1 - 1500e-6)
    rows, aux = synthetic(scale=scale, video_s=600.0, seed=3)
    found = match(*head_speed(rows, FACE_ANGLES), *gyro_speed(aux, "gyro"))
    assert found.drift_ppm == pytest.approx(1500.0, abs=300.0)
    result = AlignResult(reference="recording", target="child_raw", offset_s=found.offset_s, confidence=found.confidence,
                         overlap_s=found.overlap_s, drift_ppm=found.drift_ppm, drift_r2=found.drift_r2,
                         drift_intercept_s=found.drift_intercept_s, method=IMU)
    assert drift_corrected(result)
    offset, fitted = signal_clock(result)
    for own in (150.3, 450.0, 750.0):
        assert offset + own * fitted == pytest.approx(TRUE_OFFSET + own * scale, abs=0.1)


def write_snirf(path, aux: pd.DataFrame | None, markers=((20.0, "start"),)):
    import h5py

    own = np.arange(0.0, 1500.0, 0.1)
    with h5py.File(path, "w") as handle:
        nirs = handle.create_group("nirs")
        tags = nirs.create_group("metaDataTags")
        tags["TimeUnit"], tags["MeasurementDate"], tags["MeasurementTime"] = "s", "2020-01-01", "10:00:00"
        data = nirs.create_group("data1")
        data["dataTimeSeries"], data["time"] = np.sin(own)[:, None], own
        item = data.create_group("measurementList1")
        item["sourceIndex"], item["detectorIndex"], item["wavelengthIndex"], item["dataType"] = 1, 1, 0, 99999
        item["dataTypeLabel"] = "HbO"
        for number, (onset, name) in enumerate(markers, start=1):
            stim = nirs.create_group(f"stim{number}")
            stim["name"], stim["data"] = name, np.array([[onset, 1.0, 1.0]])
        for number, (name, axis) in enumerate(() if aux is None else aux.groupby("aux", sort=False), start=1):
            group = nirs.create_group(f"aux{number}")
            group["name"], group["dataTimeSeries"], group["time"] = name, axis["value"].to_numpy(), axis["time"].to_numpy()
    return path


@pytest.fixture
def aligned_project(tmp_path):
    from macaw.config.loader import resolve_config, scaffold_project_config
    from macaw.project.layout import layout_for
    from macaw.store.longtable import build_long_rows, write_long_table
    from macaw.store.schema import Scale
    from setup_helpers import camera, put, write_setup

    raw = tmp_path / "raw"
    put(raw, "003", "room", "room.mp4")
    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=raw, output_dir=tmp_path / "out", roles=("parent", "child"))
    write_setup(project, cameras={"room": camera()})
    rows, aux = synthetic()
    folder = tmp_path / "nirs"
    folder.mkdir()
    write_snirf(folder / "sub-c003_task-play_nirs.snirf", aux)
    write_snirf(folder / "sub-c003_task-play_desc-preproc_nirs.snirf", None)
    write_snirf(folder / "sub-p003_task-play_run-1_nirs.snirf", aux[aux["aux"].str.startswith("ACCEL")])
    (tmp_path / "signals.csv").write_text(
        "recording,person,stage,file\n003,child,raw,nirs/sub-c003_task-play_nirs.snirf\n"
        "003,child,preproc,nirs/sub-c003_task-play_desc-preproc_nirs.snirf\n003,parent,raw,nirs/sub-p003_task-play_run-1_nirs.snirf\n",
        encoding="utf-8",
    )
    config = resolve_config(project)
    layout = layout_for(config)
    head = build_long_rows(recording_id="003", source="face.au", scale=Scale.FRAME, subject_id="child",
                           values=pd.DataFrame({"t_start": rows["t_start"], "face_yaw_deg": rows["value"]}))
    write_long_table(head, layout.long_table_path("003"))
    return project, config, layout


def run_steps(config, layout):
    from macaw.project.recording import Recording
    from macaw.signals.align import SignalAlignExtractor
    from macaw.signals.read import SignalReadExtractor

    recording = Recording(recording_id="003", root=layout.output_dir)
    for step in (SignalReadExtractor, SignalAlignExtractor):
        folder = layout.artifacts_dir("003", step.name)
        folder.mkdir(parents=True, exist_ok=True)
        result = step(config).run(recording, folder)
    return recording, result


def test_each_person_takes_the_clock_their_imu_finds_and_a_person_checks_it(aligned_project):
    from typer.testing import CliRunner

    from macaw.cli.main import app
    from macaw.extractors.base import PERSON
    from macaw.media.align import MANUAL, UNALIGNED
    from macaw.signals.align import IMU, SignalReviewGate, computed_offsets, signal_offsets

    project, config, layout = aligned_project
    recording, result = run_steps(config, layout)
    found = {entry.target: entry for entry in computed_offsets(config, "003")}
    assert found["child_raw"].method == IMU and found["child_raw"].offset_s == pytest.approx(TRUE_OFFSET, abs=0.1)
    assert found["child_preproc"].offset_s == found["child_raw"].offset_s
    assert found["child_preproc"].note.startswith("same clock as child_raw")
    assert found["parent_raw"].method == UNALIGNED and "gyro" in found["parent_raw"].note
    assert found["child_raw"].metadata_offset_s is None   # the recording here has no video to read a start time from
    assert not [warning for warning in result.warnings if "markers at other times" in warning.message]

    gate = SignalReviewGate(config)
    waiting = gate.pending(recording)
    assert "child_raw (not looked at yet)" in waiting and "parent_raw (not aligned" in waiting

    runner = CliRunner()
    said = runner.invoke(app, ["accept-signal-offset", "-c", str(project), "-r", "003", "--by", "tester"])
    assert said.exit_code == 0, said.output
    assert "kept child_raw, child_preproc" in said.output and "parent_raw has no offset to keep" in said.output
    said = runner.invoke(app, ["set-signal-offset", "-c", str(project), "-r", "003", "--person", "parent", "--offset", "-140", "--by", "tester"])
    assert said.exit_code == 0, said.output
    assert gate.pending(recording) is None and gate.decided_by(recording) == PERSON

    current = signal_offsets(config, "003")
    assert current["child_raw"].accepted.startswith("tester") and current["parent_raw"].method == MANUAL
    # a rerun that finds the same keeps what the person decided
    run_steps(config, layout)
    assert gate.pending(recording) is None
    shown = runner.invoke(app, ["signal-offsets", "-c", str(project), "-r", "003"])
    assert shown.exit_code == 0 and "child_raw: -150." in shown.output and "parent_raw: -140.00 s, set by hand by tester" in shown.output


def test_marked_events_pair_by_name_where_most_of_them_agree():
    from macaw.signals.markers import match_markers

    # a trigger sent every 300 s and once at 540 s, one event listed twice, names as numbers written two ways
    markers = [("12", 320.0 + 300.0 * k) for k in range(5)] + [("2.0", 540.0), ("1.0", 20.0), ("1.0", 20.0)]
    found = match_markers([("1", 120.1)], markers)
    assert found.offset_s == pytest.approx(100.1) and found.pairs == [("1", 120.1, 20.0)] and not found.unpaired
    # one 12 alone could be any of five; the 2 beside it settles which
    found = match_markers([("12", 720.1), ("2", 640.0), ("12", 50.0), ("touched_cap", 70.0)], markers)
    assert found.offset_s == pytest.approx(100.0, abs=0.1) and len(found.pairs) == 2 and found.unpaired == ["12"]
    assert not found.alternatives and match_markers([("touched_cap", 70.0)], markers) is None
    # a 12 alone fits five places; a hint such as the IMU's offset picks one, and the others are kept to say so
    alone = match_markers([("12", 720.1)], markers, prefer=95.0)
    assert alone.offset_s == pytest.approx(100.1) and len(alone.alternatives) == 4


def test_a_mark_that_fits_several_places_waits_unless_a_sure_imu_settles_it():
    from macaw.media.align import UNALIGNED, AlignResult
    from macaw.signals.align import IMU, marker_result
    from macaw.signals.markers import match_markers

    cfg = SimpleNamespace(min_confidence=0.3, min_agreement=0.5)
    markers = [("12", 320.0 + 300.0 * k) for k in range(5)]
    none = AlignResult(reference="recording", target="child_raw", offset_s=0.0, confidence=None, method=UNALIGNED)
    alone = marker_result(match_markers([("12", 720.1)], markers), "child_raw", none, cfg)
    assert "fit 5 places as well (12 recurs in the file)" in alone.doubt
    imu = AlignResult(reference="recording", target="child_raw", offset_s=100.05, confidence=0.6, agreement=1.0, method=IMU)
    settled = marker_result(match_markers([("12", 720.1)], markers, prefer=imu.offset_s), "child_raw", imu, cfg)
    assert settled.offset_s == pytest.approx(100.1) and settled.doubt is None and settled.cross_check_s == 100.05


def test_marked_events_spread_over_the_recording_correct_a_drifting_clock():
    from macaw.media.align import UNALIGNED, AlignResult
    from macaw.signals.align import MARKERS, drift_corrected, marker_result, signal_clock
    from macaw.signals.markers import match_markers

    rate = 1 + 1500e-6
    onsets = [100.0, 1100.0, 2100.0, 3100.0]
    found = match_markers([("start", 40.0 + own * rate + jitter) for own, jitter in zip(onsets, (0.04, -0.03, 0.02, -0.04))],
                          [("start", own) for own in onsets])
    none = AlignResult(reference="recording", target="child_raw", offset_s=0.0, confidence=None, method=UNALIGNED)
    result = marker_result(found, "child_raw", none, SimpleNamespace(min_confidence=0.3, min_agreement=0.5))
    assert result.method == MARKERS and drift_corrected(result) and result.doubt is None
    offset, scale = signal_clock(result)
    for own in (100.0, 1600.0, 3100.0):
        assert offset + own * scale == pytest.approx(40.0 + own * rate, abs=0.06)
    # one mark a second off is flagged
    found.pairs[1] = (found.pairs[1][0], found.pairs[1][1] + 1.0, found.pairs[1][2])
    assert "disagree by up to" in marker_result(found, "child_raw", none, SimpleNamespace(min_confidence=0.3, min_agreement=0.5)).doubt


def test_a_file_whose_events_are_marked_on_the_video_takes_their_offset_and_the_imu_checks_it(aligned_project, tmp_path):
    from macaw.qc.boilerplate import _signals, signals_placed
    from macaw.signals.align import IMU, MARKERS, computed_offsets, describe, needs_person

    project, config, layout = aligned_project
    _, aux = synthetic()
    # the child's raw file and the parent's carry a trigger at 170.3 and 470.3 s, 20 and 320 s into the video
    events = ((170.3, "1.0"), (470.3, "1.0"))
    write_snirf(tmp_path / "nirs" / "sub-c003_task-play_nirs.snirf", aux, markers=events)
    write_snirf(tmp_path / "nirs" / "sub-p003_task-play_run-1_nirs.snirf", aux[aux["aux"].str.startswith("ACCEL")], markers=events)
    (tmp_path / "marks.csv").write_text("recording,mark,start_s,end_s\n003,1,20.05,21\n003,1,319.95,321\n003,touched_cap,40,42\n", encoding="utf-8")
    run_steps(config, layout)
    found = {entry.target: entry for entry in computed_offsets(config, "003")}
    child, parent = found["child_raw"], found["parent_raw"]
    assert child.method == MARKERS and child.offset_s == pytest.approx(TRUE_OFFSET, abs=0.01)
    assert child.cross_check_s == pytest.approx(TRUE_OFFSET, abs=0.1) and child.doubt is None
    assert "from 2 marked event(s)" in describe(child) and "the IMU finds -150.3" in describe(child)
    assert needs_person(child, config.signals) == "not looked at yet"
    # without a gyroscope the parent's file is placed by its events alone
    assert parent.method == MARKERS and parent.cross_check_s is None and parent.offset_s == pytest.approx(TRUE_OFFSET, abs=0.01)
    # the preprocessed copy carries no marked event and keeps the IMU's clock
    assert found["child_preproc"].method == IMU
    placed = signals_placed(config)
    assert (placed["markers"], placed["imu"], placed["checked"]) == (2, 1, 1)
    text = _signals(placed)
    assert "a person marked on the video" in text and "For 1 of the three signal files, the turning speed" in text

    # a mark 1.5 s off asks a person to look, and the IMU says so too
    (tmp_path / "marks.csv").write_text("recording,mark,start_s,end_s\n003,1,21.55,22\n003,1,319.95,321\n", encoding="utf-8")
    run_steps(config, layout)
    child = {entry.target: entry for entry in computed_offsets(config, "003")}["child_raw"]
    assert "disagree by up to 0.80 s" in needs_person(child, config.signals) and "s away" in child.doubt
