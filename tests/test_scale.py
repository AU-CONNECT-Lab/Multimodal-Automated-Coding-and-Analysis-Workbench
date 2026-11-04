from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError, ExternalToolError
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureManifest, FeatureSpec, write_manifest
from macaw.project.scale import (
    CALIBRATION,
    HAND,
    MEASURED,
    Measurement,
    apparent_size,
    centimetre_rows,
    in_centimetres,
    pixel_power,
    read_measurements,
    recording_scale,
    size_change,
    unusable,
    write_measurements,
)
from macaw.qc.boilerplate import describe, load_references, render
from macaw.qc.checks import check_scale
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from setup_helpers import camera, mics, put, recording_of, write_setup

from test_kinematics import person

runner = CliRunner()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for device, name in (("parent_mic", "a.wav"), ("child_mic", "a.wav"), ("room", "a.mp4"), ("side", "a.mp4")):
        put(raw, "003", device, name)
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(config, cameras={"room": camera(), "side": camera(uses=("body",))}, microphones=mics())
    return config


def settings(project, text: str, recording: str | None = None) -> None:
    """Add a top-level section to project.yaml, or write a recording's own file."""
    if recording is None:
        with open(project, "a", encoding="utf-8") as handle:
            handle.write("\n" + text)
    else:
        (project.parent / "recordings").mkdir(exist_ok=True)
        (project.parent / "recordings" / f"{recording}.yaml").write_text(text, encoding="utf-8")


def set_scale(project, *extra):
    return runner.invoke(app, ["set-scale", "-c", str(project), *extra])


def said(result) -> str:
    """What a command printed, or the error it stopped with."""
    return result.output + str(result.exception or "")


def clip(project, name="lab.mp4", size="64x48"):
    from macaw.media.ffmpeg import run_ffmpeg

    folder = project.parent / "calibration"
    folder.mkdir(exist_ok=True)
    try:
        run_ffmpeg(["-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration=2", "-pix_fmt", "yuv420p", str(folder / name)])
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    return name


def rows_of(values: dict, subject="parent", subject_b=None) -> pd.DataFrame:
    rows = build_long_rows(
        values=pd.DataFrame({"t_start": [0.0, 1.0], **values}),
        scale=Scale.FRAME, recording_id="003", source="pose.kinematics", subject_id=subject,
    )
    rows["subject_id_b"] = subject_b
    return rows


class TestUnits:
    @pytest.mark.parametrize("unit, power", [("px", 1), ("px/s", 1), ("px/s^3", 1), ("px^2", 2), ("m", None), ("s", None), (None, None), ("pixels", None)])
    def test_a_unit_counts_its_pixel_lengths(self, unit, power):
        assert pixel_power(unit) == power

    def test_centimetres_take_the_place_of_pixels(self):
        assert [in_centimetres(unit) for unit in ("px", "px/s^2", "px^2")] == ["cm", "cm/s^2", "cm^2"]

    def test_lengths_scale_once_and_areas_twice(self):
        rows = rows_of({"hand_distance": [100.0, 40.0], "extent": [400.0, 0.0], "f0_hz": [200.0, 210.0]})
        converted = centimetre_rows(rows, {"hand_distance": "px", "extent": "px^2", "f0_hz": "Hz"}, lambda first, second: 0.5)
        values = converted.groupby("feature")["value"].apply(list).to_dict()
        assert values == {"hand_distance_cm": [50.0, 20.0], "extent_cm": [100.0, 0.0]}

    def test_rows_without_a_factor_stay_in_pixels_only(self):
        rows = concat_long_tables([rows_of({"hand_distance": [100.0, 40.0]}), rows_of({"hand_distance": [10.0, 20.0]}, subject="child")])
        converted = centimetre_rows(rows, {"hand_distance": "px"}, lambda first, second: 0.5 if first == "parent" else None)
        assert set(converted["subject_id"]) == {"parent"} and converted["value"].tolist() == [50.0, 20.0]


class TestScale:
    def test_the_measurements_average_per_place_and_keep_their_spread(self, project):
        config = resolve_config(project)
        layout = layout_for(config)
        write_measurements(
            [
                Measurement("room", 5.0, [[0, 0], [30, 40]], 10.0, person="child"),
                Measurement("room", 9.0, [[0, 0], [0, 100]], 21.0, person="child"),
                Measurement("room", 9.0, [[0, 0], [0, 100]], 15.0),
            ],
            layout.scale_path("003"),
        )
        scale = recording_scale(config, layout, "003")
        assert scale.source == MEASURED and scale.video == "room"
        assert scale.by_role["child"] == pytest.approx(0.205) and scale.spread["child"] == pytest.approx(0.00707, abs=1e-4)
        assert scale.for_role("parent") == pytest.approx(0.15) and scale.spread[None] is None
        assert "0.205 ± 0.0071 for child" in scale.describe() and "0.15 for everyone" in scale.describe()

    def test_a_pair_is_converted_only_when_both_share_a_scale(self):
        from macaw.project.scale import RecordingScale

        apart = RecordingScale({"child": 0.2, "parent": 0.1}, {}, HAND, None, [])
        assert apart.factor("child", None) == 0.2 and apart.factor("child", "parent") is None and apart.factor(None, None) is None
        together = RecordingScale({None: 0.1}, {}, HAND, None, [])
        assert together.factor("child", "parent") == 0.1 and together.factor(None, None) == 0.1

    def test_numbers_written_by_hand_win_and_a_role_beats_everyone(self, project):
        layout = layout_for(resolve_config(project))
        write_measurements([Measurement("room", 5.0, [[0, 0], [30, 40]], 10.0)], layout.scale_path("003"))
        settings(project, "scale:\n  cm_per_px: 0.3\n  cm_per_px_by_role: {child: 0.4}\n", recording="003")
        assert recording_scale(resolve_config(project), layout, "003").source == MEASURED
        own = recording_scale(resolve_config(project, recording_id="003"), layout, "003")
        assert own.source == HAND and own.for_role("child") == 0.4 and own.for_role("parent") == 0.3

    def test_a_role_the_project_does_not_have_is_refused(self, project):
        settings(project, "scale:\n  cm_per_px_by_role: {teacher: 0.4}\n")
        with pytest.raises(ConfigError, match="teacher"):
            resolve_config(project)

    def test_a_calibration_clip_serves_the_recordings_that_name_it(self, project):
        layout = layout_for(resolve_config(project))
        write_measurements([Measurement("lab.mp4", 1.0, [[0, 0], [0, 50]], 10.0, frame_size=[64, 48])], layout.calibration_scale_path("lab.mp4"))
        assert recording_scale(resolve_config(project), layout, "003") is None
        settings(project, "scale:\n  calibration: lab.mp4\n")
        scale = recording_scale(resolve_config(project), layout, "003")
        assert scale.source == CALIBRATION and scale.for_role("child") == 0.2 and "calibration clip lab.mp4" in scale.describe()
        # the recording's own measurements come before a clip
        write_measurements([Measurement("room", 5.0, [[0, 0], [30, 40]], 10.0)], layout.scale_path("003"))
        assert recording_scale(resolve_config(project), layout, "003").source == MEASURED

    def test_a_clip_carries_over_only_to_video_of_its_size(self, project, monkeypatch):
        from macaw.project import scale as scale_module

        settings(project, "scale:\n  calibration: lab.mp4\n")
        config = resolve_config(project)
        layout = layout_for(config)
        write_measurements([Measurement("lab.mp4", 1.0, [[0, 0], [0, 50]], 10.0, frame_size=[64, 48])], layout.calibration_scale_path("lab.mp4"))
        scale = recording_scale(config, layout, "003")
        seen = person("parent", n_frames=3)
        monkeypatch.setattr(scale_module, "picture_size", lambda config, rid, video: (64, 48))
        assert unusable(scale, config, "003", seen) is None
        monkeypatch.setattr(scale_module, "picture_size", lambda config, rid, video: (1920, 1080))
        assert "64x48 but parent.mp4 is 1920x1080" in unusable(scale, config, "003", seen)
        two = pd.concat([seen, person("child", n_frames=3)], ignore_index=True)
        assert "stands for one camera" in unusable(scale, config, "003", two)

    def test_no_scale_is_none(self, project):
        config = resolve_config(project)
        assert recording_scale(config, layout_for(config), "003") is None


class TestCommand:
    def test_each_measurement_is_kept_with_whose_place_and_who_made_it(self, project):
        result = set_scale(
            project, "-r", "003", "--camera", "room", "--points", "0,0,30,40", "--length-cm", "10", "--at", "4.5",
            "--person", "child", "--object", "ruler", "--by", "ann",
        )
        assert result.exit_code == 0, said(result)
        assert "cm per pixel 0.2 for child" in result.output and "right only for people as far from the camera" in result.output
        assert set_scale(project, "-r", "003", "--camera", "room", "--points", "0,0,0,100", "--length-cm", "21").exit_code == 0
        stored = read_measurements(layout_for(resolve_config(project)).scale_path("003"))
        assert [item.length_px for item in stored] == [50.0, 100.0] and [item.person for item in stored] == ["child", None]
        assert stored[0].object == "ruler" and stored[0].by == "ann" and stored[0].t == 4.5 and stored[0].at

    def test_one_recording_is_measured_on_one_video(self, project):
        set_scale(project, "-r", "003", "--camera", "room", "--points", "0,0,30,40", "--length-cm", "10")
        result = set_scale(project, "-r", "003", "--camera", "side", "--points", "0,0,30,40", "--length-cm", "10")
        assert result.exit_code != 0 and "--clear" in said(result)
        assert set_scale(project, "-r", "003", "--clear").exit_code == 0
        assert set_scale(project, "-r", "003", "--camera", "side", "--points", "0,0,30,40", "--length-cm", "10").exit_code == 0

    def test_a_clip_is_measured_on_its_own_picture_and_named_in_the_settings(self, project):
        name = clip(project)
        result = set_scale(project, "--calibration", name, "--points", "0,0,0,50", "--length-cm", "10", "--person", "parent")
        assert result.exit_code == 0, said(result)
        assert "scale: {calibration: lab.mp4}" in result.output
        stored = read_measurements(layout_for(resolve_config(project)).calibration_scale_path(name))
        assert stored[0].frame_size == [64, 48] and stored[0].person == "parent"
        assert set_scale(project, "--calibration", name, "--clear").exit_code == 0
        assert read_measurements(layout_for(resolve_config(project)).calibration_scale_path(name)) == []

    @pytest.mark.parametrize("extra, reason", [
        (["-r", "003", "--camera", "parent_mic", "--points", "0,0,1,1", "--length-cm", "1"], "not a video"),
        (["-r", "003", "--points", "0,0,1,1", "--length-cm", "1"], "has 2 videos"),
        (["-r", "003", "--camera", "room", "--points", "0,0,1", "--length-cm", "1"], "four numbers"),
        (["-r", "003", "--camera", "room", "--points", "5,5,5,5", "--length-cm", "1"], "same point"),
        (["-r", "003", "--camera", "room", "--points", "0,0,1,1", "--length-cm", "0"], "above 0"),
        (["-r", "003", "--camera", "room", "--points", "0,0,1,1", "--length-cm", "1", "--person", "teacher"], "takes a role"),
        (["-r", "003", "--camera", "room"], "--points"),
        (["--points", "0,0,1,1", "--length-cm", "1"], "--calibration"),
        (["--calibration", "missing.mp4", "--points", "0,0,1,1", "--length-cm", "1"], "no calibration clip"),
    ])
    def test_a_measurement_that_cannot_be_used_is_refused(self, project, extra, reason):
        result = set_scale(project, *extra)
        assert result.exit_code != 0 and reason in said(result)


def test_an_export_gives_each_person_their_own_centimetres(project, tmp_path):
    config = resolve_config(project)
    layout = layout_for(config)
    directory = layout.long_table_path("003")
    directory.mkdir(parents=True)
    concat_long_tables([
        rows_of({"hand_distance": [100.0, 50.0], "f0_hz": [200.0, 210.0]}),
        rows_of({"hand_distance": [10.0, 20.0]}, subject="child"),
        rows_of({"shoulder_distance": [300.0, 310.0]}, subject="parent", subject_b="child"),
    ]).to_parquet(directory / "pose.kinematics.parquet", index=False)
    manifest = FeatureManifest()
    for name, unit in (("hand_distance", "px"), ("f0_hz", "Hz"), ("shoulder_distance", "px")):
        manifest.add(FeatureSpec(name=name, description="", unit=unit, scale=Scale.FRAME, group="pose", extractor="pose.kinematics", extractor_version="0.4.0"))
    write_manifest(manifest, layout.manifest_path)
    settings(project, "scale:\n  cm_per_px_by_role: {parent: 0.2, child: 0.3}\n")

    out = tmp_path / "exports" / "features.csv"
    result = runner.invoke(app, ["export", "-c", str(project), "--long", "--out", str(out)])
    assert result.exit_code == 0, said(result)
    table = pd.read_csv(out)
    converted = table[table.feature == "hand_distance_cm"].groupby("subject_id")["value"].apply(list).to_dict()
    assert converted["parent"] == pytest.approx([20.0, 10.0]) and converted["child"] == pytest.approx([3.0, 6.0])
    # the two sit at different distances, so the distance between them has no one scale
    assert "shoulder_distance_cm" not in set(table.feature) and "f0_hz_cm" not in set(table.feature)
    used = pd.read_csv(out.parent / "scale.csv")
    assert dict(zip(used.person, used.cm_per_px)) == {"parent": 0.2, "child": 0.3}
    assert set(used.source) == {HAND} and used.holds.str.contains("as far from the camera").all()


def moving_parent(fps=5.0):
    """60 s at one distance, then 30 s leaning in, so the parent looks 30% bigger: nose to shoulders 60 px, then 78."""
    near = person("parent", n_frames=int(30 * fps), fps=fps, scale=1.3)
    near["t"] += 60.0
    near["frame"] += int(60 * fps)
    return pd.concat([person("parent", n_frames=int(60 * fps), fps=fps), near], ignore_index=True)


class TestCheck:
    def results(self, project, keypoints):
        return SimpleNamespace(layout=layout_for(resolve_config(project)), keypoints=lambda: keypoints)

    def recording(self, *people):
        return recording_of([], people or ("parent",), recording_id="003")

    def test_nose_to_shoulders_is_measured_per_frame(self):
        assert apparent_size(person("parent", n_frames=3)).tolist() == pytest.approx([60.0] * 3)

    def test_a_person_who_leans_in_looks_bigger_for_a_stretch(self):
        typical, change = size_change(apparent_size(moving_parent()), 30.0)
        assert typical == pytest.approx(60.0) and change == pytest.approx(0.3)

    def test_bending_now_and_then_does_not_count_as_moving(self):
        size = apparent_size(person("parent", n_frames=450, fps=5.0))
        size.iloc[::4] *= 0.6
        assert size_change(size, 30.0)[1] == pytest.approx(0.0)

    def test_the_report_says_how_big_each_person_is_and_flags_one_who_moves(self, project):
        config = resolve_config(project, cli_overrides=["scale.cm_per_px=0.5"])
        outcome = check_scale(self.recording(), config, self.results(project, moving_parent()))
        messages = {warning.severity: warning.message for warning in outcome.warnings}
        assert "about 30 cm at this scale (60 px)" in messages["info"]
        assert "up to 30% bigger or smaller" in messages["warn"]

    def test_shorter_stretches_catch_a_brief_move_that_long_ones_pass(self, project):
        fps = 5.0
        lean = person("parent", n_frames=int(2 * fps), fps=fps, scale=1.3)
        after = person("parent", n_frames=int(28 * fps), fps=fps)
        for part, start in ((lean, 60.0), (after, 62.0)):
            part["t"] += start
            part["frame"] += int(start * fps)
        keypoints = pd.concat([person("parent", n_frames=int(60 * fps), fps=fps), lean, after], ignore_index=True)

        def flagged(window_s):
            config = resolve_config(project, cli_overrides=["scale.cm_per_px=0.5", f"scale.size_window_s={window_s}"])
            outcome = check_scale(self.recording(), config, self.results(project, keypoints))
            return [warning.message for warning in outcome.warnings if warning.severity == "warn"]

        assert flagged(30) == []
        assert "across 5 s stretches" in flagged(5)[0]

    def test_a_person_without_a_scale_at_their_place_is_said(self, project):
        settings(project, "scale:\n  cm_per_px_by_role: {child: 0.5}\n")
        both = pd.concat([person("parent", n_frames=60, fps=5.0), person("child", n_frames=60, fps=5.0)], ignore_index=True)
        outcome = check_scale(self.recording("parent", "child"), resolve_config(project), self.results(project, both))
        by_person = {warning.subject_id: warning.message for warning in outcome.warnings}
        assert "stay in pixels" in by_person["parent"] and "about 30 cm" in by_person["child"]

    def test_an_object_measured_on_another_camera_is_said(self, project):
        layout = layout_for(resolve_config(project))
        write_measurements([Measurement("room", 5.0, [[0, 0], [30, 40]], 10.0)], layout.scale_path("003"))
        outcome = check_scale(self.recording(), resolve_config(project), self.results(project, person("parent", n_frames=60, fps=5.0)))
        assert any("not only on room" in warning.message for warning in outcome.warnings)

    def test_a_clip_named_but_never_measured_is_said(self, project):
        settings(project, "scale:\n  calibration: lab.mp4\n")
        outcome = check_scale(self.recording(), resolve_config(project), self.results(project, moving_parent()))
        assert "nothing was measured on it" in outcome.warnings[0].message

    def test_without_a_scale_there_is_nothing_to_check(self, project):
        outcome = check_scale(self.recording(), resolve_config(project), self.results(project, moving_parent()))
        assert outcome.warnings == [] and outcome.skipped is None


def test_the_methods_say_how_many_recordings_were_converted(project):
    manifest = FeatureManifest()
    manifest.add(FeatureSpec(name="hand_distance", description="", unit="px", scale=Scale.FRAME, group="pose", extractor="pose.kinematics", extractor_version="0.4.0"))
    config = resolve_config(project)
    text, _ = render(describe(manifest, config, scaled=(1, 2)), load_references())
    assert "For 1 of the 2 recordings, lengths in pixels were also converted to centimetres" in text
    assert "holds only at the distance from the camera" in text
    text, _ = render(describe(manifest, config, scaled=(0, 2)), load_references())
    assert "centimetres" not in text
