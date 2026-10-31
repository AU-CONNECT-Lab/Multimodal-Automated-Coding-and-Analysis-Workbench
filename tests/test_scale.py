from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureManifest, FeatureSpec, write_manifest
from macaw.project.scale import (
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
    write_measurements,
)
from macaw.qc.boilerplate import describe, load_references, render
from macaw.qc.checks import check_scale
from macaw.store.longtable import build_long_rows
from macaw.store.schema import Scale

from test_kinematics import person

runner = CliRunner()


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw" / "003"
    raw.mkdir(parents=True)
    for name in ("003_parent.wav", "003_child.wav", "003_room.mp4", "003_side.mp4"):
        (raw / name).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    return config


def set_scale(project, *extra):
    return runner.invoke(app, ["set-scale", "-c", str(project), "-r", "003", *extra])


def said(result) -> str:
    """What a command printed, or the error it stopped with."""
    return result.output + str(result.exception or "")


class TestUnits:
    @pytest.mark.parametrize("unit, power", [("px", 1), ("px/s", 1), ("px/s^3", 1), ("px^2", 2), ("m", None), ("s", None), (None, None), ("pixels", None)])
    def test_a_unit_counts_its_pixel_lengths(self, unit, power):
        assert pixel_power(unit) == power

    def test_centimetres_take_the_place_of_pixels(self):
        assert [in_centimetres(unit) for unit in ("px", "px/s^2", "px^2")] == ["cm", "cm/s^2", "cm^2"]

    def test_lengths_scale_once_and_areas_twice(self):
        rows = build_long_rows(
            values=pd.DataFrame({"t_start": [0.0, 1.0], "hand_distance": [100.0, 40.0], "extent": [400.0, 0.0], "f0_hz": [200.0, 210.0]}),
            scale=Scale.FRAME, recording_id="003", source="pose.kinematics", subject_id="parent",
        )
        converted = centimetre_rows(rows, {"hand_distance": "px", "extent": "px^2", "f0_hz": "Hz"}, 0.5)
        values = converted.groupby("feature")["value"].apply(list).to_dict()
        assert values == {"hand_distance_cm": [50.0, 20.0], "extent_cm": [100.0, 0.0]}


class TestStore:
    def test_the_measurements_average_and_their_spread_is_kept(self, project):
        config = resolve_config(project)
        layout = layout_for(config)
        write_measurements(
            [Measurement("003_room.mp4", 5.0, [[0, 0], [30, 40]], 10.0), Measurement("003_room.mp4", 9.0, [[0, 0], [0, 100]], 21.0)],
            layout.scale_path("003"),
        )
        scale = recording_scale(config, layout, "003")
        assert scale.source == MEASURED and scale.video == "003_room.mp4"
        assert scale.cm_per_px == pytest.approx(0.205) and scale.spread == pytest.approx(0.00707, abs=1e-4)
        assert "2 measurement(s)" in scale.describe()

    def test_a_number_written_by_hand_wins_for_its_recording(self, project):
        layout = layout_for(resolve_config(project))
        write_measurements([Measurement("003_room.mp4", 5.0, [[0, 0], [30, 40]], 10.0)], layout.scale_path("003"))
        (project.parent / "recordings").mkdir()
        (project.parent / "recordings" / "003.yaml").write_text("scale:\n  cm_per_px: 0.3\n", encoding="utf-8")
        assert recording_scale(resolve_config(project), layout, "003").source == MEASURED
        own = recording_scale(resolve_config(project, recording_id="003"), layout, "003")
        assert own.source == HAND and own.cm_per_px == 0.3

    def test_no_scale_is_none(self, project):
        config = resolve_config(project)
        assert recording_scale(config, layout_for(config), "003") is None


class TestCommand:
    def test_each_measurement_is_kept_with_who_made_it(self, project):
        result = set_scale(project, "--media", "003_room.mp4", "--points", "0,0,30,40", "--length-cm", "10", "--at", "4.5", "--object", "table edge", "--by", "ann")
        assert result.exit_code == 0, result.output
        assert "0.2 cm per pixel" in result.output
        assert set_scale(project, "--media", "003_room.mp4", "--points", "0,0,0,100", "--length-cm", "21").exit_code == 0
        stored = read_measurements(layout_for(resolve_config(project)).scale_path("003"))
        assert [item.length_px for item in stored] == [50.0, 100.0]
        assert stored[0].object == "table edge" and stored[0].by == "ann" and stored[0].t == 4.5 and stored[0].at

    def test_one_recording_is_measured_on_one_video(self, project):
        set_scale(project, "--media", "003_room.mp4", "--points", "0,0,30,40", "--length-cm", "10")
        result = set_scale(project, "--media", "003_side.mp4", "--points", "0,0,30,40", "--length-cm", "10")
        assert result.exit_code != 0 and "--clear" in said(result)
        assert set_scale(project, "--clear").exit_code == 0
        assert set_scale(project, "--media", "003_side.mp4", "--points", "0,0,30,40", "--length-cm", "10").exit_code == 0

    @pytest.mark.parametrize("extra, reason", [
        (["--media", "003_parent.wav", "--points", "0,0,1,1", "--length-cm", "1"], "not a video"),
        (["--media", "003_room.mp4", "--points", "0,0,1", "--length-cm", "1"], "four numbers"),
        (["--media", "003_room.mp4", "--points", "5,5,5,5", "--length-cm", "1"], "same point"),
        (["--media", "003_room.mp4", "--points", "0,0,1,1", "--length-cm", "0"], "above 0"),
        (["--media", "003_room.mp4"], "--points"),
    ])
    def test_a_measurement_that_cannot_be_used_is_refused(self, project, extra, reason):
        result = set_scale(project, *extra)
        assert result.exit_code != 0 and reason in said(result)


def test_an_export_gives_pixel_features_again_in_centimetres(project, tmp_path):
    config = resolve_config(project)
    layout = layout_for(config)
    directory = layout.long_table_path("003")
    directory.mkdir(parents=True)
    build_long_rows(
        values=pd.DataFrame({"t_start": [0.0, 1.0], "hand_distance": [100.0, 50.0], "f0_hz": [200.0, 210.0]}),
        scale=Scale.FRAME, recording_id="003", source="pose.kinematics", subject_id="parent",
    ).to_parquet(directory / "pose.kinematics.parquet", index=False)
    manifest = FeatureManifest()
    for name, unit in (("hand_distance", "px"), ("f0_hz", "Hz")):
        manifest.add(FeatureSpec(name=name, description="", unit=unit, scale=Scale.FRAME, group="pose", extractor="pose.kinematics", extractor_version="0.4.0"))
    write_manifest(manifest, layout.manifest_path)
    set_scale(project, "--media", "003_room.mp4", "--points", "0,0,30,40", "--length-cm", "10")

    out = tmp_path / "exports" / "features.csv"
    result = runner.invoke(app, ["export", "-c", str(project), "--long", "--out", str(out)])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(out)
    assert table[table.feature == "hand_distance_cm"]["value"].tolist() == pytest.approx([20.0, 10.0])
    assert "f0_hz_cm" not in set(table.feature) and len(table[table.feature == "hand_distance"]) == 2
    used = pd.read_csv(out.parent / "scale.csv")
    assert used.loc[0, "source"] == MEASURED and used.loc[0, "cm_per_px"] == pytest.approx(0.2)


def moving_parent(fps=5.0):
    """60 s at one distance, then 30 s leaning in, so the parent looks 30% bigger: nose to shoulders 60 px, then 78."""
    near = person("parent", n_frames=int(30 * fps), fps=fps, scale=1.3)
    near["t"] += 60.0
    near["frame"] += int(60 * fps)
    return pd.concat([person("parent", n_frames=int(60 * fps), fps=fps), near], ignore_index=True)


class TestCheck:
    def results(self, project, keypoints):
        return SimpleNamespace(layout=layout_for(resolve_config(project)), keypoints=lambda: keypoints)

    def recording(self):
        return SimpleNamespace(recording_id="003", media=[], subjects=[SimpleNamespace(subject_id="parent")])

    def test_nose_to_shoulders_is_measured_per_frame(self):
        assert apparent_size(person("parent", n_frames=3)).tolist() == pytest.approx([60.0] * 3)

    def test_a_person_who_leans_in_looks_bigger_for_a_stretch(self):
        typical, change = size_change(apparent_size(moving_parent()))
        assert typical == pytest.approx(60.0) and change == pytest.approx(0.3)

    def test_bending_now_and_then_does_not_count_as_moving(self):
        size = apparent_size(person("parent", n_frames=450, fps=5.0))
        size.iloc[::4] *= 0.6
        assert size_change(size)[1] == pytest.approx(0.0)

    def test_the_report_says_how_big_each_person_is_and_flags_one_who_moves(self, project):
        config = resolve_config(project, cli_overrides=["scale.cm_per_px=0.5"])
        outcome = check_scale(self.recording(), config, self.results(project, moving_parent()))
        messages = {warning.severity: warning.message for warning in outcome.warnings}
        assert "about 30 cm at this scale (60 px)" in messages["info"]
        assert "up to 30% bigger or smaller" in messages["warn"]

    def test_a_yardstick_on_another_camera_is_said(self, project):
        layout = layout_for(resolve_config(project))
        write_measurements([Measurement("003_room.mp4", 5.0, [[0, 0], [30, 40]], 10.0)], layout.scale_path("003"))
        outcome = check_scale(self.recording(), resolve_config(project), self.results(project, person("parent", n_frames=60, fps=5.0)))
        assert any("not only on 003_room.mp4" in warning.message for warning in outcome.warnings)

    def test_without_a_scale_there_is_nothing_to_check(self, project):
        outcome = check_scale(self.recording(), resolve_config(project), self.results(project, moving_parent()))
        assert outcome.warnings == [] and outcome.skipped is None


def test_the_methods_say_how_many_recordings_were_converted(project):
    manifest = FeatureManifest()
    manifest.add(FeatureSpec(name="hand_distance", description="", unit="px", scale=Scale.FRAME, group="pose", extractor="pose.kinematics", extractor_version="0.4.0"))
    config = resolve_config(project)
    text, _ = render(describe(manifest, config, scaled=(1, 2)), load_references())
    assert "For 1 of the 2 recordings, lengths in pixels were also converted to centimetres" in text
    text, _ = render(describe(manifest, config, scaled=(0, 2)), load_references())
    assert "centimetres" not in text
