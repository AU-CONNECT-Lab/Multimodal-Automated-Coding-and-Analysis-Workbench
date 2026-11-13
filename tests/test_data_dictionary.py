from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.dictionary import COLUMNS, NOT_STATED, feature_dictionary, flat_settings, write_dictionary
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureManifest, FeatureSpec, merge_manifests, write_manifest
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from macaw.store.wide import DEFAULT_INDEX, with_segment

from test_scale import project, rows_of, said, settings  # noqa: F401

runner = CliRunner()


def one_row(feature, value, recording="003"):
    rows = build_long_rows(
        values=pd.DataFrame({"t_start": [0.0], feature: [value]}), scale=Scale.FRAME, recording_id=recording,
        source="pose.kinematics", subject_id="parent",
    )
    rows["subject_id_b"] = None
    return rows


def spec(name, unit, *, extractor="pose.kinematics", params=None, **fields):
    return FeatureSpec(
        name=name, description=f"What {name} is.", unit=unit, scale=Scale.FRAME, group="pose", extractor=extractor,
        extractor_version="0.6.0", params=params or {}, **fields,
    )


@pytest.fixture
def exported(project):  # noqa: F811
    """One recording with a pixel measure for each person and the pair, a recording-level count, and a step that says
    nothing of its empty cells."""
    layout = layout_for(resolve_config(project))
    directory = layout.long_table_path("003")
    directory.mkdir(parents=True)
    concat_long_tables([
        rows_of({"hand_distance": [100.0, 50.0]}),
        rows_of({"hand_distance": [10.0, 20.0]}, subject="child"),
        rows_of({"shoulder_distance": [300.0, 310.0]}, subject="parent", subject_b="child"),
        build_long_rows(
            values=pd.DataFrame({"t_start": [0.0], "t_end": [2.0], "n_turns": [4.0]}), scale=Scale.RECORDING,
            recording_id="003", source="turns.structure",
        ),
    ]).to_parquet(directory / "pose.kinematics.parquet", index=False)
    fragment = FeatureManifest()
    fragment.add(spec("hand_distance", "px", params={"smoothing_s": 0.2, "joints": {"from": "left_wrist"}}, missing="Empty where the hand was not seen."))
    fragment.add(spec("shoulder_distance", "px"))
    fragment.add(spec("n_turns", None, extractor="turns.structure", comparable_across_subjects=False))
    write_manifest(fragment, layout.manifest_fragment_path("003", "pose.kinematics"))
    settings(project, "scale:\n  per_px_by_role: {parent: 0.2, child: 0.3}\n")
    return project


def export(project, tmp_path, *extra, name="features.csv"):  # noqa: F811
    out = tmp_path / "exports" / name
    result = runner.invoke(app, ["export", "-c", str(project), "--out", str(out), *extra])
    assert result.exit_code == 0, said(result)
    return out, pd.read_csv(out.with_name(f"{out.stem}_dictionary.csv"), keep_default_na=False), json.loads(
        out.with_name(f"{out.stem}_dictionary.json").read_text(encoding="utf-8")
    )


def test_each_column_of_the_wide_table_has_an_entry_in_its_order(exported, tmp_path):
    out, table, sidecar = export(exported, tmp_path)
    columns = list(pd.read_csv(out).columns)
    assert list(table.column) == columns == list(sidecar)
    assert columns[: len(DEFAULT_INDEX)] == DEFAULT_INDEX and list(table.columns) == COLUMNS
    rows = table.set_index("column")
    assert rows.at["segment", "missing"].startswith("Empty for the whole analysed part")
    assert rows.at["t_start", "unit"] == "s" and sidecar["t_start"]["Units"] == "s"


def test_a_measure_says_what_it_is_whose_rows_hold_it_and_what_an_empty_cell_means(exported, tmp_path):
    _, table, sidecar = export(exported, tmp_path)
    rows = table.set_index("column")
    hand = rows.loc["hand_distance"]
    assert hand.meaning == "What hand_distance is." and hand.unit == "px" and hand.who == "person" and hand.rows == "frame"
    assert hand.missing == "Empty where the hand was not seen." and hand.step.startswith("pose.kinematics 0.6.0 (pymacaw ")
    assert hand.settings == "joints.from=left_wrist; smoothing_s=0.2"
    assert sidecar["hand_distance"]["settings"] == {"joints": {"from": "left_wrist"}, "smoothing_s": 0.2}
    assert rows.at["shoulder_distance", "who"] == "pair" and rows.at["shoulder_distance", "missing"] == NOT_STATED
    assert rows.at["n_turns", "rows"] == "recording" and rows.at["n_turns", "comparable_across_people"] == "no"
    assert sidecar["n_turns"]["comparable_across_people"] is False and "Units" not in sidecar["n_turns"]


def test_a_column_in_centimetres_names_the_pixels_it_came_from(exported, tmp_path):
    _, table, sidecar = export(exported, tmp_path)
    converted = table.set_index("column").loc["hand_distance_cm"]
    assert converted.unit == "cm" and converted.meaning == "What hand_distance is."
    assert converted.converted_from.startswith("hand_distance, in pixels") and "scale.csv" in converted.converted_from
    assert converted.missing.startswith("Empty where the hand was not seen. Also empty where the recording has no usable scale")
    assert sidecar["hand_distance_cm"]["Units"] == "cm"


def test_the_long_layout_lists_its_own_columns_then_each_measure(exported, tmp_path):
    out, table, sidecar = export(exported, tmp_path, "--long")
    assert list(table.column[:16]) == list(pd.read_csv(out).columns)
    assert set(table.column[16:]) == set(pd.read_csv(out).feature)
    assert set(sidecar["scale"]["Levels"]) == {"frame", "event", "window", "recording"}
    assert "frame: one video or audio frame" in table.set_index("column").at["scale", "meaning"]


def test_a_parquet_export_gets_the_same_dictionary(exported, tmp_path):
    _, from_csv, _ = export(exported, tmp_path)
    _, from_parquet, _ = export(exported, tmp_path, "-f", "parquet", name="features.parquet")
    pd.testing.assert_frame_equal(from_csv, from_parquet)


def test_recordings_run_under_their_own_settings_show_each_variant():
    table = with_segment(concat_long_tables([one_row("hand_distance", value, recording) for value, recording in ((1.0, "a"), (2.0, "b"), (3.0, "c"))]))
    specs = {
        "a": {"hand_distance": spec("hand_distance", "px", params={"smoothing_s": 0.2})},
        "b": {"hand_distance": spec("hand_distance", "px", params={"smoothing_s": 0.5}, missing="Empty where unseen.")},
        "c": {"hand_distance": spec("hand_distance", "px", params={"smoothing_s": 0.2})},
    }
    (entry,) = feature_dictionary(table, ["hand_distance"], specs, "cm")
    assert entry["settings"] == [
        {"recordings": ["a", "c"], "settings": {"smoothing_s": 0.2}}, {"recordings": ["b"], "settings": {"smoothing_s": 0.5}},
    ]
    assert flat_settings(entry["settings"]) == "a, c: smoothing_s=0.2 | b: smoothing_s=0.5"
    assert entry["missing"] == f"{NOT_STATED} (for a, c); Empty where unseen. (for b)"
    assert entry["meaning"] == "What hand_distance is."


def test_a_measure_no_step_described_still_gets_an_entry(tmp_path):
    table = with_segment(concat_long_tables([one_row("mystery", 1.0)]))
    entries = feature_dictionary(table, ["recording_id", "mystery"], {}, "cm")
    csv_path, json_path = write_dictionary(entries, tmp_path / "features.csv")
    assert pd.read_csv(csv_path).set_index("column").at["mystery", "missing"] == NOT_STATED
    assert json.loads(json_path.read_text(encoding="utf-8"))["mystery"]["Description"] == "No step described this measure."


def test_recordings_with_their_own_settings_merge_into_one_manifest():
    first, second = FeatureManifest(), FeatureManifest()
    first.add(spec("f0_hz", "Hz", extractor="prosody.frame", params={"floor": 75}))
    second.add(spec("f0_hz", "Hz", extractor="prosody.frame", params={"floor": 100}))
    assert merge_manifests([first, second]).specs["f0_hz"].params == {"floor": 75}
    other = FeatureManifest()
    other.add(spec("f0_hz", "Hz", extractor="prosody.voice_report"))
    with pytest.raises(MacawError, match="described by both"):
        merge_manifests([first, other])
