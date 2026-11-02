from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.coding.agreement import NO_CODE, agreement_report, build_units
from macaw.coding.imports import import_coding
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError, EafError, MacawError
from macaw.interop.coding_files import read_coding_file, seconds
from macaw.interop.eaf import export_eaf
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store.annotations import build_annotations, read_annotations, write_annotations

pytest.importorskip("pympi")


@pytest.fixture
def project(tmp_path):
    for name in ("dyad01", "dyad010"):
        (tmp_path / "raw" / name).mkdir(parents=True)
        (tmp_path / "raw" / name / "video.mp4").write_bytes(b"")
    config_path = tmp_path / "project.yaml"
    scaffold_project_config(config_path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")

    def configure(text=""):
        if text:
            config_path.write_text(config_path.read_text(encoding="utf-8") + text, encoding="utf-8")
        config = resolve_config(config_path)
        return config, build_recording("dyad01", config)

    return SimpleNamespace(root=tmp_path, path=config_path, configure=configure)


IMPORTS = """
coding:
  imports:
    Child Gaze:
      person: child
      tier: gaze_target
      labels: {Mom: parent, Away: elsewhere}
"""


def write_eaf(path, tiers):
    """tiers: name -> (participant, [(start_ms, end_ms, value)])"""
    from pympi.Elan import Eaf

    eaf = Eaf()
    eaf.remove_tier("default")
    for name, (participant, annotations) in tiers.items():
        eaf.add_tier(name, part=participant)
        for start, end, value in annotations:
            eaf.add_annotation(name, start, end, value)
    eaf.to_file(str(path))
    return path


def imported(config, recording_id="dyad01"):
    return read_annotations(layout_for(config, create=False).imported_codes_path(recording_id))


class TestReadingFiles:
    def test_an_elan_file_gives_each_annotation_with_its_tier_and_participant(self, tmp_path):
        path = write_eaf(tmp_path / "a.eaf", {"Child Gaze": ("child", [(1000, 2500, "Mom")]), "Notes": (None, [(0, 100, "x")])})
        raw = read_coding_file(path).sort_values("file_tier")
        assert raw["file_tier"].tolist() == ["Child Gaze", "Notes"]
        assert raw.iloc[0][["participant", "t_start", "t_end", "label"]].tolist() == ["child", 1.0, 2.5, "Mom"]
        assert pd.isna(raw.iloc[1]["participant"])

    @pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx"])
    def test_a_table_takes_seconds_or_clock_times(self, tmp_path, suffix):
        table = pd.DataFrame({"Tier": ["gaze", "gaze"], "Start": ["1.5", "1:02.5"], "End": ["2", "0:01:03"], "Code": ["Mom", "Away"], "Person": ["child", ""]})
        path = tmp_path / f"a{suffix}"
        if suffix == ".xlsx":
            table.to_excel(path, index=False)
        else:
            table.to_csv(path, sep="\t" if suffix == ".tsv" else ",", index=False)
        raw = read_coding_file(path)
        assert raw["t_start"].tolist() == [1.5, 62.5] and raw["t_end"].tolist() == [2.0, 63.0]
        assert raw["participant"].iloc[0] == "child" and pd.isna(raw["participant"].iloc[1]) and raw["label"].tolist() == ["Mom", "Away"]

    def test_a_table_without_the_needed_columns_or_with_unreadable_times_is_refused(self, tmp_path):
        (tmp_path / "a.csv").write_text("tier,onset,code\ngaze,1,Mom\n", encoding="utf-8")
        with pytest.raises(EafError, match="missing start, end"):
            read_coding_file(tmp_path / "a.csv")
        (tmp_path / "b.csv").write_text("tier,start,end,code\ngaze,1,2,Mom\ngaze,soon,3,Mom\n", encoding="utf-8")
        with pytest.raises(EafError, match="row\\(s\\) 3"):
            read_coding_file(tmp_path / "b.csv")

    def test_clock_times(self):
        assert seconds("1:00:01.5") == 3601.5 and seconds("2:03") == 123.0 and seconds("abc") is None


class TestImport:
    def test_a_tier_in_the_settings_takes_its_person_name_and_codes(self, project, tmp_path):
        config, recording = project.configure(IMPORTS)
        path = write_eaf(tmp_path / "a.eaf", {"Child Gaze": (None, [(0, 1000, "Mom"), (1000, 1500, "Away"), (1500, 2000, "toy")]), "Notes": (None, [(0, 100, "x")])})
        report = import_coding(config, recording, path, "anna")
        assert report.kept == {("child", "gaze_target"): 3}
        assert report.left_out == ["Notes: not in coding.imports"]
        rows = imported(config)
        assert rows["label"].tolist() == ["parent", "elsewhere", "toy"]
        assert set(rows["annotator"]) == {"anna"} and set(rows["layer"]) == {"C"} and set(rows["subject_id"]) == {"child"}

    def test_a_tier_the_file_gives_to_someone_comes_in_under_its_own_name(self, project, tmp_path):
        config, recording = project.configure()
        path = write_eaf(tmp_path / "a.eaf", {"speech": ("child", [(0, 1000, "question")]), "other": ("Mom", [(0, 100, "x")])})
        report = import_coding(config, recording, path, "anna")
        assert report.kept == {("child", "speech"): 1}
        assert report.left_out == ["other (Mom): not in coding.imports; the file gives it to Mom, who is not one of this recording's people"]

    def test_a_person_the_recording_does_not_have_is_refused(self, project, tmp_path):
        config, recording = project.configure(IMPORTS.replace("person: child", "person: baby"))
        path = write_eaf(tmp_path / "a.eaf", {"Child Gaze": (None, [(0, 1000, "Mom")])})
        with pytest.raises(ConfigError, match="baby"):
            import_coding(config, recording, path, "anna")

    def test_importing_again_replaces_only_that_coder_s_tier(self, project, tmp_path):
        config, recording = project.configure(IMPORTS)
        first = write_eaf(tmp_path / "a.eaf", {"Child Gaze": (None, [(0, 1000, "Mom")]), "speech": ("child", [(0, 500, "question")])})
        import_coding(config, recording, first, "anna")
        import_coding(config, recording, first, "ben")
        import_coding(config, recording, write_eaf(tmp_path / "b.eaf", {"Child Gaze": (None, [(0, 2000, "Away")])}), "anna")
        rows = imported(config)
        mine = rows[rows["annotator"] == "anna"].set_index("tier")["label"].to_dict()
        assert mine == {"gaze_target": "elsewhere", "speech": "question"}
        assert len(rows[rows["annotator"] == "ben"]) == 2

    def test_machine_codes_handed_back_unchanged_are_not_the_coder_s(self, project, tmp_path):
        config, recording = project.configure()
        machine = build_annotations(
            recording_id="dyad01", tier="gaze_target", layer="gaze", annotator="gaze.target", subject_id="child",
            intervals=pd.DataFrame({"t_start": [0.0, 1.0], "t_end": [1.0, 2.0], "label": ["parent", "elsewhere"]}),
        )
        write_annotations(machine, layout_for(config).annotations_path("dyad01"))
        exported = export_eaf(machine, recording, tmp_path / "export.eaf")
        report = import_coding(config, recording, exported, "anna")
        assert report.kept == {} and "the same as gaze.target's codes" in report.left_out[0]

        changed = export_eaf(machine.assign(label=["parent", "parent"]), recording, tmp_path / "changed.eaf")
        assert import_coding(config, recording, changed, "anna").kept == {("child", "gaze_target"): 2}

    def test_gestures_read_the_imported_coding_only(self, project, tmp_path):
        from macaw.extractors.video.pose.gestures import GestureExtractor

        config, recording = project.configure("kinematics:\n  gestures:\n    tier: gesture\n    still_speed: 10\n    peak_prominence: 20\n")
        step = GestureExtractor(config)
        assert [path.name for path in step.inputs_for(recording)] == ["pose.keypoints.parquet"]
        import_coding(config, recording, write_eaf(tmp_path / "a.eaf", {"gesture": ("parent", [(0, 1000, "iconic")])}), "anna")
        assert [path.name for path in step.inputs_for(recording)] == ["pose.keypoints.parquet", "imported.parquet"]


class TestCoderUnits:
    def frame(self, rows):
        return pd.DataFrame(
            [{"recording_id": "r", "subject_id": "child", "tier": "acts", "t_start": s, "t_end": e, "label": label, "annotator": who}
             for who, s, e, label in rows]
        )

    def test_each_unit_takes_the_code_covering_most_of_it(self):
        frame = self.frame([
            ("model", 0.0, 2.0, "question"), ("model", 2.0, 3.0, "answer"), ("model", 5.0, 6.0, "answer"),
            ("anna", 0.0, 0.5, "answer"), ("anna", 0.5, 2.2, "question"), ("anna", 2.2, 3.0, "answer"),
        ])
        units = build_units(frame, units_of="model").pivot(index="unit_id", columns="annotator", values="label")
        assert units["model"].tolist() == ["question", "answer", "answer"]
        assert units["anna"].tolist() == ["question", "answer", NO_CODE]

    def test_a_coder_who_coded_nothing_for_that_person_gives_no_rating(self):
        frame = pd.concat([self.frame([("model", 0.0, 1.0, "question")]), self.frame([("anna", 0.0, 1.0, "question")]).assign(subject_id="parent")])
        units = build_units(frame, units_of="model")
        assert units["annotator"].tolist() == ["model"]

    def test_the_report_names_the_mode_and_refuses_an_unknown_coder(self):
        frame = self.frame([("model", 0.0, 1.0, "question"), ("anna", 0.0, 1.0, "question")])
        report = agreement_report(frame, units_of="model")
        assert report["unit_mode"] == "coder" and report["units_of"] == "model" and report["n_pairable"] == 1
        with pytest.raises(MacawError, match="coders are: anna, model"):
            agreement_report(frame, units_of="judge")


class TestCommand:
    def test_files_named_after_their_recording_come_in_from_a_folder(self, project, tmp_path):
        project.configure(IMPORTS)
        folder = tmp_path / "coding"
        folder.mkdir()
        write_eaf(folder / "dyad01_anna.eaf", {"Child Gaze": (None, [(0, 1000, "Mom")])})
        write_eaf(folder / "dyad010 anna.eaf", {"Child Gaze": (None, [(0, 1000, "Away")])})
        result = CliRunner().invoke(app, ["import-codes", str(folder), "--coder", "anna", "-c", str(project.path)])
        assert result.exit_code == 0, result.output
        config = resolve_config(project.path)
        assert imported(config, "dyad01")["label"].tolist() == ["parent"]
        assert imported(config, "dyad010")["label"].tolist() == ["elsewhere"]

    def test_a_file_named_after_no_recording_needs_r(self, project, tmp_path):
        project.configure(IMPORTS)
        path = write_eaf(tmp_path / "family1.eaf", {"Child Gaze": (None, [(0, 1000, "Mom")])})
        result = CliRunner().invoke(app, ["import-codes", str(path), "--coder", "anna", "-c", str(project.path)])
        assert result.exit_code == 1 and "family1.eaf" in str(result.exception)
        result = CliRunner().invoke(app, ["import-codes", str(path), "--coder", "anna", "-r", "dyad01", "-c", str(project.path)])
        assert result.exit_code == 0 and "child gaze_target 1" in result.output

    def test_two_coders_compared_into_a_folder(self, project, tmp_path):
        project.configure(IMPORTS)
        for coder, code in (("anna", "Mom"), ("ben", "Away")):
            path = write_eaf(tmp_path / f"dyad01_{coder}.eaf", {"Child Gaze": (None, [(0, 1000, code), (2000, 3000, "Mom")])})
            CliRunner().invoke(app, ["import-codes", str(path), "--coder", coder, "-c", str(project.path)])
        out = tmp_path / "agreement"
        result = CliRunner().invoke(app, ["agreement", "--units", "anna", "--out", str(out), "-r", "dyad01", "-c", str(project.path)])
        assert result.exit_code == 0, result.output
        units = pd.read_csv(out / "units.csv")
        assert units[["anna", "ben"]].values.tolist() == [["parent", "elsewhere"], ["parent", "parent"]]
        assert (out / "report.json").is_file()
        result = CliRunner().invoke(app, ["agreement", "--out", str(out / "report.json"), "-c", str(project.path)])
        assert result.exit_code == 1 and "takes a folder" in str(result.exception)

    def test_two_coders_compared_inside_a_segment(self, project, tmp_path):
        project.configure(IMPORTS)
        for coder, code in (("anna", "Mom"), ("ben", "Away")):
            for rid in ("dyad01", "dyad010"):
                path = write_eaf(tmp_path / f"{rid}_{coder}.eaf", {"Child Gaze": (None, [(0, 1000, code), (2000, 3000, "Mom")])})
                CliRunner().invoke(app, ["import-codes", str(path), "--coder", coder, "-r", rid, "-c", str(project.path)])
        CliRunner().invoke(app, ["set-segment", "-r", "dyad01", "--name", "early", "--start", "0.5", "--end", "2.5", "-c", str(project.path)])
        out = tmp_path / "agreement"
        result = CliRunner().invoke(app, ["agreement", "--segment", "early", "--units", "anna", "--out", str(out), "-c", str(project.path)])
        assert result.exit_code == 0, result.output
        assert "left out, no segment early: dyad010" in result.output and "segment=early" in result.output
        units = pd.read_csv(out / "units.csv")
        assert units[["recording_id", "t_start", "t_end", "anna", "ben"]].values.tolist() == [
            ["dyad01", 0.5, 1.0, "parent", "elsewhere"], ["dyad01", 2.0, 2.5, "parent", "parent"],
        ]
        result = CliRunner().invoke(app, ["agreement", "--segment", "late", "-c", str(project.path)])
        assert result.exit_code == 1 and "the segments are: early" in str(result.exception)
