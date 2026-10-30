from __future__ import annotations

import pandas as pd
import pytest
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.coding.describe import describe_codes
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.project.layout import layout_for
from macaw.store.annotations import write_annotations
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame


def coded(labels, *, annotator="llm", recording_id="r1", subject_id="child", tier="acts", unit_s=2.0):
    """One unit per label, back to back; None is a unit the coder abstained on."""
    return pd.DataFrame(
        {
            "recording_id": recording_id, "subject_id": subject_id, "tier": tier, "layer": "B", "annotator": annotator,
            "t_start": [index * unit_s for index in range(len(labels))],
            "t_end": [(index + 1) * unit_s for index in range(len(labels))],
            "label": [label for label in labels], "abstained": [label is None for label in labels],
        }
    )


def row(table, label, **keys):
    match = table[table["label"] == label]
    for key, value in keys.items():
        match = match[match[key] == value]
    assert len(match) == 1
    return match.iloc[0]


class TestRuns:
    def test_a_run_is_one_code_over_consecutive_units(self):
        table = describe_codes(coded(["ask", "ask", "tell", "ask", "tell", "tell", "tell"]))
        ask, tell = row(table, "ask"), row(table, "tell")
        assert (ask.n, ask.runs, ask.mean_run_units, ask.single_unit_share) == (3, 2, 1.5, 0.5)
        assert (tell.n, tell.runs, tell.mean_run_units, tell.single_unit_share) == (4, 2, 2.0, 0.5)
        assert ask.sequence_transitions == 3 and ask.sequence_units == 7

    def test_time_share_is_of_the_coded_time(self):
        table = describe_codes(coded(["ask", "tell", "tell", "tell"]))
        assert row(table, "ask").time_s == 2.0 and row(table, "ask").time_share == pytest.approx(0.25)

    def test_an_abstention_ends_a_run_and_is_no_switch(self):
        table = describe_codes(coded(["ask", None, "ask", "ask"]))
        ask = row(table, "ask")
        assert (ask.n, ask.runs, ask.single_unit_share, ask.sequence_transitions) == (3, 2, 0.5, 0)
        assert ask.time_share == pytest.approx(1.0) and ask.sequence_abstained_share == pytest.approx(0.25)

    def test_units_are_ordered_by_time_whatever_order_they_were_stored_in(self):
        table = describe_codes(coded(["ask", "ask", "tell"]).iloc[::-1])
        assert row(table, "ask").runs == 1


class TestNothingPooled:
    def test_recordings_annotators_and_people_are_kept_apart(self):
        frames = [
            coded(["ask", "ask"]), coded(["ask", "tell"], recording_id="r2"),
            coded(["tell", "tell"], annotator="ann"), coded(["ask"], subject_id="parent"),
        ]
        table = describe_codes(pd.concat(frames, ignore_index=True))
        assert row(table, "ask", recording_id="r1", annotator="llm", subject_id="child").runs == 1
        assert row(table, "ask", recording_id="r2").sequence_transitions == 1
        assert row(table, "tell", annotator="ann").mean_run_units == 2.0
        assert len(table) == 5

    def test_a_tier_without_a_person_is_kept(self):
        table = describe_codes(coded(["interruption", "interruption"], subject_id=None))
        assert table["subject_id"].isna().all() and row(table, "interruption").n == 2

    def test_nothing_coded_gives_an_empty_table(self):
        assert describe_codes(coded([]).iloc[0:0]).empty


def test_the_command_prints_the_table_and_writes_it(tmp_path):
    (tmp_path / "raw" / "003").mkdir(parents=True)
    (tmp_path / "raw" / "003" / "003_parent.wav").write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    rows = coded(["ask", "ask", "tell"], recording_id="003", subject_id="parent").assign(
        group_id="003", session_id=None, task=None, run=None, confidence=1.0, needs_review=False, execution_id="e", config_hash="h"
    )
    write_annotations(coerce_frame(rows, ANNOTATION_SCHEMA, "annotations"), layout_for(resolve_config(config)).annotations_path("003"))
    out = tmp_path / "codes.csv"
    outcome = CliRunner().invoke(app, ["describe-codes", "-c", str(config), "-r", "003", "--out", str(out)])
    assert outcome.exit_code == 0, outcome.output
    assert "ask" in outcome.output and "67%" in outcome.output
    written = pd.read_csv(out)
    assert written.set_index("label").loc["ask", "runs"] == 1 and written.set_index("label").loc["tell", "single_unit_share"] == 1.0
