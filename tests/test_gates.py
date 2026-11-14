from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import PipelineSection
from macaw.exceptions import RecordingNotFoundError
from macaw.execution import runner as runner_module
from macaw.execution.dag import build_dag
from macaw.execution.runner import check_gates, plan
from macaw.extractors import get_extractor
from macaw.extractors.base import Phase
from macaw.extractors.prep import AlignReviewGate
from macaw.media.align import (
    GCC_PHAT,
    MANUAL,
    UNALIGNED,
    AlignResult,
    keep_accepted,
    read_offset_table,
    write_offset_table,
)
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from setup_helpers import camera, mics, put, write_setup


@pytest.fixture
def project(tmp_path):
    """Two recordings, each with a mic per person and a room camera without audio."""
    raw = tmp_path / "raw"
    for name in ("003", "004"):
        put(raw, name, "parent_mic", "a.wav")
        put(raw, name, "child_mic", "a.wav")
        put(raw, name, "room", "a.mp4")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    write_setup(config, cameras={"room": camera()}, microphones=mics())
    return SimpleNamespace(config=config, raw=raw, resolved=lambda: resolve_config(config))


def configure(project, section, **values):
    data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    data.setdefault(section, {}).update(values)
    project.config.write_text(yaml.safe_dump(data), encoding="utf-8")


def result(target, offset_s=0.0, *, method=GCC_PHAT, confidence=0.9, **extra):
    return AlignResult(
        reference="parent_mic", target=target, offset_s=offset_s, confidence=confidence, method=method, **extra
    )


def table(project, rows, recording="003"):
    write_offset_table(rows, layout_for(project.resolved()).offset_table_path(recording))


def align_pending(project, recording="003"):
    config = project.resolved()
    return AlignReviewGate(config).pending(build_recording(recording, config))


GOOD = [
    result("parent_mic", confidence=1.0),
    result("child_mic", 0.003, confidence=0.44, agreement=0.95),
    result("room", method=UNALIGNED, confidence=None, note="no audio stream", accepted="ann 2026-10-25T10:00:00"),
]


class TestAlignReview:
    def test_without_a_table_it_says_how_to_make_one_or_skip_it(self, project):
        reason = align_pending(project)
        assert "macaw align -r 003" in reason and "align.method: none" in reason

    def test_a_clean_table_clears_it(self, project):
        table(project, GOOD)
        assert align_pending(project) is None

    def test_a_blunt_peak_the_segments_do_not_back_holds_it(self, project):
        table(project, [*GOOD[:1], result("child_mic", confidence=0.3, agreement=0.1), GOOD[2]])
        reason = align_pending(project)
        assert "child_mic (low confidence)" in reason
        assert "macaw accept-offset -r 003" in reason

    def test_a_camera_without_audio_holds_it_until_kept(self, project):
        table(project, [*GOOD[:2], result("room", method=UNALIGNED, confidence=None, note="no audio stream")])
        assert "room (not aligned: no audio stream)" in align_pending(project)

    def test_drift_that_moves_the_offset_holds_it(self, project):
        drifting = result("child_mic", drift_ppm=120.0, drift_r2=1.0, overlap_s=1200.0)
        table(project, [GOOD[0], drifting, GOOD[2]])
        assert "child_mic (drift)" in align_pending(project)

    def test_a_hand_set_offset_needs_nothing_more(self, project):
        table(project, [*GOOD[:2], result("room", 1.5, method=MANUAL, confidence=None)])
        assert align_pending(project) is None

    def test_a_file_added_after_aligning_asks_for_a_rerun(self, project):
        table(project, GOOD[:2])
        assert "room came after aligning" in align_pending(project)

    @pytest.mark.parametrize("section, values", [("align", {"review": "skip"}), ("align", {"method": "none"})])
    def test_skipping_or_files_recorded_in_sync_clear_it(self, project, section, values):
        configure(project, section, **values)
        assert align_pending(project) is None

    def test_always_holds_a_clean_table_until_a_person_keeps_it(self, project):
        configure(project, "align", review="always")
        table(project, GOOD)
        reason = align_pending(project)
        assert "child_mic (not looked at yet)" in reason
        assert "parent_mic" not in reason and "room" not in reason
        outcome = CliRunner().invoke(app, ["accept-offset", "-c", str(project.config), "-r", "003", "--by", "ann"])
        assert outcome.exit_code == 0, outcome.output
        assert align_pending(project) is None

    def test_a_single_file_has_nothing_to_line_up(self, project):
        for name in ("child_mic", "room"):
            for path in (project.raw / "003" / name).iterdir():
                path.unlink()
        assert align_pending(project) is None


class TestKeepAccepted:
    def test_the_same_result_stays_accepted(self):
        old = [result("child_mic", 0.003, accepted="ann")]
        fresh, dropped = keep_accepted([result("child_mic", 0.010)], old)
        assert fresh[0].accepted == "ann" and dropped == []

    def test_a_moved_offset_has_to_be_looked_at_again(self):
        fresh, dropped = keep_accepted([result("child_mic", 0.5)], [result("child_mic", 0.003, accepted="ann")])
        assert fresh[0].accepted is None and dropped == ["child_mic"]

    def test_a_camera_without_audio_stays_accepted(self):
        old = [result("room", method=UNALIGNED, confidence=None, accepted="ann")]
        fresh, _ = keep_accepted([result("room", method=UNALIGNED, confidence=None)], old)
        assert fresh[0].accepted == "ann"


class TestAcceptOffset:
    def test_every_flagged_file_is_kept_and_named(self, project):
        table(project, [*GOOD[:2], result("room", method=UNALIGNED, confidence=None)])
        outcome = CliRunner().invoke(app, ["accept-offset", "-c", str(project.config), "-r", "003", "--by", "ann"])
        assert "kept room" in outcome.output
        kept = {r.target: r for r in read_offset_table(layout_for(project.resolved()).offset_table_path("003"))}
        assert kept["room"].accepted.startswith("ann ")
        assert kept["child_mic"].accepted is None
        assert align_pending(project) is None

    def test_a_file_in_no_table_is_an_error(self, project):
        table(project, GOOD)
        outcome = CliRunner().invoke(app, ["accept-offset", "-c", str(project.config), "--device", "nope"])
        assert outcome.exit_code != 0


class TestWiring:
    def test_cutting_waits_for_the_offsets(self):
        assert AlignReviewGate.name in get_extractor("prep.media").requires
        assert get_extractor("align.review").phase == Phase.INTERACTIVE


@pytest.fixture
def fake_line(monkeypatch, tmp_path):
    """prep -> gate -> after, with a gate that is cleared for every recording but rec2."""
    import macaw.extractors as extractors

    class FakeGate:
        name, requires, phase, config_sections = "gate", ("prep",), Phase.INTERACTIVE, ()
        runs_after = ()

        def __init__(self, config):
            pass

        @classmethod
        def dependencies(cls, config):
            return cls.requires

        @classmethod
        def ordered_after(cls, config):
            return cls.runs_after

        @classmethod
        def brings_along(cls, config):
            return ()

        def pending(self, recording):
            return "look at it" if recording.recording_id == "rec2" else None

        def notes(self, recording):
            return []

        def clear_by_itself(self, recording):
            return None

    def plain(name, requires, phase):
        return type(name, (), {"name": name, "requires": requires, "phase": phase, "config_sections": (), "dependencies": classmethod(lambda cls, config: cls.requires), "runs_after": (), "wanted_by_default": classmethod(lambda cls, config: True),
                                 "ordered_after": classmethod(lambda cls, config: cls.runs_after), "brings_along": classmethod(lambda cls, config: ())})

    registry = {
        "prep": plain("prep", (), Phase.PREP),
        "gate": FakeGate,
        "after": plain("after", ("gate",), Phase.EXTRACT),
    }
    monkeypatch.setattr(extractors, "_REGISTRY", registry)
    monkeypatch.setattr(extractors, "_builtins_loaded", True)
    monkeypatch.setattr(
        runner_module, "build_recording", lambda recording_id, config: SimpleNamespace(recording_id=recording_id)
    )
    # no recording has a settings file of its own, so the plan is the project's for every one
    return SimpleNamespace(project=SimpleNamespace(excluded_recordings=[], recording_config_dir=tmp_path / "recordings"), pipeline=PipelineSection())


class TestRunnerGates:
    def test_a_waiting_gate_holds_back_only_its_own_recording(self, fake_line):
        assert plan(fake_line, ["rec1", "rec2"], ["after"]) == ["rec1:prep", "rec2:prep", "rec1:after"]

    def test_the_plan_leaves_out_what_is_already_finished(self, fake_line):
        assert plan(fake_line, ["rec1", "rec2"], ["after"], {"rec1:prep", "rec2:prep"}) == ["rec1:after"]

    def test_the_wait_says_which_recording_and_why(self, fake_line):
        waiting, _, _ = check_gates(fake_line, build_dag(fake_line, ["after"]), ["rec1", "rec2"])
        assert [(w.recording_id, w.gate, w.message) for w in waiting] == [("rec2", "gate", "look at it")]

    def test_work_behind_a_cleared_gate_still_waits_for_what_the_gate_needed(self, fake_line):
        dag = build_dag(fake_line, ["after"])
        assert runner_module._job_dependencies(dag, "after", {"gate"}) == {"prep"}

    def test_a_gate_whose_input_is_still_to_come_is_deferred_not_put_to_a_person(self, fake_line):
        waiting, blocked, _ = check_gates(fake_line, build_dag(fake_line, ["after"]), ["rec1"], finished=set())
        assert [(w.gate, w.deferred) for w in waiting] == [("gate", True)]
        assert "prep" in waiting[0].message
        assert "rec1:after" in blocked and "rec1:prep" not in blocked

    def test_once_its_input_exists_the_gate_decides(self, fake_line):
        dag = build_dag(fake_line, ["after"])
        waiting, _, _ = check_gates(fake_line, dag, ["rec1", "rec2"], finished={"rec1:prep", "rec2:prep"})
        assert [(w.recording_id, w.deferred) for w in waiting] == [("rec2", False)]

    def test_a_recording_that_cannot_be_read_waits_with_the_reason(self, fake_line, monkeypatch):
        def broken(recording_id, config):
            raise RecordingNotFoundError(f"{recording_id}: no directory")

        monkeypatch.setattr(runner_module, "build_recording", broken)
        waiting, blocked, _ = check_gates(fake_line, build_dag(fake_line, ["after"]), ["rec1"])
        assert waiting[0].message == "rec1: no directory"
        assert blocked == {"rec1:gate", "rec1:after"}


class TestGateNotes:
    def test_a_gate_that_now_waits_loses_what_it_said_when_it_last_cleared(self, tmp_path):
        from macaw.execution.runner import GateWait, record_gate_notes
        from macaw.extractors.base import QCWarning
        from macaw.project.layout import make_layout

        layout = make_layout(tmp_path / "out", tmp_path / "work")
        skipped = QCWarning(check="gate", recording_id="rec1", subject_id=None, severity="info", message="kept without a look")
        record_gate_notes(layout, {("rec1", "gate"): [skipped]}, [])
        assert layout.warnings_path("rec1", "gate").is_file()
        record_gate_notes(layout, {}, [GateWait(recording_id="rec1", gate="gate", message="look at it")])
        assert not layout.warnings_path("rec1", "gate").exists()


class TestRunCommand:
    def test_a_dry_run_names_what_waits_and_leaves_it_out(self, project):
        outcome = CliRunner().invoke(app, ["run", "-c", str(project.config), "-e", "prep.media", "--dry-run"])
        assert outcome.exit_code == 0
        assert "003:prep.media" not in outcome.output
        assert "003  align.review: no offset table yet" in outcome.output
