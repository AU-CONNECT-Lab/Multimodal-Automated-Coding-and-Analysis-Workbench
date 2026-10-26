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
from macaw.extractors.prep import AlignReviewGate, IdentityConfirmGate
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


@pytest.fixture
def project(tmp_path):
    """Two recordings, each with a mic per person and a room camera without audio."""
    raw = tmp_path / "raw"
    for name in ("003", "004"):
        (raw / name).mkdir(parents=True)
        for file in (f"{name}_parent.wav", f"{name}_child.wav", f"{name}_room.mp4"):
            (raw / name / file).write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=tmp_path / "out")
    return SimpleNamespace(config=config, raw=raw, resolved=lambda: resolve_config(config))


def configure(project, section, **values):
    data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    data.setdefault(section, {}).update(values)
    project.config.write_text(yaml.safe_dump(data), encoding="utf-8")


def result(target, offset_s=0.0, *, method=GCC_PHAT, confidence=0.9, **extra):
    return AlignResult(
        reference="003_parent.wav", target=target, offset_s=offset_s, confidence=confidence, method=method, **extra
    )


def table(project, rows, recording="003"):
    write_offset_table(rows, layout_for(project.resolved()).offset_table_path(recording))


def align_pending(project, recording="003"):
    config = project.resolved()
    return AlignReviewGate(config).pending(build_recording(recording, config))


GOOD = [
    result("003_parent.wav", confidence=1.0),
    result("003_child.wav", 0.003, confidence=0.44, agreement=0.95),
    result("003_room.mp4", method=UNALIGNED, confidence=None, note="no audio stream", accepted="ann 2026-10-25T10:00:00"),
]


class TestAlignReview:
    def test_without_a_table_it_says_how_to_make_one_or_skip_it(self, project):
        reason = align_pending(project)
        assert "macaw align -r 003" in reason and "align.method: none" in reason

    def test_a_clean_table_clears_it(self, project):
        table(project, GOOD)
        assert align_pending(project) is None

    def test_a_blunt_peak_the_segments_do_not_back_holds_it(self, project):
        table(project, [*GOOD[:1], result("003_child.wav", confidence=0.3, agreement=0.1), GOOD[2]])
        reason = align_pending(project)
        assert "003_child.wav (low confidence)" in reason
        assert "macaw accept-offset -r 003" in reason

    def test_a_camera_without_audio_holds_it_until_kept(self, project):
        table(project, [*GOOD[:2], result("003_room.mp4", method=UNALIGNED, confidence=None, note="no audio stream")])
        assert "003_room.mp4 (not aligned: no audio stream)" in align_pending(project)

    def test_drift_that_moves_the_offset_holds_it(self, project):
        drifting = result("003_child.wav", drift_ppm=120.0, drift_r2=1.0, overlap_s=1200.0)
        table(project, [GOOD[0], drifting, GOOD[2]])
        assert "003_child.wav (drift)" in align_pending(project)

    def test_a_hand_set_offset_needs_nothing_more(self, project):
        table(project, [*GOOD[:2], result("003_room.mp4", 1.5, method=MANUAL, confidence=None)])
        assert align_pending(project) is None

    def test_a_file_added_after_aligning_asks_for_a_rerun(self, project):
        table(project, GOOD[:2])
        assert "003_room.mp4 came after aligning" in align_pending(project)

    @pytest.mark.parametrize("section, values", [("align", {"review": "skip"}), ("align", {"method": "none"})])
    def test_skipping_or_files_recorded_in_sync_clear_it(self, project, section, values):
        configure(project, section, **values)
        assert align_pending(project) is None

    def test_a_single_file_has_nothing_to_line_up(self, project):
        for name in ("003_child.wav", "003_room.mp4"):
            (project.raw / "003" / name).unlink()
        assert align_pending(project) is None


class TestKeepAccepted:
    def test_the_same_result_stays_accepted(self):
        old = [result("003_child.wav", 0.003, accepted="ann")]
        fresh, dropped = keep_accepted([result("003_child.wav", 0.010)], old)
        assert fresh[0].accepted == "ann" and dropped == []

    def test_a_moved_offset_has_to_be_looked_at_again(self):
        fresh, dropped = keep_accepted([result("003_child.wav", 0.5)], [result("003_child.wav", 0.003, accepted="ann")])
        assert fresh[0].accepted is None and dropped == ["003_child.wav"]

    def test_a_camera_without_audio_stays_accepted(self):
        old = [result("003_room.mp4", method=UNALIGNED, confidence=None, accepted="ann")]
        fresh, _ = keep_accepted([result("003_room.mp4", method=UNALIGNED, confidence=None)], old)
        assert fresh[0].accepted == "ann"


def identity_gate(project, recording="003"):
    config = project.resolved()
    gate = IdentityConfirmGate(config)
    return gate, build_recording(recording, config)


class TestIdentityConfirm:
    def test_unconfirmed_it_points_at_the_command(self, project):
        gate, recording = identity_gate(project)
        assert "macaw identify" in gate.pending(recording)

    def test_one_confirmation_clears_every_recording(self, project):
        assert CliRunner().invoke(app, ["identify", "-c", str(project.config), "--yes"]).exit_code == 0
        for recording in ("003", "004"):
            gate, built = identity_gate(project, recording)
            assert gate.pending(built) is None

    def test_a_changed_naming_rule_needs_a_new_confirmation(self, project):
        CliRunner().invoke(app, ["identify", "-c", str(project.config), "--yes"])
        configure(project, "project", media_pattern="{subject_id}_*.*")
        gate, recording = identity_gate(project)
        assert "again" in gate.pending(recording)

    def test_skipping_clears_it_and_says_so_every_run(self, project):
        configure(project, "identity", confirm="skip")
        gate, recording = identity_gate(project)
        assert gate.pending(recording) is None
        assert [note.check for note in gate.notes(recording)] == ["identity_unconfirmed"]


class TestIdentifyCommand:
    def test_it_shows_who_owns_what_and_records_names_only(self, project):
        output = CliRunner().invoke(app, ["identify", "-c", str(project.config), "--yes", "--by", "ann"]).output
        assert "parent: 003_parent.wav" in output and "shared: 003_room.mp4" in output
        path = layout_for(project.resolved()).identity_path
        text = path.read_text(encoding="utf-8")
        assert '"confirmed_by": "ann"' in text
        assert str(project.raw) not in text and "\\\\" not in text

    def test_saying_no_records_nothing(self, project):
        outcome = CliRunner().invoke(app, ["identify", "-c", str(project.config)], input="n\n")
        assert outcome.exit_code == 1
        assert not layout_for(project.resolved()).identity_path.is_file()


class TestAcceptOffset:
    def test_every_flagged_file_is_kept_and_named(self, project):
        table(project, [*GOOD[:2], result("003_room.mp4", method=UNALIGNED, confidence=None)])
        outcome = CliRunner().invoke(app, ["accept-offset", "-c", str(project.config), "-r", "003", "--by", "ann"])
        assert "kept 003_room.mp4" in outcome.output
        kept = {r.target: r for r in read_offset_table(layout_for(project.resolved()).offset_table_path("003"))}
        assert kept["003_room.mp4"].accepted.startswith("ann ")
        assert kept["003_child.wav"].accepted is None
        assert align_pending(project) is None

    def test_a_file_in_no_table_is_an_error(self, project):
        table(project, GOOD)
        outcome = CliRunner().invoke(app, ["accept-offset", "-c", str(project.config), "--media", "nope.wav"])
        assert outcome.exit_code != 0


class TestWiring:
    def test_cutting_waits_for_the_offsets_and_turns_wait_for_identity(self):
        assert AlignReviewGate.name in get_extractor("prep.media").requires
        assert IdentityConfirmGate.name in get_extractor("turns.structure").requires
        assert get_extractor("align.review").phase == get_extractor("identity.confirm").phase == Phase.INTERACTIVE


@pytest.fixture
def fake_line(monkeypatch):
    """prep -> gate -> after, with a gate that is cleared for every recording but rec2."""
    import macaw.extractors as extractors

    class FakeGate:
        name, requires, phase, config_sections = "gate", ("prep",), Phase.INTERACTIVE, ()

        def __init__(self, config):
            pass

        def pending(self, recording):
            return "look at it" if recording.recording_id == "rec2" else None

        def notes(self, recording):
            return []

    def plain(name, requires, phase):
        return type(name, (), {"name": name, "requires": requires, "phase": phase, "config_sections": ()})

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
    return SimpleNamespace(project=SimpleNamespace(excluded_recordings=[]), pipeline=PipelineSection())


class TestRunnerGates:
    def test_a_waiting_gate_holds_back_only_its_own_recording(self, fake_line):
        assert plan(fake_line, ["rec1", "rec2"], ["after"]) == ["rec1:prep", "rec2:prep", "rec1:after"]

    def test_the_wait_says_which_recording_and_why(self, fake_line):
        waiting, _, _ = check_gates(fake_line, build_dag(fake_line, ["after"]), ["rec1", "rec2"])
        assert [(w.recording_id, w.gate, w.message) for w in waiting] == [("rec2", "gate", "look at it")]

    def test_work_behind_a_cleared_gate_still_waits_for_what_the_gate_needed(self, fake_line):
        dag = build_dag(fake_line, ["after"])
        assert runner_module._job_dependencies(dag, "after", {"gate"}) == {"prep"}

    def test_a_recording_that_cannot_be_read_waits_with_the_reason(self, fake_line, monkeypatch):
        def broken(recording_id, config):
            raise RecordingNotFoundError(f"{recording_id}: no directory")

        monkeypatch.setattr(runner_module, "build_recording", broken)
        waiting, blocked, _ = check_gates(fake_line, build_dag(fake_line, ["after"]), ["rec1"])
        assert waiting[0].message == "rec1: no directory"
        assert blocked == {"rec1:gate", "rec1:after"}


class TestRunCommand:
    def test_a_dry_run_names_what_waits_and_leaves_it_out(self, project):
        outcome = CliRunner().invoke(app, ["run", "-c", str(project.config), "-e", "prep.media", "--dry-run"])
        assert outcome.exit_code == 0
        assert "003:prep.media" not in outcome.output
        assert "003  align.review: no offset table yet" in outcome.output
