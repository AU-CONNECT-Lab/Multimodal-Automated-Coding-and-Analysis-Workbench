from __future__ import annotations

import sys

import pytest

from macaw.execution.backends.base import JobSpec
from macaw.execution.backends.local import LocalBackend
from macaw.execution.job_db import JobDB, JobRecord

EXECUTION = "exec-1"


def _command(marker, *, code=0):
    body = (
        "import pathlib, time, sys; "
        f"pathlib.Path(r'{marker}').write_text(repr(time.time())); "
        f"sys.exit({code})"
    )
    return [sys.executable, "-c", body]


@pytest.fixture
def database(tmp_path):
    db = JobDB(tmp_path / "jobs")
    db.initialise()
    return db


@pytest.fixture
def make_spec(tmp_path, database):
    def build(task_id, *, depends_on=(), code=0):
        marker = tmp_path / "markers" / f"{task_id.replace(':', '_')}.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        recording_id, extractor = task_id.split(":", 1)
        database.upsert(
            JobRecord(
                task_id=task_id,
                recording_id=recording_id,
                extractor=extractor,
                execution_id=EXECUTION,
                status="pending",
            )
        )
        return JobSpec(
            task_id=task_id,
            recording_id=recording_id,
            extractor=extractor,
            command=_command(marker, code=code),
            resources=None,
            log_path=tmp_path / "logs" / f"{task_id.replace(':', '_')}.log",
            depends_on=list(depends_on),
        ), marker

    return build


def _run(database, specs, *, n_workers=4):
    backend = LocalBackend(n_workers=n_workers, job_db=database)
    backend.wait(backend.submit(specs))
    return backend


def _finished_at(marker):
    return float(marker.read_text(encoding="utf-8"))


class TestRunning:
    def test_a_task_runs_and_is_marked_done(self, database, make_spec):
        spec, marker = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert marker.is_file()
        assert database.get("rec1:prep.media").status == "done"

    def test_the_exit_code_is_recorded(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media", code=3)
        _run(database, [spec])
        record = database.get("rec1:prep.media")
        assert (record.status, record.exit_code) == ("failed", 3)

    def test_a_log_file_records_the_command(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert "-c" in spec.log_path.read_text(encoding="utf-8")

    def test_a_duration_is_recorded(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert database.get("rec1:prep.media").duration_s >= 0.0

    def test_independent_tasks_all_run(self, database, make_spec):
        specs = [make_spec(f"rec{index}:prep.media") for index in range(4)]
        _run(database, [spec for spec, _ in specs])
        assert all(marker.is_file() for _, marker in specs)


class TestDependencies:
    def test_a_dependency_runs_first(self, database, make_spec):
        first, first_marker = make_spec("rec1:prep.media")
        second, second_marker = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        _run(database, [second, first])
        assert _finished_at(first_marker) < _finished_at(second_marker)

    def test_a_chain_does_not_deadlock_with_one_worker(self, database, make_spec):
        """Each worker waiting on its own predecessors is what would deadlock here; the
        dispatch loop exists to keep that from happening."""
        a, a_marker = make_spec("rec1:a")
        b, b_marker = make_spec("rec1:b", depends_on=["rec1:a"])
        c, c_marker = make_spec("rec1:c", depends_on=["rec1:b"])
        _run(database, [c, b, a], n_workers=1)
        assert _finished_at(a_marker) < _finished_at(b_marker) < _finished_at(c_marker)

    def test_a_chain_finishes_with_fewer_workers_than_tasks(self, database, make_spec):
        specs = [make_spec("rec1:a")]
        specs.append(make_spec("rec1:b", depends_on=["rec1:a"]))
        specs.append(make_spec("rec1:c", depends_on=["rec1:a"]))
        _run(database, [spec for spec, _ in specs], n_workers=2)
        assert all(database.get(spec.task_id).status == "done" for spec, _ in specs)


class TestFailure:
    def test_a_downstream_task_is_cancelled_not_left_hanging(self, database, make_spec):
        """Without this the dispatch loop would spin forever waiting on a predecessor that
        is never going to arrive."""
        failing, _ = make_spec("rec1:prep.media", code=1)
        downstream, downstream_marker = make_spec(
            "rec1:prosody.frame", depends_on=["rec1:prep.media"]
        )
        _run(database, [failing, downstream])
        assert database.get("rec1:prep.media").status == "failed"
        assert database.get("rec1:prosody.frame").status == "cancelled"
        assert not downstream_marker.is_file()

    def test_a_cancelled_task_says_why(self, database, make_spec):
        failing, _ = make_spec("rec1:prep.media", code=1)
        downstream, _ = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        _run(database, [failing, downstream])
        assert "upstream" in database.get("rec1:prosody.frame").message

    def test_another_recording_is_not_dragged_down(self, database, make_spec):
        """Recordings are independent, so one failure must not stop the other 30."""
        failing, _ = make_spec("rec1:prep.media", code=1)
        blocked, _ = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        healthy, healthy_marker = make_spec("rec2:prep.media")
        _run(database, [failing, blocked, healthy])
        assert healthy_marker.is_file()
        assert database.get("rec2:prep.media").status == "done"

    def test_the_failure_message_points_at_the_log(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media", code=1)
        _run(database, [spec])
        assert str(spec.log_path) in database.get("rec1:prep.media").message


class TestStatus:
    def test_status_reports_each_task(self, database, make_spec):
        ok, _ = make_spec("rec1:a")
        bad, _ = make_spec("rec2:a", code=1)
        backend = _run(database, [ok, bad])
        assert backend.status(["rec1:a", "rec2:a"]) == {"rec1:a": "done", "rec2:a": "failed"}

    def test_an_unknown_task_reads_as_pending(self, database, make_spec):
        spec, _ = make_spec("rec1:a")
        backend = _run(database, [spec])
        assert backend.status(["rec9:a"]) == {"rec9:a": "pending"}
