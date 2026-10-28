from __future__ import annotations

import json
import os
import threading

import pytest

from macaw.exceptions import MacawError
from macaw.execution.job_db import STATUSES, JobDB, JobRecord


@pytest.fixture
def database(tmp_path):
    db = JobDB(tmp_path / "jobs")
    db.initialise()
    return db


@pytest.fixture
def make_record():
    def build(task_id="rec1:prep.media", *, execution_id="exec-1", status="pending", **rest):
        recording_id, extractor = task_id.split(":", 1)
        return JobRecord(
            task_id=task_id,
            recording_id=recording_id,
            extractor=extractor,
            execution_id=execution_id,
            status=status,
            **rest,
        )

    return build


class TestOneFilePerTask:
    def test_a_record_comes_back_unchanged(self, database, make_record):
        database.upsert(make_record(exit_code=0, message="fine"))
        found = database.get("rec1:prep.media")
        assert (found.status, found.exit_code, found.message) == ("pending", 0, "fine")

    def test_the_task_id_becomes_a_safe_file_name(self, database, make_record):
        """Task ids carry a colon and dots, neither of which every filesystem accepts."""
        database.upsert(make_record("sub-003_ses-12mo:prosody.frame"))
        written = list((database.path / "exec-1").glob("*.json"))
        assert len(written) == 1
        assert ":" not in written[0].name

    def test_two_executions_do_not_share_a_file(self, database, make_record):
        """A rerun must not rewrite history; that is what the per-execution directory is for."""
        database.upsert(make_record(execution_id="exec-1", status="failed"))
        database.upsert(make_record(execution_id="exec-2", status="done"))
        assert sorted(database.executions()) == ["exec-1", "exec-2"]
        assert database.query(execution_id="exec-1")[0].status == "failed"
        assert database.query(execution_id="exec-2")[0].status == "done"

    def test_get_returns_the_most_recent_execution(self, database, make_record):
        database.upsert(make_record(execution_id="exec-1", status="failed"))
        database.upsert(make_record(execution_id="exec-2", status="done"))
        # forced rather than slept for: two writes in the same millisecond are a real tie on
        # some filesystems, and the point here is which one get() picks
        older = database.record_path("exec-1", "rec1:prep.media")
        os.utime(older, (0, 0))
        assert database.get("rec1:prep.media").status == "done"

    def test_no_temporary_file_is_left_behind(self, database, make_record):
        database.upsert(make_record())
        assert list((database.path / "exec-1").glob("*.tmp*")) == []

    def test_a_missing_task_is_none_not_an_error(self, database):
        assert database.get("nobody:nothing") is None


class TestStatus:
    def test_set_status_updates_in_place(self, database, make_record):
        database.upsert(make_record())
        database.set_status("rec1:prep.media", "done", exit_code=0)
        record = database.get("rec1:prep.media")
        assert (record.status, record.exit_code) == ("done", 0)

    @pytest.mark.parametrize("status", STATUSES)
    def test_every_declared_status_is_accepted(self, database, make_record, status):
        database.upsert(make_record(status=status))
        assert database.get("rec1:prep.media").status == status

    def test_an_unknown_status_is_refused_on_upsert(self, database, make_record):
        with pytest.raises(MacawError, match="Unknown job status"):
            database.upsert(make_record(status="halfway"))

    def test_an_unknown_status_is_refused_on_update(self, database, make_record):
        database.upsert(make_record())
        with pytest.raises(MacawError, match="Unknown job status"):
            database.set_status("rec1:prep.media", "halfway")

    def test_an_unknown_field_is_refused(self, database, make_record):
        """A typo in a field name would otherwise be written and silently never read."""
        database.upsert(make_record())
        with pytest.raises(MacawError, match="Unknown job field"):
            database.set_status("rec1:prep.media", "done", exti_code=0)

    def test_updating_a_task_that_was_never_started_is_an_error(self, database):
        with pytest.raises(MacawError, match="No job record"):
            database.set_status("rec1:prep.media", "done")


class TestQuerying:
    @pytest.fixture
    def populated(self, database, make_record):
        database.upsert(make_record("rec1:prep.media", status="done"))
        database.upsert(make_record("rec2:prep.media", status="failed"))
        database.upsert(make_record("rec3:prep.media", status="pending"))
        database.upsert(make_record("rec4:prep.media", execution_id="exec-2", status="done"))
        return database

    def test_query_filters_by_execution(self, populated):
        assert len(populated.query(execution_id="exec-1")) == 3

    def test_query_filters_by_status(self, populated):
        assert [r.task_id for r in populated.query(status="failed")] == ["rec2:prep.media"]

    def test_query_without_a_filter_spans_executions(self, populated):
        assert len(populated.query()) == 4

    def test_pending_and_failed_are_shorthands(self, populated):
        assert [r.task_id for r in populated.pending("exec-1")] == ["rec3:prep.media"]
        assert [r.task_id for r in populated.failed("exec-1")] == ["rec2:prep.media"]

    def test_summary_reports_every_status_including_the_empty_ones(self, populated):
        summary = populated.summary("exec-1")
        assert summary["done"] == 1 and summary["failed"] == 1 and summary["pending"] == 1
        assert summary["cached"] == 0
        assert set(summary) == set(STATUSES)

    def test_results_are_ordered_not_left_to_the_filesystem(self, populated):
        ids = [record.task_id for record in populated.query(execution_id="exec-1")]
        assert ids == sorted(ids)


class TestPartialWrites:
    def test_an_unreadable_file_is_treated_as_absent(self, database, make_record):
        """A job killed mid-write must not bring down whatever is reporting progress."""
        database.upsert(make_record("rec1:prep.media", status="done"))
        (database.path / "exec-1" / "rec2_prep.media.json").write_text("{ truncated", encoding="utf-8")
        assert len(database.query(execution_id="exec-1")) == 1

    def test_a_record_missing_fields_still_loads(self, database):
        directory = database.path / "exec-1"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rec1_prep.media.json").write_text(
            json.dumps({"task_id": "rec1:prep.media", "status": "done"}), encoding="utf-8"
        )
        record = database.query(execution_id="exec-1")[0]
        assert record.status == "done"
        assert record.exit_code is None


class TestConcurrentWriters:
    def test_many_writers_lose_nothing(self, database, make_record):
        """One task is only ever run in one place, so each is the sole writer of its own file.
        This is the property that lets the whole thing work without a lock."""
        n = 20

        def write(index):
            database.upsert(make_record(f"rec{index}:prep.media", status="done"))

        threads = [threading.Thread(target=write, args=(index,)) for index in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        records = database.query(execution_id="exec-1")
        assert len(records) == n
        assert all(record.status == "done" for record in records)
