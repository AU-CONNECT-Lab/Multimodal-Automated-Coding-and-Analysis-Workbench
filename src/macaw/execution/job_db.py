"""SQLite job state. Adapted from neuroimage-pipe job_db, keyed by task_id rather than
subject/recording columns; macaw's unit of work is recording x extractor."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from macaw.exceptions import MacawError

STATUSES = ("pending", "running", "done", "failed", "cached", "cancelled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    task_id        TEXT NOT NULL,
    recording_id   TEXT NOT NULL,
    extractor      TEXT NOT NULL,
    execution_id   TEXT NOT NULL,
    status         TEXT NOT NULL,
    backend_job_id TEXT,
    exit_code      INTEGER,
    started_at     TEXT,
    finished_at    TEXT,
    duration_s     REAL,
    log_path       TEXT,
    node_name      TEXT,
    config_hash    TEXT,
    message        TEXT,
    PRIMARY KEY (execution_id, task_id)
);
CREATE INDEX IF NOT EXISTS jobs_by_execution ON jobs (execution_id, status);
CREATE INDEX IF NOT EXISTS jobs_by_task ON jobs (task_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class JobRecord:
    task_id: str
    recording_id: str
    extractor: str
    execution_id: str
    status: str                    # pending | running | done | failed | cached | cancelled
    backend_job_id: str | None = None
    exit_code: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    log_path: str | None = None
    node_name: str | None = None
    config_hash: str | None = None
    message: str | None = None


_COLUMNS = tuple(f.name for f in fields(JobRecord))


class JobDB:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            # several tasks report at once even on one machine; make waiting the default
            # rather than an immediate "database is locked"
            connection.execute("PRAGMA busy_timeout = 30000")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialise(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    def upsert(self, record: JobRecord) -> None:
        if record.status not in STATUSES:
            raise MacawError(f"Unknown job status '{record.status}'")
        data = asdict(record)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO jobs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                [data[name] for name in _COLUMNS],
            )

    def set_status(self, task_id: str, status: str, **fields) -> None:
        """Updates the most recent execution of that task, so a rerun never rewrites history."""
        if status not in STATUSES:
            raise MacawError(f"Unknown job status '{status}'")
        unknown = set(fields) - set(_COLUMNS)
        if unknown:
            raise MacawError(f"Unknown job column(s): {', '.join(sorted(unknown))}")

        assignments = ", ".join(f"{name} = ?" for name in ("status", *fields))
        values = [status, *fields.values(), task_id, task_id]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE task_id = ? AND rowid = "
                "(SELECT MAX(rowid) FROM jobs WHERE task_id = ?)",
                values,
            )

    def _row(self, row: sqlite3.Row) -> JobRecord:
        return JobRecord(**{name: row[name] for name in _COLUMNS})

    def get(self, task_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE task_id = ? ORDER BY rowid DESC LIMIT 1", (task_id,)
            ).fetchone()
        return self._row(row) if row else None

    def query(
        self, *, execution_id: str | None = None, status: str | None = None
    ) -> list[JobRecord]:
        clauses, values = [], []
        if execution_id is not None:
            clauses.append("execution_id = ?")
            values.append(execution_id)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY task_id", values
            ).fetchall()
        return [self._row(row) for row in rows]

    def pending(self, execution_id: str) -> list[JobRecord]:
        return self.query(execution_id=execution_id, status="pending")

    def failed(self, execution_id: str) -> list[JobRecord]:
        return self.query(execution_id=execution_id, status="failed")

    def summary(self, execution_id: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS n FROM jobs WHERE execution_id = ? GROUP BY status",
                (execution_id,),
            ).fetchall()
        counts = {status: 0 for status in STATUSES}
        counts.update({row["status"]: row["n"] for row in rows})
        return counts

    def array_indices_for_rerun(self, execution_id: str) -> list[int]:
        """Turns the failed set into a --array=3,7,12 argument."""
        from macaw.execution.array_map import read_map

        map_path = self.path.parent / "array_map.tsv"
        if not map_path.is_file():
            raise MacawError(
                f"No array_map.tsv beside {self.path}. Reruns need the mapping the original "
                "submission used, because array indices must stay stable across arrays."
            )
        mapping = read_map(map_path)
        indices = {
            mapping[record.recording_id]
            for record in self.failed(execution_id)
            if record.recording_id in mapping
        }
        return sorted(indices)
