"""Job state as one small file per task.

Not SQLite: on a cluster the state lives on a shared filesystem and is written by every
array element at once, where NFS locking is unreliable enough to corrupt the database. One
file per task removes locking from the problem entirely, because a task only ever runs in
one place at a time and so is the only writer of its own file. Same rule as the long table
and the manifest.

Reads glob and merge. A run of 31 recordings x 20 extractors is 620 files, which is nothing
to a directory listing.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from macaw.exceptions import MacawError

STATUSES = ("pending", "running", "done", "failed", "cached", "cancelled")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


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


def _slug(task_id: str) -> str:
    return _UNSAFE.sub("_", task_id)


class JobDB:
    def __init__(self, path: Path):
        self.path = Path(path)

    def initialise(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def record_path(self, execution_id: str, task_id: str) -> Path:
        return self.path / execution_id / f"{_slug(task_id)}.json"

    def _write(self, record: JobRecord) -> None:
        target = self.record_path(record.execution_id, record.task_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        # atomic swap, so a reader polling for progress never sees half a record
        staging = target.with_name(f"{target.name}.tmp.{os.getpid()}")
        staging.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        os.replace(staging, target)

    def _read(self, path: Path) -> JobRecord | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # a task that died mid-write leaves an unreadable file; treat it as absent
            # rather than bringing down whatever is reporting progress
            return None
        return JobRecord(**{name: payload.get(name) for name in _COLUMNS})

    def _paths(self, execution_id: str | None = None, task_id: str | None = None) -> list[Path]:
        pattern = f"{_slug(task_id)}.json" if task_id else "*.json"
        root = self.path / execution_id if execution_id else self.path
        if not root.is_dir():
            return []
        return sorted(root.glob(pattern) if execution_id else root.glob(f"*/{pattern}"))

    def upsert(self, record: JobRecord) -> None:
        if record.status not in STATUSES:
            raise MacawError(f"Unknown job status '{record.status}'")
        self._write(record)

    def set_status(self, task_id: str, status: str, **updates) -> None:
        """Updates the most recent execution of that task, so a rerun never rewrites history."""
        if status not in STATUSES:
            raise MacawError(f"Unknown job status '{status}'")
        unknown = set(updates) - set(_COLUMNS)
        if unknown:
            raise MacawError(f"Unknown job field(s): {', '.join(sorted(unknown))}")

        record = self.get(task_id)
        if record is None:
            raise MacawError(f"No job record for task '{task_id}'")
        record.status = status
        for name, value in updates.items():
            setattr(record, name, value)
        self._write(record)

    def get(self, task_id: str) -> JobRecord | None:
        paths = self._paths(task_id=task_id)
        if not paths:
            return None
        newest = max(paths, key=lambda path: path.stat().st_mtime)
        return self._read(newest)

    def query(
        self, *, execution_id: str | None = None, status: str | None = None
    ) -> list[JobRecord]:
        records = [record for path in self._paths(execution_id) if (record := self._read(path))]
        if status is not None:
            records = [record for record in records if record.status == status]
        return sorted(records, key=lambda record: (record.execution_id, record.task_id))

    def pending(self, execution_id: str) -> list[JobRecord]:
        return self.query(execution_id=execution_id, status="pending")

    def failed(self, execution_id: str) -> list[JobRecord]:
        return self.query(execution_id=execution_id, status="failed")

    def summary(self, execution_id: str) -> dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        for record in self.query(execution_id=execution_id):
            counts[record.status] = counts.get(record.status, 0) + 1
        return counts

    def executions(self) -> list[str]:
        return sorted(item.name for item in self.path.glob("*") if item.is_dir())
