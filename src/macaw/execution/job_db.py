"""SQLite job state. Adapted from neuroimage-pipe job_db, keyed by task_id rather than
subject/recording columns; macaw's unit of work is recording x extractor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


class JobDB:
    def __init__(self, path: Path):
        self.path = path

    def initialise(self) -> None:
        raise NotImplementedError

    def upsert(self, record: JobRecord) -> None:
        raise NotImplementedError

    def set_status(self, task_id: str, status: str, **fields) -> None:
        raise NotImplementedError

    def get(self, task_id: str) -> JobRecord | None:
        raise NotImplementedError

    def query(self, *, execution_id: str | None = None, status: str | None = None) -> list[JobRecord]:
        raise NotImplementedError

    def pending(self, execution_id: str) -> list[JobRecord]:
        raise NotImplementedError

    def failed(self, execution_id: str) -> list[JobRecord]:
        raise NotImplementedError

    def summary(self, execution_id: str) -> dict[str, int]:
        raise NotImplementedError

    def array_indices_for_rerun(self, execution_id: str) -> list[int]:
        """Turns the failed set into a --array=3,7,12 argument."""
        raise NotImplementedError
