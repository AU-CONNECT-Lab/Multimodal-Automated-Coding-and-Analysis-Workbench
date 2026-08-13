from __future__ import annotations

from macaw.execution.job_db import JobRecord


def layout() -> object:
    raise NotImplementedError


def status_table(records: list[JobRecord]) -> object:
    raise NotImplementedError


def progress_summary(counts: dict[str, int]) -> object:
    raise NotImplementedError


def log_viewer(log_path: str | None) -> object:
    raise NotImplementedError
