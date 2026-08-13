"""The backend most users will actually run. Same job_db bookkeeping as the HPC ones."""

from __future__ import annotations

from macaw.exec.backends.base import ExecBackend, JobSpec


class LocalBackend(ExecBackend):
    name = "local"

    def __init__(self, n_workers: int = 4, job_db=None):
        self.n_workers = n_workers
        self.job_db = job_db

    def submit(self, specs: list[JobSpec]) -> list[str]:
        """Process pool, dependency-aware. Each task still gets a job_db row and a log file."""
        raise NotImplementedError

    def wait(self, job_ids: list[str], *, polling_interval: int = 5) -> None:
        raise NotImplementedError

    def cancel(self, job_ids: list[str]) -> None:
        raise NotImplementedError

    def status(self, job_ids: list[str]) -> dict[str, str]:
        raise NotImplementedError

    def _run_one(self, spec: JobSpec) -> int:
        raise NotImplementedError
