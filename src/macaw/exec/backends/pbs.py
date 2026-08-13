from __future__ import annotations

from pathlib import Path

from macaw.exec.backends.base import ExecBackend, JobSpec


class PBSBackend(ExecBackend):
    name = "pbs"

    def __init__(self, cfg, job_db=None):
        self.cfg = cfg
        self.job_db = job_db

    def submit(self, specs: list[JobSpec]) -> list[str]:
        raise NotImplementedError

    def wait(self, job_ids: list[str], *, polling_interval: int = 60) -> None:
        raise NotImplementedError

    def cancel(self, job_ids: list[str]) -> None:
        raise NotImplementedError

    def status(self, job_ids: list[str]) -> dict[str, str]:
        raise NotImplementedError


def generate_array_script(specs: list[JobSpec], cfg, out_path: Path) -> Path:
    raise NotImplementedError
