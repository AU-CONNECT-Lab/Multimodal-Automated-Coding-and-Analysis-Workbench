from __future__ import annotations

from pathlib import Path

from macaw.config.schema import MacawConfig
from macaw.execution.backends.base import ExecBackend, JobSpec
from macaw.execution.script import ArrayJob, SubmitScript


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


def build_jobs(config: MacawConfig, extractors: list[str], array_spec: str) -> list[ArrayJob]:
    raise NotImplementedError


def generate(
    config: MacawConfig,
    config_path: Path,
    session_ids: list[str],
    extractors: list[str],
    out_path: Path,
) -> SubmitScript:
    """No element-wise dependency exists here, so each array waits for the previous one in
    full. Sessions no longer advance independently; the generator warns about this."""
    raise NotImplementedError


def dependency_flag(job_ids: list[str], kind: str = "afterokarray") -> str:
    raise NotImplementedError
