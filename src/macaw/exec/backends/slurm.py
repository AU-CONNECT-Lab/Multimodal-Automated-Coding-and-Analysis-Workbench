"""Script generation only, no automatic submission."""

from __future__ import annotations

from pathlib import Path

from macaw.config.schema import MacawConfig
from macaw.exec.backends.base import ExecBackend, JobSpec
from macaw.exec.script import ArrayJob, SubmitScript


class SlurmBackend(ExecBackend):
    name = "slurm"

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
    """One array per extractor. Grouping by topological layer instead would force a whole
    layer to request GPUs because one member needs them."""
    raise NotImplementedError


def generate(
    config: MacawConfig,
    config_path: Path,
    session_ids: list[str],
    extractors: list[str],
    out_path: Path,
) -> SubmitScript:
    """Chained sbatch calls: each captures --parsable and feeds the next --dependency."""
    raise NotImplementedError


def dependency_flag(job_ids: list[str], kind: str = "aftercorr") -> str:
    """aftercorr pairs element N with element N, so each session advances independently.
    afterok would make the slowest session hold up every other one."""
    raise NotImplementedError


def generate_rerun(
    config: MacawConfig,
    config_path: Path,
    run_id: str,
    out_path: Path,
) -> SubmitScript:
    raise NotImplementedError


def shell_quote(value) -> str:
    raise NotImplementedError
