"""Scheduler abstraction. Local runs and cluster submission share this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from macaw.config.hpc import ResourceProfile


@dataclass
class JobSpec:
    task_id: str                       # "<session_id>:<extractor>", macaw's unit of work
    session_id: str
    extractor: str
    command: list[str]
    resources: ResourceProfile
    log_path: Path
    depends_on: list[str] = None


class ExecBackend(ABC):
    name: str

    @abstractmethod
    def submit(self, specs: list[JobSpec]) -> list[str]: ...

    @abstractmethod
    def wait(self, job_ids: list[str], *, polling_interval: int = 5) -> None: ...

    @abstractmethod
    def cancel(self, job_ids: list[str]) -> None: ...

    @abstractmethod
    def status(self, job_ids: list[str]) -> dict[str, str]: ...


def get_backend(name: str, cfg, hpc=None) -> ExecBackend:
    """hpc is only consulted by cluster backends; the local one ignores it."""
    raise NotImplementedError
