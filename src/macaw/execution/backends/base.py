"""Scheduler abstraction. Local runs and cluster submission share this interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from macaw.config.hpc import ResourceProfile


@dataclass
class JobSpec:
    task_id: str                       # "<recording_id>:<extractor>", macaw's unit of work
    recording_id: str
    extractor: str
    command: list[str]
    resources: ResourceProfile
    log_path: Path
    depends_on: list[str] = None
    env: dict[str, str] | None = None  # set on top of this process's environment


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
