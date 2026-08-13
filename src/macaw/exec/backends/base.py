"""Adapted from neuroimage-pipe hpc_utils.HPCBackend (Apache-2.0), generalised to local runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Resources:
    cpus: int = 1
    mem: str | None = None
    time: str | None = None
    gpu: str | None = None


@dataclass
class JobSpec:
    task_id: str                       # "<session_id>:<extractor>" -- macaw's unit of work
    session_id: str
    extractor: str
    command: list[str]
    resources: Resources
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


def get_backend(name: str, cfg) -> ExecBackend:
    raise NotImplementedError
