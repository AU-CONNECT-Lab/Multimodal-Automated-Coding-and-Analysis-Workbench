"""Job script rendering, shared by the cluster backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from macaw.config.hpc import HPCConfig, ResourceProfile
from macaw.config.schema import MacawConfig


@dataclass
class ArrayJob:
    extractor: str
    array_spec: str
    resources: ResourceProfile
    depends_on: list[str] = field(default_factory=list)   # extractor names, resolved to job ids
    dependency_kind: str = "aftercorr"


@dataclass
class SubmitScript:
    path: Path
    jobs: list[ArrayJob]
    map_path: Path


def preamble(config: MacawConfig, hpc: HPCConfig) -> list[str]:
    """Module loads, environment activation, HF_HOME, scratch. Runs before every task."""
    raise NotImplementedError


def task_command(config_path: Path, extractor: str, map_path: Path) -> str:
    """The one line an array element runs: look up its recording id, then `macaw run-task`."""
    raise NotImplementedError


def llm_server_lines(config: MacawConfig) -> list[str]:
    """Bring up a local model server on the compute node and wait for it to answer."""
    raise NotImplementedError


def render(
    jobs: list[ArrayJob],
    config: MacawConfig,
    hpc: HPCConfig,
    config_path: Path,
    map_path: Path,
) -> str:
    raise NotImplementedError


def write_script(text: str, path: Path) -> Path:
    raise NotImplementedError
