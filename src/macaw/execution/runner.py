"""Ties config, recordings, DAG, cache and backend together. The one entry point `macaw run` uses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from macaw.config.schema import MacawConfig
from macaw.extractors.base import QCWarning


@dataclass
class RunSummary:
    execution_id: str
    n_tasks: int
    n_done: int
    n_cached: int
    n_failed: int
    warnings: list[QCWarning]
    output_dir: Path


def run(
    config: MacawConfig,
    *,
    recordings: list[str] | None = None,
    extractors: list[str] | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> RunSummary:
    raise NotImplementedError


def run_task(config: MacawConfig, recording_id: str, extractor_name: str, execution_id: str) -> None:
    """The single-task entry point cluster array elements call back into."""
    raise NotImplementedError


def plan(config: MacawConfig, recording_ids: list[str], extractors: list[str]) -> list[str]:
    """What would run, in order, without running it."""
    raise NotImplementedError


def resolve_recordings(config: MacawConfig, requested: str | None) -> list[str]:
    """A comma list, a file of ids, or None meaning every detected recording minus exclusions."""
    raise NotImplementedError
