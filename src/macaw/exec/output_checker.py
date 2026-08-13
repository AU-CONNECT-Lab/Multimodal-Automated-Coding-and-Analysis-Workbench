"""Expected outputs declared in YAML (glob + min_size + count). Adapted from neuroimage-pipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutputRule:
    extractor: str
    pattern: str
    min_size_kb: float = 0.0
    min_count: int = 1
    optional: bool = False


@dataclass
class CheckResult:
    task_id: str
    complete: bool
    missing: list[str]
    undersized: list[str]


def load_rules(path: Path) -> list[OutputRule]:
    raise NotImplementedError


def check_task(session_dir: Path, extractor: str, rules: list[OutputRule], task_id: str) -> CheckResult:
    raise NotImplementedError


def completed_tasks(output_dir: Path, rules: list[OutputRule]) -> set[str]:
    """Drives resume: a task whose outputs already validate is skipped."""
    raise NotImplementedError
