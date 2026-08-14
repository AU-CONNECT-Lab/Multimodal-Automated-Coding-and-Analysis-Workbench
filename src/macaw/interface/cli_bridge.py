"""The GUI never recomputes anything. Batch actions shell out to the `macaw` CLI, and
progress is read back from job_db and the log files. Only interactive editing
(alignment tuning, manual coding) touches the store layer directly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class LaunchedCommand:
    argv: list[str]
    pid: int
    execution_id: str | None
    log_path: Path
    started_at: str


def build_command(
    subcommand: list[str],
    *,
    config_path: Path,
    recordings: list[str] | None = None,
    extractors: list[str] | None = None,
    overrides: dict[str, str] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Assemble the exact argv a user could paste into a terminal; show it in the UI."""
    raise NotImplementedError


def launch(argv: list[str], log_path: Path, *, cwd: Path | None = None) -> LaunchedCommand:
    raise NotImplementedError


def is_running(cmd: LaunchedCommand) -> bool:
    raise NotImplementedError


def cancel(cmd: LaunchedCommand) -> None:
    raise NotImplementedError


def tail_log(log_path: Path, *, n_lines: int = 200) -> list[str]:
    raise NotImplementedError


def poll_status(config_path: Path, execution_id: str) -> dict[str, int]:
    """Counts by status, straight out of job_db; the same source `macaw status` reads."""
    raise NotImplementedError
