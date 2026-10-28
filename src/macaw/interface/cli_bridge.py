"""The GUI never recomputes anything. Batch actions shell out to the `macaw` CLI, and
progress is read back from job_db and the log files."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from macaw.config.loader import resolve_config
from macaw.execution.job_db import JobDB
from macaw.project.layout import layout_for

# the interpreter running the GUI also runs its commands, so a GUI started from one environment never reaches another
PREFIX = [sys.executable, "-m", "macaw"]


@dataclass
class LaunchedCommand:
    argv: list[str]
    pid: int
    log_path: Path
    started_at: str
    process: subprocess.Popen | None = field(default=None, repr=False)


def build_command(
    subcommand: list[str],
    *,
    config_path: Path,
    recordings: list[str] | None = None,
    extractors: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """The exact argv a user could paste into a terminal, which is also what the page shows."""
    argv = ["macaw", *subcommand, "-c", str(config_path)]
    if recordings:
        argv += ["-r", ",".join(recordings)]
    for name in extractors or ():
        argv += ["-e", name]
    return argv + list(extra_flags or ())


def _environment() -> dict[str, str]:
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


def launch(argv: list[str], log_path: Path, *, cwd: Path | None = None) -> LaunchedCommand:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    # its own process group, so stopping it stops what it started as well
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    process = subprocess.Popen(
        [*PREFIX, *argv[1:]], stdout=handle, stderr=subprocess.STDOUT, cwd=cwd, env=_environment(), **group
    )
    handle.close()
    return LaunchedCommand(argv, process.pid, log_path, datetime.now().isoformat(timespec="seconds"), process)


def run_now(argv: list[str], *, cwd: Path | None = None, timeout_s: float = 600) -> tuple[int, str]:
    """For the short commands a button stands for: keep, rename, confirm."""
    done = subprocess.run(
        [*PREFIX, *argv[1:]], capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=cwd, env=_environment(), timeout=timeout_s,
    )
    return done.returncode, (done.stdout + done.stderr).strip()


def is_running(cmd: LaunchedCommand) -> bool:
    return cmd.process is not None and cmd.process.poll() is None


def exit_code(cmd: LaunchedCommand) -> int | None:
    return None if cmd.process is None else cmd.process.poll()


def cancel(cmd: LaunchedCommand) -> None:
    if not is_running(cmd):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(cmd.pid)], capture_output=True)
    else:
        os.killpg(cmd.pid, signal.SIGTERM)


def tail_log(log_path: Path, *, n_lines: int = 200) -> list[str]:
    if not log_path.is_file():
        return []
    return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:]


def poll_status(config_path: Path) -> tuple[str | None, dict[str, int]]:
    """The latest execution's counts by status, straight out of job_db; the same source `macaw status` reads."""
    database = JobDB(layout_for(resolve_config(config_path), create=False).job_db_path)
    known = database.executions()
    if not known:
        return None, {}
    return known[-1], database.summary(known[-1])


def failed_last_time(config_path: Path) -> set[str]:
    database = JobDB(layout_for(resolve_config(config_path), create=False).job_db_path)
    known = database.executions()
    return {record.task_id for record in database.failed(known[-1])} if known else set()
