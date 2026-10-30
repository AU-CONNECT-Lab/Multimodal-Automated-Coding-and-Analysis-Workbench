"""The backend most users will actually run. Same job_db bookkeeping as the HPC ones."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from macaw.execution.backends.base import ExecBackend, JobSpec
from macaw.execution.job_db import now


class LocalBackend(ExecBackend):
    name = "local"

    def __init__(self, n_workers: int = 4, job_db=None):
        self.n_workers = n_workers
        self.job_db = job_db
        self._results: dict[str, int] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._scheduler: threading.Thread | None = None
        self._cancelled = threading.Event()

    def submit(self, specs: list[JobSpec]) -> list[str]:
        """Process pool, dependency-aware. Each task still gets a job_db row and a log file."""
        self._results.clear()
        self._cancelled.clear()
        self._scheduler = threading.Thread(target=self._schedule, args=(list(specs),), daemon=True)
        self._scheduler.start()
        return [spec.task_id for spec in specs]

    def _schedule(self, specs: list[JobSpec]) -> None:
        """A dispatch loop rather than each worker waiting on its own dependencies: with N
        workers all blocked on predecessors the pool would deadlock."""
        pending = {spec.task_id: spec for spec in specs}
        finished: set[str] = set()

        with ThreadPoolExecutor(max_workers=max(self.n_workers, 1)) as pool:
            running: dict[str, object] = {}
            while pending or running:
                if self._cancelled.is_set():
                    break
                ready = [
                    task_id
                    for task_id, spec in pending.items()
                    if set(spec.depends_on or ()) <= finished
                ]
                for task_id in ready:
                    if len(running) >= max(self.n_workers, 1):
                        break
                    running[task_id] = pool.submit(self._run_one, pending.pop(task_id))

                if not running:
                    if pending:
                        # everything left depends on something that failed
                        for task_id in list(pending):
                            self._finish(task_id, None, "cancelled", "upstream task did not finish")
                            pending.pop(task_id)
                    break

                done = [task_id for task_id, future in running.items() if future.done()]
                if not done:
                    time.sleep(0.05)
                    continue
                for task_id in done:
                    code = running.pop(task_id).result()
                    self._results[task_id] = code
                    if code == 0:
                        finished.add(task_id)

    def wait(self, job_ids: list[str], *, polling_interval: int = 5) -> None:
        if self._scheduler is not None:
            self._scheduler.join()

    def cancel(self, job_ids: list[str]) -> None:
        self._cancelled.set()
        with self._lock:
            for task_id in job_ids:
                process = self._processes.get(task_id)
                if process is not None and process.poll() is None:
                    process.terminate()

    def status(self, job_ids: list[str]) -> dict[str, str]:
        states = {}
        for task_id in job_ids:
            if task_id in self._results:
                states[task_id] = "done" if self._results[task_id] == 0 else "failed"
            elif task_id in self._processes:
                states[task_id] = "running"
            else:
                states[task_id] = "pending"
        return states

    def _run_one(self, spec: JobSpec) -> int:
        log_path = Path(spec.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        if self.job_db is not None:
            self.job_db.set_status(
                spec.task_id, "running", started_at=now(), log_path=str(log_path)
            )

        try:
            with open(log_path, "w", encoding="utf-8") as log:
                log.write(f"$ {' '.join(spec.command)}\n\n")
                log.flush()
                env = {**os.environ, **spec.env} if spec.env else None
                process = subprocess.Popen(spec.command, stdout=log, stderr=subprocess.STDOUT, env=env)
                with self._lock:
                    self._processes[spec.task_id] = process
                code = process.wait()
        except OSError as error:
            self._finish(spec.task_id, time.monotonic() - started, "failed", str(error))
            return 1

        status = "done" if code == 0 else "failed"
        message = None if code == 0 else f"exit code {code}, see {log_path}"
        self._finish(spec.task_id, time.monotonic() - started, status, message, exit_code=code)
        return code

    def _finish(
        self,
        task_id: str,
        duration_s: float | None,
        status: str,
        message: str | None,
        *,
        exit_code: int | None = None,
    ) -> None:
        if self.job_db is None:
            return
        self.job_db.set_status(
            task_id,
            status,
            finished_at=now(),
            duration_s=duration_s,
            exit_code=exit_code,
            message=message,
        )
