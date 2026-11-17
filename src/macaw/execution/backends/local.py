"""The backend most users will actually run. Same job_db bookkeeping as the HPC ones."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from macaw.execution.backends.base import ExecBackend, JobSpec
from macaw.execution.job_db import now


def visible_cards() -> list[str]:
    """The graphics cards a run may use: those CUDA_VISIBLE_DEVICES names where it is set, else every card nvidia-smi lists."""
    chosen = os.environ.get("CUDA_VISIBLE_DEVICES")
    if chosen is not None:
        return [part.strip() for part in chosen.split(",") if part.strip()]
    try:
        listed = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if listed.returncode != 0:
        return []
    return [str(index) for index, line in enumerate(line for line in listed.stdout.splitlines() if line.startswith("GPU "))]


def gpu_limit(gpu_workers: int | None, cards: list[str]) -> int | None:
    """How many steps that use a card run at once; None, without cards and without a number set, leaves them to n_workers."""
    return gpu_workers if gpu_workers is not None else (len(cards) or None)


def gpu_notice(steps: list[str], n_workers: int, gpu_workers: int | None, cards: list[str]) -> str | None:
    """What the run plan says about the steps that use a card, where more than one task runs at a time."""
    if not steps or n_workers <= 1:
        return None
    limit = gpu_limit(gpu_workers, cards)
    if limit is None:
        return f"no NVIDIA card was found, so {', '.join(steps)} run side by side like any other step"
    where = f"on card{'s' if len(cards) > 1 else ''} {', '.join(cards)}" if cards else "no NVIDIA card was found"
    return (
        f"{', '.join(steps)} use a graphics card: {min(limit, n_workers)} at a time (execution.gpu_workers), {where}; "
        "CUDA_VISIBLE_DEVICES picks the cards"
    )


class LocalBackend(ExecBackend):
    name = "local"

    def __init__(self, n_workers: int = 4, job_db=None, *, gpu_workers: int | None = None, cards: list[str] | None = None):
        self.n_workers = n_workers
        self.job_db = job_db
        self.cards = list(cards or [])
        self.gpu_limit = gpu_limit(gpu_workers, self.cards)
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
        # tasks on each card; a step that uses one gets the least busy, and sees only that one
        load = {card: 0 for card in self.cards}
        on_card: dict[str, str | None] = {}

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
                # the card is the scarcer resource, so its steps go first
                ready.sort(key=lambda task_id: not pending[task_id].gpu)
                for task_id in ready:
                    if len(running) >= max(self.n_workers, 1):
                        break
                    spec = pending[task_id]
                    if spec.gpu and self.gpu_limit is not None:
                        if len(on_card) >= self.gpu_limit:
                            continue
                        card = min(load, key=load.get) if load else None
                        if card is not None:
                            load[card] += 1
                            spec.env = {**(spec.env or {}), "CUDA_VISIBLE_DEVICES": card}
                        on_card[task_id] = card
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
                    try:
                        code = running.pop(task_id).result()
                    except Exception as error:
                        # bookkeeping around the task broke; it counts as failed, and the tasks left still get their turn
                        code = 1
                        self._record_broken(task_id, error)
                    if task_id in on_card and (card := on_card.pop(task_id)) is not None:
                        load[card] -= 1
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
        # a task that found its result in the cache records that itself, and a clean exit keeps it
        if code == 0 and self.job_db is not None and getattr(self.job_db.get(spec.task_id), "status", None) == "cached":
            status = "cached"
        message = None if code == 0 else f"exit code {code}, see {log_path}"
        self._finish(spec.task_id, time.monotonic() - started, status, message, exit_code=code)
        return code

    def _record_broken(self, task_id: str, error: Exception) -> None:
        try:
            self._finish(task_id, None, "failed", f"{type(error).__name__}: {error}")
        except Exception:
            logging.getLogger("macaw").error("%s failed and its record could not be written: %s", task_id, error)

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
