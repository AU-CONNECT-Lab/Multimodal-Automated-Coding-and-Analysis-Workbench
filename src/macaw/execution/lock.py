"""One `macaw run` per output directory at a time; the operating system drops the lock when the process ends."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

from macaw.exceptions import MacawError

LOCK_NAME = ".macaw_run.lock"


@contextmanager
def run_lock(output_dir: Path):
    path = Path(output_dir) / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        try:
            _acquire(handle)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "another process"
            raise MacawError(
                f"another macaw run ({holder}) is writing to {output_dir}; wait for it to finish or stop it"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid {os.getpid()}")
        handle.flush()
        yield
    finally:
        handle.close()


def _acquire(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        # a byte far past the pid text, so the holder can still rewrite the text it locks out others with
        handle.seek(1 << 20)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
