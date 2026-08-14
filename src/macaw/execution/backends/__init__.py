from macaw.execution.backends.base import ExecBackend, JobSpec, get_backend
from macaw.execution.backends.local import LocalBackend
from macaw.execution.backends.slurm import SlurmBackend

__all__ = [
    "ExecBackend",
    "JobSpec",
    "get_backend",
    "LocalBackend",
    "SlurmBackend",
]
