from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.extractors.base import Backend


class ProsodyBackend(Backend):
    def f0(self, audio: np.ndarray, sr: float, f0_min: float, f0_max: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def intensity(self, audio: np.ndarray, sr: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def hnr(self, audio: np.ndarray, sr: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def formants(self, audio: np.ndarray, sr: float, step_s: float, n_formants: int = 4) -> np.ndarray:
        raise NotImplementedError


class ParselmouthBackend(ProsodyBackend):
    name = "parselmouth"

    def check_available(self) -> None:
        raise NotImplementedError


class LibrosaBackend(ProsodyBackend):
    name = "librosa"

    def check_available(self) -> None:
        raise NotImplementedError


class BuiltinBackend(ProsodyBackend):
    """Autocorrelation f0. Last resort so a run never dies for lack of a dependency."""

    name = "builtin"

    def check_available(self) -> None:
        raise NotImplementedError


def resolve_backend(chain: list[str]) -> ProsodyBackend:
    """First available backend in the chain; logs which one was picked."""
    raise NotImplementedError


def load_audio(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, float]:
    raise NotImplementedError
