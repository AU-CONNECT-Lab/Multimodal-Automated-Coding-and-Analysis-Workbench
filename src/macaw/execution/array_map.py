"""Stable recording to array-index mapping.

aftercorr pairs element N of one array with element N of the next, so the mapping must be
identical across every array in a run. Cached recordings still occupy their index and run a
task that exits immediately; indices are never compacted."""

from __future__ import annotations

from pathlib import Path


def build_map(recording_ids: list[str]) -> dict[str, int]:
    raise NotImplementedError


def write_map(mapping: dict[str, int], path: Path) -> Path:
    raise NotImplementedError


def read_map(path: Path) -> dict[str, int]:
    raise NotImplementedError


def recording_for_index(mapping: dict[str, int], index: int) -> str:
    raise NotImplementedError


def array_spec(mapping: dict[str, int], *, max_concurrent: int | None = None) -> str:
    """Renders 0-30 or 0-30%4 for the --array argument."""
    raise NotImplementedError


def rerun_spec(indices: list[int], *, max_concurrent: int | None = None) -> str:
    """Renders 3,7,12 for reruns of failed elements."""
    raise NotImplementedError
