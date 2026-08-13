"""Model weights must be downloaded before submission. Compute nodes often have no
internet, and the failure mode is every array element dying at once."""

from __future__ import annotations

from pathlib import Path

from macaw.config.schema import MacawConfig


def required_models(config: MacawConfig, extractors: list[str]) -> dict[str, str]:
    """Extractor name to the model identifier its configured backend will ask for."""
    raise NotImplementedError


def prefetch(config: MacawConfig, dest: Path, extractors: list[str]) -> list[Path]:
    raise NotImplementedError


def verify(config: MacawConfig, dest: Path, extractors: list[str]) -> list[str]:
    """Names still missing from the shared cache, reported before anything is submitted."""
    raise NotImplementedError
