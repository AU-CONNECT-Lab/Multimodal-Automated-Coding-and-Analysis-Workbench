from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from macaw.config.schema import MacawConfig


def canonical(value: Any) -> str:
    """Stable serialisation: sorted keys, normalised numbers, no absolute paths. Absolute
    paths would give the same data a different hash on a laptop and on the cluster, and
    every cached result would be recomputed after moving machines."""
    raise NotImplementedError


def hash_referenced_file(path: Path) -> str:
    """Content hash for keys listed in CONTENT_HASHED_PATHS."""
    raise NotImplementedError


def hash_section(config: MacawConfig, section: str) -> str:
    raise NotImplementedError


def hash_sections(config: MacawConfig, sections: Sequence[str]) -> str:
    raise NotImplementedError


def hash_extractor_config(config: MacawConfig, extractor: str) -> str:
    """Reads Extractor.config_sections, the single place that mapping is written down."""
    raise NotImplementedError


def hash_full(config: MacawConfig) -> str:
    raise NotImplementedError


def diff_configs(a: MacawConfig, b: MacawConfig) -> dict[str, tuple]:
    """Which sections changed, and therefore which extractors a rerun would recompute."""
    raise NotImplementedError
