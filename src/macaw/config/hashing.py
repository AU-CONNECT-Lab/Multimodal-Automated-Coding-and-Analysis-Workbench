from __future__ import annotations

from typing import Sequence

from macaw.config.schema import MacawConfig


def hash_section(config: MacawConfig, section: str) -> str:
    raise NotImplementedError


def hash_sections(config: MacawConfig, sections: Sequence[str]) -> str:
    """Combined hash for an extractor; changing f0 params must not invalidate pose."""
    raise NotImplementedError


def hash_full(config: MacawConfig) -> str:
    raise NotImplementedError


def diff_configs(a: MacawConfig, b: MacawConfig) -> dict[str, tuple]:
    """Which sections changed, for reporting what a rerun will recompute."""
    raise NotImplementedError
