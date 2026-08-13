"""Content-addressed cache: (input hash, config-section hash, extractor version)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CacheKey:
    extractor: str
    extractor_version: str
    input_hash: str
    config_hash: str

    def as_dirname(self) -> str:
        raise NotImplementedError


def hash_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash size + mtime + head/tail bytes; full hashing of multi-GB video is too slow."""
    raise NotImplementedError


def hash_inputs(paths: Sequence[Path]) -> str:
    raise NotImplementedError


def make_key(
    *,
    extractor: str,
    extractor_version: str,
    inputs: Sequence[Path],
    config_hash: str,
) -> CacheKey:
    raise NotImplementedError


def lookup(cache_root: Path, key: CacheKey) -> Path | None:
    raise NotImplementedError


def store(cache_root: Path, key: CacheKey, payload_dir: Path) -> Path:
    raise NotImplementedError


def invalidate(cache_root: Path, *, extractor: str | None = None) -> int:
    raise NotImplementedError
