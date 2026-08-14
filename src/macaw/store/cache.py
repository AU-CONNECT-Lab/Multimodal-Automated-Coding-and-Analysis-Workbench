"""Content-addressed cache: (input hash, config-section hash, extractor version)."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from macaw.exceptions import CacheCorruptError, InputNotFoundError

HASH_LENGTH = 12
COMPLETE_MARKER = ".complete"
# how much of each end of a file goes into its hash
_EDGE_BYTES = 1 << 20


@dataclass(frozen=True)
class CacheKey:
    extractor: str
    extractor_version: str
    input_hash: str
    config_hash: str

    def as_dirname(self) -> str:
        return f"{self.extractor_version}-{self.input_hash}-{self.config_hash}"


def hash_file(path: Path, *, chunk_size: int = _EDGE_BYTES) -> str:
    """Size plus the first and last chunk. Hashing multi-GB video end to end is too slow,
    and mtime is deliberately left out: copying a dataset to the cluster changes every
    mtime and would throw away the whole cache."""
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"Cannot hash a file that does not exist: {path}")

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("utf-8"))
    with open(path, "rb") as handle:
        digest.update(handle.read(chunk_size))
        if size > chunk_size:
            handle.seek(max(size - chunk_size, chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:HASH_LENGTH]


def hash_inputs(paths: Sequence[Path]) -> str:
    """Only the file names go in, never the directories they sit in, for the same reason
    absolute paths stay out of the config hash."""
    parts = sorted(f"{Path(path).name}:{hash_file(Path(path))}" for path in paths)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:HASH_LENGTH]


def make_key(
    *,
    extractor: str,
    extractor_version: str,
    inputs: Sequence[Path],
    config_hash: str,
) -> CacheKey:
    return CacheKey(
        extractor=extractor,
        extractor_version=extractor_version,
        input_hash=hash_inputs(inputs),
        config_hash=config_hash,
    )


def entry_path(cache_root: Path, key: CacheKey) -> Path:
    return Path(cache_root) / key.extractor / key.as_dirname()


def lookup(cache_root: Path, key: CacheKey) -> Path | None:
    """A hit means a complete, replayable payload. A directory without the marker is the
    wreckage of a job that died mid-write, so it counts as a miss."""
    entry = entry_path(cache_root, key)
    if not entry.is_dir():
        return None
    return entry if (entry / COMPLETE_MARKER).is_file() else None


def store(cache_root: Path, key: CacheKey, payload_dir: Path) -> Path:
    payload_dir = Path(payload_dir)
    if not payload_dir.is_dir():
        raise CacheCorruptError(f"Nothing to cache, payload directory is missing: {payload_dir}")

    entry = entry_path(cache_root, key)
    entry.parent.mkdir(parents=True, exist_ok=True)

    staging = entry.with_name(entry.name + f".partial.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(payload_dir, staging)
    (staging / COMPLETE_MARKER).write_text(key.as_dirname(), encoding="utf-8")

    if entry.exists():
        shutil.rmtree(entry)
    os.replace(staging, entry)
    return entry


def restore(entry: Path, dest_dir: Path) -> Path:
    """Copy a hit back out. The caller then sees exactly what a real run would have left."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(entry).iterdir():
        if item.name == COMPLETE_MARKER:
            continue
        target = dest_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    return dest_dir


def invalidate(cache_root: Path, *, extractor: str | None = None) -> int:
    root = Path(cache_root)
    if not root.is_dir():
        return 0

    targets = [root / extractor] if extractor else [p for p in root.iterdir() if p.is_dir()]
    removed = 0
    for target in targets:
        if not target.is_dir():
            continue
        for entry in target.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
                removed += 1
    return removed
