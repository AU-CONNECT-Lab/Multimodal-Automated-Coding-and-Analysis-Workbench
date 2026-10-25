from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from macaw.config.schema import CONTENT_HASHED_PATHS, MacawConfig

HASH_LENGTH = 12
_CHUNK = 1 << 20


def canonical(value: Any) -> str:
    """Stable serialisation: sorted keys, normalised numbers, no absolute paths. Absolute
    paths would give the same data a different hash on a laptop and on the cluster, and
    every cached result would be recomputed after moving machines."""
    return json.dumps(_normalise(value), sort_keys=True, ensure_ascii=False, default=str)


def _normalise(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalise(value.model_dump(mode="python"))
    if isinstance(value, dict):
        # a path collapses to None rather than dropping its key: work_dir is Path | None, and
        # dropping would make "configured" and "left empty" two different shapes of dict
        return {k: (None if isinstance(v, Path) else _normalise(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalise(v) for v in value if not isinstance(v, Path)]
    if isinstance(value, float):
        return repr(value)          # 0.1 and 0.10 must not produce two different hashes
    return value


def hash_referenced_file(path: Path) -> str:
    """Content hash for keys listed in CONTENT_HASHED_PATHS."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()[:HASH_LENGTH]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_LENGTH]


def _prepared(config: MacawConfig) -> dict[str, Any]:
    data = config.model_dump(mode="python")
    for dotted in CONTENT_HASHED_PATHS:
        _substitute_content(data, dotted)
    return data


def _substitute_content(data: dict[str, Any], dotted: str) -> None:
    """Swap a referenced file's path for a hash of what is inside it, so editing the
    codebook invalidates the cache even though its path never changed."""
    parts = dotted.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            return
        cursor = cursor.get(part)
    if not isinstance(cursor, dict):
        return

    value = cursor.get(parts[-1])
    if value is None:
        return
    path = Path(value)
    # a value that is not a file on disk is a backend-specific name, not a reference
    cursor[parts[-1]] = hash_referenced_file(path) if path.is_file() else str(value)


def _section_value(data: dict[str, Any], name: str) -> Any:
    """A dotted name reaches a subsection, so two extractors sharing a config section can still
    have different cache keys: one that only reads coding.llm is not invalidated by a codebook.
    Keys may hold dots themselves (plugins are keyed by extractor name), so at each level the
    longest key matching the rest of the path wins."""
    cursor: Any = data
    parts = name.split(".")
    index = 0
    while index < len(parts):
        if not isinstance(cursor, dict):
            return None
        for end in range(len(parts), index, -1):
            key = ".".join(parts[index:end])
            if key in cursor:
                cursor, index = cursor[key], end
                break
        else:
            return None
    return cursor


def hash_section(config: MacawConfig, section: str) -> str:
    return _digest(canonical(_section_value(_prepared(config), section)))


def hash_sections(config: MacawConfig, sections: Sequence[str]) -> str:
    data = _prepared(config)
    parts = [f"{name}:{canonical(_section_value(data, name))}" for name in sorted(set(sections))]
    return _digest("|".join(parts))


def hash_extractor_config(config: MacawConfig, extractor: str) -> str:
    """Reads Extractor.config_sections, the single place that mapping is written down."""
    from macaw.extractors import get_extractor

    return hash_sections(config, get_extractor(extractor).config_sections)


def hash_full(config: MacawConfig) -> str:
    return _digest(canonical(_prepared(config)))


def diff_configs(a: MacawConfig, b: MacawConfig) -> dict[str, tuple]:
    """Which sections changed, and therefore which extractors a rerun would recompute."""
    left, right = _prepared(a), _prepared(b)
    changed: dict[str, tuple] = {}
    for section in sorted(set(left) | set(right)):
        one, two = _digest(canonical(left.get(section))), _digest(canonical(right.get(section)))
        if one != two:
            changed[section] = (one, two)
    return changed
