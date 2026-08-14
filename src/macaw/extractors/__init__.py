"""Extractor registry. Third-party extractors register through the same entry point."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable, Type

from macaw.exceptions import BackendNotAvailableError, ExtractorNotFoundError
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning

ENTRY_POINT_GROUP = "macaw.extractors"

_REGISTRY: dict[str, Type[Extractor]] = {}
_BACKENDS: dict[str, dict[str, Type[Backend]]] = {}


def register_extractor(cls: Type[Extractor]) -> Type[Extractor]:
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls:
        raise ValueError(f"Extractor name '{cls.name}' is already taken by {existing.__module__}")
    _REGISTRY[cls.name] = cls
    return cls


def register_backend(group: str) -> Callable[[Type[Backend]], Type[Backend]]:
    def decorator(cls: Type[Backend]) -> Type[Backend]:
        _BACKENDS.setdefault(group, {})[cls.name] = cls
        return cls

    return decorator


def get_extractor(name: str) -> Type[Extractor]:
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise ExtractorNotFoundError(f"Unknown extractor '{name}'. Available: {available}") from None


def get_backend(group: str, name: str) -> Type[Backend]:
    try:
        return _BACKENDS[group][name]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS.get(group, {}))) or "none registered"
        raise BackendNotAvailableError(
            f"No '{name}' backend for {group}. Available: {available}"
        ) from None


def list_extractors(group: str | None = None) -> list[str]:
    names = [n for n, cls in _REGISTRY.items() if group is None or cls.group == group]
    return sorted(names)


def list_backends(group: str) -> list[str]:
    return sorted(_BACKENDS.get(group, {}))


def load_entry_point_extractors() -> int:
    """Discover extractors published by other packages under the macaw.extractors group."""
    before = len(_REGISTRY)
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        loaded = entry.load()
        if isinstance(loaded, type) and issubclass(loaded, Extractor):
            register_extractor(loaded)
        elif callable(loaded):
            loaded()               # a module-level hook that registers several at once
    return len(_REGISTRY) - before


__all__ = [
    "Extractor",
    "ExtractorResult",
    "Backend",
    "QCWarning",
    "register_extractor",
    "register_backend",
    "get_extractor",
    "get_backend",
    "list_extractors",
    "list_backends",
]
