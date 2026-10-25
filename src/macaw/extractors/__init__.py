"""Extractor registry. Third-party extractors register through the same entry point."""

from __future__ import annotations

import logging
from importlib import import_module
from importlib.metadata import entry_points
from typing import Callable, Type

from macaw.exceptions import BackendNotAvailableError, ExtractorNotFoundError
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning

ENTRY_POINT_GROUP = "macaw.extractors"

# Modules holding extractors that ship with the package. A name appears here only once its
# extractor is implemented, so the registry always describes what can actually run.
BUILTIN_MODULES = (
    "macaw.extractors.prep",
    "macaw.extractors.audio.prosody.frame",
    "macaw.extractors.audio.content.asr",
    "macaw.turns.segmentation",
    "macaw.turns.transitions",
    "macaw.turns.entropy",
    "macaw.coding.layer_a",
    "macaw.extractors.video.pose.keypoints",
    "macaw.extractors.video.pose.kinematics",
)

_REGISTRY: dict[str, Type[Extractor]] = {}
_BACKENDS: dict[str, dict[str, Type[Backend]]] = {}
_builtins_loaded = False


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


def load_builtin_extractors() -> None:
    """Import the modules that ship with the package so their classes register themselves.
    Lazy rather than at package import: each of those modules imports this one back."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True          # set before importing, the imports re-enter here
    for module in BUILTIN_MODULES:
        import_module(module)
    load_entry_point_extractors()


def get_extractor(name: str) -> Type[Extractor]:
    load_builtin_extractors()
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise ExtractorNotFoundError(f"Unknown extractor '{name}'. Available: {available}") from None


def get_backend(group: str, name: str) -> Type[Backend]:
    load_builtin_extractors()
    try:
        return _BACKENDS[group][name]
    except KeyError:
        available = ", ".join(sorted(_BACKENDS.get(group, {}))) or "none registered"
        raise BackendNotAvailableError(
            f"No '{name}' backend for {group}. Available: {available}"
        ) from None


def list_extractors(group: str | None = None) -> list[str]:
    load_builtin_extractors()
    names = [n for n, cls in _REGISTRY.items() if group is None or cls.group == group]
    return sorted(names)


def list_backends(group: str) -> list[str]:
    load_builtin_extractors()
    return sorted(_BACKENDS.get(group, {}))


def list_backend_groups() -> list[str]:
    load_builtin_extractors()
    return sorted(_BACKENDS)


def load_entry_point_extractors() -> int:
    """Discover extractors published by other packages under the macaw.extractors group."""
    before = len(_REGISTRY)
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        # one broken plugin must not take every macaw command down with it
        try:
            loaded = entry.load()
        except Exception as error:
            logging.getLogger("macaw").warning("extractor plugin %s failed to load: %s", entry.name, error)
            continue
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
    "load_builtin_extractors",
    "get_extractor",
    "get_backend",
    "list_extractors",
    "list_backends",
    "list_backend_groups",
]
