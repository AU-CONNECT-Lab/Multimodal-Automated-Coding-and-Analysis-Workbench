"""Extractor registry. Third-party extractors register through the same entry point."""

from __future__ import annotations

import logging
import re
from importlib import import_module
from importlib.metadata import entry_points
from typing import Callable, Type

from macaw.exceptions import BackendNotAvailableError, ExtractorNotFoundError
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning

ENTRY_POINT_GROUP = "macaw.extractors"
BACKEND_ENTRY_POINT_GROUP = "macaw.backends"

# Modules holding extractors that ship with the package. A name appears here only once its
# extractor is implemented, so the registry always describes what can actually run.
BUILTIN_MODULES = (
    "macaw.extractors.prep",
    "macaw.extractors.audio.prosody.frame",
    "macaw.extractors.audio.content.asr",
    "macaw.extractors.audio.content.line_speakers",
    "macaw.extractors.audio.content.features",
    "macaw.extractors.audio.content.syntax",
    "macaw.turns.segmentation",
    "macaw.turns.transitions",
    "macaw.turns.entropy",
    "macaw.coding.layer_a",
    "macaw.coding.llm.judge",
    "macaw.extractors.video.pose.keypoints",
    "macaw.extractors.video.pose.kinematics",
    "macaw.extractors.video.motion.mea",
    "macaw.extractors.audio.prosody.standard_sets",
    "macaw.extractors.audio.prosody.voice_report",
    "macaw.extractors.audio.prosody.turn_pairs",
    "macaw.extractors.video.face.au",
    "macaw.extractors.video.face.movement",
    "macaw.extractors.video.face.shares",
    "macaw.extractors.video.pose.listening",
    "macaw.extractors.video.pose.moving",
    "macaw.extractors.video.objects.track",
    "macaw.extractors.video.objects.review",
    "macaw.extractors.video.objects.touch",
    "macaw.extractors.video.objects.position",
    "macaw.extractors.segments",
    "macaw.signals.read",
    "macaw.signals.align",
)

_REGISTRY: dict[str, Type[Extractor]] = {}
_BACKENDS: dict[str, dict[str, Type[Backend]]] = {}
# a step's name is the name of its files and its job's, so it keeps to what a file name can hold everywhere
STEP_NAME = re.compile(r"[A-Za-z0-9._-]+")
_builtins_loaded = False


def _read_again(existing: type, cls: type) -> bool:
    """A project script read again after an edit replaces what it registered before."""
    from macaw.project_scripts import MODULE_PREFIX

    return existing.__module__ == cls.__module__ and cls.__module__.startswith(f"{MODULE_PREFIX}.")


def register_extractor(cls: Type[Extractor]) -> Type[Extractor]:
    if not STEP_NAME.fullmatch(cls.name):
        raise ValueError(f"Extractor name '{cls.name}' may hold only letters, digits, dots, dashes and underscores")
    existing = _REGISTRY.get(cls.name)
    if existing is not None and existing is not cls and not _read_again(existing, cls):
        raise ValueError(f"Extractor name '{cls.name}' is already taken by {existing.__module__}")
    _REGISTRY[cls.name] = cls
    return cls


def register_backend(group: str) -> Callable[[Type[Backend]], Type[Backend]]:
    def decorator(cls: Type[Backend]) -> Type[Backend]:
        existing = _BACKENDS.get(group, {}).get(cls.name)
        if existing is not None and existing is not cls and not _read_again(existing, cls):
            raise ValueError(f"{group} backend name '{cls.name}' is already taken by {existing.__module__}")
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
    load_entry_point_backends()


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


def load_entry_point_backends() -> int:
    """Import the modules other packages publish under macaw.backends; their classes register
    themselves with register_backend as they are imported."""
    before = sum(len(names) for names in _BACKENDS.values())
    for entry in entry_points(group=BACKEND_ENTRY_POINT_GROUP):
        try:
            entry.load()
        except Exception as error:
            logging.getLogger("macaw").warning("backend plugin %s failed to load: %s", entry.name, error)
    return sum(len(names) for names in _BACKENDS.values()) - before


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
