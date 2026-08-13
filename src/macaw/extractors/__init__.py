"""Extractor registry. Third-party extractors register through the same entry point."""

from __future__ import annotations

from typing import Callable, Type

from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning

_REGISTRY: dict[str, Type[Extractor]] = {}
_BACKENDS: dict[str, dict[str, Type[Backend]]] = {}


def register_extractor(cls: Type[Extractor]) -> Type[Extractor]:
    raise NotImplementedError


def register_backend(group: str) -> Callable[[Type[Backend]], Type[Backend]]:
    raise NotImplementedError


def get_extractor(name: str) -> Type[Extractor]:
    raise NotImplementedError


def get_backend(group: str, name: str) -> Type[Backend]:
    raise NotImplementedError


def list_extractors(group: str | None = None) -> list[str]:
    raise NotImplementedError


def list_backends(group: str) -> list[str]:
    raise NotImplementedError


def load_entry_point_extractors() -> int:
    """Discover extractors published by other packages under the macaw.extractors group."""
    raise NotImplementedError


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
