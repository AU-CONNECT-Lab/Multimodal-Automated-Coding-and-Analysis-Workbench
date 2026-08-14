from __future__ import annotations

from macaw.config.hpc import HPCConfig, ResourceProfile


def resolve(hpc: HPCConfig, extractor: str) -> ResourceProfile:
    """Looks up Extractor.resource_profile, then the tier by that name in hpc.yaml."""
    raise NotImplementedError


def needs_gpu(hpc: HPCConfig, extractor: str) -> bool:
    raise NotImplementedError


def validate(hpc: HPCConfig, extractors: list[str]) -> list[str]:
    """Warnings for requests a scheduler will reject, checked before anything is submitted."""
    raise NotImplementedError
