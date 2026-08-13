from __future__ import annotations

from macaw.config.schema import MacawConfig, ResourceSpec


def resolve(config: MacawConfig, extractor: str) -> ResourceSpec:
    """Per-extractor override on top of default_resources. A GPU pose job and a CPU
    prosody job must not share one request."""
    raise NotImplementedError


def needs_gpu(config: MacawConfig, extractor: str) -> bool:
    raise NotImplementedError


def validate(config: MacawConfig, extractors: list[str]) -> list[str]:
    """Warnings for requests a scheduler will reject, checked before anything is submitted."""
    raise NotImplementedError
