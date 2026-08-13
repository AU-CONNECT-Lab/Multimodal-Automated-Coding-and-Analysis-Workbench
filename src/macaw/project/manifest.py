"""Feature manifest. Without it nobody knows what a column means."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from macaw.store.schema import Scale


@dataclass
class FeatureSpec:
    name: str
    description: str
    unit: str | None
    scale: Scale
    group: str                          # prosody | text | pose | face | motion | turns | coding
    extractor: str
    extractor_version: str
    backend: str | None = None
    backend_version: str | None = None
    params: dict = field(default_factory=dict)
    interpolation: str | None = None    # how to fill gaps at frame scale
    aggregation: str | None = None      # how to roll up to window/session scale
    comparable_across_subjects: bool = True   # False for intensity; mic distance confounds it
    comparable_across_backends: bool = True   # False for AU; never merge across backends
    validation: str = "none"            # none | synthetic | against_reference | published
    primary: bool = False
    notes: str | None = None


@dataclass
class FeatureManifest:
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def add(self, spec: FeatureSpec) -> None:
        raise NotImplementedError

    def extend(self, specs: list[FeatureSpec]) -> None:
        raise NotImplementedError

    def get(self, name: str) -> FeatureSpec:
        raise NotImplementedError

    def by_group(self, group: str) -> list[FeatureSpec]:
        raise NotImplementedError

    def aggregation_map(self, scale: Scale) -> dict[str, str]:
        """Feeds aggregate_to_window so each feature rolls up the way its spec says."""
        raise NotImplementedError

    def mark_primary(self, names: list[str]) -> None:
        raise NotImplementedError


def read_manifest(path: Path) -> FeatureManifest:
    raise NotImplementedError


def write_manifest(manifest: FeatureManifest, path: Path) -> Path:
    raise NotImplementedError


def merge_manifests(manifests: list[FeatureManifest]) -> FeatureManifest:
    raise NotImplementedError
