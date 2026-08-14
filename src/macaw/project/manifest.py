"""Feature manifest. Without it nobody knows what a column means."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from macaw.exceptions import MacawError
from macaw.store.schema import Scale

# frame and event both sit below window; anything finer than the target scale rolls up
_SCALE_RANK = {Scale.FRAME: 0, Scale.EVENT: 0, Scale.WINDOW: 1, Scale.RECORDING: 2}

DEFAULT_AGGREGATION = "mean"


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
    aggregation: str | None = None      # how to roll up to window/recording scale
    comparable_across_subjects: bool = True   # False for intensity; mic distance confounds it
    comparable_across_backends: bool = True   # False for AU; never merge across backends
    validation: str = "none"            # none | synthetic | against_reference | published
    primary: bool = False
    notes: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scale"] = Scale(self.scale).value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        known = {f.name for f in fields(cls)}
        unexpected = set(data) - known
        if unexpected:
            raise MacawError(f"Manifest entry has unknown keys: {', '.join(sorted(unexpected))}")
        return cls(**{**data, "scale": Scale(data["scale"])})


@dataclass
class FeatureManifest:
    specs: dict[str, FeatureSpec] = field(default_factory=dict)

    def add(self, spec: FeatureSpec) -> None:
        existing = self.specs.get(spec.name)
        # the same extractor rerun over another recording produces an identical spec; two
        # different extractors claiming one name is a bug that must not resolve silently
        if existing is not None and existing.to_dict() != spec.to_dict():
            raise MacawError(
                f"Feature '{spec.name}' is described differently by "
                f"{existing.extractor} and {spec.extractor}"
            )
        self.specs[spec.name] = spec

    def extend(self, specs: list[FeatureSpec]) -> None:
        for spec in specs:
            self.add(spec)

    def get(self, name: str) -> FeatureSpec:
        try:
            return self.specs[name]
        except KeyError:
            raise MacawError(f"No manifest entry for feature '{name}'") from None

    def by_group(self, group: str) -> list[FeatureSpec]:
        return sorted(
            (spec for spec in self.specs.values() if spec.group == group), key=lambda s: s.name
        )

    def aggregation_map(self, scale: Scale) -> dict[str, str]:
        """Feeds aggregate_to_window so each feature rolls up the way its spec says."""
        target = _SCALE_RANK[Scale(scale)]
        return {
            name: spec.aggregation or DEFAULT_AGGREGATION
            for name, spec in self.specs.items()
            if _SCALE_RANK[Scale(spec.scale)] < target
        }

    def mark_primary(self, names: list[str]) -> None:
        for name in names:
            self.get(name).primary = True


def read_manifest(path: Path) -> FeatureManifest:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FeatureManifest(
        specs={entry["name"]: FeatureSpec.from_dict(entry) for entry in payload["features"]}
    )


def write_manifest(manifest: FeatureManifest, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": [manifest.specs[name].to_dict() for name in sorted(manifest.specs)],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def merge_manifests(manifests: list[FeatureManifest]) -> FeatureManifest:
    merged = FeatureManifest()
    for manifest in manifests:
        merged.extend(list(manifest.specs.values()))
    return merged
