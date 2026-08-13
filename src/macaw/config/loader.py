"""Three-layer merge: packaged defaults -> project.yaml -> session override -> CLI --set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from macaw.config.schema import MacawConfig

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
RESOLVED_FILENAME = "resolved_config.yaml"


def load_defaults() -> dict[str, Any]:
    raise NotImplementedError


def load_yaml(path: Path) -> dict[str, Any]:
    raise NotImplementedError


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def parse_cli_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """--set prosody.f0_max_hz=450 into a nested dict."""
    raise NotImplementedError


def resolve_config(
    project_yaml: Path,
    *,
    session_override: Path | None = None,
    cli_overrides: Sequence[str] = (),
) -> MacawConfig:
    """Merge all layers and validate. Unknown keys raise UnknownConfigKeyError."""
    raise NotImplementedError


def load_config(path: Path) -> MacawConfig:
    """Load an already-resolved config (e.g. from an output directory) without merging."""
    raise NotImplementedError


def archive_resolved(config: MacawConfig, output_dir: Path) -> Path:
    """Reproducibility depends entirely on this file existing."""
    raise NotImplementedError


def scaffold_project_config(dest: Path, *, force: bool = False) -> Path:
    raise NotImplementedError
