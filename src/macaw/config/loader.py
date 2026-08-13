"""Layered merge: packaged defaults, then project.yaml, then sessions/<id>.yaml, then --set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from macaw.config.schema import MacawConfig

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
RESOLVED_FILENAME = "resolved_config.yaml"


@dataclass(frozen=True)
class KeySource:
    """Where a value came from, so an error can point at a file and a line."""

    layer: str                  # defaults | project | session | cli
    path: Path | None = None
    line: int | None = None


def load_defaults() -> dict[str, Any]:
    raise NotImplementedError


def load_yaml(path: Path) -> tuple[dict[str, Any], dict[str, KeySource]]:
    """Values plus a dotted-key to source map built from the YAML node line numbers."""
    raise NotImplementedError


def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
    *,
    prefix: str = "",
    sources: dict[str, KeySource] | None = None,
) -> dict[str, Any]:
    """Dicts merge key by key; lists replace wholesale. Write an explicit null to clear."""
    raise NotImplementedError


def parse_cli_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Each value is parsed as YAML, so 2.5 is a float and [a,b] is a list."""
    raise NotImplementedError


def check_session_overridable(data: dict[str, Any], path: Path) -> None:
    """Raises when a session file touches a section outside SESSION_OVERRIDABLE."""
    raise NotImplementedError


def suggest_key(unknown: str, candidates: Sequence[str]) -> str | None:
    raise NotImplementedError


def describe_unknown_key(key: str, sources: dict[str, KeySource], candidates: Sequence[str]) -> str:
    """Builds the 'project.yaml:14: unknown key ... did you mean ...' message."""
    raise NotImplementedError


def resolve_config(
    project_yaml: Path,
    *,
    session_id: str | None = None,
    cli_overrides: Sequence[str] = (),
) -> MacawConfig:
    """Without session_id this is the project baseline; with one it is what that session runs."""
    raise NotImplementedError


def load_config(path: Path) -> MacawConfig:
    """Load an already-resolved config from an output directory, without merging anything."""
    raise NotImplementedError


def session_override_path(project_yaml: Path, session_id: str) -> Path | None:
    raise NotImplementedError


def archive_resolved(config: MacawConfig, output_dir: Path) -> Path:
    """Reproducibility depends entirely on this file existing."""
    raise NotImplementedError


def archive_session_resolved(config: MacawConfig, output_dir: Path, session_id: str) -> Path:
    """The per-session copy is the one that explains a given session's numbers."""
    raise NotImplementedError


def scaffold_project_config(dest: Path, *, force: bool = False) -> Path:
    raise NotImplementedError
