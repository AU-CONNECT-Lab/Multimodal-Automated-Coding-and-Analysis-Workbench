from __future__ import annotations

from pathlib import Path

from dash import Dash

from macaw.config.schema import MacawConfig


def register(app: Dash, config: MacawConfig, config_path: Path) -> None:
    """Estimation runs via `macaw align`; manual nudging edits the offset table in place."""
    raise NotImplementedError
