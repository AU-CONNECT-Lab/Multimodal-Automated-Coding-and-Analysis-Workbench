from __future__ import annotations

from pathlib import Path

from dash import Dash

from macaw.config.schema import MacawConfig


def register(app: Dash, config: MacawConfig, config_path: Path) -> None:
    """Run/QC/code buttons build a `macaw ...` argv and launch it -- see cli_bridge."""
    raise NotImplementedError
