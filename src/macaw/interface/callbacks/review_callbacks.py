from __future__ import annotations

from pathlib import Path

from dash import Dash

from macaw.config.schema import MacawConfig


def register(app: Dash, config: MacawConfig, config_path: Path) -> None:
    """Interactive coding: writes decisions straight to the annotation store, no CLI involved."""
    raise NotImplementedError
