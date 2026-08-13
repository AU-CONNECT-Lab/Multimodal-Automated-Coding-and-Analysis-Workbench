"""Dash app. Batch work goes through cli_bridge; interactive editors use the store layer.
No pipeline logic lives in this package."""

from __future__ import annotations

from pathlib import Path

from dash import Dash

from macaw.config.schema import MacawConfig


def create_app(config: MacawConfig, config_path: Path) -> Dash:
    raise NotImplementedError


def build_layout(config: MacawConfig) -> object:
    raise NotImplementedError


def register_callbacks(app: Dash, config: MacawConfig, config_path: Path) -> None:
    raise NotImplementedError
