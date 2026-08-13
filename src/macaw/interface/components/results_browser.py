from __future__ import annotations

from macaw.config.schema import MacawConfig


def layout(config: MacawConfig) -> object:
    raise NotImplementedError


def feature_picker(manifest_names: list[str]) -> object:
    raise NotImplementedError


def timeseries_plot(t: list[float], y: list[float], label: str) -> object:
    raise NotImplementedError


def manifest_table(rows: list[dict]) -> object:
    raise NotImplementedError
