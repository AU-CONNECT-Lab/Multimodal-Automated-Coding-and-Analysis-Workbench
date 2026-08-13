"""Offset review. The waveform correlation plot is the evidence; video is human sanity check."""

from __future__ import annotations

from macaw.config.schema import MacawConfig


def layout(config: MacawConfig) -> object:
    raise NotImplementedError


def correlation_plot(lags: list[float], values: list[float]) -> object:
    raise NotImplementedError


def waveform_overlay(t: list[float], a: list[float], b: list[float], offset_s: float) -> object:
    """Downsampled before plotting; raw audio at 100 Hz over an hour stalls the browser."""
    raise NotImplementedError


def offset_table(rows: list[dict]) -> object:
    raise NotImplementedError
