"""Clock drift: cheap recorders run slightly fast or slow, so a first-frame offset walks off."""

from __future__ import annotations

from pathlib import Path

from macaw.config.schema import AlignSection


def segment_offsets(
    reference: Path,
    target: Path,
    *,
    segment_s: float,
    max_lag_s: float,
) -> list[tuple[float, float]]:
    """(segment_midpoint_s, offset_s) pairs."""
    raise NotImplementedError


def estimate_drift(offsets: list[tuple[float, float]]) -> tuple[float, float]:
    """Linear fit; returns (drift_ppm, r_squared). High r2 with nonzero slope means real drift."""
    raise NotImplementedError


def check_drift(reference: Path, target: Path, cfg: AlignSection) -> tuple[float, float, bool]:
    raise NotImplementedError
