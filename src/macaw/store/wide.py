"""Export for downstream statistics users (R, MATLAB)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.store.schema import Scale


def to_wide(
    df: pd.DataFrame,
    *,
    index: list[str] | None = None,
    scale: Scale | None = None,
) -> pd.DataFrame:
    """Pivot long rows to one column per feature."""
    raise NotImplementedError


def export_csv(df: pd.DataFrame, path: Path, *, wide: bool = True) -> Path:
    raise NotImplementedError


def export_rmd_template(out_dir: Path, csv_name: str, manifest_name: str) -> Path:
    """Emit a runnable .Rmd next to the CSV so R users have a starting point."""
    raise NotImplementedError
