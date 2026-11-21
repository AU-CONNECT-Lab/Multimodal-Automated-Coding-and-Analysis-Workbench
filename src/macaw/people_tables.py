"""Tables people fill in by hand, often in a spreadsheet program."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.exceptions import ConfigError


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        raise ConfigError(
            f"{Path(path).name} is not saved as UTF-8, as a spreadsheet program's plain CSV is on many computers. "
            "Save it again as CSV UTF-8 (in Excel: Save As, CSV UTF-8 (Comma delimited))."
        ) from None


def write_csv(table: pd.DataFrame, path: Path) -> None:
    # the byte-order mark makes a spreadsheet program open letters such as é as they are
    table.to_csv(path, index=False, encoding="utf-8-sig")
