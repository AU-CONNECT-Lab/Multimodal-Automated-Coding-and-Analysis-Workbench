"""Tables and files people fill in or edit by hand, often in a spreadsheet program or ELAN."""

from __future__ import annotations

import re
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


def set_aside(path: Path) -> Path | None:
    """A file about to be written again, moved beside it under the time it is set aside, as a person may have been editing it."""
    from datetime import datetime

    path = Path(path)
    if not path.is_file():
        return None
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    kept = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    count = 1
    while kept.exists():   # set aside twice in one second: the first one kept stays
        count += 1
        kept = path.with_name(f"{path.stem}.{stamp}-{count}{path.suffix}")
    path.replace(kept)
    return kept


_SET_ASIDE = re.compile(r"\.\d{8}-\d{6}(-\d+)?$")


def is_set_aside(path: Path) -> bool:
    """A file set_aside kept, which reading a folder of people's files leaves out."""
    return bool(_SET_ASIDE.search(Path(path).stem))


def said_aside(kept: Path | None) -> str:
    return f"; the earlier one kept as {kept.name}" if kept is not None else ""


def write_csv(table: pd.DataFrame, path: Path) -> None:
    # the byte-order mark makes a spreadsheet program open letters such as é as they are
    table.to_csv(path, index=False, encoding="utf-8-sig")
