"""A person's coding as it stands in the file they made: one row per annotation, with the tier and person the file names."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from macaw.exceptions import EafError, InputNotFoundError
from macaw.interop.eaf import _elan

SUFFIXES = (".eaf", ".csv", ".tsv", ".xlsx")
RAW_COLUMNS = ("file_tier", "participant", "annotator", "linguistic_type", "t_start", "t_end", "label")
TABLE_COLUMNS = ("tier", "start", "end", "code")
_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$")


def read_coding_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"coding file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".eaf":
        return _read_eaf(path)
    if suffix in (".csv", ".tsv", ".xlsx"):
        return _read_table(path, suffix)
    raise EafError(f"{path.name}: coding is read from {', '.join(SUFFIXES)} files")


def _read_eaf(path: Path) -> pd.DataFrame:
    Eaf = _elan()
    try:
        eaf = Eaf(str(path))
    except Exception as error:
        raise EafError(f"{path}: could not be read as ELAN ({error})") from error
    rows = []
    for name in eaf.get_tier_names():
        attributes = eaf.tiers[name][2] if name in eaf.tiers else {}
        for annotation in eaf.get_annotation_data_for_tier(name):
            rows.append(
                {
                    "file_tier": name,
                    "participant": attributes.get("PARTICIPANT") or None,
                    "annotator": attributes.get("ANNOTATOR") or None,
                    "linguistic_type": attributes.get("LINGUISTIC_TYPE_REF") or None,
                    "t_start": annotation[0] / 1000.0,
                    "t_end": annotation[1] / 1000.0,
                    "label": str(annotation[2] or "").strip(),
                }
            )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


def _read_table(path: Path, suffix: str) -> pd.DataFrame:
    if suffix == ".xlsx":
        table = pd.read_excel(path, dtype=str)
    else:
        table = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False)
    table.columns = [str(column).strip().lower() for column in table.columns]
    missing = [column for column in TABLE_COLUMNS if column not in table.columns]
    if missing:
        raise EafError(
            f"{path.name}: needs the columns {', '.join(TABLE_COLUMNS)} (person is optional); missing {', '.join(missing)}"
        )
    starts, ends = table["start"].map(seconds), table["end"].map(seconds)
    unread = table.index[starts.isna() | ends.isna()]
    if len(unread):
        rows = ", ".join(str(index + 2) for index in unread[:5])
        raise EafError(f"{path.name}: start and end must be seconds or a clock time such as 1:02.5; not on row(s) {rows}")
    person = table["person"] if "person" in table.columns else pd.Series(None, index=table.index, dtype=object)
    return pd.DataFrame(
        {
            "file_tier": table["tier"].fillna("").astype(str).str.strip(),
            "participant": person.map(lambda value: str(value).strip() if isinstance(value, str) and value.strip() else None),
            "annotator": None,
            "linguistic_type": None,
            "t_start": starts.astype(float),
            "t_end": ends.astype(float),
            "label": table["code"].fillna("").astype(str).str.strip(),
        },
        columns=list(RAW_COLUMNS),
    )


def seconds(value) -> float | None:
    """A number of seconds, or a clock time h:mm:ss.s or m:ss.s."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = _CLOCK.match(text)
    if match is None:
        return None
    hours, minutes, rest = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + float(rest)
