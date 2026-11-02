"""Units people coded, drawn into a table the language model is shown as examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.exceptions import CodebookError, InputNotFoundError

COLUMNS = ("code", "speaker", "text", "before", "why", "recording_id", "subject_id", "t_start", "t_end")
_SUFFIXES = (".csv", ".tsv", ".xlsx")


@dataclass
class DrawReport:
    # code -> (drawn, available)
    counts: dict[str, tuple[int, int]] = field(default_factory=dict)
    no_words: int = 0
    not_in_codebook: dict[str, int] = field(default_factory=dict)


def read_examples(path: Path, codebook: Codebook) -> pd.DataFrame:
    """The table as people left it: code and text are needed, the other columns may be blank or missing."""
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"examples file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        raise CodebookError(f"{path.name}: examples are read from {', '.join(_SUFFIXES)} files")
    if suffix == ".xlsx":
        table = pd.read_excel(path, dtype=str).fillna("")
    else:
        table = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False)
    table.columns = [str(column).strip().lower() for column in table.columns]
    missing = [column for column in ("code", "text") if column not in table.columns]
    if missing:
        raise CodebookError(f"{path.name}: no {' or '.join(missing)} column")
    for column in COLUMNS:
        if column not in table.columns:
            table[column] = ""
    table = table[list(COLUMNS)].astype(str).apply(lambda column: column.str.strip())
    table = table[table["text"] != ""].reset_index(drop=True)
    unknown = sorted(set(table["code"]) - set(codebook.labels()))
    if unknown:
        raise CodebookError(f"{path.name}: {', '.join(unknown)} not in codebook '{codebook.name}'")
    return table


def draw_examples(
    coded: pd.DataFrame, lines: dict[str, pd.DataFrame], codebook: Codebook, *, per_code: int, seed: int = 0
) -> tuple[pd.DataFrame, DrawReport]:
    """per_code units of each code at random from one person's annotations. coded holds their rows;
    lines maps each recording to its transcript lines as coding_units gives them."""
    report = DrawReport()
    found = []
    for row in coded.itertuples(index=False):
        if row.label not in codebook.labels():
            report.not_in_codebook[row.label] = report.not_in_codebook.get(row.label, 0) + 1
            continue
        said = _words(lines.get(str(row.recording_id)), row.subject_id, float(row.t_start), float(row.t_end))
        if said is None:
            report.no_words += 1
            continue
        found.append({"code": row.label, **said, "why": "", "recording_id": str(row.recording_id),
                      "subject_id": None if pd.isna(row.subject_id) else str(row.subject_id),
                      "t_start": float(row.t_start), "t_end": float(row.t_end)})
    pool = pd.DataFrame(found, columns=list(COLUMNS))
    rng = np.random.default_rng(seed)
    drawn = []
    for code in codebook.labels():
        mine = pool[pool["code"] == code]
        take = min(per_code, len(mine))
        report.counts[code] = (take, len(mine))
        if take:
            drawn.append(mine.iloc[np.sort(rng.choice(len(mine), size=take, replace=False))])
    table = pd.concat(drawn, ignore_index=True) if drawn else pool
    return table, report


def _words(lines: pd.DataFrame | None, subject_id, t_start: float, t_end: float) -> dict | None:
    """The person's lines whose middle falls inside the unit, else the one overlapping it most, and the line before."""
    if lines is None or lines.empty:
        return None
    own = lines if pd.isna(subject_id) else lines[lines["subject_id"] == str(subject_id)]
    middle = (own["t_start"] + own["t_end"]) / 2
    inside = own[(middle >= t_start) & (middle <= t_end)]
    if inside.empty:
        overlap = np.minimum(own["t_end"], t_end) - np.maximum(own["t_start"], t_start)
        if own.empty or overlap.max() <= 0:
            return None
        inside = own.loc[[overlap.idxmax()]]
    first = inside.index[0]
    earlier = lines.loc[: first - 1] if first > lines.index[0] else lines.iloc[0:0]
    before = f"{earlier.iloc[-1]['speaker']}: {earlier.iloc[-1]['text']}" if not earlier.empty else ""
    return {"speaker": str(inside.iloc[0]["speaker"]), "text": " ".join(inside["text"].astype(str)), "before": before}
