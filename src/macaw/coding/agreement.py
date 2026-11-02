"""Inter-annotator agreement. Layer B has no acceptance criterion without it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from macaw.exceptions import InputNotFoundError, MacawError
from macaw.project.layout import layout_for
from macaw.store.annotations import read_annotations
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, empty_frame

UNIT_KEY = ("recording_id", "subject_id", "tier", "t_start", "t_end")
NO_CODE = "_none"  # grid and coder modes: this annotator left the cell or unit blank
REST = "_rest"     # the other side of a one-vs-rest split

_UNIT_COLUMNS = ["unit_id", "annotator", "label", *UNIT_KEY]
_GROUP_KEY = ("recording_id", "subject_id", "tier")
_SMALL_SAMPLE = 30
_BLOCK = 500


def _empty_units() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in _UNIT_COLUMNS})


def build_units(annotations: pd.DataFrame, *, grid_s: float | None = None, units_of: str | None = None) -> pd.DataFrame:
    """One row per (unit, annotator). Without grid_s the interval itself is the unit, which
    holds whenever the coders worked from a shared list of candidates; with it the timeline is
    cut into fixed cells so freely drawn boundaries can still be compared; with units_of each of
    that coder's intervals is a unit, and the others give it the code covering most of it."""
    if grid_s is not None and units_of is not None:
        raise ValueError("compare on a grid or on one coder's units, not both")
    if annotations is None or annotations.empty:
        return _empty_units()

    frame = annotations
    if "abstained" in frame.columns:
        # an abstention is missing data, not a category: two models that both declined to answer
        # are not agreeing about anything
        frame = frame[~frame["abstained"].fillna(False).astype(bool)]
    frame = frame.dropna(subset=["label", "annotator"])
    if frame.empty:
        return _empty_units()

    if units_of is not None:
        units = _coder_units(frame, units_of)
    else:
        units = _exact_units(frame) if grid_s is None else _grid_units(frame, float(grid_s))
    duplicates = int(units.duplicated(subset=["unit_id", "annotator"]).sum())
    out = units.drop_duplicates(subset=["unit_id", "annotator"], keep="first").reset_index(drop=True)
    out.attrs["duplicate_ratings"] = duplicates
    out.attrs["grid_ties"] = units.attrs.get("grid_ties", 0)
    return out


def _exact_units(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in UNIT_KEY if column in frame.columns]
    keys = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        keys = keys + "|" + frame[column].astype(str)
    return pd.DataFrame(
        {
            "unit_id": keys.to_numpy(),
            "annotator": frame["annotator"].astype(str).to_numpy(),
            "label": frame["label"].astype(str).to_numpy(),
            **{column: frame[column].to_numpy(dtype=object) if column in frame.columns else None for column in UNIT_KEY},
        }
    )


def _grid_units(frame: pd.DataFrame, grid_s: float) -> pd.DataFrame:
    if grid_s <= 0:
        raise ValueError("grid_s must be positive")

    columns = [column for column in _GROUP_KEY if column in frame.columns]
    grouped = (
        frame.groupby(columns, dropna=False, sort=False, observed=True)
        if columns
        else [((), frame)]
    )
    rows: list[tuple] = []
    ties = 0
    for keys, group in grouped:
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = "|".join(str(key) for key in keys)
        where = _where(columns, keys)
        covered = _cell_coverage(group, grid_s)
        if not covered:
            continue
        present = sorted({annotator for _, annotator in covered})
        for cell in sorted({cell for cell, _ in covered}):
            span = (round(cell * grid_s, 6), round((cell + 1) * grid_s, 6))
            for annotator in present:
                choices = covered.get((cell, annotator))
                if not choices:
                    # a cell nobody covered is not a unit at all, but a cell one coder marked
                    # and another left blank is a real disagreement
                    rows.append((f"{prefix}|{cell}", annotator, NO_CODE, *where, *span))
                    continue
                top = max(choices.values())
                winners = sorted(label for label, seconds in choices.items() if seconds == top)
                if len(winners) > 1:
                    ties += 1
                rows.append((f"{prefix}|{cell}", annotator, winners[0], *where, *span))

    out = pd.DataFrame(rows, columns=_UNIT_COLUMNS)
    out.attrs["grid_ties"] = int(ties)
    return out


def _coder_units(frame: pd.DataFrame, coder: str) -> pd.DataFrame:
    """A coder who coded nothing for this person and tier gives no rating; one who coded elsewhere gives NO_CODE."""
    columns = [column for column in _GROUP_KEY if column in frame.columns]
    rows: list[tuple] = []
    ties = 0
    for keys, group in frame.groupby(columns, dropna=False, sort=False, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        prefix = "|" + "|".join(str(key) for key in keys)
        where = _where(columns, keys)
        names = group["annotator"].astype(str)
        own = group[names == coder]
        if own.empty:
            continue
        starts, ends = own["t_start"].to_numpy(dtype=float), own["t_end"].to_numpy(dtype=float)
        ids = [f"{prefix}|{start}|{end}" for start, end in zip(own["t_start"], own["t_end"])]
        spans = dict(zip(ids, zip(starts, ends)))
        rows += [(unit, coder, label, *where, *spans[unit]) for unit, label in zip(ids, own["label"].astype(str))]
        for other, theirs in group[names != coder].groupby(names[names != coder], sort=True):
            labels = theirs["label"].astype(str).to_numpy()
            # how long each of their annotations overlaps each unit, a block of units at a time
            for first in range(0, len(own), _BLOCK):
                block = slice(first, first + _BLOCK)
                overlap = np.clip(
                    np.minimum(ends[block, None], theirs["t_end"].to_numpy(dtype=float)[None, :])
                    - np.maximum(starts[block, None], theirs["t_start"].to_numpy(dtype=float)[None, :]),
                    0.0, None,
                )
                for unit, seconds in zip(ids[block], overlap):
                    if not (seconds > 0).any():
                        rows.append((unit, str(other), NO_CODE, *where, *spans[unit]))
                        continue
                    totals = pd.Series(seconds).groupby(labels).sum()
                    winners = sorted(totals.index[totals == totals.max()])
                    ties += len(winners) > 1
                    rows.append((unit, str(other), winners[0], *where, *spans[unit]))
    out = pd.DataFrame(rows, columns=_UNIT_COLUMNS)
    out.attrs["grid_ties"] = int(ties)
    return out


def _where(columns: list[str], keys: tuple) -> tuple:
    """Recording, person and tier of a group, None for a column the frame lacks."""
    values = dict(zip(columns, keys))
    return tuple(values.get(column) for column in _GROUP_KEY)


def _cell_coverage(group: pd.DataFrame, grid_s: float) -> dict[tuple[int, str], dict[str, float]]:
    starts = group["t_start"].to_numpy(dtype=float)
    ends = group["t_end"].to_numpy(dtype=float)
    labels = group["label"].astype(str).to_numpy()
    annotators = group["annotator"].astype(str).to_numpy()

    covered: dict[tuple[int, str], dict[str, float]] = {}
    for start, end, label, annotator in zip(starts, ends, labels, annotators):
        if not (np.isfinite(start) and np.isfinite(end)) or end <= start:
            continue
        for cell in range(int(np.floor(start / grid_s)), int(np.ceil(end / grid_s))):
            low = max(start, cell * grid_s)
            high = min(end, (cell + 1) * grid_s)
            if high <= low:
                continue
            bucket = covered.setdefault((cell, annotator), {})
            bucket[label] = bucket.get(label, 0.0) + (high - low)
    return covered


def _wide(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    return units.pivot(index="unit_id", columns="annotator", values="label")


def krippendorff_alpha(units: pd.DataFrame) -> float:
    """Nominal alpha. Missing ratings are fine; a unit rated once cannot be paired and drops out."""
    if units is None or units.empty:
        return float("nan")

    coincidence: dict[tuple[str, str], float] = {}
    for _, group in units.groupby("unit_id", sort=False):
        values = group["label"].astype(str).to_numpy()
        pairable = values.size - 1
        if pairable < 1:
            continue
        for left in values:
            for right in values:
                pair = (left, right)
                coincidence[pair] = coincidence.get(pair, 0.0) + 1.0 / pairable
        for value in values:
            coincidence[(value, value)] -= 1.0 / pairable  # a value never pairs with itself

    if not coincidence:
        return float("nan")

    labels = sorted({value for pair in coincidence for value in pair})
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for (left, right), value in coincidence.items():
        matrix[index[left], index[right]] += value

    marginals = matrix.sum(axis=1)
    total = float(marginals.sum())
    expected = total**2 - float((marginals**2).sum())
    if total <= 1.0 or expected <= 0.0:
        return float("nan")
    disagreement = total - float(np.trace(matrix))
    return float(1.0 - (total - 1.0) * disagreement / expected)


def _kappa(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if left.size == 0:
        return float("nan"), float("nan")
    observed = float((left == right).mean())
    expected = sum(
        float((left == label).mean()) * float((right == label).mean())
        for label in sorted(set(left.tolist()) | set(right.tolist()))
    )
    if expected >= 1.0:
        return observed, float("nan")
    return observed, float((observed - expected) / (1.0 - expected))


def _paired(units: pd.DataFrame, annotator_a: str, annotator_b: str) -> tuple[np.ndarray, np.ndarray]:
    wide = _wide(units)
    if annotator_a not in wide.columns or annotator_b not in wide.columns:
        return np.empty(0, dtype=object), np.empty(0, dtype=object)
    both = wide.loc[:, [annotator_a, annotator_b]].dropna()
    return (
        both[annotator_a].astype(str).to_numpy(),
        both[annotator_b].astype(str).to_numpy(),
    )


def cohen_kappa(units: pd.DataFrame, annotator_a: str, annotator_b: str) -> float:
    if units is None or units.empty or annotator_a == annotator_b:
        return float("nan")
    return _kappa(*_paired(units, annotator_a, annotator_b))[1]


def pairwise_kappa(units: pd.DataFrame) -> pd.DataFrame:
    columns = ["annotator_a", "annotator_b", "n_units", "observed_agreement", "kappa"]
    if units is None or units.empty:
        return pd.DataFrame(columns=columns)

    names = sorted(units["annotator"].astype(str).unique().tolist())
    rows = []
    for position, left in enumerate(names):
        for right in names[position + 1 :]:
            values_left, values_right = _paired(units, left, right)
            observed, kappa = _kappa(values_left, values_right)
            rows.append(
                {
                    "annotator_a": left,
                    "annotator_b": right,
                    "n_units": int(values_left.size),
                    "observed_agreement": observed,
                    "kappa": kappa,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _codes(units: pd.DataFrame) -> list[str]:
    return sorted(label for label in units["label"].astype(str).unique().tolist() if label != NO_CODE)


def per_code_alpha(units: pd.DataFrame) -> pd.DataFrame:
    """One code against everything else, all coders together; an overall coefficient hides which code disagrees."""
    columns = ["code", "n_units", "n_ratings", "alpha"]
    if units is None or units.empty:
        return pd.DataFrame(columns=columns)

    labels = units["label"].astype(str)
    rows = []
    for code in _codes(units):
        hit = labels.to_numpy() == code
        binary = pd.DataFrame(
            {
                "unit_id": units["unit_id"].to_numpy(),
                "annotator": units["annotator"].astype(str).to_numpy(),
                "label": np.where(hit, code, REST),
            }
        )
        rows.append(
            {
                "code": code,
                "n_units": int(units.loc[hit, "unit_id"].nunique()),
                "n_ratings": int(hit.sum()),
                "alpha": krippendorff_alpha(binary),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def per_code_pairs(units: pd.DataFrame) -> pd.DataFrame:
    """One code against the rest for every pair of coders; n_a and n_b count it on the units both rated."""
    columns = ["code", "annotator_a", "annotator_b", "n_units", "n_a", "n_b", "kappa"]
    if units is None or units.empty:
        return pd.DataFrame(columns=columns)

    wide = _wide(units)
    names = sorted(str(name) for name in wide.columns)
    pairs = {
        (left, right): wide.loc[:, [left, right]].dropna().astype(str)
        for position, left in enumerate(names)
        for right in names[position + 1 :]
    }
    rows = []
    for code in _codes(units):
        for (left, right), both in pairs.items():
            hit_a, hit_b = both[left].to_numpy() == code, both[right].to_numpy() == code
            rows.append(
                {
                    "code": code,
                    "annotator_a": left,
                    "annotator_b": right,
                    "n_units": len(both),
                    "n_a": int(hit_a.sum()),
                    "n_b": int(hit_b.sum()),
                    "kappa": _kappa(np.where(hit_a, code, REST), np.where(hit_b, code, REST))[1],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def unit_table(units: pd.DataFrame) -> pd.DataFrame:
    """A row per unit, a column per coder: blank for no rating, NO_CODE where they coded around the unit."""
    if units is None or units.empty:
        return pd.DataFrame(columns=list(UNIT_KEY))
    where = units.drop_duplicates("unit_id").set_index("unit_id").reindex(columns=list(UNIT_KEY))
    where = where.astype({"t_start": float, "t_end": float})
    table = where.join(_wide(units)).sort_values(list(UNIT_KEY), kind="stable", na_position="first")
    table.columns.name = None
    return table.reset_index(drop=True)


def _abstention_rate(annotations: pd.DataFrame) -> dict[str, float]:
    if annotations.empty or "abstained" not in annotations.columns:
        return {}
    flags = annotations["abstained"].fillna(False).astype(bool)
    grouped = flags.groupby(annotations["annotator"].astype(str), sort=True).mean()
    return {str(name): float(value) for name, value in grouped.items()}


def agreement_report(
    annotations: pd.DataFrame,
    *,
    tier: str | None = None,
    layer: str | None = None,
    grid_s: float | None = None,
    units_of: str | None = None,
) -> dict:
    """Everything a methods section needs about one tier. n_units is reported rather than turned
    into a pass/fail: how much agreement is enough is the analyst's call."""
    frame = annotations if annotations is not None else empty_frame(ANNOTATION_SCHEMA)
    if tier is not None and not frame.empty:
        frame = frame[frame["tier"].astype(str) == tier]
    if layer is not None and not frame.empty:
        frame = frame[frame["layer"].astype(str) == layer]
    if units_of is not None and units_of not in set(frame["annotator"].dropna().astype(str)):
        coders = ", ".join(sorted(frame["annotator"].dropna().astype(str).unique())) or "none"
        raise MacawError(f"no coder named {units_of} here; the coders are: {coders}")

    units = build_units(frame, grid_s=grid_s, units_of=units_of)
    names = sorted(units["annotator"].astype(str).unique().tolist()) if not units.empty else []
    n_units = int(units["unit_id"].nunique()) if not units.empty else 0
    sizes = units.groupby("unit_id", sort=False).size() if not units.empty else pd.Series(dtype=int)
    wide = _wide(units)
    complete = int(wide.notna().all(axis=1).sum()) if not wide.empty else 0

    report = {
        "tier": tier,
        "layer": layer,
        "unit_mode": "coder" if units_of is not None else "exact" if grid_s is None else "grid",
        "grid_s": grid_s,
        "units_of": units_of,
        "n_units": n_units,
        # the N alpha actually ran on: a unit only one coder touched carries no agreement
        "n_pairable": int((sizes >= 2).sum()),
        "n_annotators": len(names),
        "annotators": names,
        "alpha": krippendorff_alpha(units),
        "pairwise_kappa": pairwise_kappa(units),
        "per_code": per_code_alpha(units),
        "per_code_pairs": per_code_pairs(units),
        "abstention_rate": _abstention_rate(frame),
        "coverage": (complete / n_units) if n_units else float("nan"),
        "warnings": [],
        "units": unit_table(units),
    }
    report["warnings"] = _report_warnings(report, units)
    return report


def _report_warnings(report: dict, units: pd.DataFrame) -> list[str]:
    warnings = []
    if report["n_annotators"] < 2:
        warnings.append(
            f"{report['n_annotators']} annotator(s) present, so there is nothing to compare. "
            "Agreement needs the same material coded twice."
        )
    if report["n_pairable"] == 0 and report["n_annotators"] >= 2:
        warnings.append(
            "No unit was rated twice. In exact mode the coders have to work from the same "
            "interval boundaries; pass grid_s or units_of to compare freely drawn annotations instead."
        )
    elif 0 < report["n_pairable"] < _SMALL_SAMPLE:
        warnings.append(
            f"Only {report['n_pairable']} pairable units. Both coefficients are unstable this "
            "far down; the value is reported as is rather than withheld."
        )
    coverage = report["coverage"]
    if np.isfinite(coverage) and coverage < 1.0:
        warnings.append(
            f"{coverage:.0%} of units were rated by every annotator; the rest are treated as "
            "missing, which alpha handles and kappa does not."
        )
    duplicates = units.attrs.get("duplicate_ratings", 0)
    if duplicates:
        warnings.append(
            f"{duplicates} rating(s) dropped: one annotator gave a unit more than one label. "
            "Agreement assumes a single label per unit per coder."
        )
    ties = units.attrs.get("grid_ties", 0)
    if ties and report["unit_mode"] == "coder":
        warnings.append(
            f"{ties} unit(s) were covered evenly by two of a coder's labels; the first in alphabetical order was taken."
        )
    elif ties:
        warnings.append(
            f"{ties} grid cell(s) were split evenly between labels; the first in alphabetical "
            "order was taken. A finer grid_s removes the ambiguity."
        )
    return warnings


def format_report(report: dict) -> str:
    lines = [
        f"tier={report['tier'] or 'all'}  layer={report['layer'] or 'all'}  "
        f"units={report['unit_mode']}" + (f"({report['grid_s']}s)" if report["grid_s"] else "")
        + (f"({report['units_of']})" if report.get("units_of") else ""),
        f"n_units={report['n_units']}  n_pairable={report['n_pairable']}  "
        f"n_annotators={report['n_annotators']}  coverage={report['coverage']:.2f}",
        f"krippendorff_alpha={report['alpha']:.3f}",
    ]
    if not report["pairwise_kappa"].empty:
        lines.append("")
        lines.append(report["pairwise_kappa"].to_string(index=False))
    for table in ("per_code", "per_code_pairs"):
        if not report[table].empty:
            lines.append("")
            lines.append(report[table].to_string(index=False))
    if report["abstention_rate"]:
        lines.append("")
        lines.append(
            "abstention: "
            + ", ".join(f"{name}={rate:.2f}" for name, rate in report["abstention_rate"].items())
        )
    for warning in report["warnings"]:
        lines.append("")
        lines.append(f"! {warning}")
    return "\n".join(lines)


def report_to_json(report: dict) -> dict:
    """Frames flattened and every non-finite value turned into null, so a strict parser can read
    the file back."""
    out = {
        key: (None if isinstance(value, float) and not np.isfinite(value) else value)
        for key, value in report.items()
        if key != "units"
    }
    for table in ("pairwise_kappa", "per_code", "per_code_pairs"):
        out[table] = json.loads(report[table].to_json(orient="records"))
    return out


def add_words(config, table: pd.DataFrame) -> pd.DataFrame:
    """The unit table with what its person said inside each unit and the line before it, from the transcript,
    so the units people disagree on can be read where they are listed. Blank where there is no transcript."""
    from macaw.extractors.audio.content.asr import load_transcript

    words, before = [""] * len(table), [""] * len(table)
    for rid, rows in table.groupby("recording_id", sort=False):
        try:
            lines = load_transcript(config, str(rid), others=True).sort_values("t_start", kind="stable").reset_index(drop=True)
        except (MacawError, OSError):   # coded but never transcribed, such as gaze in a silent task
            continue
        middle = ((lines["t_start"] + lines["t_end"]) / 2).to_numpy()
        for position, row in zip(rows.index, rows.itertuples(index=False)):
            mine = (lines["subject_id"] == str(row.subject_id)).to_numpy() if pd.notna(row.subject_id) else np.ones(len(lines), bool)
            inside = np.flatnonzero(mine & (middle >= row.t_start) & (middle <= row.t_end))
            words[table.index.get_loc(position)] = " ".join(lines["text"].iloc[inside].astype(str))
            opens = lines["t_start"].iloc[inside[0]] if inside.size else row.t_start
            earlier = lines[lines["t_start"] < opens - 1e-9]
            if not earlier.empty:
                before[table.index.get_loc(position)] = f"{earlier.iloc[-1]['subject_id']}: {earlier.iloc[-1]['text']}"
    at = list(table.columns).index("t_end") + 1
    out = table.copy()
    out.insert(at, "before", before)
    out.insert(at, "words", words)
    return out


def write_report(report: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report_to_json(report), indent=2), encoding="utf-8")
    report["units"].to_csv(folder / "units.csv", index=False)


def load_annotations(config, recording_ids: Sequence[str]) -> pd.DataFrame:
    """Where the agreement command starts, the way turn extractors start at load_turns.

    Reliability is usually measured on a sample, so a recording nobody has coded yet is skipped
    rather than fatal; a request where none of them exists is still an error."""
    layout = layout_for(config, create=False)
    paths = [layout.annotations_path(recording_id) for recording_id in recording_ids]
    present = [path for path in paths if path.exists()]
    if paths and not present:
        raise InputNotFoundError(
            f"none of the {len(paths)} requested recordings has an annotations directory: "
            f"{paths[0].parent.parent}"
        )

    frames = [frame for frame in (read_annotations(path) for path in present) if not frame.empty]
    if not frames:
        return empty_frame(ANNOTATION_SCHEMA)
    return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")
