"""How each code was used, per recording, annotator, tier and person; nothing is pooled across recordings."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEQUENCE_KEYS = ["recording_id", "annotator", "tier", "subject_id"]
COLUMNS = [
    *SEQUENCE_KEYS, "label", "n", "time_s", "time_share", "runs", "mean_run_units", "single_unit_share",
    "sequence_units", "sequence_transitions", "sequence_abstained_share",
]


def describe_codes(annotations: pd.DataFrame) -> pd.DataFrame:
    """One row per code in each sequence of units, the units of one annotator on one tier for one person
    in the order coded."""
    if annotations.empty:
        return pd.DataFrame(columns=COLUMNS)
    # a recording-level tier has no person, and groupby would drop it
    frame = annotations.assign(subject_id=annotations["subject_id"].astype("string").fillna(""))
    rows = []
    for keys, sequence in frame.groupby(SEQUENCE_KEYS, sort=True, observed=True):
        rows.extend(_describe_sequence(dict(zip(SEQUENCE_KEYS, keys)), sequence))
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["subject_id"] = out["subject_id"].replace("", pd.NA)
    return out


def _describe_sequence(keys: dict, sequence: pd.DataFrame) -> list[dict]:
    ordered = sequence.sort_values(["t_start", "t_end"], kind="stable")
    abstained = ordered["abstained"].fillna(False).astype(bool).to_numpy()
    valid = ordered["label"].notna().to_numpy() & ~abstained
    labels = ordered["label"].fillna("").astype(str).to_numpy()
    durations = (ordered["t_end"] - ordered["t_start"]).to_numpy(dtype=float)

    # a run is one code over consecutive units; a unit nobody could code ends it
    runs: list[tuple[str, int]] = []
    previous = None
    for label, ok in zip(labels, valid):
        if ok and label == previous:
            runs[-1] = (label, runs[-1][1] + 1)
        elif ok:
            runs.append((label, 1))
        previous = label if ok else None
    switches = int((valid[:-1] & valid[1:] & (labels[:-1] != labels[1:])).sum())
    coded_s = float(durations[valid].sum())

    rows = []
    for label in sorted({label for label, _ in runs}):
        own = valid & (labels == label)
        lengths = np.array([length for code, length in runs if code == label])
        rows.append(
            {
                **keys,
                "label": label,
                "n": int(own.sum()),
                "time_s": float(durations[own].sum()),
                "time_share": float(durations[own].sum()) / coded_s if coded_s > 0 else np.nan,
                "runs": int(lengths.size),
                "mean_run_units": float(lengths.mean()),
                "single_unit_share": float((lengths == 1).mean()),
                "sequence_units": int(len(labels)),
                "sequence_transitions": switches,
                "sequence_abstained_share": float(1.0 - valid.mean()),
            }
        )
    return rows


def format_description(table: pd.DataFrame) -> str:
    if table.empty:
        return "no codes found"
    shown = table.drop(columns=["sequence_abstained_share"]).copy()
    for column in ("time_s", "mean_run_units"):
        shown[column] = shown[column].map(lambda value: f"{value:.1f}")
    for column in ("time_share", "single_unit_share"):
        shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.0%}")
    shown["subject_id"] = shown["subject_id"].fillna("")
    return shown.to_string(index=False)
