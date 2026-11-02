"""Partial automation plus human review. Full automation is a fantasy."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import InputNotFoundError
from macaw.store import _io
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, empty_frame

NAME = "review queue"

ABSTAINED = "abstained"
LOW_CONFIDENCE = "low_confidence"
FLAGGED = "flagged"
# a unit the model was sure of, drawn at random: the only way to learn how often sure answers are wrong
RANDOM_CHECK = "random_check"

# what a decision points back at. The queue is a file that outlives the frame it came from, so
# it keys on the row's own content rather than on a position.
MATCH_KEY = ("recording_id", "subject_id", "tier", "t_start", "t_end", "annotator", "label")

_CARRIED = (
    "recording_id",
    "group_id",
    "session_id",
    "task",
    "run",
    "subject_id",
    "tier",
    "t_start",
    "t_end",
    "label",
    "annotator",
    "layer",
    "confidence",
    "abstained",
)

REVIEW_QUEUE_SCHEMA: dict[str, str] = {
    **{column: ANNOTATION_SCHEMA[column] for column in _CARRIED},
    "reason": "string",        # abstained | low_confidence | flagged
    "decision": "string",      # the reviewer's label; empty until somebody decides
    "reviewer": "string",
}

# an annotation is identified again after a parquet or csv round trip, so the time key is
# rounded rather than compared as a raw float
_TIME_FORMAT = "{:.6f}"


def _keys(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in MATCH_KEY:
        values = frame[column]
        if column in ("t_start", "t_end"):
            parts.append(pd.to_numeric(values, errors="coerce").map(_TIME_FORMAT.format))
        else:
            parts.append(values.astype("string").fillna(""))
    joined = parts[0].astype(str)
    for part in parts[1:]:
        joined = joined + "|" + part.astype(str)
    return joined


def _reviewer_name(value) -> str:
    # the annotator column is what agreement groups on, so an unnamed reviewer still needs a name
    return "human" if value is None or pd.isna(value) or not str(value).strip() else str(value)


def _decided(frame: pd.DataFrame) -> pd.Series:
    values = frame["decision"]
    return values.notna() & (values.astype("string").fillna("").str.strip() != "")


def build_queue(annotations: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Low confidence and every abstention go to the queue."""
    if annotations is None or annotations.empty:
        return empty_frame(REVIEW_QUEUE_SCHEMA)

    frame = coerce_frame(annotations, ANNOTATION_SCHEMA, "annotations")
    abstained = frame["abstained"].fillna(False).astype(bool)
    flagged = frame["needs_review"].fillna(False).astype(bool)
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    # a missing confidence is a rule or a human, not a hesitant model, so it never queues alone
    low = confidence.notna() & (confidence < float(threshold))

    reason = pd.Series(pd.NA, index=frame.index, dtype="object")
    reason[flagged] = FLAGGED
    reason[low] = LOW_CONFIDENCE
    reason[abstained] = ABSTAINED       # no label came back at all, so nothing else applies

    keep = reason.notna()
    if not keep.any():
        return empty_frame(REVIEW_QUEUE_SCHEMA)

    queue = frame.loc[keep, list(_CARRIED)].copy()
    queue["reason"] = reason[keep].to_numpy()
    queue["decision"] = pd.NA
    queue["reviewer"] = pd.NA
    queue = queue.sort_values(["recording_id", "t_start", "subject_id", "tier"], kind="stable")
    return coerce_frame(queue.reset_index(drop=True), REVIEW_QUEUE_SCHEMA, NAME)


def random_checks(annotations: pd.DataFrame, queue: pd.DataFrame, *, per_recording: int, seed: int) -> pd.DataFrame:
    """per_recording of the units not already queued, drawn the same way every time for the same units."""
    if per_recording <= 0 or annotations is None or annotations.empty:
        return empty_frame(REVIEW_QUEUE_SCHEMA)
    frame = coerce_frame(annotations, ANNOTATION_SCHEMA, "annotations")
    sure = frame[~_unit_keys(frame).isin(set(_unit_keys(queue))).to_numpy()]
    drawn = []
    for recording_id, units in sure.groupby("recording_id", sort=True):
        units = units.sort_values(["t_start", "subject_id", "tier"], kind="stable")
        rng = np.random.default_rng([seed, zlib.crc32(str(recording_id).encode("utf-8"))])
        drawn.append(units.iloc[np.sort(rng.choice(len(units), size=min(per_recording, len(units)), replace=False))])
    if not drawn:
        return empty_frame(REVIEW_QUEUE_SCHEMA)
    checks = pd.concat(drawn, ignore_index=True)[list(_CARRIED)]
    checks["reason"] = RANDOM_CHECK
    checks["decision"] = pd.NA
    checks["reviewer"] = pd.NA
    return coerce_frame(checks, REVIEW_QUEUE_SCHEMA, NAME)


def write_queue(queue: pd.DataFrame, path: Path) -> Path:
    """One file, not a directory: only the interactive step writes it, so nothing runs
    concurrently here. A .csv suffix writes csv, which a reviewer without the GUI can edit."""
    path = Path(path)
    frame = coerce_frame(queue, REVIEW_QUEUE_SCHEMA, NAME)
    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8")
        return path
    return _io.write_file(frame, path, REVIEW_QUEUE_SCHEMA, NAME)


def read_queue(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"{NAME} not found: {path}")
    if path.suffix.lower() == ".csv":
        # ids such as 003 must stay text, not become the number 3
        text = {column: "string" for column, kind in REVIEW_QUEUE_SCHEMA.items() if kind == "string"}
        return coerce_frame(pd.read_csv(path, dtype=text), REVIEW_QUEUE_SCHEMA, NAME)
    return _io.read_file(path, REVIEW_QUEUE_SCHEMA, NAME)


def apply_decisions(annotations: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """The reviewer's label is added as a second annotation instead of being written over the
    model's. Keeping both is what lets agreement treat the human and the model as two
    annotators, which is the only evidence that the automatic pass was worth trusting."""
    frame = coerce_frame(annotations, ANNOTATION_SCHEMA, "annotations")
    if decisions is None or decisions.empty:
        return frame

    decided = decisions.loc[_decided(decisions)]
    if decided.empty:
        return frame

    verdicts = {
        key: (str(label), _reviewer_name(reviewer))
        for key, label, reviewer in zip(
            _keys(decided), decided["decision"], decided["reviewer"]
        )
    }

    keys = _keys(frame)
    touched = keys.isin(verdicts)
    frame = frame.copy()
    # a row that has been looked at is no longer waiting, whatever the reviewer concluded
    frame.loc[touched, "needs_review"] = False

    added = frame.loc[touched].copy()
    added["label"] = [verdicts[key][0] for key in keys[touched]]
    added["annotator"] = [verdicts[key][1] for key in keys[touched]]
    added["confidence"] = float("nan")
    added["abstained"] = False
    added["needs_review"] = False
    # a human decision does not belong to any pipeline execution
    added["execution_id"] = pd.NA
    added["config_hash"] = pd.NA

    out = pd.concat([frame, added], ignore_index=True)
    # applying the same decisions twice must not double the human rows
    out = out.drop_duplicates(
        subset=["recording_id", "subject_id", "tier", "t_start", "t_end", "annotator", "label"],
        keep="last",
    )
    out = coerce_frame(out.reset_index(drop=True), ANNOTATION_SCHEMA, "annotations")
    out.attrs["unmatched_decisions"] = int(len(verdicts) - keys[touched].nunique())
    return out


def queue_summary(queue: pd.DataFrame) -> dict:
    if queue is None or queue.empty:
        return {
            "n_items": 0,
            "n_decided": 0,
            "n_pending": 0,
            "by_reason": {},
            "by_tier": {},
            "by_annotator": {},
            "mean_confidence": float("nan"),
        }

    decided = _decided(queue)
    return {
        "n_items": int(len(queue)),
        "n_decided": int(decided.sum()),
        "n_pending": int((~decided).sum()),
        "by_reason": _counts(queue["reason"]),
        "by_tier": _counts(queue["tier"]),
        "by_annotator": _counts(queue["annotator"]),
        "mean_confidence": float(pd.to_numeric(queue["confidence"], errors="coerce").mean()),
    }


def _counts(values: pd.Series) -> dict[str, int]:
    counted = values.astype("string").fillna("").value_counts()
    return {str(name): int(count) for name, count in counted.items()}


_UNIT =["recording_id", "subject_id", "tier", "t_start", "t_end"]


def _unit_keys(frame: pd.DataFrame) -> pd.Series:
    return _keys(frame.assign(annotator="", label=""))


def export_review(config, recording_ids: list[str]) -> pd.DataFrame:
    """The queue across recordings, without units a person has already decided."""
    queue = _full_queue(config, recording_ids)
    decided = _decisions(config, recording_ids)
    if not decided.empty and not queue.empty:
        queue = queue[~_unit_keys(queue).isin(set(_unit_keys(decided))).to_numpy()].reset_index(drop=True)
    return queue


def random_check_result(config, recording_ids: list[str]) -> tuple[int, int]:
    """(units drawn at random that a person has decided, how many of them the person coded otherwise)."""
    queue = _full_queue(config, recording_ids)
    checks = queue[queue["reason"].astype(str) == RANDOM_CHECK]
    decided = _decisions(config, recording_ids)
    if checks.empty or decided.empty:
        return 0, 0
    given = dict(zip(_unit_keys(decided), decided["label"].astype("string")))
    said = [given.get(key) for key in _unit_keys(checks)]
    looked = [(person, model) for person, model in zip(said, checks["label"].astype("string")) if person is not None]
    return len(looked), sum(person != model for person, model in looked)


def _full_queue(config, recording_ids: list[str]) -> pd.DataFrame:
    from macaw.coding.agreement import load_annotations
    from macaw.coding.llm.judge import LAYER

    annotations = load_annotations(config, recording_ids)
    # the model's own coding: not the versions under coding.judges, which are there to be compared, nor the
    # people's decisions kept beside it under the same layer
    annotations = annotations[annotations["layer"].astype(str) == LAYER]
    decided = _decisions(config, recording_ids)
    if not decided.empty:
        annotations = annotations[~_keys(annotations).isin(set(_keys(decided))).to_numpy()]
    if annotations.empty:
        return empty_frame(REVIEW_QUEUE_SCHEMA)
    llm = config.coding.llm
    queue = build_queue(annotations, threshold=llm.review_threshold)
    checks = random_checks(annotations, queue, per_recording=llm.review_random, seed=llm.seed)
    both = pd.concat([queue, checks], ignore_index=True) if not checks.empty else queue
    return coerce_frame(both.sort_values(["recording_id", "t_start", "subject_id", "tier"], kind="stable").reset_index(drop=True), REVIEW_QUEUE_SCHEMA, NAME)


def _decisions(config, recording_ids: list[str]) -> pd.DataFrame:
    from macaw.project.layout import layout_for

    layout = layout_for(config, create=False)
    decided = [
        _io.read_file(path, ANNOTATION_SCHEMA, "annotations")
        for path in (layout.reviewed_codes_path(recording_id) for recording_id in recording_ids)
        if path.is_file()
    ]
    return pd.concat(decided, ignore_index=True) if decided else empty_frame(ANNOTATION_SCHEMA)


def apply_review(config, queue: pd.DataFrame) -> dict[str, int]:
    """Decided rows become a person's annotations in reviewed.parquet. Per recording, how many were added
    and, under '<id> (not found)', how many decisions matched no annotation."""
    from macaw.project.layout import layout_for

    layout = layout_for(config, create=False)
    counts: dict[str, int] = {}
    for recording_id, decisions in queue.loc[_decided(queue)].groupby("recording_id", sort=True):
        directory = layout.annotations_path(str(recording_id))
        present = directory.is_dir() and any(directory.glob("*.parquet"))
        annotations = _io.read_partitioned(directory, ANNOTATION_SCHEMA, "annotations") if present else empty_frame(ANNOTATION_SCHEMA)
        target = layout.reviewed_codes_path(str(recording_id))
        earlier = _io.read_file(target, ANNOTATION_SCHEMA, "annotations") if target.is_file() else empty_frame(ANNOTATION_SCHEMA)
        annotations = annotations[~_keys(annotations).isin(set(_keys(earlier))).to_numpy()].reset_index(drop=True)
        merged = apply_decisions(annotations, decisions)
        # apply_decisions appends the people's rows after the ones it was given
        added = merged.iloc[len(annotations):]
        if merged.attrs.get("unmatched_decisions"):
            counts[f"{recording_id} (not found)"] = int(merged.attrs["unmatched_decisions"])
        if added.empty:
            continue
        # a unit decided again replaces the earlier decision
        both = pd.concat([earlier, added], ignore_index=True).drop_duplicates(subset=[*_UNIT, "annotator"], keep="last")
        _io.write_file(both.reset_index(drop=True), target, ANNOTATION_SCHEMA, "annotations")
        counts[str(recording_id)] = int(len(added))
    return counts
