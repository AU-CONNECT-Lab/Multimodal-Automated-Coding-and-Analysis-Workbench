"""Two-way .eaf interchange, annotation layer only. Signals are not annotations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from macaw.exceptions import EafError, InputNotFoundError
from macaw.logging_setup import get_logger
from macaw.project.recording import Recording
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, empty_frame

logger = get_logger("interop.eaf")

# ELAN stores whole milliseconds, so a round trip is exact only down to that
TIME_TOLERANCE_S = 5e-4
LINGUISTIC_TYPE_PREFIX = "macaw-"
DEFAULT_LAYER = "C"
DEFAULT_ANNOTATOR = "human"

# the columns an ELAN tier can carry. confidence, needs_review and the execution stamps have
# nowhere to live in the format and are dropped on the way out.
ROUNDTRIP_COLUMNS = ("subject_id", "tier", "layer", "annotator", "label", "t_start", "t_end")

_MIME = {
    ".wav": "audio/x-wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def _elan():
    try:
        from pympi.Elan import Eaf
    except ImportError:
        raise EafError(
            "reading and writing .eaf needs pympi-ling, which ships with the core install: "
            "pip install pympi-ling"
        ) from None
    return Eaf


def _text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _ms(value) -> int:
    return int(round(float(value) * 1000.0))


def tier_for_subject(subject_id: str, kind: str) -> str:
    """ELAN tier ids have to be unique across the file, so the subject goes into the name. On
    the way back in it is read from the tier's PARTICIPANT attribute, never by splitting this."""
    return kind if not _text(subject_id) else f"{subject_id}_{kind}"


def _link_media(eaf, recording: Recording, out_path: Path) -> None:
    for item in recording.media:
        path = Path(item.path)
        try:
            relative = path.resolve().relative_to(out_path.parent.resolve()).as_posix()
        except (ValueError, OSError):
            relative = None
        mimetype = _MIME.get(
            path.suffix.lower(), "video/mp4" if item.kind == "video" else "audio/x-wav"
        )
        eaf.add_linked_file(path.absolute().as_uri(), relpath=relative, mimetype=mimetype)


def _attach_vocabulary(eaf, linguistic_type: str, cv_id: str, labels: list[str]) -> None:
    if not labels:
        return
    # the vocabulary API has moved between library versions; a failure here is worth reporting
    # rather than silently producing a file without the dropdown the caller asked for
    try:
        eaf.add_controlled_vocabulary(cv_id)
        for index, label in enumerate(labels):
            eaf.add_cv_entry(cv_id, f"cveid{index}", [(str(label), "und", "")])
        eaf.linguistic_types[linguistic_type]["CONTROLLED_VOCABULARY_REF"] = cv_id
    except Exception as error:
        raise EafError(f"could not build the ELAN controlled vocabulary '{cv_id}': {error}") from error


def _tier_names(frame: pd.DataFrame) -> dict[tuple, str]:
    """One ELAN tier per (subject, tier, layer, annotator). Layer and annotator only enter the
    name when the same tier carries more than one of them, because then the ids would collide."""
    variants: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in frame.loc[:, ["subject_id", "tier", "layer", "annotator"]].itertuples(index=False):
        key = (_text(row.subject_id), _text(row.tier))
        variants.setdefault(key, set()).add((_text(row.layer), _text(row.annotator)))

    names: dict[tuple, str] = {}
    for (subject_id, tier), combinations in variants.items():
        base = tier_for_subject(subject_id, tier)
        for layer, annotator in combinations:
            name = base if len(combinations) == 1 else f"{base}_{layer}_{annotator}"
            names[(subject_id, tier, layer, annotator)] = name
    return names


def export_eaf(
    annotations: pd.DataFrame,
    recording: Recording,
    out_path: Path,
    *,
    tiers: list[str] | None = None,
    controlled_vocabulary: dict[str, list[str]] | None = None,
) -> Path:
    Eaf = _elan()
    frame = coerce_frame(annotations, ANNOTATION_SCHEMA, "annotations")
    if tiers is not None:
        frame = frame[frame["tier"].astype(str).isin(list(tiers))]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eaf = Eaf()
    if "default" in eaf.get_tier_names():
        eaf.remove_tier("default")      # the empty tier a fresh file starts with
    _link_media(eaf, recording, out_path)

    names = _tier_names(frame) if not frame.empty else {}
    skipped = 0
    grouped = (
        frame.groupby(["subject_id", "tier", "layer", "annotator"], dropna=False, sort=True)
        if not frame.empty
        else []
    )
    for key, group in grouped:
        subject_id, tier, layer, annotator = (_text(part) for part in key)
        name = names[(subject_id, tier, layer, annotator)]
        linguistic_type = f"{LINGUISTIC_TYPE_PREFIX}{layer or DEFAULT_LAYER}"
        if linguistic_type not in eaf.linguistic_types:
            eaf.add_linguistic_type(linguistic_type)
            if controlled_vocabulary and tier in controlled_vocabulary:
                _attach_vocabulary(
                    eaf, linguistic_type, f"cv-{tier}", list(controlled_vocabulary[tier])
                )
        eaf.add_tier(name, ling=linguistic_type, part=subject_id or None, ann=annotator or None)

        for row in group.itertuples(index=False):
            start, end = _ms(row.t_start), _ms(row.t_end)
            if end <= start:
                skipped += 1
                continue
            eaf.add_annotation(name, start, end, _text(row.label))

    if skipped:
        logger.warning(
            "%d annotation(s) dropped from %s: ELAN has no zero-length annotation", skipped, out_path.name
        )

    if hasattr(eaf, "to_file"):
        eaf.to_file(str(out_path))
    else:
        from pympi.Elan import to_eaf

        to_eaf(str(out_path), eaf)
    return out_path


def import_eaf(
    path: Path,
    *,
    recording_id: str,
    default_layer: str = DEFAULT_LAYER,
    default_annotator: str = DEFAULT_ANNOTATOR,
) -> pd.DataFrame:
    Eaf = _elan()
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f".eaf not found: {path}")
    try:
        eaf = Eaf(str(path))
    except Exception as error:
        raise EafError(f"{path}: could not be read as ELAN ({error})") from error

    rows = []
    for name in eaf.get_tier_names():
        attributes = eaf.tiers[name][2] if name in eaf.tiers else {}
        subject_id = attributes.get("PARTICIPANT") or None
        annotator = attributes.get("ANNOTATOR") or default_annotator
        reference = attributes.get("LINGUISTIC_TYPE_REF", "") or ""
        layer = (
            reference[len(LINGUISTIC_TYPE_PREFIX) :]
            if reference.startswith(LINGUISTIC_TYPE_PREFIX)
            else default_layer
        )
        tier = _strip_affixes(name, subject_id, layer, annotator)

        for annotation in eaf.get_annotation_data_for_tier(name):
            start, end, value = annotation[0], annotation[1], annotation[2]
            rows.append(
                {
                    "subject_id": subject_id,
                    "tier": tier,
                    "t_start": float(start) / 1000.0,
                    "t_end": float(end) / 1000.0,
                    "label": value,
                    "annotator": annotator,
                    "layer": layer,
                }
            )

    if not rows:
        return empty_frame(ANNOTATION_SCHEMA)

    frame = pd.DataFrame(rows)
    frame["recording_id"] = recording_id
    # the file itself knows nothing about the session or task it came from
    for column in ("group_id", "session_id", "task", "run", "execution_id", "config_hash"):
        frame[column] = pd.NA
    frame["confidence"] = float("nan")
    frame["abstained"] = False
    frame["needs_review"] = False
    return coerce_frame(frame, ANNOTATION_SCHEMA, "annotations")


def _strip_affixes(name: str, subject_id: str | None, layer: str, annotator: str | None) -> str:
    """Undo the composition done on export, using values the tier itself declares."""
    tier = name
    if subject_id and tier.startswith(f"{subject_id}_"):
        tier = tier[len(subject_id) + 1 :]
    suffix = f"_{layer}_{annotator}"
    if annotator and tier.endswith(suffix):
        tier = tier[: -len(suffix)]
    return tier


def _comparable(frame: pd.DataFrame) -> list[tuple]:
    if frame is None or frame.empty:
        return []
    rows = []
    for row in frame.loc[:, list(ROUNDTRIP_COLUMNS)].itertuples(index=False):
        if _ms(row.t_end) <= _ms(row.t_start):
            continue        # ELAN cannot hold it, so it is not part of what a round trip promises
        rows.append(
            (
                _text(row.subject_id),
                _text(row.tier),
                _text(row.layer),
                _text(row.annotator),
                _text(row.label),
                _ms(row.t_start),
                _ms(row.t_end),
            )
        )
    return sorted(rows)


def roundtrip_check(annotations: pd.DataFrame, recording: Recording, tmp_dir: Path) -> list[str]:
    """Export then re-import must reproduce every annotation exactly, on the columns .eaf can
    carry; see ROUNDTRIP_COLUMNS for the ones it cannot. Empty means lossless."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"{recording.recording_id}.eaf"

    export_eaf(annotations, recording, target)
    returned = import_eaf(target, recording_id=recording.recording_id)

    before = Counter(_comparable(coerce_frame(annotations, ANNOTATION_SCHEMA, "annotations")))
    after = Counter(_comparable(returned))
    problems = [f"lost on the way out: {item}" for item in sorted((before - after).elements())]
    problems += [f"appeared on the way in: {item}" for item in sorted((after - before).elements())]
    return problems
