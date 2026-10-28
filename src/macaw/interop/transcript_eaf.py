"""The transcript out to ELAN for a person to correct, and the corrected one back in."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.exceptions import EafError, InputNotFoundError
from macaw.interop.eaf import _elan, _link_media, _ms, _text, tier_for_subject
from macaw.project.recording import Recording

TRANSCRIPT_TYPE = "macaw-transcript"
# what the software left out; moving an annotation from here to a transcript tier puts it back
DROPPED_TYPE = "macaw-dropped"
CORRECTED_COLUMNS = ("subject_id", "t_start", "t_end", "text")


def export_transcript_eaf(
    transcript: pd.DataFrame, dropped: pd.DataFrame, recording: Recording, out_path: Path
) -> Path:
    Eaf = _elan()
    out_path = Path(out_path)
    eaf = Eaf()
    if "default" in eaf.get_tier_names():
        eaf.remove_tier("default")
    for kind in (TRANSCRIPT_TYPE, DROPPED_TYPE):
        eaf.add_linguistic_type(kind)
    _link_media(eaf, recording, out_path)
    subjects = [subject.subject_id for subject in recording.subjects]
    # a further speaker the transcript holds, such as other, keeps a tier of its own
    subjects += sorted(set(transcript["subject_id"].dropna().astype(str)) - set(subjects))
    for kind, frame, name in ((TRANSCRIPT_TYPE, transcript, "transcript"), (DROPPED_TYPE, dropped, "dropped")):
        for subject_id in subjects:
            rows = frame[frame["subject_id"].astype(str) == subject_id]
            if kind == DROPPED_TYPE and rows.empty:
                continue
            tier = tier_for_subject(subject_id, name)
            eaf.add_tier(tier, ling=kind, part=subject_id)
            for row in rows.itertuples(index=False):
                start, end = _ms(row.t_start), _ms(row.t_end)
                if end > start:
                    text = _text(row.text) if kind == TRANSCRIPT_TYPE else f"{_text(row.text)} [{row.reason}: {_text(row.detail)}]"
                    eaf.add_annotation(tier, start, end, text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(eaf, "to_file"):
        eaf.to_file(str(out_path))
    else:
        from pympi.Elan import to_eaf

        to_eaf(str(out_path), eaf)
    return out_path


def read_transcript_eaf(path: Path, subjects: list[str]) -> pd.DataFrame:
    """Every annotation on a transcript tier, given to the tier's participant. Other tiers are ignored."""
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
        attributes = eaf.tiers[name][2]
        if attributes.get("LINGUISTIC_TYPE_REF") != TRANSCRIPT_TYPE:
            continue
        subject_id = attributes.get("PARTICIPANT") or ""
        if subject_id not in subjects:
            raise EafError(f"{path}: transcript tier '{name}' belongs to '{subject_id}', who is not a subject ({', '.join(subjects)})")
        for start, end, value, *_ in eaf.get_annotation_data_for_tier(name):
            if str(value).strip():
                rows.append({"subject_id": subject_id, "t_start": start / 1000.0, "t_end": end / 1000.0, "text": str(value).strip()})
    if not rows:
        raise EafError(f"{path}: no annotation on any transcript tier")
    return pd.DataFrame(rows, columns=list(CORRECTED_COLUMNS)).sort_values(["t_start", "subject_id"], ignore_index=True)
