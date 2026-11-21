"""The transcript out to ELAN for a person to correct, and the corrected one back in."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.exceptions import EafError, InputNotFoundError
from macaw.interop.eaf import _elan, _link_media, _ms, _text, elan_zero_of, tier_for_subject, timed_annotations
from macaw.project.recording import Recording
from macaw.qc.transcript_check import KINDS

TRANSCRIPT_TYPE = "macaw-transcript"
# what the software left out; moving an annotation from here to a transcript tier puts it back
DROPPED_TYPE = "macaw-dropped"
# why each was left out, on a tier of its own so that the words moved back carry nothing else; never read back
WHY_TYPE = "macaw-dropped-why"
# lines a model marked as possibly misheard or cut off, for the person correcting; never read back
DOUBT_TYPE = "macaw-doubt"
CORRECTED_COLUMNS = ("subject_id", "t_start", "t_end", "text")


def export_transcript_eaf(
    transcript: pd.DataFrame, dropped: pd.DataFrame, recording: Recording, out_path: Path, doubts: pd.DataFrame | None = None,
) -> Path:
    Eaf = _elan()
    out_path = Path(out_path)
    eaf = Eaf()
    if "default" in eaf.get_tier_names():
        eaf.remove_tier("default")
    doubts = doubts if doubts is not None else pd.DataFrame(columns=["subject_id", "t_start", "t_end", "text", "kind", "why"])
    for kind in (TRANSCRIPT_TYPE, DROPPED_TYPE, WHY_TYPE, *([DOUBT_TYPE] if not doubts.empty else [])):
        eaf.add_linguistic_type(kind)
    zero = _link_media(eaf, recording, out_path)
    subjects = [subject.subject_id for subject in recording.subjects]
    # a further speaker the transcript holds, such as other, keeps a tier of its own
    subjects += sorted(set(transcript["subject_id"].dropna().astype(str)) - set(subjects))
    tiers = (
        (TRANSCRIPT_TYPE, transcript, "transcript"), (DROPPED_TYPE, dropped, "dropped"), (WHY_TYPE, dropped, "why dropped"),
        (DOUBT_TYPE, doubts, "doubtful"),
    )
    for kind, frame, name in tiers:
        # words no voice took belong to nobody yet, on a tier without a participant
        nobody = [""] if kind in (DROPPED_TYPE, WHY_TYPE) else []
        for subject_id in [*subjects, *nobody]:
            rows = frame[frame["subject_id"].astype("string").fillna("") == subject_id]
            if kind != TRANSCRIPT_TYPE and rows.empty:
                continue
            tier = tier_for_subject(subject_id, name)
            eaf.add_tier(tier, ling=kind, part=subject_id)
            for row in rows.itertuples(index=False):
                # before ELAN's time 0, where the last file began; the recogniser's lines there are kept on reading back
                if float(row.t_start) < zero:
                    continue
                start = _ms(row.t_start - zero)
                # a line of no length still goes out, or a corrected transcript read back would lose it
                end = max(_ms(row.t_end - zero), start + 1)
                text = (
                    _text(row.text) if kind in (TRANSCRIPT_TYPE, DROPPED_TYPE) else f"{KINDS[row.kind]}: {_text(row.why)}" if kind == DOUBT_TYPE
                    else f"{row.reason}: {_text(row.detail)}"
                )
                eaf.add_annotation(tier, start, end, text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eaf.to_file(str(out_path))
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
    rows, zero = [], elan_zero_of(eaf)
    for name in eaf.get_tier_names():
        attributes = eaf.tiers[name][2]
        if attributes.get("LINGUISTIC_TYPE_REF") != TRANSCRIPT_TYPE:
            continue
        subject_id = attributes.get("PARTICIPANT") or ""
        if subject_id not in subjects:
            raise EafError(f"{path}: transcript tier '{name}' belongs to '{subject_id}', who is not a subject ({', '.join(subjects)})")
        timed, untimed = timed_annotations(eaf, name)
        if untimed:
            raise EafError(
                f"{path}: {untimed} annotation(s) on tier {name} have no time of their own; give each line its own start and "
                "end on the transcript tier in ELAN"
            )
        for start, end, value, *_ in timed:
            if str(value).strip():
                rows.append({"subject_id": subject_id, "t_start": start / 1000.0 + zero, "t_end": end / 1000.0 + zero, "text": str(value).strip()})
    if not rows:
        raise EafError(f"{path}: no annotation on any transcript tier")
    out = pd.DataFrame(rows, columns=list(CORRECTED_COLUMNS)).sort_values(["t_start", "subject_id"], ignore_index=True)
    out.attrs["zero"] = zero
    return out
