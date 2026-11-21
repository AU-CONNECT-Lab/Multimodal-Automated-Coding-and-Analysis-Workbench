"""A person's own coding brought in beside the machine's, where macaw agreement, describe-codes and the gesture measures read it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from macaw.exceptions import ConfigError
from macaw.interop.coding_files import read_coding_file
from macaw.interop.eaf import LINGUISTIC_TYPE_PREFIX, _ms, _strip_affixes, _text
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.annotations import build_annotations, read_annotations
from macaw.store.schema import ANNOTATION_SCHEMA, coerce_frame, empty_frame
from macaw.store._io import write_text_atomic

LAYER = "C"
_NAME = "annotations"


@dataclass
class ImportReport:
    recording_id: str
    coder: str
    # (person, tier) -> annotations brought in
    kept: dict[tuple[str | None, str], int] = field(default_factory=dict)
    left_out: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def import_coding(config, recording: Recording, path: Path, coder: str, against: str | None = None) -> ImportReport:
    """Importing again replaces this coder's codes for the same person and tier, and those the same file brought in
    before, so a tier given to another person or renamed leaves nothing behind; other coders' stay. With against, the
    file's times are on that device's own clock and are put on the recording's."""
    raw = read_coding_file(path)
    if against is not None:
        raw = _on_recording_clock(raw, config, recording, against, path)
    people = [str(subject.subject_id) for subject in recording.subjects]
    layout = layout_for(config, create=False)
    directory = layout.annotations_path(recording.recording_id)
    held = read_annotations(directory) if directory.is_dir() else empty_frame(ANNOTATION_SCHEMA)
    report = ImportReport(recording.recording_id, coder)
    for tier, count in raw.attrs.get("untimed", {}).items():
        report.notes.append(f"{tier}: {count} annotation(s) without a time of their own (a Time Subdivision or Included In tier) left out")
    frames = []
    for (file_tier, participant), rows in raw.groupby(["file_tier", "participant"], dropna=False, sort=True):
        participant = None if pd.isna(participant) else str(participant)
        placed = _place(str(file_tier), participant, rows, config.coding.imports, people, coder, held)
        if isinstance(placed, str):
            report.left_out.append(f"{file_tier}{f' ({participant})' if participant else ''}: {placed}")
            continue
        person, tier, labels = placed
        lasting = rows["t_end"] > rows["t_start"]
        if not lasting.all():
            report.notes.append(f"{file_tier}: {int((~lasting).sum())} annotation(s) without length left out")
        rows = rows[lasting]
        codes = rows["label"].map(lambda value: labels.get(value, value))
        if (codes == "").any():
            report.notes.append(f"{file_tier}: {int((codes == '').sum())} annotation(s) without a code, kept as an empty code")
        intervals = pd.DataFrame({"t_start": rows["t_start"].to_numpy(), "t_end": rows["t_end"].to_numpy(), "label": codes.to_numpy()})
        frames.append(
            build_annotations(
                recording_id=recording.recording_id, tier=tier, layer=LAYER, annotator=coder, intervals=intervals,
                entities=recording.entities(), subject_id=person,
            )
        )
        report.kept[(person, tier)] = report.kept.get((person, tier), 0) + len(intervals)
    new = coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, _NAME) if frames else empty_frame(ANNOTATION_SCHEMA)
    sources = layout.imported_sources_path(recording.recording_id)
    found = json.loads(sources.read_text(encoding="utf-8")) if sources.is_file() else {}
    before = {(person, tier) for person, tier in found.get(coder, {}).get(Path(path).name, [])}
    now = {(_text(person), tier) for person, tier in zip(new["subject_id"], new["tier"])}
    gone = _store(layout.imported_codes_path(recording.recording_id), new, coder, before - now)
    for person, tier in sorted(gone, key=str):
        report.notes.append(f"{person or 'everyone'} {tier}: brought in from this file before and not now, removed")
    found.setdefault(coder, {})[Path(path).name] = sorted([list(key) for key in now], key=str)
    sources.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(sources, json.dumps(found, indent=1, ensure_ascii=False))
    return report


def _on_recording_clock(raw: pd.DataFrame, config, recording: Recording, against: str, path: Path) -> pd.DataFrame:
    from macaw.media.align import apply_offset_table

    if raw.attrs.get("zero"):
        raise ConfigError(f"{Path(path).name} was written by macaw and is on the recording's clock already; leave out --against")
    timed = apply_offset_table(recording, layout_for(config, create=False).offset_table_path(recording.recording_id))
    names = [item.name for item in timed.media]
    if against not in names:
        raise ConfigError(f"--against takes a device of {recording.recording_id}: {', '.join(names)}")
    item = timed.media_named(against)
    # where ELAN plays that file from a moment of its own, its time 0 is that moment
    origin = raw.attrs.get("origins", {}).get(Path(item.path).name, 0.0)
    out = raw.copy()
    for column in ("t_start", "t_end"):
        out[column] = item.offset_s + (out[column] + origin) * item.clock_scale
    out.attrs = dict(raw.attrs)
    return out


def _place(file_tier: str, participant: str | None, rows: pd.DataFrame, imports: dict, people: list[str], coder: str, held: pd.DataFrame):
    """(person, tier, labels) for a tier to bring in, or why it stays out. The person is the setting's, else the file's."""
    entry = imports.get(file_tier)
    stranger = participant is not None and participant not in people
    if entry is not None:
        person = entry.person if entry.person is not None else participant
        if person is not None and person not in people:
            where = "coding.imports gives it to" if entry.person is not None else "the file gives it to"
            raise ConfigError(
                f"tier '{file_tier}': {where} {person}, who is not one of this recording's people ({', '.join(people)}); "
                "set its person in coding.imports"
            )
        return person, entry.tier or file_tier, dict(entry.labels)
    if participant is None or stranger:
        named = f"; the file gives it to {participant}, who is not one of this recording's people" if stranger else ""
        return f"not in coding.imports{named}"
    kind, annotator = _text(rows["linguistic_type"].iloc[0]), _text(rows["annotator"].iloc[0]) or None
    # a tier MACAW wrote carries its layer in the type name and its subject in the tier name
    layer = kind[len(LINGUISTIC_TYPE_PREFIX):].split("-", 1)[0] if kind.startswith(LINGUISTIC_TYPE_PREFIX) else LAYER
    tier = _strip_affixes(file_tier, participant, layer, annotator)
    if annotator and annotator != coder and _unchanged(rows, held, participant, tier, annotator):
        return f"the same as {annotator}'s codes already here, so not this coder's own"
    return participant, tier, {}


def _unchanged(rows: pd.DataFrame, held: pd.DataFrame, person: str, tier: str, annotator: str) -> bool:
    theirs = held[(held["subject_id"] == person) & (held["tier"] == tier) & (held["annotator"] == annotator)]
    if theirs.empty:
        return False

    def spans(frame):
        return sorted((_ms(row.t_start), _ms(row.t_end), _text(row.label)) for row in frame.itertuples(index=False))

    return spans(rows) == spans(theirs)


def _store(target: Path, new: pd.DataFrame, coder: str, dropped: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """Writes new over the same coder, person and tier, and removes the coder's dropped ones; returns those that held codes."""
    gone: set[tuple[str, str]] = set()
    if target.is_file():
        old = _io.read_file(target, ANNOTATION_SCHEMA, _NAME)
        replaced = set(zip(new["annotator"], new["subject_id"].map(_text), new["tier"])) | {(coder, *key) for key in dropped}
        keys = list(zip(old["annotator"], old["subject_id"].map(_text), old["tier"]))
        gone = {(person, tier) for annotator, person, tier in keys if annotator == coder and (person, tier) in dropped}
        new = pd.concat([old[[key not in replaced for key in keys]], new], ignore_index=True)
    if new.empty and not target.is_file():
        return gone
    _io.write_file(new.sort_values(["annotator", "subject_id", "tier", "t_start"], na_position="first").reset_index(drop=True), target, ANNOTATION_SCHEMA, _NAME)
    return gone
