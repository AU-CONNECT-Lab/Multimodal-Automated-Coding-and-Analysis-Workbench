"""A person's own coding brought in beside the machine's, where macaw agreement, describe-codes and the gesture measures read it."""

from __future__ import annotations

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


def import_coding(config, recording: Recording, path: Path, coder: str) -> ImportReport:
    """Importing again replaces this coder's codes for the same person and tier; other coders' stay."""
    raw = read_coding_file(path)
    people = [str(subject.subject_id) for subject in recording.subjects]
    layout = layout_for(config, create=False)
    directory = layout.annotations_path(recording.recording_id)
    held = read_annotations(directory) if directory.is_dir() else empty_frame(ANNOTATION_SCHEMA)
    report = ImportReport(recording.recording_id, coder)
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
    if frames:
        _store(layout.imported_codes_path(recording.recording_id), coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, _NAME))
    return report


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


def _store(target: Path, new: pd.DataFrame) -> None:
    if target.is_file():
        old = _io.read_file(target, ANNOTATION_SCHEMA, _NAME)
        replaced = set(zip(new["annotator"], new["subject_id"].map(_text), new["tier"]))
        stays = [key not in replaced for key in zip(old["annotator"], old["subject_id"].map(_text), old["tier"])]
        new = pd.concat([old[stays], new], ignore_index=True)
    _io.write_file(new.sort_values(["annotator", "subject_id", "tier", "t_start"], na_position="first").reset_index(drop=True), target, ANNOTATION_SCHEMA, _NAME)
