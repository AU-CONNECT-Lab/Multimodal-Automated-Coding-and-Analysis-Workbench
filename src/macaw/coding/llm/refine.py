"""A clearer codebook from the units coders disagree on, for a person to read and edit before any coding uses it."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.coding.agreement import NO_CODE, build_units, load_annotations, unit_table
from macaw.coding.codebook import Codebook, CodeDefinition
from macaw.coding.llm.backends import LLMBackend
from macaw.coding.llm.judge import LAYER, SAMPLES_NAME, VERSION_LAYER, JudgementExtractor, coding_units
from macaw.coding.llm.parsing import answer_object
from macaw.coding.llm.prompts import ABSTAIN, codebook_lines
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store import _io
from macaw.store.schema import ANNOTATION_SCHEMA

MODEL = "model"
EVIDENCE = ["recording_id", "subject_id", "t_start", "t_end", "speaker", "words", "before", "person", "model", "answers"]
# per kind of unit shown for one code, so a prompt stays short enough for a small local model
PER_KIND = 12
NEW_EXAMPLES = 3
FLATTERS = "Judge it on other recordings: agreement on the units it was made from flatters it."
_LISTS = ("criteria", "boundary_cases")


@dataclass
class Evidence:
    units: pd.DataFrame
    coder: str | None          # the person the model is compared with; None when the model is read alone
    models: list[str]


@dataclass
class Revision:
    code: str
    before: CodeDefinition
    after: CodeDefinition
    why: str = ""
    shown: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["id", "kind", *EVIDENCE]))
    dropped_ids: int = 0       # examples the model named that are not units of the right kind

    @property
    def changed(self) -> bool:
        return self.before != self.after


def _model_codes(config, recording_ids: list[str], tier: str) -> pd.DataFrame:
    """The model's own coding, layer B, from the file only coding.judge writes; people's review decisions share the layer."""
    layout = layout_for(config, create=False)
    frames = []
    for rid in recording_ids:
        path = layout.annotations_path(rid) / f"{LAYER}.parquet"
        if path.is_file():
            frames.append(_io.read_file(path, ANNOTATION_SCHEMA, "annotations"))
    found = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(ANNOTATION_SCHEMA))
    return found[found["tier"].astype(str) == tier]


def _answers(config, recording_ids: list[str]) -> dict[tuple, str]:
    """Each unit's answers, as "inform 3, question 2", by recording, person and times."""
    layout = layout_for(config, create=False)
    said = {}
    for rid in recording_ids:
        path = layout.artifacts_dir(rid, JudgementExtractor.name) / SAMPLES_NAME
        if not path.is_file():
            continue
        samples = pd.read_parquet(path)
        for (subject, start, end), group in samples.groupby(["subject_id", "t_start", "t_end"], sort=False):
            counted = group["label"].astype(str).value_counts()
            said[(rid, str(subject), round(float(start), 3), round(float(end), 3))] = ", ".join(f"{label} {n}" for label, n in counted.items())
    return said


def gather_evidence(config, codebook: Codebook, recording_ids: list[str], *, coder: str | None = None) -> Evidence:
    """Every unit with the words in it, the line before, the person's code where a person coded it, the model's, and its answers."""
    tier = codebook.name
    model = _model_codes(config, recording_ids, tier)
    if model.empty:
        raise MacawError(
            f"no codes by the language model on the tier {tier} in these recordings; run coding.judge first, "
            "or draft a codebook with macaw codebook draft"
        )
    annotations = load_annotations(config, recording_ids)
    on_tier = annotations[annotations["tier"].astype(str) == tier]
    machine = set(model["annotator"].astype(str)) | set(on_tier.loc[on_tier["layer"].astype(str).str.startswith(VERSION_LAYER), "annotator"].astype(str))
    people = on_tier[~on_tier["annotator"].astype(str).isin(machine)]
    coders = sorted(people["annotator"].dropna().astype(str).unique())
    if coder is None and len(coders) > 1:
        raise MacawError(f"several people coded {tier}: {', '.join(coders)}; say whose codes to go by with --coder")
    if coder is not None and coder not in coders:
        raise MacawError(f"no codes by {coder} on the tier {tier}; the people who coded it: {', '.join(coders) or 'none'}")
    coder = coder if coder is not None else (coders[0] if coders else None)

    renamed = model.assign(annotator=MODEL)
    if coder is not None:
        units = build_units(pd.concat([people[people["annotator"].astype(str) == coder], renamed], ignore_index=True), units_of=coder)
        table = unit_table(units)
        person = table[coder]
    else:
        table = unit_table(build_units(renamed))
        person = pd.Series(pd.NA, index=table.index, dtype="object")
    said = _said(config, table, codebook.unit)
    answers = _answers(config, recording_ids) if coder is None else {}
    keys = zip(table["recording_id"].astype(str), table["subject_id"].astype(str), table["t_start"].round(3), table["t_end"].round(3))
    out = pd.DataFrame({
        "recording_id": table["recording_id"].astype(str), "subject_id": table["subject_id"].astype(str),
        "t_start": table["t_start"].astype(float), "t_end": table["t_end"].astype(float),
        "speaker": said["speaker"], "words": said["words"], "before": said["before"],
        "person": [_code(value) for value in person],
        "model": [_code(value) for value in table.get(MODEL, pd.Series(pd.NA, index=table.index))],
        "answers": [answers.get(key, "") for key in keys],
    })
    # a unit with no words inside cannot be shown to anyone
    out = out[out["words"] != ""].reset_index(drop=True)
    return Evidence(units=out, coder=coder, models=sorted(set(model["annotator"].astype(str))))


def _said(config, table: pd.DataFrame, unit: str) -> pd.DataFrame:
    """The speaker's letter, the words inside each unit and the line before it, as the model is shown lines when it codes."""
    out = pd.DataFrame({"speaker": "", "words": "", "before": ""}, index=table.index)
    for rid, rows in table.groupby(table["recording_id"].astype(str), sort=False):
        try:
            lines = coding_units(config, build_recording(rid, config), unit)
        except (MacawError, OSError):   # coded but never transcribed
            continue
        middle = ((lines["t_start"] + lines["t_end"]) / 2).to_numpy()
        for index, row in rows.iterrows():
            inside = np.flatnonzero((lines["subject_id"] == str(row["subject_id"])).to_numpy() & (middle >= row["t_start"]) & (middle <= row["t_end"]))
            if not inside.size:
                continue
            out.loc[index, "speaker"] = str(lines["speaker"].iloc[inside[0]])
            out.loc[index, "words"] = " ".join(lines["text"].iloc[inside].astype(str)).strip()
            if inside[0] > 0:
                earlier = lines.iloc[inside[0] - 1]
                out.loc[index, "before"] = f"{earlier['speaker']}: {earlier['text']}"
    return out


def _code(value) -> str | None:
    return None if pd.isna(value) or str(value) in ("", NO_CODE, ABSTAIN) else str(value)


def _split(answers: str) -> list[str]:
    return [part.rsplit(" ", 1)[0] for part in answers.split(", ") if part]


def units_for(evidence: Evidence, code: str, *, per_kind: int = PER_KIND, rng: np.random.Generator) -> pd.DataFrame:
    """The units that bear on one code, by kind: with a person, the ones they and the model coded differently and some
    they agreed on; without one, the ones the model was split on and some it was sure of."""
    units = evidence.units
    if evidence.coder is not None:
        kinds = {
            "missed": (units["person"] == code) & (units["model"] != code),
            "taken": (units["model"] == code) & units["person"].notna() & (units["person"] != code),
            "agreed": (units["person"] == code) & (units["model"] == code),
        }
    else:
        split = units["answers"].map(lambda text: len(_split(text)) > 1 and code in _split(text))
        kinds = {
            "split": split,
            "agreed": (units["model"] == code) & ~units["answers"].map(lambda text: len(_split(text)) > 1),
        }
    parts = []
    for kind, chosen in kinds.items():
        found = units[chosen.fillna(False).astype(bool)]
        if len(found) > per_kind:
            found = found.iloc[np.sort(rng.choice(len(found), size=per_kind, replace=False))]
        parts.append(found.assign(kind=kind))
    shown = pd.concat(parts, ignore_index=True) if parts else units.iloc[0:0].assign(kind="")
    return shown.assign(id=[f"e{index + 1}" for index in range(len(shown))])


def _reference(evidence: Evidence, row) -> str | None:
    """What the unit is taken to be: the person's code, or without a person the model's when it was sure."""
    if evidence.coder is not None:
        return row.person
    return row.model if len(_split(row.answers)) <= 1 else None


_HEADINGS = {
    "missed": "Units the person coded {code} and the model coded otherwise",
    "taken": "Units the model coded {code} and the person coded otherwise",
    "split": "Units where the model's answers were split between {code} and other codes",
    "agreed": "Units both coded {code}",
}
_SURE = "Units the model coded {code} every time it was asked"


def refine_prompt(codebook: Codebook, code: str, shown: pd.DataFrame, *, with_person: bool, notes: str = "") -> tuple[str, str]:
    system = [
        "You help researchers make a coding scheme clearer, so that coders who read only the codebook code alike.",
        *([f"Notes from the researchers: {notes.strip()}"] if notes.strip() else []),
        "The codebook as it stands:", "", *codebook_lines(codebook),
        f"Rewrite the definition of {code} only. Keep what is right in it and change what the units show to be unclear; "
        "do not change what the code is about.",
        ("A coder reading your definition should code these units as the person did." if with_person else
         f"Make the boundary between {code} and the codes the model hesitated between clear."),
        'Answer with one JSON object and nothing else: {"description": "<one or two sentences>", "criteria": ["<what must hold>"], '
        '"boundary_cases": ["<how to decide a case near another code>"], "positive_examples": ["<unit id>"], '
        '"negative_examples": ["<unit id>"], "why": "<what in the units led to the changes>"}.',
        f"Examples are ids of the units below: positive ones are units that are {code}, negative ones units that are not; "
        f"at most {NEW_EXAMPLES} of each.",
    ]
    user = []
    for kind in _HEADINGS:
        rows = shown[shown["kind"] == kind]
        if rows.empty:
            continue
        heading = _SURE if kind == "agreed" and not with_person else _HEADINGS[kind]
        user += ["", heading.format(code=code) + ":"]
        for row in rows.itertuples(index=False):
            coded = (
                f"person: {row.person or 'nothing'}, model: {row.model or 'nothing'}" if with_person
                else f"model's answers: {row.answers or row.model}"
            )
            before = f" (just before, {row.before})" if row.before else ""
            user.append(f"id {row.id} | {coded} | speaker {row.speaker}{before}: {row.words}")
    return "\n".join(system), "\n".join(user).strip()


def refine_code(
    backend: LLMBackend, codebook: Codebook, evidence: Evidence, code: str, *, rng: np.random.Generator,
    per_kind: int = PER_KIND, notes: str = "", seed: int | None = None,
) -> Revision:
    before = codebook.get(code)
    shown = units_for(evidence, code, per_kind=per_kind, rng=rng)
    unclear = shown["kind"].isin(["missed", "taken", "split"])
    if not unclear.any():
        return Revision(code, before, deepcopy(before), shown=shown)
    system, user = refine_prompt(codebook, code, shown, with_person=evidence.coder is not None, notes=notes)
    answer = answer_object(backend.complete(system, user, seed=seed).text)
    after = deepcopy(before)
    description = str(answer.get("description") or "").strip()
    if description:
        after.description = description
    for name in _LISTS:
        given = answer.get(name)
        if isinstance(given, list) and any(str(item).strip() for item in given):
            setattr(after, name, [str(item).strip() for item in given if str(item).strip()])
    by_id = {row.id: row for row in shown.itertuples(index=False)}
    dropped = 0
    for name, wanted in (("positive_examples", True), ("negative_examples", False)):
        ids = answer.get(name) if isinstance(answer.get(name), list) else []
        added = []
        for unit_id in ids[:NEW_EXAMPLES]:
            row = by_id.get(str(unit_id).strip())
            reference = _reference(evidence, row) if row is not None else None
            if row is None or reference is None or (reference == code) != wanted:
                dropped += 1
                continue
            if row.words not in getattr(after, name) and row.words not in added:
                added.append(row.words)
        setattr(after, name, [*getattr(after, name), *added])
    return Revision(code, before, after, why=str(answer.get("why") or "").strip(), shown=shown, dropped_ids=dropped)


def refine_codebook(
    backend: LLMBackend, codebook: Codebook, evidence: Evidence, *, seed: int = 0, per_kind: int = PER_KIND, notes: str = "",
    takes_seed: bool = True,
) -> tuple[Codebook, list[Revision]]:
    rng = np.random.default_rng(seed)
    revisions = [
        refine_code(backend, codebook, evidence, definition.code, rng=rng, per_kind=per_kind, notes=notes, seed=seed if takes_seed else None)
        for definition in codebook.codes
    ]
    return Codebook(codebook.name, codebook.unit, [revision.after for revision in revisions], codebook.mutually_exclusive), revisions


def _listed(title: str, before: list[str], after: list[str]) -> list[str]:
    added = [item for item in after if item not in before]
    removed = [item for item in before if item not in after]
    lines = [f"{title} added:", *(f"- {item}" for item in added)] if added else []
    return lines + ([f"{title} removed:", *(f"- {item}" for item in removed)] if removed else [])


def changes_text(source: Path, evidence: Evidence, revisions: list[Revision], model: str) -> str:
    against = f"the codes of {evidence.coder}" if evidence.coder else "the model's own answers (no person coded this tier)"
    lines = [
        f"# Changes to {source.name}", "",
        f"Proposed by {model} from {against}, against the codes of {', '.join(evidence.models)}, in "
        f"{', '.join(sorted(evidence.units['recording_id'].unique()))}. {FLATTERS} "
        "The examples are participants' own words from the transcripts; check them before sharing the codebook.", "",
    ]
    for revision in revisions:
        lines += [f"## {revision.code}", ""]
        if not revision.changed:
            unclear = revision.shown["kind"].isin(["missed", "taken", "split"]).any() if not revision.shown.empty else False
            lines += ["Unchanged." if unclear else "Unchanged: no unit showed a problem with it.", ""]
            continue
        if revision.why:
            lines += [f"Why: {revision.why}", ""]
        if revision.after.description != revision.before.description:
            lines += [f"Description before: {revision.before.description or '(none)'}", f"Description now: {revision.after.description}", ""]
        for name, title in (("criteria", "Criteria"), ("boundary_cases", "Boundary cases"),
                            ("positive_examples", "Examples"), ("negative_examples", "Examples of what it is not")):
            listed = _listed(title, getattr(revision.before, name), getattr(revision.after, name))
            lines += [*listed, ""] if listed else []
        if revision.dropped_ids:
            lines += [f"{revision.dropped_ids} example(s) the model named were left out: not units of the kind it was asked for.", ""]
        lines += ["Units shown to the model:", ""]
        for row in revision.shown.itertuples(index=False):
            coded = f"person {row.person or '-'}, model {row.model or '-'}" if evidence.coder else f"answers {row.answers or row.model}"
            lines.append(f"- {row.recording_id} {row.subject_id} at {row.t_start:.1f} s ({row.kind}; {coded}): {row.words}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def changes_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.changes.md")
