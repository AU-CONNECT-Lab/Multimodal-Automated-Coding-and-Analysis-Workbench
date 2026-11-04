"""A first codebook drafted from the transcripts: the model says what each sampled unit does, then gathers those
descriptions into codes. Choosing and defining the codes in the end is the researchers' work."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macaw.coding.codebook import RESERVED_LABELS, Codebook, CodeDefinition
from macaw.coding.llm.backends import LLMBackend
from macaw.coding.llm.parsing import answer_object

N_UNITS = 200
MAX_CODES = 12
EXAMPLES = 3


@dataclass
class Draft:
    codebook: Codebook
    units: pd.DataFrame                                   # the sample, with each unit's description and code
    gathered: dict[str, list[str]] = field(default_factory=dict)   # code -> the descriptions it gathers
    left_out: dict[str, int] = field(default_factory=dict)         # description -> units, gathered into no code


def sample_units(lines: dict[str, pd.DataFrame], n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Units drawn at random across the recordings, each with the line before it; lines are coding_units' per recording."""
    pool = []
    for rid, frame in lines.items():
        frame = frame.reset_index(drop=True)
        before = (frame["speaker"].astype(str) + ": " + frame["text"].astype(str)).shift().fillna("")
        own = frame.assign(recording_id=rid, before=before)
        pool.append(own[own["unit_id"].notna()])
    units = pd.concat(pool, ignore_index=True) if pool else pd.DataFrame(columns=["recording_id", "unit_id", "speaker", "text", "before"])
    if len(units) > n:
        units = units.iloc[np.sort(rng.choice(len(units), size=n, replace=False))]
    units = units.sort_values(["recording_id", "t_start"], kind="stable").reset_index(drop=True)
    return units.assign(id=[f"d{index + 1}" for index in range(len(units))])


def describe_prompt(aim: str, batch: pd.DataFrame) -> tuple[str, str]:
    system = "\n".join([
        "You read units of recorded conversations to help researchers build a coding scheme. Speakers are named by letters.",
        f"What the codes are for: {aim.strip()}",
        "For each unit, say in two to five words what the speaker does with it, as it bears on that aim. "
        "Use the same words for the same thing across units.",
        'Answer with one JSON object and nothing else: {"labels": [{"id": "<unit id>", "label": "<two to five words>"}]}, '
        "one entry per unit, in the order given.",
    ])
    user = ["Units:"] + [
        f"id {row.id} | speaker {row.speaker}{f' (just before, {row.before})' if row.before else ''}: {row.text}"
        for row in batch.itertuples(index=False)
    ]
    return system, "\n".join(user)


def gather_prompt(aim: str, counts: pd.Series) -> tuple[str, str]:
    system = "\n".join([
        "You help researchers build a coding scheme from what speakers did in recorded conversations.",
        f"What the codes are for: {aim.strip()}",
        f"Below are short descriptions of units, each with how many units it was given to. Gather them into between 3 and "
        f"{MAX_CODES} codes that cover most units, each unit taking exactly one code. A description that fits no code is left out.",
        'Answer with one JSON object and nothing else: {"codes": [{"code": "<short name, lowercase with underscores>", '
        '"description": "<one or two sentences>", "criteria": ["<what must hold>"], '
        '"boundary_cases": ["<how to tell it from the nearest code>"], "labels": ["<descriptions it gathers, as written below>"]}]}.',
    ])
    return system, "\n".join(["Descriptions:", *(f"- {label} ({n})" for label, n in counts.items())])


def _plain(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _code_name(text) -> str:
    return re.sub(r"[^0-9a-z]+", "_", _plain(text)).strip("_")


def describe_units(backend: LLMBackend, aim: str, units: pd.DataFrame, *, batch_size: int, seed: int | None) -> pd.Series:
    """Each unit's description, blank where the model gave none."""
    said: dict[str, str] = {}
    for start in range(0, len(units), batch_size):
        batch = units.iloc[start : start + batch_size]
        answer = answer_object(backend.complete(*describe_prompt(aim, batch), seed=seed).text)
        for item in answer.get("labels") or []:
            if isinstance(item, dict) and _plain(item.get("label")):
                said[str(item.get("id", "")).strip()] = _plain(item.get("label"))
    return pd.Series([said.get(unit_id, "") for unit_id in units["id"]], index=units.index, dtype="object")


def draft_codebook(
    backend: LLMBackend, aim: str, units: pd.DataFrame, *, name: str, unit: str, batch_size: int = 20, seed: int = 0,
    takes_seed: bool = True,
) -> Draft:
    rng = np.random.default_rng(seed)
    request_seed = seed if takes_seed else None
    units = units.assign(description=describe_units(backend, aim, units, batch_size=batch_size, seed=request_seed))
    counts = units.loc[units["description"] != "", "description"].value_counts()
    answer = answer_object(backend.complete(*gather_prompt(aim, counts), seed=request_seed).text) if not counts.empty else {}

    claimed: dict[str, str] = {}
    definitions, gathered = [], {}
    for item in answer.get("codes") or []:
        if not isinstance(item, dict):
            continue
        code = _code_name(item.get("code"))
        if not code or code in gathered or code in RESERVED_LABELS:
            continue
        # a description given to two codes stays with the first
        mine = [label for label in map(_plain, item.get("labels") or []) if label in counts.index and label not in claimed]
        claimed.update(dict.fromkeys(mine, code))
        gathered[code] = mine
        members = units[units["description"].isin(mine)]
        picked = members.iloc[np.sort(rng.choice(len(members), size=min(EXAMPLES, len(members)), replace=False))] if len(members) else members
        definitions.append(CodeDefinition(
            code=code, description=str(item.get("description") or "").strip(),
            criteria=[str(text).strip() for text in item.get("criteria") or [] if str(text).strip()],
            boundary_cases=[str(text).strip() for text in item.get("boundary_cases") or [] if str(text).strip()],
            positive_examples=list(dict.fromkeys(picked["text"].astype(str).str.strip())),
        ))
    units = units.assign(code=[claimed.get(description) for description in units["description"]])
    left_out = {label: int(n) for label, n in counts.items() if label not in claimed}
    return Draft(Codebook(name=name, unit=unit, codes=definitions), units, gathered, left_out)


def draft_text(draft: Draft, aim: str, model: str, recordings: int) -> str:
    units = draft.units
    lines = [
        f"# Draft codebook {draft.codebook.name}", "",
        f"Drafted by {model} from {len(units)} {draft.codebook.unit}s drawn at random from {recordings} recording(s), for: {aim.strip()}. "
        "Each unit was first described in a few words, and the descriptions were then gathered into codes. "
        "Choosing, merging and defining the codes is left to you. The examples are participants' own words from the "
        "transcripts; check them before sharing the codebook.", "",
    ]
    counts = units["description"].value_counts()
    for definition in draft.codebook.codes:
        mine = draft.gathered.get(definition.code, [])
        lines += [f"## {definition.code} ({int(counts.reindex(mine).fillna(0).sum())} units)", "", definition.description or "(no description)", ""]
        lines += ["Gathers:", *(f"- {label} ({int(counts.get(label, 0))})" for label in mine), ""] if mine else ["Gathers no description of a sampled unit.", ""]
    if draft.left_out:
        lines += ["## Left out", "", "Descriptions gathered into no code:", *(f"- {label} ({n})" for label, n in draft.left_out.items()), ""]
    blank = int((units["description"] == "").sum())
    if blank:
        lines += [f"{blank} unit(s) got no description from the model.", ""]
    return "\n".join(lines).rstrip() + "\n"
