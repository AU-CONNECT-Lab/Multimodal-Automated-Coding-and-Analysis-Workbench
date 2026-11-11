"""A first codebook drafted from the transcripts: the model says what each sampled unit does, then gathers those
descriptions into codes. Choosing and defining the codes in the end is the researchers' work."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macaw.coding.codebook import RESERVED_LABELS, Codebook, CodeDefinition
from macaw.coding.llm.backends import LLMBackend
from macaw.coding.llm.parsing import whole_answer

N_UNITS = 200
MAX_CODES = 12
EXAMPLES = 3
# requests to merge codes when the model gives more than MAX_CODES
MERGE_TRIES = 2
# descriptions shown in one request to place those gathered into no code
PLACE_BATCH = 60


@dataclass
class Draft:
    codebook: Codebook
    units: pd.DataFrame                                   # the sample, with each unit's description and code
    gathered: dict[str, list[str]] = field(default_factory=dict)   # code -> the descriptions it gathers
    left_out: dict[str, int] = field(default_factory=dict)         # description -> units, gathered into no code
    placed: dict[str, str] = field(default_factory=dict)           # description -> the code it was placed in when shown again
    proposed: list[int] = field(default_factory=list)              # how many codes each gathering or merging answer held


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


def merge_prompt(aim: str, codes: list[dict], counts: pd.Series) -> tuple[str, str]:
    system = "\n".join([
        "You help researchers build a coding scheme from what speakers did in recorded conversations.",
        f"What the codes are for: {aim.strip()}",
        f"Below are {len(codes)} codes, each with how many units it holds; at most {MAX_CODES} are wanted. Merge codes that "
        f"do the same thing so that no more than {MAX_CODES} remain, each code below going into exactly one.",
        'Answer with one JSON object and nothing else: {"codes": [{"code": "<short name, lowercase with underscores>", '
        '"description": "<one or two sentences>", "criteria": ["<what must hold>"], '
        '"boundary_cases": ["<how to tell it from the nearest code>"], "merges": ["<codes it takes, as written below>"]}]}.',
    ])
    lines = [f"- {code['code']} ({_size(code, counts)}): {code['description']}" for code in codes]
    return system, "\n".join(["Codes:", *lines])


def place_prompt(aim: str, codes: list[dict], rest: pd.Series) -> tuple[str, str]:
    system = "\n".join([
        "You help researchers build a coding scheme from what speakers did in recorded conversations.",
        f"What the codes are for: {aim.strip()}",
        f"The scheme has these {len(codes)} codes and no others: {', '.join(code['code'] for code in codes)}. Below are their "
        "definitions, and numbered descriptions of units that are in no code yet, each with how many units it was given to. "
        "For each description, choose the one code of the scheme whose definition it meets, or none if it meets none of them. "
        "Do not stretch a code to take a description in.",
        'Answer with one JSON object and nothing else: {"placed": [{"n": <number of the description>, '
        f'"code": "<{" | ".join(code["code"] for code in codes)} | none>"}}]}}, one entry per description.',
    ])
    listed = [f"{number}. {label} ({n})" for number, (label, n) in enumerate(rest.items(), start=1)]
    return system, "\n".join(["Codes:", *(f"- {code['code']}: {code['description']}" for code in codes), "", "Descriptions to place:", *listed])


def _size(code: dict, counts: pd.Series) -> int:
    return int(counts.reindex(code["labels"]).fillna(0).sum())


def _codes(items, labels_of) -> list[dict]:
    """The codes an answer gives, named and cleaned; a description given to two codes stays with the first."""
    claimed: set[str] = set()
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        code = _code_name(item.get("code"))
        if not code or code in RESERVED_LABELS or any(known["code"] == code for known in out):
            continue
        mine = [label for label in dict.fromkeys(labels_of(item)) if label not in claimed]
        claimed.update(mine)
        out.append({
            "code": code, "description": str(item.get("description") or "").strip(), "labels": mine,
            "criteria": [str(text).strip() for text in item.get("criteria") or [] if str(text).strip()],
            "boundary_cases": [str(text).strip() for text in item.get("boundary_cases") or [] if str(text).strip()],
        })
    return out


def _merged(backend: LLMBackend, aim: str, codes: list[dict], counts: pd.Series, seed) -> list[dict]:
    """The codes merged as the model says; a code it places nowhere stays as it was."""
    before = {code["code"]: code for code in codes}
    answer = whole_answer(backend.complete(*merge_prompt(aim, codes, counts), seed=seed))

    known = {code: [code] for code in before}

    def labels_of(item):
        # a merged code that keeps the name of one before holds that one too, listed again or not
        named = [_code_name(item.get("code")), *(old for text in item.get("merges") or [] for old in _listed(text, known, _code_name))]
        return [label for old in dict.fromkeys(named) if old in before for label in before[old]["labels"]]

    merged = [code for code in _codes(answer.get("codes"), labels_of) if code["labels"]]
    placed = {label for code in merged for label in code["labels"]}
    names = {code["code"] for code in merged}
    return merged + [code for code in codes if not set(code["labels"]) & placed and code["code"] not in names]


def _placed(backend: LLMBackend, aim: str, codes: list[dict], rest: pd.Series, seed) -> dict[str, str]:
    """Where the model puts each description gathered into no code; one it calls none, or answers nothing for, is absent."""
    names = {code["code"] for code in codes}
    placed: dict[str, str] = {}
    for start in range(0, len(rest), PLACE_BATCH):
        batch = rest.iloc[start : start + PLACE_BATCH]
        answer = whole_answer(backend.complete(*place_prompt(aim, codes, batch), seed=seed))
        for item in answer.get("placed") or []:
            if not isinstance(item, dict) or not str(item.get("n", "")).strip().isdigit():
                continue
            number, code = int(str(item["n"]).strip()), _code_name(item.get("code"))
            if 1 <= number <= len(batch) and code in names:
                placed.setdefault(str(batch.index[number - 1]), code)
    return placed


def _plain(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _code_name(text) -> str:
    return re.sub(r"[^0-9a-z]+", "_", _plain(text)).strip("_")


def _loose(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text).strip()


def _listed(text, known: dict[str, list[str]], key=_loose) -> list[str]:
    """The listed entries the model means by one it copied, with or without the count the list gave beside it."""
    plain = _plain(text)
    return known.get(key(plain)) or known.get(key(re.sub(r"\(\d+\)$", "", plain))) or []


def describe_units(backend: LLMBackend, aim: str, units: pd.DataFrame, *, batch_size: int, seed: int | None) -> pd.Series:
    """Each unit's description, blank where the model gave none."""
    said: dict[str, str] = {}
    for start in range(0, len(units), batch_size):
        batch = units.iloc[start : start + batch_size]
        answer = whole_answer(backend.complete(*describe_prompt(aim, batch), seed=seed))
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
    answer = whole_answer(backend.complete(*gather_prompt(aim, counts), seed=request_seed)) if not counts.empty else {}
    known: dict[str, list[str]] = {}
    for label in counts.index:
        known.setdefault(_loose(label), []).append(label)
    codes = _codes(answer.get("codes"), lambda item: [label for text in item.get("labels") or [] for label in _listed(text, known)])
    proposed = [len(codes)]
    for _ in range(MERGE_TRIES):
        if len(codes) <= MAX_CODES:
            break
        codes = _merged(backend, aim, [code for code in codes if code["labels"]], counts, request_seed)
        proposed.append(len(codes))
    # still too many: the codes holding the most units stay, the others' descriptions are left out
    if len(codes) > MAX_CODES:
        codes = sorted(codes, key=lambda code: -_size(code, counts))[:MAX_CODES]

    rest = counts[~counts.index.isin([label for code in codes for label in code["labels"]])]
    placed = _placed(backend, aim, codes, rest, request_seed) if codes and len(rest) else {}
    claimed: dict[str, str] = {}
    definitions, gathered = [], {}
    for code in codes:
        first = code["labels"]
        code["labels"] = first + [label for label, where in placed.items() if where == code["code"]]
        claimed.update(dict.fromkeys(code["labels"], code["code"]))
        gathered[code["code"]] = code["labels"]
        # examples come from the units the code was made from, where it has any
        members = units[units["description"].isin(first or code["labels"])]
        picked = members.iloc[np.sort(rng.choice(len(members), size=min(EXAMPLES, len(members)), replace=False))] if len(members) else members
        definitions.append(CodeDefinition(
            code=code["code"], description=code["description"], criteria=code["criteria"], boundary_cases=code["boundary_cases"],
            positive_examples=list(dict.fromkeys(picked["text"].astype(str).str.strip())),
        ))
    units = units.assign(code=[claimed.get(description) for description in units["description"]])
    left_out = {label: int(n) for label, n in counts.items() if label not in claimed}
    return Draft(Codebook(name=name, unit=unit, codes=definitions), units, gathered, left_out, placed, proposed)


def draft_text(draft: Draft, aim: str, model: str, recordings: int) -> str:
    units = draft.units
    lines = [
        f"# Draft codebook {draft.codebook.name}", "",
        f"Drafted by {model} from {len(units)} {draft.codebook.unit}s drawn at random from {recordings} recording(s), for: {aim.strip()}. "
        "Each unit was first described in a few words, and the descriptions were then gathered into codes. "
        "Choosing, merging and defining the codes is left to you. The examples are participants' own words from the "
        "transcripts; check them before sharing the codebook.", "",
    ]
    if len(draft.proposed) > 1:
        rounds = ", then ".join(str(n) for n in draft.proposed[1:])
        capped = draft.proposed[-1] > MAX_CODES
        lines += [
            f"The model first proposed {draft.proposed[0]} codes, more than the {MAX_CODES} asked for, and merged them into {rounds} "
            f"when asked{f'; the {MAX_CODES} holding the most units are kept, and the descriptions of the others are under Left out' if capped else ''}.",
            "",
        ]
    if draft.placed:
        lines += [
            f"{len(draft.placed)} description(s) the model first gathered into no code were shown to the model again and placed "
            "in a code; they are marked below. Those it placed in none are under Left out.", "",
        ]
    counts = units["description"].value_counts()
    for definition in draft.codebook.codes:
        mine = draft.gathered.get(definition.code, [])
        lines += [f"## {definition.code} ({int(counts.reindex(mine).fillna(0).sum())} units)", "", definition.description or "(no description)", ""]
        gathers = [f"- {label} ({int(counts.get(label, 0))}){', placed afterwards' if label in draft.placed else ''}" for label in mine]
        lines += ["Gathers:", *gathers, ""] if mine else ["Gathers no description of a sampled unit.", ""]
    if draft.left_out:
        lines += ["## Left out", "", "Descriptions gathered into no code:", *(f"- {label} ({n})" for label, n in draft.left_out.items()), ""]
    blank = int((units["description"] == "").sum())
    if blank:
        lines += [f"{blank} unit(s) got no description from the model.", ""]
    return "\n".join(lines).rstrip() + "\n"
