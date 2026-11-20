"""The lines an answer names as its evidence, checked mechanically against the transcript it was shown."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from macaw.coding.llm.parsing import answer_object
from macaw.coding.scales import ALL, RatingScale
from macaw.extractors.base import QCWarning

EVIDENCE_NAME = "evidence.parquet"
EVIDENCE_COLUMNS = [
    "recording_id", "segment", "scale", "rater", "sample", "line", "speaker", "t_start", "t_end", "quote", "said", "problem",
]
NO_LIST = "no list of evidence"
NO_LINE, OTHER_SPEAKER, NO_QUOTE, NOT_IN_LINE = "no such line", "said by another person", "no words copied", "words not in the line"
RUNS_ON = "words run on into the next or last line"
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def cited(text: str) -> list | None:
    """The evidence list an answer holds, None when it holds none."""
    found = answer_object(text).get("evidence")
    return found if isinstance(found, list) else None


def normalised(text: str) -> str:
    """Lower case, punctuation and apostrophes gone, single spaces: what two copies of the same words share."""
    return " ".join(_NOT_WORD.sub("", str(text).replace("’", "'").lower()).split())


def _number(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def check_cited(items: list, lines: pd.DataFrame, scale: RatingScale, start: float) -> list[dict]:
    """A row per item: the line it names with its speaker, recording-clock times and words, and what failed. A line has to
    exist, be said by the person a one-person scale rates, and hold the copied words."""
    by_number = {int(row.line): row for row in lines.itertuples(index=False)}
    rows = []
    for item in items:
        number = _number(item.get("line") if isinstance(item, dict) else item)
        quote = str(item.get("quote") or "").strip() if isinstance(item, dict) else ""
        row = by_number.get(number) if number is not None else None
        found = {"line": number, "speaker": None, "t_start": np.nan, "t_end": np.nan, "quote": quote, "said": None}
        problems = []
        if row is None:
            problems.append(NO_LINE)
        else:
            found.update(speaker=str(row.speaker), t_start=float(row.t_start) + start, t_end=float(row.t_end) + start, said=str(row.text))
            if scale.person != ALL and str(row.speaker) != scale.person:
                problems.append(OTHER_SPEAKER)
            if not normalised(quote):
                problems.append(NO_QUOTE)
            elif not _holds(row.text, quote):
                neighbours = [by_number.get(number + step) for step in (-1, 1)]
                problems.append(RUNS_ON if _runs_on(row.text, quote, neighbours) else NOT_IN_LINE)
        rows.append({**found, "problem": "; ".join(problems)})
    return rows


def _holds(text: str, quote: str) -> bool:
    return f" {normalised(quote)} " in f" {normalised(text)} "


def _runs_on(text: str, quote: str, neighbours: list) -> bool:
    """The quotation starts in the cited line and ends in the next, or starts in the last line and ends in the cited one."""
    before, after = (neighbour.text if neighbour is not None else None for neighbour in neighbours)
    if before is not None and not _holds(before, quote) and _holds(f"{before} {text}", quote):
        return True
    return after is not None and not _holds(after, quote) and _holds(f"{text} {after}", quote)


def evidence_table(answers: pd.DataFrame, prompts: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cited line of every answer, and per answer how many it cited and what failed: (evidence, answers with
    n_evidence and evidence_problem)."""
    shown = {(prompt[0], prompt[3].scale): prompt for prompt in prompts}
    rows, counts, problems = [], [], []
    for answer in answers.itertuples(index=False):
        prompt = shown[(answer.segment, answer.scale)]
        items = cited(answer.answer)
        if items is None:
            counts.append(pd.NA)
            problems.append(NO_LIST)
            continue
        checked = check_cited(items, prompt[6], prompt[3], prompt[1])
        rows += [
            {"recording_id": answer.recording_id, "segment": answer.segment, "scale": answer.scale, "rater": answer.rater, "sample": answer.sample, **row}
            for row in checked
        ]
        counts.append(len(checked))
        problems.append(_tally([row["problem"] for row in checked]))
    table = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS).astype({"line": "Int64", "sample": "int64", "t_start": float, "t_end": float})
    return table, answers.assign(n_evidence=pd.array(counts, dtype="Int64"), evidence_problem=problems)


def _tally(problems: list[str]) -> str:
    kinds = pd.Series([kind for problem in problems if problem for kind in problem.split("; ")], dtype="object")
    return "; ".join(f"{kind} {count}" for kind, count in kinds.value_counts().sort_index().items())


def evidence_warnings(table: pd.DataFrame, answers: pd.DataFrame, recording_id: str) -> list[QCWarning]:
    bad = table[table["problem"] != ""]
    missing = int((answers["evidence_problem"] == NO_LIST).sum())
    if bad.empty and not missing:
        return []
    said = []
    if len(bad):
        said.append(f"{len(bad)} of the {len(table)} lines the model cited failed the check ({_tally(bad['problem'].tolist())})")
    if missing:
        said.append(f"{missing} of {len(answers)} answers gave no list of evidence")
    return [
        QCWarning(
            check="rating_evidence_unchecked", recording_id=recording_id, subject_id=None, severity="warn",
            message="; ".join(said) + ". The scores were kept; evidence.parquet marks each cited line.",
        )
    ]
