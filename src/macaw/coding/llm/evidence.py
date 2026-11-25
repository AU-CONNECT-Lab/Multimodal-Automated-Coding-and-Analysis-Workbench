"""The lines and frames an answer names as its evidence, checked mechanically against what it was shown."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from macaw.coding.llm.parsing import answer_object
from macaw.coding.scales import ALL, RatingScale
from macaw.extractors.base import QCWarning

EVIDENCE_NAME = "evidence.parquet"
EVIDENCE_COLUMNS = [
    "recording_id", "segment", "scale", "rater", "sample", "kind", "line", "speaker", "t_start", "t_end", "quote", "said",
    "frame", "frame_t", "frame_from", "frame_from_t", "seen", "problem",
]
LINE, FRAME = "line", "frame"
# the frames a prompt showed, numbered as it showed them
SHOWN_FRAME_COLUMNS = ["frame", "t", "from_recording", "from_t"]
NO_LIST = "no list of evidence"
NO_LINE, OTHER_SPEAKER, NO_QUOTE, NOT_IN_LINE = "no such line", "said by another person", "no words copied", "words not in the line"
RUNS_ON = "words run on into the next or last line"
NO_FRAME, NOTHING_SEEN = "no such frame", "nothing said of what was seen"
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)
# the label each line of the transcript ends with, copied as the line's number
_LINE_LABEL = re.compile(r"\[?\s*line\s*(\d+)\s*\]?", re.IGNORECASE)
# the label before each frame, copied as the frame's number
_FRAME_LABEL = re.compile(r"\[?\s*frame\s*(\d+)\s*(?:,\s*t\s*=\s*[\d.]+\s*s)?\s*\]?", re.IGNORECASE)


def cited(text: str) -> list | None:
    """The evidence list an answer holds, None when it holds none."""
    found = answer_object(text).get("evidence")
    return found if isinstance(found, list) else None


def normalised(text: str) -> str:
    """Lower case, punctuation and apostrophes gone, single spaces: what two copies of the same words share."""
    return " ".join(_NOT_WORD.sub("", str(text).replace("’", "'").lower()).split())


def _number(value, label: re.Pattern = _LINE_LABEL) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    labelled = label.fullmatch(text)
    if labelled:
        return int(labelled.group(1))
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def is_frame(item) -> bool:
    return isinstance(item, dict) and "frame" in item and "line" not in item


def check_cited(items: list, lines: pd.DataFrame, scale: RatingScale, frames: pd.DataFrame | None = None) -> list[dict]:
    """A row per item: the line it names with its speaker, recording-clock times and words, or the frame it names with its
    time and what the answer saw, and what failed. A line has to exist, be said by the person a one-person scale rates,
    and hold the copied words; a frame has to have been shown and come with words on what it shows."""
    by_number = {int(row.line): row for row in lines.itertuples(index=False)}
    by_frame = {} if frames is None else {int(row.frame): row for row in frames.itertuples(index=False)}
    rows = []
    for item in items:
        if is_frame(item):
            rows.append(_checked_frame(item, by_frame))
            continue
        number = _number(item.get("line") if isinstance(item, dict) else item)
        quote = str(item.get("quote") or "").strip() if isinstance(item, dict) else ""
        row = by_number.get(number) if number is not None else None
        found = {**_EMPTY, "kind": LINE, "line": number, "quote": quote}
        problems = []
        if row is None:
            problems.append(NO_LINE)
        else:
            found.update(speaker=str(row.speaker), t_start=float(row.t_start), t_end=float(row.t_end), said=str(row.text))
            if scale.person != ALL and str(row.speaker) != scale.person:
                problems.append(OTHER_SPEAKER)
            if not normalised(quote):
                problems.append(NO_QUOTE)
            elif not _holds(row.text, quote):
                neighbours = [by_number.get(number + step) for step in (-1, 1)]
                problems.append(RUNS_ON if _runs_on(row.text, quote, neighbours) else NOT_IN_LINE)
        rows.append({**found, "problem": "; ".join(problems)})
    return rows


_EMPTY = {
    "kind": None, "line": None, "speaker": None, "t_start": np.nan, "t_end": np.nan, "quote": "", "said": None,
    "frame": None, "frame_t": np.nan, "frame_from": None, "frame_from_t": np.nan, "seen": "",
}


def _checked_frame(item: dict, by_frame: dict) -> dict:
    """No speaker is checked: a frame shows whoever is in it."""
    number = _number(item.get("frame"), _FRAME_LABEL)
    seen = str(item.get("seen") or "").strip()
    row = by_frame.get(number) if number is not None else None
    found = {**_EMPTY, "kind": FRAME, "frame": number, "seen": seen}
    problems = []
    if row is None:
        problems.append(NO_FRAME)
    else:
        found.update(frame_t=float(row.t), frame_from=str(row.from_recording), frame_from_t=float(row.from_t))
    if not normalised(seen):
        problems.append(NOTHING_SEEN)
    return {**found, "problem": "; ".join(problems)}


def _holds(text: str, quote: str) -> bool:
    return f" {normalised(quote)} " in f" {normalised(text)} "


def _runs_on(text: str, quote: str, neighbours: list) -> bool:
    """The quotation starts in the cited line and ends in the next, or starts in the last line and ends in the cited one."""
    before, after = (neighbour.text if neighbour is not None else None for neighbour in neighbours)
    if before is not None and not _holds(before, quote) and _holds(f"{before} {text}", quote):
        return True
    return after is not None and not _holds(after, quote) and _holds(f"{text} {after}", quote)


def evidence_table(answers: pd.DataFrame, prompts: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cited line and frame of every answer, and per answer how many it cited of each and what failed: (evidence,
    answers with n_evidence, n_evidence_lines, n_evidence_frames and evidence_problem)."""
    shown = {(prompt[0], prompt[3].scale): prompt for prompt in prompts}
    rows, counts, problems = [], [], []
    for answer in answers.itertuples(index=False):
        prompt = shown[(answer.segment, answer.scale)]
        items = cited(answer.answer)
        if items is None:
            counts.append((pd.NA, pd.NA, pd.NA))
            problems.append(NO_LIST)
            continue
        checked = check_cited(items, prompt[6], prompt[3], getattr(prompt, "frames", None))
        rows += [
            {"recording_id": answer.recording_id, "segment": answer.segment, "scale": answer.scale, "rater": answer.rater, "sample": answer.sample, **row}
            for row in checked
        ]
        framed = sum(row["kind"] == FRAME for row in checked)
        counts.append((len(checked), len(checked) - framed, framed))
        problems.append(_tally([row["problem"] for row in checked]))
    table = pd.DataFrame(rows, columns=EVIDENCE_COLUMNS).astype({
        "line": "Int64", "frame": "Int64", "sample": "int64", "t_start": float, "t_end": float, "frame_t": float, "frame_from_t": float,
    })
    total, lines, frames = (pd.array([count[index] for count in counts], dtype="Int64") for index in range(3))
    return table, answers.assign(n_evidence=total, n_evidence_lines=lines, n_evidence_frames=frames, evidence_problem=problems)


def _tally(problems: list[str]) -> str:
    kinds = pd.Series([kind for problem in problems if problem for kind in problem.split("; ")], dtype="object")
    return "; ".join(f"{kind} {count}" for kind, count in kinds.value_counts().sort_index().items())


def evidence_warnings(table: pd.DataFrame, answers: pd.DataFrame, recording_id: str) -> list[QCWarning]:
    bad = table[table["problem"] != ""]
    missing = int((answers["evidence_problem"] == NO_LIST).sum())
    if bad.empty and not missing:
        return []
    said = []
    cited_items = "lines and frames" if (table["kind"] == FRAME).any() else "lines"
    if len(bad):
        said.append(f"{len(bad)} of the {len(table)} {cited_items} the model cited failed the check ({_tally(bad['problem'].tolist())})")
    if missing:
        said.append(f"{missing} of {len(answers)} answers gave no list of evidence")
    each = "line and frame" if cited_items != "lines" else "line"
    return [
        QCWarning(
            check="rating_evidence_unchecked", recording_id=recording_id, subject_id=None, severity="warn",
            message="; ".join(said) + f". The scores were kept; evidence.parquet marks each cited {each}.",
        )
    ]
