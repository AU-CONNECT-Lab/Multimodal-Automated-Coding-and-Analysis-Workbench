from __future__ import annotations

import json
import re

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.coding.llm.prompts import ABSTAIN
from macaw.exceptions import LLMBackendError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# reasoning models write their thinking first, braces and all
_THINKING = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
_CLOSERS = {"{": "}", "[": "]"}


def parse_annotation_response(
    text: str, codebook: Codebook, expected_ids: list[str], *, allow_abstain: bool = True
) -> pd.DataFrame:
    """A row per expected unit: its code or abstain, the reason, and where the answer gave neither, no label, what was
    wrong and the label it gave. reason and given are blank where none came."""
    by_code = {code.casefold(): code for code in codebook.labels()}
    if allow_abstain:
        by_code[ABSTAIN] = ABSTAIN
    items = [item for item in _items(text) if isinstance(item, dict)]
    labels: dict[str, str | None] = dict.fromkeys(expected_ids)
    reasons, given = dict.fromkeys(expected_ids, ""), dict.fromkeys(expected_ids, "")
    problems = dict.fromkeys(expected_ids, NO_ANSWER_FOR_UNIT if items else UNREADABLE)
    for item in items:
        unit_id = str(item.get("id", "")).strip()
        if unit_id not in labels or labels[unit_id] is not None:
            continue
        said = str(item.get("label", "")).strip()
        label = by_code.get(said.casefold())
        given[unit_id] = said
        if label is None:
            problems[unit_id] = NOT_A_CODE
            continue
        labels[unit_id], problems[unit_id], reasons[unit_id] = label, "", str(item.get("reason") or "").strip()
    return pd.DataFrame({
        "unit_id": list(labels), "label": pd.array(list(labels.values()), dtype="string"), "reason": list(reasons.values()),
        "problem": list(problems.values()), "given": list(given.values()),
    })


UNREADABLE, NOT_WHOLE, OUTSIDE = "unreadable", "not a whole number", "outside the range"
NO_ANSWER_FOR_UNIT, NOT_A_CODE, CUT_OFF = "no answer for this unit", "a label not in the codebook", "cut off before this unit"


def parse_rating(text: str, low: int, high: int) -> tuple[int | None, str, str]:
    """(score, reason, what was wrong). A score that is not whole or lies outside low to high is no score; it is never mended."""
    found = answer_object(text)
    if "score" not in found:
        return None, "", UNREADABLE
    value, reason = found["score"], str(found.get("reason") or "").strip()
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None, reason, NOT_WHOLE
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not float(value).is_integer():
        return None, reason, NOT_WHOLE
    if not low <= int(value) <= high:
        return None, reason, OUTSIDE
    return int(value), reason, ""


def answer_object(text: str) -> dict:
    """The JSON object an answer holds, or an empty one when there is none."""
    try:
        parsed = json.loads(repair_json(text))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def whole_answer(response) -> dict:
    """The JSON object of an answer that came to its end; one cut off is refused, since what is left reads as a shorter answer."""
    if response.cut_off:
        raise LLMBackendError(
            "the model's answer was cut off at coding.llm.max_tokens before it ended. A model that reasons before it "
            "answers spends tokens on that first; raise coding.llm.max_tokens."
        )
    return answer_object(response.text)


def _items(text: str) -> list:
    try:
        parsed = json.loads(repair_json(text))
    except ValueError:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("labels", next((value for value in parsed.values() if isinstance(value, list)), []))
    return parsed if isinstance(parsed, list) else []


def repair_json(text: str) -> str:
    """Small local models often wrap JSON in prose or truncate it."""
    text = _THINKING.sub("", text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    starts = [index for index, char in enumerate(text) if char in _CLOSERS]
    if not starts:
        return text.strip()
    first = None
    for start in starts:
        end, _, in_string = _scan(text[start:])
        if end is None:
            # the first value that never closes is a cut-off answer
            text = text[start:]
            break
        value = text[start: start + end + 1]
        if _answer_shaped(value):
            return value
        first = first or value
    else:
        # bracketed prose before the answer, such as "[line 12]", is passed over
        return first
    if in_string:
        text += '"'
    # a truncated last item is dropped rather than half kept
    text = re.sub(r",\s*\{[^{}]*$", "", text.rstrip())
    text = re.sub(r",\s*$", "", text)
    return text + "".join(reversed(_scan(text)[1]))


def _answer_shaped(text: str) -> bool:
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(item, dict) for item in value))


def _scan(text: str) -> tuple[int | None, list[str], bool]:
    """(index where the first value closes or None, closers still open, inside a string at the end)."""
    stack: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in _CLOSERS:
            stack.append(_CLOSERS[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index, stack, False
    return None, stack, in_string
