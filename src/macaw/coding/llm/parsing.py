from __future__ import annotations

import json
import re

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.coding.llm.prompts import ABSTAIN

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CLOSERS = {"{": "}", "[": "]"}


def parse_annotation_response(
    text: str, codebook: Codebook, expected_ids: list[str], *, allow_abstain: bool = True
) -> pd.DataFrame:
    """Missing or unparseable items become abstentions, never silent drops."""
    by_code = {code.casefold(): code for code in codebook.labels()}
    if allow_abstain:
        by_code[ABSTAIN] = ABSTAIN
    labels = dict.fromkeys(expected_ids, ABSTAIN)
    for item in _items(text):
        if not isinstance(item, dict):
            continue
        unit_id = str(item.get("id", "")).strip()
        label = by_code.get(str(item.get("label", "")).strip().casefold())
        if unit_id in labels and label is not None:
            labels[unit_id] = label
    return pd.DataFrame({"unit_id": list(labels), "label": list(labels.values())})


def _items(text: str) -> list:
    try:
        parsed = json.loads(repair_json(text))
    except ValueError:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("labels", next((value for value in parsed.values() if isinstance(value, list)), []))
    return parsed if isinstance(parsed, list) else []


def parse_observation_response(text: str, expected_ids: list[str]) -> pd.DataFrame:
    raise NotImplementedError


def parse_cleanup_response(text: str, expected_ids: list[str]) -> pd.DataFrame:
    raise NotImplementedError


def parse_refinement_response(text: str) -> Codebook:
    raise NotImplementedError


def repair_json(text: str) -> str:
    """Small local models often wrap JSON in prose or truncate it."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text.strip()
    text = text[min(starts):]

    end, _, in_string = _scan(text)
    if end is not None:
        return text[: end + 1]
    if in_string:
        text += '"'
    # a truncated last item is dropped rather than half kept
    text = re.sub(r",\s*\{[^{}]*$", "", text.rstrip())
    text = re.sub(r",\s*$", "", text)
    return text + "".join(reversed(_scan(text)[1]))


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
