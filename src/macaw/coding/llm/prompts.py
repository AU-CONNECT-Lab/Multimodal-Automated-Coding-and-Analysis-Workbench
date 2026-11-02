"""One prompt set for every backend; switching provider must not change coding logic."""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.extractors.audio.content.diarization import OTHER

ABSTAIN = "abstain"
VOICE = ("pitch", "loudness", "rate")
_VOICE_TERMS = {"pitch": "pitch in semitones", "loudness": "loudness in dB", "rate": "speech rate in percent"}


def build_annotation_prompt(
    codebook: Codebook,
    units: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    allow_abstain: bool = True,
    show: Sequence[str] = ("timing",),
) -> tuple[str, str]:
    """Returns (system, user). units and context carry unit_id, speaker, t_start, t_end, gap_before and text, and
    the voice items of show; a row of units without a unit_id is shown in its place as context."""
    lines = ["You code units of a recorded conversation with the codebook below. Speakers are named by letters."]
    shown = units["speaker"] if context is None else pd.concat([context["speaker"], units["speaker"]])
    if (shown == OTHER).any():
        lines.append(f"The speaker {OTHER} is someone besides the people recorded, such as a researcher in the room.")
    voice = [item for item in VOICE if item in show]
    if voice:
        terms = ", ".join(_VOICE_TERMS[item] for item in voice)
        lines.append(
            f"Each line gives its {terms}, against the median of that speaker's lines in this conversation; "
            "one that could not be measured is left out."
        )
    lines += [f"Codebook: {codebook.name}. Each unit gets exactly one code.", ""]
    for definition in codebook.codes:
        lines.append(f"## {definition.code}")
        lines.append(definition.description)
        for title, items in (
            ("Criteria", definition.criteria),
            ("Examples of this code", definition.positive_examples),
            ("Not this code", definition.negative_examples),
            ("Boundary cases", definition.boundary_cases),
        ):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"- {item}" for item in items)
        lines.append("")
    choices = ", ".join(f'"{code}"' for code in codebook.labels())
    if allow_abstain:
        lines.append(f'If a unit gives too little to decide, answer "{ABSTAIN}" rather than guess.')
        choices += f', "{ABSTAIN}"'
    lines.append(
        'Answer with one JSON object and nothing else: {"labels": [{"id": "<unit id>", "label": "<code>"}]}, '
        f"one entry per unit to code, in the order given. A label is one of {choices}."
    )

    user = []
    if context is not None and not context.empty:
        user.append("Earlier in the conversation (context only, do not code):")
        user.extend(_describe(row, show) for row in context.itertuples(index=False))
        user.append("")
    if units["unit_id"].isna().any():
        user.append("Units to code, with lines marked context in between that are not to be coded:")
    else:
        user.append("Units to code:")
    user.extend(
        f"context | {_describe(row, show)}" if pd.isna(row.unit_id) else f"id {row.unit_id} | {_describe(row, show)}"
        for row in units.itertuples(index=False)
    )
    return "\n".join(lines), "\n".join(user)


def _describe(row, show: Sequence[str]) -> str:
    parts = [f"speaker {row.speaker}"]
    if "timing" in show:
        gap = row.gap_before
        if pd.isna(gap):
            timing = "opens the conversation"
        elif gap < 0:
            timing = f"overlaps the talk before by {-gap:.1f} s"
        else:
            timing = f"after a {gap:.1f} s pause"
        parts.append(f"at {row.t_start:.1f} s, lasts {row.t_end - row.t_start:.1f} s, {timing}")
    formats = {"pitch": "pitch {:+.1f} st", "loudness": "loudness {:+.1f} dB", "rate": "rate {:+.0f}%"}
    voice = [
        formats[item].format(value)
        for item in VOICE
        if item in show and pd.notna(value := getattr(row, item, None)) and math.isfinite(value)
    ]
    if voice:
        parts.append(", ".join(voice))
    return f"{' | '.join(parts)}: {row.text}"


def build_observation_prompt(utterances: pd.DataFrame, context: pd.DataFrame | None = None) -> tuple[str, str]:
    """No codebook argument on purpose: the observation pass must not know the categories."""
    raise NotImplementedError
