"""One prompt set for every backend; switching provider must not change coding logic."""

from __future__ import annotations

import pandas as pd

from macaw.coding.codebook import Codebook

ABSTAIN = "abstain"


def build_annotation_prompt(
    codebook: Codebook,
    units: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    allow_abstain: bool = True,
) -> tuple[str, str]:
    """Returns (system, user). units and context carry unit_id, speaker, t_start, t_end, gap_before and text."""
    lines = [
        "You code units of a recorded conversation with the codebook below. Speakers are named by letters.",
        f"Codebook: {codebook.name}. Each unit gets exactly one code.",
        "",
    ]
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
        user.extend(_describe(row) for row in context.itertuples(index=False))
        user.append("")
    user.append("Units to code:")
    user.extend(f"id {row.unit_id} | {_describe(row)}" for row in units.itertuples(index=False))
    return "\n".join(lines), "\n".join(user)


def _describe(row) -> str:
    gap = row.gap_before
    if pd.isna(gap):
        timing = "opens the conversation"
    elif gap < 0:
        timing = f"overlaps the talk before by {-gap:.1f} s"
    else:
        timing = f"after a {gap:.1f} s pause"
    return f"speaker {row.speaker} | at {row.t_start:.1f} s, lasts {row.t_end - row.t_start:.1f} s, {timing}: {row.text}"


def build_observation_prompt(utterances: pd.DataFrame, context: pd.DataFrame | None = None) -> tuple[str, str]:
    """No codebook argument on purpose: the observation pass must not know the categories."""
    raise NotImplementedError
