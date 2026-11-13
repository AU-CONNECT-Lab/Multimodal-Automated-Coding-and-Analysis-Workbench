"""One prompt set for every backend; switching provider must not change coding logic."""

from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from macaw.coding.codebook import Codebook
from macaw.coding.scales import ALL, RatingScale
from macaw.extractors.audio.content.diarization import OTHER

ABSTAIN = "abstain"
VOICE = ("pitch", "loudness", "rate")
FRAMES = "frames"
# where in a unit its frames are taken, as shares of its length
FRAME_AT = (1 / 6, 1 / 2, 5 / 6)
_VOICE_TERMS = {"pitch": "pitch in semitones", "loudness": "loudness in dB", "rate": "speech rate in percent"}


def build_annotation_prompt(
    codebook: Codebook,
    units: pd.DataFrame,
    context: pd.DataFrame | None = None,
    *,
    allow_abstain: bool = True,
    show: Sequence[str] = ("timing",),
    notes: str = "",
    examples: pd.DataFrame | None = None,
    reasons: bool = False,
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
    if FRAMES in show:
        lines.append(
            f"The unit to code comes with up to {len(FRAME_AT)} frames from the video, taken near its start, middle and end, "
            "in that order; one the video does not reach comes with fewer or none."
        )
    if notes.strip():
        lines.append(f"Notes from the researchers: {notes.strip()}")
    lines += codebook_lines(codebook)
    if examples is not None and not examples.empty:
        lines.append("Units people coded in other conversations, as examples:")
        for row in examples.itertuples(index=False):
            said = f"{row.speaker or 'Unit'}: {row.text}"
            lines.append(f"- Before it, {row.before}\n  {said}" if row.before else f"- {said}")
            lines.append(f"  Code: {row.code}." + (f" Why: {row.why}" if row.why else ""))
        lines.append("")
    choices = ", ".join(f'"{code}"' for code in codebook.labels())
    if allow_abstain:
        lines.append(f'If a unit gives too little to decide, answer "{ABSTAIN}" rather than guess.')
        choices += f', "{ABSTAIN}"'
    entry = '{"id": "<unit id>", "label": "<code>", "reason": "<what it rests on>"}' if reasons else '{"id": "<unit id>", "label": "<code>"}'
    lines.append(
        f'Answer with one JSON object and nothing else: {{"labels": [{entry}]}}, '
        f"one entry per unit to code, in the order given. A label is one of {choices}."
    )
    if reasons:
        lines.append("A reason names in a few words what the code rests on: the words, the timing, the voice or the picture.")

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


def codebook_lines(codebook: Codebook) -> list[str]:
    lines = [f"Codebook: {codebook.name}. Each unit gets exactly one code.", ""]
    for definition in codebook.codes:
        lines.append(f"## {definition.code}")
        if definition.description:
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
    return lines


def _describe(row, show: Sequence[str]) -> str:
    return f"{' | '.join([f'speaker {row.speaker}', *_details(row, show)])}: {row.text}"


def _details(row, show: Sequence[str]) -> list[str]:
    """The timing and voice figures of a line that show asks for."""
    parts = []
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
    return parts


def build_rating_prompt(
    scale: RatingScale,
    lines: pd.DataFrame,
    *,
    people: Sequence[str],
    minutes: float,
    show: Sequence[str] = (),
    notes: str = "",
    reasons: bool = False,
) -> tuple[str, str]:
    """Returns (system, user). lines carry speaker (a role), t_start and t_end from the start of what is rated, gap_before,
    text and the voice items of show. Only the end of the user part names the scale, so every scale asked about one
    conversation starts the same and a server can reuse what it read."""
    system = [
        "You rate a recorded conversation on one rating scale, as a trained observer would. The transcript gives what was "
        "said in time order; each line starts with the role of the person who said it.",
        f"The people recorded: {', '.join(people)}.",
    ]
    if (lines["speaker"] == OTHER).any():
        system.append(f"The speaker {OTHER} is someone besides the people recorded, such as a researcher in the room.")
    if "timing" in show:
        system.append(
            "Each line gives its start and length in seconds from the start of the transcript, and the pause before it or "
            "how far it overlaps the talk before."
        )
    voice = [item for item in VOICE if item in show]
    if voice:
        terms = ", ".join(_VOICE_TERMS[item] for item in voice)
        system.append(
            f"Each line gives its {terms}, against the median of that speaker's lines in the whole recording; one that could "
            "not be measured is left out."
        )
    if notes.strip():
        system.append(f"Notes from the researchers: {notes.strip()}")
    entry = '{"score": <score>, "reason": "<what it rests on>"}' if reasons else '{"score": <score>}'
    system.append(f"Answer with one JSON object and nothing else: {entry}, the score a whole number in the scale's range.")
    if reasons:
        system.append("The reason names in a few words what the score rests on: what was said, the timing or the voice.")

    user = [f"Transcript ({minutes:.1f} minutes):"]
    user.extend(f"{' | '.join([str(row.speaker), *_details(row, show)])}: {row.text}" for row in lines.itertuples(index=False))
    user += ["", "Rate the conversation on this scale.", f"Scale: {scale.scale}", f"Who is rated: {_rated(scale, people)}"]
    user.append(f"Scores: whole numbers from {scale.low} to {scale.high}.")
    if scale.definition:
        user.append(f"Definition: {scale.definition}")
    if scale.anchors:
        user.append("Anchors:")
        user.extend(f"{score}: {text}" for score, text in sorted(scale.anchors.items()))
        if len(scale.anchors) < scale.high - scale.low + 1:
            user.append("Scores without an anchor lie between the anchors around them.")
    if scale.rules:
        user.append("Rules:")
        user.extend(f"- {rule}" for rule in scale.rules)
    user.append("Answer with the JSON object.")
    return "\n".join(system), "\n".join(user)


def _rated(scale: RatingScale, people: Sequence[str]) -> str:
    if scale.person != ALL:
        return f"the {scale.person}"
    named = [f"the {person}" for person in people]
    return (named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]) + " together"
