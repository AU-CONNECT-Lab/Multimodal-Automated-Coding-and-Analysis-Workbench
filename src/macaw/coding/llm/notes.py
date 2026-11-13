"""The researchers' note a prompt holds for one recording: the project's own, then the one notes_by gives each of the
recording's entity values."""

from __future__ import annotations

from collections import Counter

from macaw.config.schema import LLMSection
from macaw.exceptions import ConfigError

NOTE_KEYS = ("notes", "notes_by")


def notes_by_key(config, kind: str, version: str | None) -> str:
    """The setting a version's notes_by comes from: its own, or coding.llm's; kind is judges or ratings."""
    own = version is not None and getattr(config.coding, kind)[version].notes_by is not None
    return f"coding.{kind}.{version}.notes_by" if own else "coding.llm.notes_by"


def taken_notes(cfg: LLMSection, entities: dict) -> tuple[dict[str, dict[str, str]], list[tuple[str, str | None]]]:
    """(the notes_by entries this recording takes, each entity whose value notes_by has no note for)."""
    taken, absent = {}, []
    for entity, values in cfg.notes_by.items():
        value = entities.get(entity)
        if value is not None and value in values:
            taken[entity] = {value: values[value]}
        else:
            absent.append((entity, value))
    return taken, absent


def given_note(cfg: LLMSection, entities: dict, where: str) -> str:
    """The note as the prompt holds it; where names the notes_by setting, for the error when a value has none."""
    taken, absent = taken_notes(cfg, entities)
    if absent:
        raise ConfigError(absent_message(where, {pair: [] for pair in absent}))
    parts = [cfg.notes.strip(), *(text.strip() for values in taken.values() for text in values.values())]
    return "\n".join(part for part in parts if part)


def notes_params(cfg: LLMSection, recording) -> dict:
    """notes_by as a step's manifest keeps it: only the notes this recording took."""
    if not cfg.notes_by:
        return {}
    taken, _ = taken_notes(cfg, recording.entities())
    return {"notes_by": {entity: {value: text.strip() for value, text in values.items()} for entity, values in taken.items()}}


def absent_message(where: str, absent: dict[tuple[str, str | None], list[str]]) -> str:
    said = "; ".join(
        f"{entity} {value if value is not None else '(not in the recording name)'}" + (f" ({', '.join(rids)})" if rids else "")
        for (entity, value), rids in absent.items()
    )
    return f"{where} has no note for {said}; add each, with \"\" where it needs none"


def notes_plan(step: str, where: str, cfg: LLMSection, entities: dict[str, dict]) -> str | None:
    """What a run tells the model by entity, for the plan: each value, how many recordings take it, and its note."""
    if not cfg.notes_by:
        return None
    parts = []
    for entity, values in cfg.notes_by.items():
        counts = Counter(own.get(entity) for own in entities.values())
        listed = "; ".join(
            f"{value} ({counts.get(value, 0)} recording(s)): " + (f'"{text.strip()}"' if text.strip() else "none")
            for value, text in values.items()
        )
        parts.append(f"one by {entity} ({where}): {listed}")
    return f"{step} adds to the researchers' note " + "; and ".join(parts)
