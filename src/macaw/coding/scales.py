"""Rating scales: one score per scale for a whole recording or a named part of it. Like the codebook, a YAML file that
goes into the config hash."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from macaw import plain_yaml
from macaw.exceptions import CodebookError, InputNotFoundError

# rates everyone in the recording together
ALL = "all"

_SCALE_FIELDS = frozenset({"scale", "person", "range", "definition", "anchors", "rules"})


@dataclass
class RatingScale:
    scale: str
    person: str
    low: int
    high: int
    definition: str = ""
    anchors: dict[int, str] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)

    def described(self) -> bool:
        return bool(self.definition.strip() or self.anchors or self.rules)


@dataclass
class ScaleSet:
    name: str
    scales: list[RatingScale] = field(default_factory=list)

    def names_only(self) -> bool:
        """Every scale given by its name, person and range alone."""
        return bool(self.scales) and not any(scale.described() for scale in self.scales)

    def names(self) -> list[str]:
        return [scale.scale for scale in self.scales]

    def get(self, name: str) -> RatingScale:
        for scale in self.scales:
            if scale.scale == name:
                return scale
        raise CodebookError(f"'{name}' is not a scale of '{self.name}'. Defined: {', '.join(self.names()) or 'none'}")


def _whole(value: Any, what: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
        raise CodebookError(f"{path}: {what} has to be a whole number, got {value!r}")
    return int(value)


def _scale_from(entry: Any, path: Path) -> RatingScale:
    if not isinstance(entry, dict):
        raise CodebookError(f"{path}: every entry under 'scales' must be a mapping, got {entry!r}")
    if "scale" not in entry:
        raise CodebookError(f"{path}: a scale entry has no 'scale' key")
    name = str(entry["scale"])
    unknown = sorted(set(entry) - _SCALE_FIELDS)
    if unknown:
        raise CodebookError(f"{path}: scale '{name}' has unknown key(s) {', '.join(unknown)}. Available: {', '.join(sorted(_SCALE_FIELDS))}")
    for needed in ("person", "range"):
        if entry.get(needed) is None:
            raise CodebookError(f"{path}: scale '{name}' has no '{needed}'")
    span = entry["range"]
    if not isinstance(span, list) or len(span) != 2:
        raise CodebookError(f"{path}: the range of scale '{name}' is [lowest, highest], got {span!r}")
    anchors = entry.get("anchors") or {}
    if not isinstance(anchors, dict):
        raise CodebookError(f"{path}: the anchors of scale '{name}' are a mapping from a score to what it means")
    rules = entry.get("rules") or []
    return RatingScale(
        scale=name,
        person=str(entry["person"]),
        low=_whole(span[0], f"the lowest score of '{name}'", path),
        high=_whole(span[1], f"the highest score of '{name}'", path),
        definition=str(entry.get("definition") or "").strip(),
        anchors={_whole(score, f"an anchor of '{name}'", path): str(text).strip() for score, text in anchors.items()},
        rules=[str(rules)] if isinstance(rules, str) else [str(rule).strip() for rule in rules],
    )


def read_scales(path: Path) -> ScaleSet:
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"Rating scales not found: {path}")
    try:
        data = plain_yaml.load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CodebookError(f"{path}: not readable as YAML ({error})") from None
    if not isinstance(data, dict):
        raise CodebookError(f"{path}: expected a mapping at the top level")
    entries = data.get("scales") or []
    if not isinstance(entries, list):
        raise CodebookError(f"{path}: 'scales' must be a list")
    return ScaleSet(name=str(data.get("name") or path.stem), scales=[_scale_from(entry, path) for entry in entries])


def validate_scales(scales: ScaleSet, roles: list[str]) -> list[str]:
    """Problems in reading order; empty means usable."""
    problems: list[str] = []
    if not scales.name.strip():
        problems.append("the scales have no name")
    if not scales.scales:
        problems.append("no scales are defined")
    # scales given by name alone are a choice; some described and some not is a draft
    described = not scales.names_only()
    seen: set[str] = set()
    for scale in scales.scales:
        name = scale.scale
        if not name.strip():
            problems.append("a scale has an empty name")
        elif name in seen:
            problems.append(f"scale '{name}' is defined more than once")
        seen.add(name)
        if scale.person != ALL and scale.person not in roles:
            problems.append(f"scale '{name}' rates {scale.person}, which is not one of project.roles ({', '.join(roles)}) nor {ALL}")
        if scale.high <= scale.low:
            problems.append(f"scale '{name}' runs from {scale.low} to {scale.high}; the highest score has to be above the lowest")
        outside = sorted(score for score in scale.anchors if not scale.low <= score <= scale.high)
        if outside:
            problems.append(f"scale '{name}' has anchors outside its range {scale.low} to {scale.high}: {', '.join(map(str, outside))}")
        if any(not text for text in scale.anchors.values()):
            problems.append(f"scale '{name}' has an anchor without words")
        if described and not scale.described():
            problems.append(f"scale '{name}' has no definition, anchors or rules while other scales do")
    return problems


@dataclass
class ScaleExample:
    """A part of another conversation people rated on a scale: its lines with their speakers' roles, or a description of
    it, the score they gave, and their reason where they wrote one."""

    score: int
    lines: list[tuple[str, str]] = field(default_factory=list)
    description: str = ""
    why: str = ""


_EXAMPLE_FIELDS = frozenset({"lines", "description", "score", "why"})


def read_examples(path: Path, scales: ScaleSet, speakers: list[str]) -> dict[str, list[ScaleExample]]:
    """Examples by scale from a YAML file whose examples key maps each scale's name to its list; speakers are the roles a
    line may name."""
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"Rating examples not found: {path}")
    try:
        data = plain_yaml.load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CodebookError(f"{path}: not readable as YAML ({error})") from None
    given = data.get("examples") if isinstance(data, dict) else None
    if not isinstance(given, dict):
        raise CodebookError(f"{path}: expected examples: a mapping from each scale's name to a list of examples")
    found: dict[str, list[ScaleExample]] = {}
    for name, entries in given.items():
        scale = scales.get(str(name))
        if not isinstance(entries, list):
            raise CodebookError(f"{path}: the examples of {name} must be a list")
        found[scale.scale] = [_example_from(entry, scale, speakers, path, number) for number, entry in enumerate(entries, 1)]
    return found


def _example_from(entry: Any, scale: RatingScale, speakers: list[str], path: Path, number: int) -> ScaleExample:
    where = f"{path}: example {number} of {scale.scale}"
    if not isinstance(entry, dict):
        raise CodebookError(f"{where} must be a mapping")
    unknown = sorted(set(entry) - _EXAMPLE_FIELDS)
    if unknown:
        raise CodebookError(f"{where} has unknown key(s) {', '.join(unknown)}. Available: {', '.join(sorted(_EXAMPLE_FIELDS))}")
    if bool(entry.get("lines")) == bool(str(entry.get("description") or "").strip()):
        raise CodebookError(f"{where} needs either lines or a description, not both")
    if entry.get("score") is None:
        raise CodebookError(f"{where} has no score")
    score = _whole(entry["score"], f"the score of {where.split(': ', 1)[1]}", path)
    if not scale.low <= score <= scale.high:
        raise CodebookError(f"{where} scores {score}, outside the scale's range {scale.low} to {scale.high}")
    lines = []
    for line in entry.get("lines") or []:
        if not isinstance(line, dict) or len(line) != 1:
            raise CodebookError(f"{where}: each line is one role and what they said, such as - {speakers[0] if speakers else 'role'}: How was it?")
        (role, text), = line.items()
        if str(role) not in speakers:
            raise CodebookError(f"{where}: {role} is not one of the speakers ({', '.join(speakers)})")
        lines.append((str(role), str(text).strip()))
    return ScaleExample(score=score, lines=lines, description=str(entry.get("description") or "").strip(), why=str(entry.get("why") or "").strip())
