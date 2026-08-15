"""The codebook is data, not code. It lives in a YAML file that goes into the config hash, so
editing a definition invalidates the cache of every layer that read it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from macaw.exceptions import CodebookError, InputNotFoundError

UNITS = ("utterance", "turn", "window")
LAYERS = ("A", "B", "C")

# the agreement report gives these two labels a meaning of their own; a code sharing a name
# would be counted as a blank cell instead of a category
RESERVED_LABELS = ("_none", "_rest")

_LIST_FIELDS = ("criteria", "positive_examples", "negative_examples", "boundary_cases")
_CODE_FIELDS = frozenset({"code", "description", "layer", *_LIST_FIELDS})


@dataclass
class CodeDefinition:
    code: str
    description: str
    criteria: list[str] = field(default_factory=list)
    positive_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    boundary_cases: list[str] = field(default_factory=list)
    layer: str = "B"


@dataclass
class Codebook:
    name: str
    unit: str                                   # utterance | turn | window
    codes: list[CodeDefinition] = field(default_factory=list)
    mutually_exclusive: bool = True

    def get(self, code: str) -> CodeDefinition:
        for definition in self.codes:
            if definition.code == code:
                return definition
        known = ", ".join(self.labels()) or "none"
        raise CodebookError(f"'{code}' is not in codebook '{self.name}'. Defined: {known}")

    def labels(self) -> list[str]:
        return [definition.code for definition in self.codes]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _definition_from(entry: Any, path: Path) -> CodeDefinition:
    if not isinstance(entry, dict):
        raise CodebookError(f"{path}: every entry under 'codes' must be a mapping, got {entry!r}")
    if "code" not in entry:
        raise CodebookError(f"{path}: a code entry has no 'code' key")
    unknown = sorted(set(entry) - _CODE_FIELDS)
    if unknown:
        raise CodebookError(
            f"{path}: code '{entry['code']}' has unknown key(s) {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_CODE_FIELDS))}"
        )
    return CodeDefinition(
        code=str(entry["code"]),
        description=str(entry.get("description") or ""),
        layer=str(entry.get("layer") or "B"),
        **{name: _as_list(entry.get(name)) for name in _LIST_FIELDS},
    )


def read_codebook(path: Path) -> Codebook:
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"Codebook not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CodebookError(f"{path}: not readable as YAML ({error})") from None

    if not isinstance(data, dict):
        raise CodebookError(f"{path}: expected a mapping at the top level")
    codes = data.get("codes") or []
    if not isinstance(codes, list):
        raise CodebookError(f"{path}: 'codes' must be a list")

    return Codebook(
        name=str(data.get("name") or path.stem),
        unit=str(data.get("unit") or "turn"),
        mutually_exclusive=bool(data.get("mutually_exclusive", True)),
        codes=[_definition_from(entry, path) for entry in codes],
    )


def write_codebook(codebook: Codebook, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "name": codebook.name,
        "unit": codebook.unit,
        "mutually_exclusive": codebook.mutually_exclusive,
        # empty fields are left out so a hand-written file does not grow noise on a round trip
        "codes": [
            {key: value for key, value in asdict(definition).items() if value not in ([], "")}
            for definition in codebook.codes
        ],
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def validate_codebook(codebook: Codebook) -> list[str]:
    """Problems in reading order; empty means usable. Nothing here raises, because a codebook
    still being drafted has to stay printable."""
    problems: list[str] = []
    if not codebook.name.strip():
        problems.append("codebook has no name")
    if codebook.unit not in UNITS:
        problems.append(f"unit '{codebook.unit}' is not one of: {', '.join(UNITS)}")
    if not codebook.codes:
        problems.append("codebook defines no codes")

    seen: set[str] = set()
    for definition in codebook.codes:
        label = definition.code
        if not label.strip():
            problems.append("a code has an empty name")
        elif label in seen:
            problems.append(f"code '{label}' is defined more than once")
        elif label in RESERVED_LABELS:
            problems.append(
                f"code '{label}' is reserved by the agreement report and would be read as a "
                "blank cell; rename it"
            )
        seen.add(label)

        if definition.layer not in LAYERS:
            problems.append(
                f"code '{label}': layer '{definition.layer}' is not one of {', '.join(LAYERS)}"
            )
        if not definition.description.strip():
            problems.append(f"code '{label}' has no description")
        if not definition.criteria and not definition.positive_examples:
            problems.append(
                f"code '{label}' gives neither criteria nor a positive example, so nothing "
                "distinguishes it from the other codes"
            )
    return problems
