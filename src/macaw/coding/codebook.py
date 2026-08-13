from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
        raise NotImplementedError

    def labels(self) -> list[str]:
        raise NotImplementedError


def read_codebook(path: Path) -> Codebook:
    raise NotImplementedError


def write_codebook(codebook: Codebook, path: Path) -> Path:
    raise NotImplementedError


def validate_codebook(codebook: Codebook) -> list[str]:
    raise NotImplementedError
