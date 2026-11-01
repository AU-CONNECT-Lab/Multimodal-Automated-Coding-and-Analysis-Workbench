"""Frames a person labelled with whom or what each person looks at: the check on gaze.target, and with the angles
backend the directions it reads targets from."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ELSEWHERE = "elsewhere"
# a frame where the labeller could not tell; kept so it is not offered again, never compared
UNSURE = "unsure"


@dataclass
class GazeLabel:
    t: float                # recording time of the frame
    subject: str            # the role who looks
    target: str             # a role, a name from gaze.targets, elsewhere, or unsure
    by: str | None = None
    at: str | None = None


def read_labels(path: Path) -> list[GazeLabel]:
    if not path.is_file():
        return []
    return [GazeLabel(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def write_labels(labels: list[GazeLabel], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(labels, key=lambda label: (label.t, label.subject))
    path.write_text(json.dumps([asdict(label) for label in ordered], indent=2), encoding="utf-8")
    return path


def add_label(labels: list[GazeLabel], label: GazeLabel, period_s: float) -> list[GazeLabel]:
    """A new label replaces one for the same person within half a frame."""
    kept = [old for old in labels if not (old.subject == label.subject and abs(old.t - label.t) < period_s / 2)]
    return [*kept, label]


def labels_frame(labels: list[GazeLabel]) -> pd.DataFrame:
    """t, subject, target for the labels that say something, unsure ones left out."""
    rows = [asdict(label) for label in labels if label.target != UNSURE]
    return pd.DataFrame(rows, columns=["t", "subject", "target", "by", "at"])[["t", "subject", "target"]]
