"""Derived quantities. These go to the long table; raw keypoints do not."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class KinematicsExtractor(Extractor):
    name = "pose.kinematics"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose",)
    requires = ("pose.keypoints",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        raise NotImplementedError


def hand_distance(kp: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def velocity(kp: pd.DataFrame, joint: str) -> pd.Series:
    raise NotImplementedError


def acceleration(kp: pd.DataFrame, joint: str) -> pd.Series:
    raise NotImplementedError


def jerk(kp: pd.DataFrame, joint: str) -> pd.Series:
    raise NotImplementedError


def head_orientation(kp: pd.DataFrame) -> pd.DataFrame:
    """yaw/pitch/roll. Degrees when z is present, screen-relative proxy when it is NaN."""
    raise NotImplementedError


def nod_events(head: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def interpersonal_distance(kp_a: pd.DataFrame, kp_b: pd.DataFrame) -> pd.Series:
    """Metres with depth, pixels without; the unit goes into the manifest either way."""
    raise NotImplementedError


def postural_openness(kp: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def envelope_overlap(kp_a: pd.DataFrame, kp_b: pd.DataFrame) -> pd.Series:
    raise NotImplementedError
