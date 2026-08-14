from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_keypoints(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    raise NotImplementedError


def read_keypoints(
    path: Path,
    *,
    recording_id: str | None = None,
    subject_id: str | None = None,
    joints: list[str] | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    raise NotImplementedError


def to_array(df: pd.DataFrame, joint_order: list[str]):
    """Long keypoint rows to a (n_frames, n_joints, 4) array for kinematics code."""
    raise NotImplementedError


def has_depth(df: pd.DataFrame) -> bool:
    """False when z is entirely NaN; feature code takes the 2D degenerate branch."""
    raise NotImplementedError
