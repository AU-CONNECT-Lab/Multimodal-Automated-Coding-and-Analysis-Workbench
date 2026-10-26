from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from macaw.store import _io
from macaw.store.schema import KEYPOINT_SCHEMA, coerce_frame, empty_frame

NAME = "keypoints"


def write_keypoints(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    return _io.write_partitioned(df, path, KEYPOINT_SCHEMA, NAME, "source", append=append)


def write_no_keypoints(path: Path, source: str) -> Path:
    """A run that found nobody still leaves its file; a missing one would read as never run."""
    return _io.write_file(empty_frame(KEYPOINT_SCHEMA), Path(path) / f"{source}.parquet", KEYPOINT_SCHEMA, NAME)


def read_keypoints(
    path: Path,
    *,
    recording_id: str | None = None,
    subject_id: str | None = None,
    joints: list[str] | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    predicate = _io.all_of(
        _io.equals("recording_id", recording_id),
        _io.equals("subject_id", subject_id),
        _io.is_in("joint", joints),
        _sampled_within(t_range),
    )
    return _io.read_partitioned(path, KEYPOINT_SCHEMA, NAME, predicate)


def _sampled_within(t_range: tuple[float, float] | None):
    """Keypoints are instants, not intervals, so this is containment not overlap."""
    if t_range is None:
        return None
    low, high = t_range
    return (ds.field("t") >= low) & (ds.field("t") < high)


def to_array(df: pd.DataFrame, joint_order: list[str]):
    """Long keypoint rows to a (n_frames, n_joints, 4) array for kinematics code."""
    frame = coerce_frame(df, KEYPOINT_SCHEMA, NAME)
    frames = np.sort(frame["frame"].unique())
    frame_position = {value: index for index, value in enumerate(frames)}
    joint_position = {name: index for index, name in enumerate(joint_order)}

    out = np.full((len(frames), len(joint_order), 4), np.nan)
    rows = frame[frame["joint"].isin(joint_position)]
    if rows.empty:
        return out

    row_index = rows["frame"].map(frame_position).to_numpy()
    column_index = rows["joint"].map(joint_position).to_numpy()
    for depth, column in enumerate(("x", "y", "z", "confidence")):
        out[row_index, column_index, depth] = rows[column].to_numpy(dtype=float)
    return out


def has_depth(df: pd.DataFrame) -> bool:
    """False when z is entirely NaN; feature code takes the 2D degenerate branch."""
    return "z" in df.columns and not df["z"].isna().all()
