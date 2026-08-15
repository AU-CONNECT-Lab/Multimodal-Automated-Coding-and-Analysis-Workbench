from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from macaw.store import _io
from macaw.store.schema import BODY_MODEL_SCHEMA, coerce_frame

NAME = "body_model"

SHAPE_COLUMNS = [name for name in BODY_MODEL_SCHEMA if name.startswith("shape_")]
POSE_COLUMNS = [name for name in BODY_MODEL_SCHEMA if name.startswith("pose_")]
CAMERA_COLUMNS = [name for name in BODY_MODEL_SCHEMA if name.startswith("camera_")]


def write_body_model(df: pd.DataFrame, path: Path, *, append: bool = False) -> Path:
    return _io.write_partitioned(df, path, BODY_MODEL_SCHEMA, NAME, "source", append=append)


def read_body_model(
    path: Path,
    *,
    recording_id: str | None = None,
    subject_id: str | None = None,
    t_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    predicate = _io.all_of(
        _io.equals("recording_id", recording_id),
        _io.equals("subject_id", subject_id),
        _sampled_within(t_range),
    )
    return _io.read_partitioned(path, BODY_MODEL_SCHEMA, NAME, predicate)


def _sampled_within(t_range: tuple[float, float] | None):
    """Parameters describe an instant, not an interval, so this is containment."""
    if t_range is None:
        return None
    low, high = t_range
    return (ds.field("t") >= low) & (ds.field("t") < high)


def build_body_model_rows(
    recording_id: str,
    subject_id: str,
    source: str,
    body_model: str,
    frames: np.ndarray,
    times: np.ndarray,
    shape: np.ndarray,
    pose: np.ndarray,
    camera: np.ndarray,
) -> pd.DataFrame:
    """Backends hand back arrays per person; this is the only place they become rows."""
    frames = np.asarray(frames)
    for name, block, columns in (
        ("shape", shape, SHAPE_COLUMNS),
        ("pose", pose, POSE_COLUMNS),
        ("camera", camera, CAMERA_COLUMNS),
    ):
        if np.asarray(block).shape != (len(frames), len(columns)):
            raise ValueError(
                f"{name} must be ({len(frames)}, {len(columns)}), "
                f"got {np.asarray(block).shape}"
            )

    data = {
        "recording_id": recording_id,
        "subject_id": subject_id,
        "source": source,
        "frame": frames,
        "t": np.asarray(times, dtype=float),
        "body_model": body_model,
    }
    for block, columns in ((shape, SHAPE_COLUMNS), (pose, POSE_COLUMNS), (camera, CAMERA_COLUMNS)):
        block = np.asarray(block, dtype=float)
        data.update({name: block[:, index] for index, name in enumerate(columns)})
    return coerce_frame(pd.DataFrame(data), BODY_MODEL_SCHEMA, NAME)


def to_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rows back to the (n_frames, n_params) blocks a mesh reconstruction expects."""
    frame = coerce_frame(df, BODY_MODEL_SCHEMA, NAME).sort_values("frame")
    return {
        "frame": frame["frame"].to_numpy(),
        "t": frame["t"].to_numpy(dtype=float),
        "shape": frame[SHAPE_COLUMNS].to_numpy(dtype=float),
        "pose": frame[POSE_COLUMNS].to_numpy(dtype=float),
        "camera": frame[CAMERA_COLUMNS].to_numpy(dtype=float),
    }


def has_body_model(df: pd.DataFrame) -> bool:
    """False when every row is "none"; feature code takes the keypoint-only branch."""
    if df.empty or "body_model" not in df.columns:
        return False
    return bool((df["body_model"] != "none").any())
