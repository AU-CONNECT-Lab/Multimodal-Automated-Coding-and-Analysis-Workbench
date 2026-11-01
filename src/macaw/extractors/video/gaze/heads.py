from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FACE_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
SHOULDERS = ("left_shoulder", "right_shoulder")
# a head box is this many times the spread of the face points, or of half the shoulder width when that is larger;
# hand-drawn head boxes, which take in the hair, measure about this
HEAD_SCALE = 2.0
SHOULDER_SHARE = 0.5
# the box centre sits this share of its side above the mean of the face points
CENTRE_RISE = 0.12


def head_boxes(keypoints: pd.DataFrame) -> pd.DataFrame:
    """One box per person, video and frame where at least two face points were found: subject_id, video,
    frame, t, x0, y0, x1, y1 in the video's pixels."""
    wanted = keypoints[keypoints["joint"].isin(FACE_JOINTS + SHOULDERS) & keypoints["x"].notna()]
    columns = ["subject_id", "video", "frame", "t", "x0", "y0", "x1", "y1"]
    if wanted.empty:
        return pd.DataFrame(columns=columns)
    wide = wanted.pivot_table(index=["subject_id", "video", "frame"], columns="joint", values=["x", "y", "t"], aggfunc="first", observed=True)
    face_x = wide["x"].reindex(columns=list(FACE_JOINTS)).to_numpy(dtype=float)
    face_y = wide["y"].reindex(columns=list(FACE_JOINTS)).to_numpy(dtype=float)
    shoulder_x = wide["x"].reindex(columns=list(SHOULDERS)).to_numpy(dtype=float)
    shoulder_y = wide["y"].reindex(columns=list(SHOULDERS)).to_numpy(dtype=float)
    enough = np.isfinite(face_x).sum(axis=1) >= 2
    # rows with fewer than two face points are all-NaN slices here and are dropped below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.fmax(np.nanmax(face_x, axis=1) - np.nanmin(face_x, axis=1), np.nanmax(face_y, axis=1) - np.nanmin(face_y, axis=1))
        shoulder_width = np.hypot(shoulder_x[:, 0] - shoulder_x[:, 1], shoulder_y[:, 0] - shoulder_y[:, 1])
        side = HEAD_SCALE * np.fmax(spread, SHOULDER_SHARE * shoulder_width)
        centre_x = np.nanmean(face_x, axis=1)
        centre_y = np.nanmean(face_y, axis=1) - CENTRE_RISE * side
    index = wide.index.to_frame(index=False)
    out = pd.DataFrame(
        {
            "subject_id": index["subject_id"].astype(str), "video": index["video"].astype(str), "frame": index["frame"].astype(int),
            "t": wide["t"].bfill(axis=1).iloc[:, 0].to_numpy(dtype=float),
            "x0": centre_x - side / 2, "y0": centre_y - side / 2, "x1": centre_x + side / 2, "y1": centre_y + side / 2,
        }
    )
    return out[enough & (side > 0)].reset_index(drop=True)[columns]
