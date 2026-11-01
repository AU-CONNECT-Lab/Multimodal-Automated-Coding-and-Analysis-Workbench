"""Whom a person looks at from their head and eye angles: for a camera that does not show the people they look at."""

from __future__ import annotations

import numpy as np
import pandas as pd

CAMERA = "camera"
# a person needs this many labelled frames looking at a target before those frames, not the seats, say where it is
MIN_LABELS = 3


def seat_direction(seats: dict, looker: str, target: str) -> tuple[float, float]:
    """(yaw, pitch) in degrees, signed as the face models sign head angles, of a face in the looker's seat pointing at the
    target's eyes. Seats are [x, depth, eye height] seen from a camera that looks along depth, level, with x to its right."""
    gap = np.subtract(seats[target], seats[looker]).astype(float)
    # facing the camera, a person's own right is the camera's left
    yaw = np.degrees(np.arctan2(-gap[0], -gap[1]))
    pitch = np.degrees(np.arctan2(gap[2], np.hypot(gap[0], gap[1])))
    return float(yaw), float(pitch)


def unit_vectors(yaw, pitch) -> np.ndarray:
    yaw, pitch = np.radians(np.asarray(yaw, dtype=float)), np.radians(np.asarray(pitch, dtype=float))
    return np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], axis=-1)


def angle_between(yaw_a, pitch_a, yaw_b, pitch_b) -> np.ndarray:
    cos = np.sum(unit_vectors(yaw_a, pitch_a) * unit_vectors(yaw_b, pitch_b), axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def directions(face_rows: pd.DataFrame) -> pd.DataFrame:
    """subject_id, t, yaw, pitch, source from face.au's long table: the eyes' gaze where read, else the head's angles."""
    wide = face_rows.pivot_table(index=["subject_id", "t_start"], columns="feature", values="value", observed=True)
    head = wide.reindex(columns=["face_yaw_deg", "face_pitch_deg"])
    eyes = wide.reindex(columns=["gaze_yaw_deg", "gaze_pitch_deg"])
    use_eyes = eyes.notna().all(axis=1).to_numpy()
    out = wide.index.to_frame(index=False).rename(columns={"t_start": "t"})
    out["yaw"] = np.where(use_eyes, eyes.iloc[:, 0], head.iloc[:, 0])
    out["pitch"] = np.where(use_eyes, eyes.iloc[:, 1], head.iloc[:, 1])
    out["source"] = np.where(use_eyes, "eyes", "head")
    return out.dropna(subset=["yaw", "pitch"]).reset_index(drop=True)


def labelled_directions(measured: pd.DataFrame, labels: pd.DataFrame, period_s: float) -> pd.DataFrame:
    """Each label (t, subject, target) with the direction measured in the frame nearest it, within half a frame."""
    if labels.empty or measured.empty:
        return pd.DataFrame(columns=["subject", "target", "yaw", "pitch"])
    found = []
    for subject, mine in labels.groupby("subject", sort=False):
        theirs = measured[measured["subject_id"] == subject].sort_values("t")
        if theirs.empty:
            continue
        joined = pd.merge_asof(mine.sort_values("t"), theirs[["t", "yaw", "pitch"]], on="t", direction="nearest", tolerance=period_s / 2)
        found.append(joined.dropna(subset=["yaw"]))
    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=["subject", "target", "yaw", "pitch"])


def expected_directions(looker: str, targets: list[str], seats: dict, labelled: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """target -> (yaw, pitch) for one looker: from their labelled frames where there are enough, else from the seats,
    shifted by how far their labelled frames sit from where the seats put them."""
    mine = labelled[labelled["subject"] == looker] if not labelled.empty else labelled
    seated = looker in seats
    shifts = [
        np.subtract((row.yaw, row.pitch), seat_direction(seats, looker, row.target))
        for row in mine.itertuples() if seated and row.target in seats
    ]
    shift = np.median(np.array(shifts), axis=0) if shifts else np.zeros(2)
    expected = {}
    for target in targets:
        frames = mine[mine["target"] == target] if not mine.empty else mine
        if len(frames) >= MIN_LABELS:
            expected[target] = (float(frames["yaw"].median()), float(frames["pitch"].median()))
        elif seated and target in seats:
            yaw, pitch = seat_direction(seats, looker, target)
            expected[target] = (yaw + float(shift[0]), pitch + float(shift[1]))
    return expected


def name_by_angles(measured: pd.DataFrame, expected: dict[str, tuple[float, float]], tolerance_deg: float, elsewhere: str) -> pd.DataFrame:
    """Adds target, the nearest expected direction within tolerance_deg, and off_<target>, the angle to each in degrees."""
    out = measured.copy()
    if not expected:
        out["target"] = elsewhere
        return out
    offs = {target: angle_between(out["yaw"], out["pitch"], yaw, pitch) for target, (yaw, pitch) in expected.items()}
    for target, off in offs.items():
        out[f"off_{target}"] = off
    table = np.column_stack(list(offs.values()))
    nearest = np.array(list(offs))[np.argmin(table, axis=1)]
    out["target"] = np.where(table.min(axis=1) <= tolerance_deg, nearest, elsewhere)
    return out
