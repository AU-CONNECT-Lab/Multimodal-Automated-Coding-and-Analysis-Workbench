"""Where a wrist is against the body, frame by frame: its height level, gesture space and the sector around the centre."""

from __future__ import annotations

import numpy as np
import pandas as pd

SIDES = ("left", "right")
BODY = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "nose", "left_eye", "right_eye")
WRIST_PLACES = ("height_level", "gesture_space", "gesture_sector")


def wrist_places(points: pd.DataFrame) -> pd.DataFrame:
    """Per frame of points (an (x|y, joint) column per BODY joint and wrist, image y down), each wrist's level, space and sector."""
    places = {}
    for side in SIDES:
        wrist = _xy(points, f"{side}_wrist")
        places[f"{side}_wrist_height_level"] = height_levels(wrist, points)
        places[f"{side}_wrist_gesture_space"] = gesture_spaces(wrist, points)
        places[f"{side}_wrist_gesture_sector"] = gesture_sectors(wrist, points)
    return pd.DataFrame(places, index=points.index)


def _xy(points: pd.DataFrame, joint: str) -> np.ndarray:
    return np.column_stack([points[("x", joint)].to_numpy(), points[("y", joint)].to_numpy()])


def _midpoint(points: pd.DataFrame, a: str, b: str) -> np.ndarray:
    return (_xy(points, a) + _xy(points, b)) / 2.0


def frontal_plane(found: dict[str, np.ndarray], frames: np.ndarray) -> pd.DataFrame:
    """Each landmark and wrist in the person's own frontal plane, laid out like a picture taken from in front of them:
    x toward their left, y down their trunk, from the hip midpoint. NaN without both hips and both shoulders."""
    missing = np.full((len(frames), 3), np.nan)
    hips = (found.get("left_hip", missing) + found.get("right_hip", missing)) / 2.0
    up = (found.get("left_shoulder", missing) + found.get("right_shoulder", missing)) / 2.0 - hips
    across = found.get("left_hip", missing) - found.get("right_hip", missing)
    with np.errstate(invalid="ignore", divide="ignore"):
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        across -= np.sum(across * up, axis=1, keepdims=True) * up
        across /= np.linalg.norm(across, axis=1, keepdims=True)
    joints = [*BODY, *(f"{side}_wrist" for side in SIDES)]
    columns = {}
    for joint in joints:
        offset = found.get(joint, missing) - hips
        columns[("x", joint)] = np.sum(offset * across, axis=1)
        columns[("y", joint)] = -np.sum(offset * up, axis=1)
    plane = pd.DataFrame(columns, index=pd.Index(frames, name="frame"))
    return plane.reindex(columns=pd.MultiIndex.from_product([["x", "y"], joints]))


def height_levels(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, the vertical level the wrist is at, 0 to 5; image y grows downward."""
    up = -wrist[:, 1]
    shoulders = -_midpoint(body, "left_shoulder", "right_shoulder")[:, 1]
    hips = -_midpoint(body, "left_hip", "right_hip")[:, 1]
    nose = -body[("y", "nose")].to_numpy()
    eyes = -_midpoint(body, "left_eye", "right_eye")[:, 1]
    midline = (shoulders + hips) / 2.0
    chest = (midline + shoulders) / 2.0
    top = eyes + 2.0 * (eyes - nose)
    bounds = np.column_stack([midline, chest, shoulders, nose, top])
    known = np.isfinite(up) & np.isfinite(bounds).all(axis=1)
    with np.errstate(invalid="ignore"):
        levels = (up[:, None] > bounds).sum(axis=1).astype(float)
    return np.where(known, levels, np.nan)


def gesture_spaces(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, the gesture space the wrist is in, centre-centre 1 to extreme periphery 4."""
    x, up = wrist[:, 0], -wrist[:, 1]
    low, high, centre_up = _hips_box(body)
    eyes = -_midpoint(body, "left_eye", "right_eye")[:, 1]
    nose = -body[("y", "nose")].to_numpy()
    face = 2.0 * np.abs(body[("x", "left_eye")].to_numpy() - body[("x", "right_eye")].to_numpy())
    width = high - low
    top = eyes + (eyes - nose)
    with np.errstate(invalid="ignore"):
        centre_centre = (x > low) & (x < high) & (np.abs(up - centre_up) < width / 2.0)
        centre = (x > low - width / 2.0) & (x < high + width / 2.0) & (np.abs(up - centre_up) < width)
        periphery = (x > low - width / 2.0 - face) & (x < high + width / 2.0 + face) & (np.abs(up - centre_up) < top - centre_up)
    spaces = np.select([centre_centre, centre, periphery], [1.0, 2.0, 3.0], 4.0)
    known = np.isfinite(np.column_stack([x, up, low, high, centre_up, top, face])).all(axis=1)
    return np.where(known, spaces, np.nan)


def gesture_sectors(wrist: np.ndarray, body: pd.DataFrame) -> np.ndarray:
    """Per frame, which of the eight cells around the centre space the wrist is in, 1 to 8, or 0 within it."""
    x, up = wrist[:, 0], -wrist[:, 1]
    low, high, centre_up = _hips_box(body)
    width = high - low
    with np.errstate(invalid="ignore"):
        column = np.where(x <= low - width / 2.0, 0, np.where(x >= high + width / 2.0, 2, 1))
        row = np.where(up >= centre_up + width, 0, np.where(up <= centre_up - width, 2, 1))
    # the edges of the centre space extended outward cut the periphery into a 3 x 3 grid less its middle
    cell = 3 * row + column
    sectors = np.where(cell < 4, cell + 1, cell).astype(float)
    spaces = gesture_spaces(wrist, body)
    return np.where(np.isfinite(spaces), np.where(spaces >= 3, sectors, 0.0), np.nan)


def _hips_box(body: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, the hips' outer x on each side and the height halfway between shoulders and hips, up positive."""
    hips_x = np.column_stack([body[("x", "left_hip")].to_numpy(), body[("x", "right_hip")].to_numpy()])
    centre_up = -(_midpoint(body, "left_shoulder", "right_shoulder")[:, 1] + _midpoint(body, "left_hip", "right_hip")[:, 1]) / 2.0
    return hips_x.min(axis=1), hips_x.max(axis=1), centre_up
