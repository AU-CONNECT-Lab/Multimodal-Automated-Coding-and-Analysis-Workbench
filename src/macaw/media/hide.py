"""Copies of a camera's video with the people hidden: heads blurred in the picture, or only the skeletons on a plain ground."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from macaw.exceptions import UnreadableMediaError
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.media.ffmpeg import encode_frames
from macaw.media.probe import probe
from macaw.qc.overlay import cv2_module, draw_overlay

HIDE = ("faces", "everything")
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")
SHOULDERS = ("left_shoulder", "right_shoulder")
# a head found late or lost for a moment stays hidden this long before, after and across the gap
HOLD_S = 1.0
# the circle reaches this far past the outermost head point, to take in chin, forehead and hair
_PAST_HEAD_POINTS = 2.2
# and is never smaller than this share of the shoulder width, for a head seen from the side
_OF_SHOULDERS = 0.5
# a head with no points of its own sits this far above the shoulder line, in shoulder widths
_ABOVE_SHOULDERS = 0.6
# side of the grid a head is reduced to before it is drawn back
_BLUR_CELLS = 4
_GROUND = 128
CIRCLE_COLUMNS = ["who", "frame", "x", "y", "r"]
GAP_COLUMNS = ["subject_id", "t_start", "t_end"]


def head_circles(people: pd.DataFrame) -> pd.DataFrame:
    """A circle over each head found on one camera; who is the person's id, or their track when nobody named it."""
    found = people[np.isfinite(people["x"]) & np.isfinite(people["y"])]
    if found.empty:
        return pd.DataFrame(columns=CIRCLE_COLUMNS)
    named = found["subject_id"].astype(object)
    who = named.where(named.notna(), "track " + found["track_id"].astype("Int64").astype(str))
    found = found.assign(who=who.astype(str))
    keys = ["who", "frame"]
    head = found[found["joint"].isin(HEAD_JOINTS)]
    centre = head.groupby(keys)[["x", "y"]].mean()
    middle = head.join(centre, on=keys, rsuffix="_centre")
    spread = np.hypot(middle["x"] - middle["x_centre"], middle["y"] - middle["y_centre"]).groupby([middle[key] for key in keys]).max()
    circles = centre.assign(r=_PAST_HEAD_POINTS * spread)
    wide = found[found["joint"].isin(SHOULDERS)].pivot_table(index=keys, columns="joint", values=["x", "y"])
    if all((axis, joint) in wide.columns for axis in ("x", "y") for joint in SHOULDERS):
        wide = wide.dropna()
        width = np.hypot(wide[("x", SHOULDERS[0])] - wide[("x", SHOULDERS[1])], wide[("y", SHOULDERS[0])] - wide[("y", SHOULDERS[1])])
        above = pd.DataFrame({
            "x": wide["x"].mean(axis=1), "y": wide["y"].mean(axis=1) - _ABOVE_SHOULDERS * width, "r": _ABOVE_SHOULDERS * width,
        })
        circles = circles.reindex(circles.index.union(above.index))
        circles = circles.fillna(above.reindex(circles.index))
        circles["r"] = np.fmax(circles["r"], (_OF_SHOULDERS * width).reindex(circles.index))
    # one point alone spans nothing, so such a head takes the size this person's head usually has
    usual = circles["r"].where(circles["r"] > 0).groupby(level="who").transform("median")
    circles["r"] = np.fmax(circles["r"], 0.5 * usual)
    return circles[circles["r"] > 0].reset_index()[CIRCLE_COLUMNS]


def held_circles(circles: pd.DataFrame, n_frames: int, fps: float) -> pd.DataFrame:
    """A circle in every frame within HOLD_S of one where the head was found, moving evenly between the two found ones it lies between."""
    hold = int(round(HOLD_S * fps))
    filled = []
    for who, own in circles.groupby("who", sort=False):
        own = own.sort_values("frame")
        at = own["frame"].to_numpy(dtype=int)
        frames = np.arange(max(at[0] - hold, 0), min(at[-1] + hold, n_frames - 1) + 1)
        after = np.clip(np.searchsorted(at, frames), 0, len(at) - 1)
        before = np.clip(after - 1, 0, len(at) - 1)
        # a head lost for longer than this on both sides may have left the picture
        frames = frames[np.minimum(np.abs(frames - at[before]), np.abs(at[after] - frames)) <= hold]
        filled.append(pd.DataFrame({
            "who": who, "frame": frames,
            **{column: np.interp(frames, at, own[column].to_numpy(dtype=float)) for column in ("x", "y", "r")},
        }))
    return pd.concat(filled, ignore_index=True) if filled else pd.DataFrame(columns=CIRCLE_COLUMNS)


def not_hidden(circles: pd.DataFrame, subjects: list[str], n_frames: int, fps: float) -> pd.DataFrame:
    """Each stretch of the video in which a named person has no circle: out of the picture, or in it and not found."""
    rows = []
    for subject_id in subjects:
        covered = np.zeros(n_frames, dtype=bool)
        frames = circles.loc[circles["who"] == subject_id, "frame"].to_numpy(dtype=int)
        covered[frames[(frames >= 0) & (frames < n_frames)]] = True
        edges = np.flatnonzero(np.diff(np.r_[False, ~covered, False]))
        rows += [(subject_id, start / fps, end / fps) for start, end in zip(edges[::2], edges[1::2])]
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def blur_heads(image: np.ndarray, circles) -> np.ndarray:
    """Each (x, y, r) circle of the picture replaced, in place, by a few flat cells smoothed into each other."""
    cv2 = cv2_module()
    height, width = image.shape[:2]
    for x, y, r in circles:
        x0, x1 = max(int(x - r), 0), min(int(np.ceil(x + r)) + 1, width)
        y0, y1 = max(int(y - r), 0), min(int(np.ceil(y + r)) + 1, height)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = image[y0:y1, x0:x1]
        coarse = cv2.resize(region, (_BLUR_CELLS, _BLUR_CELLS), interpolation=cv2.INTER_AREA)
        soft = cv2.resize(coarse, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        inside = np.zeros(region.shape[:2], dtype=np.uint8)
        cv2.circle(inside, (int(round(x - x0)), int(round(y - y0))), int(np.ceil(r)), 255, -1)
        region[inside > 0] = soft[inside > 0]
    return image


def _pictures(source: Path, n_frames: int) -> Iterator[np.ndarray]:
    cv2 = cv2_module()
    capture = cv2.VideoCapture(str(source))
    try:
        for _ in range(n_frames):
            ok, image = capture.read()
            if not ok:
                return
            yield image
    finally:
        capture.release()


def video_shape(source: Path) -> tuple[int, int, int, float]:
    """(width, height, frames, frames per second) of a video as opencv reads it."""
    cv2 = cv2_module()
    capture = cv2.VideoCapture(str(source))
    try:
        shape = tuple(int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(shape) <= 0 or not fps > 0:
        raise UnreadableMediaError(f"{source}: could not read the video's size, length or frame rate")
    return (*shape, fps)


def write_hidden(source: Path, dest: Path, people: pd.DataFrame, subjects: list[str], *, hide: str, sound: bool) -> pd.DataFrame:
    """Writes the hidden copy of one camera's video; returns the stretches a named person's head was not blurred in."""
    width, height, n_frames, fps = video_shape(source)
    heard = source if sound and probe(source).audio_streams() else None
    if hide == "everything":
        standard = people[people["joint"].isin(STANDARD_JOINTS)]
        by_frame = {int(frame): rows for frame, rows in standard.groupby("frame")}
        nobody = standard.iloc[:0]

        def drawn() -> Iterator[np.ndarray]:
            for frame in range(n_frames):
                ground = np.full((height, width, 3), _GROUND, dtype=np.uint8)
                yield draw_overlay(ground, by_frame.get(frame, nobody), subjects, [], labels=False, boxes=False)

        encode_frames(drawn(), dest, size=(width, height), fps=fps, sound_from=heard)
        return pd.DataFrame(columns=GAP_COLUMNS)

    circles = held_circles(head_circles(people), n_frames, fps)
    by_frame = {int(frame): rows[["x", "y", "r"]].to_numpy() for frame, rows in circles.groupby("frame")}

    def blurred() -> Iterator[np.ndarray]:
        for frame, image in enumerate(_pictures(source, n_frames)):
            yield blur_heads(image, by_frame.get(frame, ()))

    encode_frames(blurred(), dest, size=(width, height), fps=fps, sound_from=heard)
    return not_hidden(circles, subjects, n_frames, fps)
