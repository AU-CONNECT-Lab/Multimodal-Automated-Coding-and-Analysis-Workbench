"""Who is who on a shared camera from each person's own mic: the person whose head moves while a
voice is talking is likely the one talking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from macaw.extractors.audio.prosody.backends import load_audio

BIN_S = 0.5
ENERGY_FRAME_S = 0.02
# a mic is taken to carry its wearer when it is this far above its own quiet level
ABOVE_FLOOR_DB = 15.0
# and no further than this below the loudest mic at that moment, which drops a neighbour's crosstalk
BELOW_LOUDEST_DB = 10.0
# the null shifts the voice against the motion by at least this much, past any real coupling
MIN_SHIFT_S = 10.0
# a weaker pairing can be a listener turning toward the talker rather than the talker
CLEAR_Z = 5.0
CLEAR_MARGIN = 2.0
HEAD_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


@dataclass
class VoiceMatch:
    people: dict[str, int] = field(default_factory=dict)       # subject -> index of the person on camera
    z: dict[str, float] = field(default_factory=dict)          # of that pairing against shifted voices
    runner_up: dict[str, float] = field(default_factory=dict)  # best z of any other person for that voice
    clear: list[str] = field(default_factory=list)


def bin_edges(start: float, end: float, step: float = BIN_S) -> np.ndarray:
    return start + np.arange(int(np.floor((end - start) / step)) + 1) * step


def speech_activity(tracks: dict[str, Path], t_origin: float, edges: np.ndarray) -> dict[str, np.ndarray]:
    """Share of each bin in which a person's own mic carries their voice; tracks start at t_origin."""
    levels = {}
    for subject, path in tracks.items():
        audio, rate = load_audio(Path(path))
        hop = max(int(ENERGY_FRAME_S * rate), 1)
        frames = audio[: audio.size // hop * hop].reshape(-1, hop)
        levels[subject] = 10 * np.log10(np.mean(np.square(frames, dtype=float), axis=1) + 1e-12)
    if not levels:
        return {}
    length = min(level.size for level in levels.values())
    stacked = np.vstack([level[:length] for level in levels.values()])
    floors = np.percentile(stacked, 10, axis=1, keepdims=True)
    active = (stacked > floors + ABOVE_FLOOR_DB) & (stacked > stacked.max(axis=0) - BELOW_LOUDEST_DB)
    times = t_origin + np.arange(length) * ENERGY_FRAME_S
    return {subject: _binned(times, row.astype(float), edges, "mean") for subject, row in zip(levels, active)}


def head_motion(keypoints: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Median head speed per bin, in body sizes per frame, of one person's rows (one track per frame)."""
    head = keypoints[keypoints["joint"].isin(HEAD_JOINTS) & keypoints["x"].notna()]
    if head.empty:
        return np.full(edges.size - 1, np.nan)
    body = keypoints[keypoints["x"].notna()].groupby("frame")
    size = float(np.nanmedian(np.hypot(body["x"].max() - body["x"].min(), body["y"].max() - body["y"].min())))
    x = head.pivot_table(index="frame", columns="joint", values="x")
    y = head.pivot_table(index="frame", columns="joint", values="y")
    step = np.hypot(x.diff(), y.diff()).mean(axis=1)
    # a jump across a gap in detection is not a movement
    step[np.diff(x.index.to_numpy(), prepend=x.index[0]) != 1] = np.nan
    times = head.groupby("frame")["t"].first().reindex(x.index).to_numpy(dtype=float)
    return _binned(times, step.to_numpy(dtype=float) / max(size, 1.0), edges, "median")


def _binned(times: np.ndarray, values: np.ndarray, edges: np.ndarray, how: str) -> np.ndarray:
    index = np.digitize(times, edges) - 1
    keep = (index >= 0) & (index < edges.size - 1)
    grouped = pd.Series(values[keep]).groupby(index[keep])
    out = grouped.mean() if how == "mean" else grouped.median()
    return out.reindex(range(edges.size - 1)).to_numpy(dtype=float)


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def shift_z(voice: np.ndarray, motion: np.ndarray, min_shift: int) -> float:
    """The real pairing against the same voice shifted in a circle by min_shift bins or more."""
    observed = _rank_correlation(voice, motion)
    shifts = range(min_shift, voice.size - min_shift + 1)
    null = np.array([_rank_correlation(np.roll(voice, shift), motion) for shift in shifts])
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size < 10 or null.std() == 0:
        return float("nan")
    return float((observed - null.mean()) / null.std())


def match_voices(activity: dict[str, np.ndarray], motion: list[np.ndarray], step: float = BIN_S) -> VoiceMatch:
    """One person per voice, the pairing with the highest total z; clear where its z is high and
    beats every other person for that voice by a margin."""
    subjects = [subject for subject, voice in activity.items() if np.nansum(voice) > 0]
    if not subjects or not motion:
        return VoiceMatch()
    min_shift = int(round(MIN_SHIFT_S / step))
    z = np.array([[shift_z(activity[subject], person, min_shift) for person in motion] for subject in subjects])
    rows, columns = linear_sum_assignment(-np.nan_to_num(z, nan=-10.0))
    match = VoiceMatch()
    for row, column in zip(rows, columns):
        subject, value = subjects[row], z[row, column]
        if not np.isfinite(value):
            continue
        others = np.delete(z[row], column)
        rival = float(np.nanmax(others)) if np.isfinite(others).any() else float("-inf")
        match.people[subject], match.z[subject], match.runner_up[subject] = int(column), float(value), rival
        if value >= CLEAR_Z and value - rival >= CLEAR_MARGIN:
            match.clear.append(subject)
    return match
