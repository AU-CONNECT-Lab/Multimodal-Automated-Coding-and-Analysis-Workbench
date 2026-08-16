"""body_id swapping mid-recording. No existing tool tells you this happened."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS

# A crossing only counts when the swapped pairing beats the straight one by this much of the
# total, relative rather than absolute so it does not depend on frame size or on how fast the
# people happen to be moving.
SWAP_MARGIN = 0.2


@dataclass
class SwapEvent:
    t: float
    frame: int
    track_a: int
    track_b: int
    evidence: str
    score: float


def _centroids(keypoints: pd.DataFrame) -> pd.DataFrame:
    """One point per track per frame: the mean of the standard joints that were measured.
    Extra joints are left out so a backend with hands does not weight the mean toward them."""
    standard = keypoints[keypoints["joint"].isin(STANDARD_JOINTS)]
    standard = standard[standard["track_id"].notna()]
    grouped = standard.groupby(["frame", "track_id"], dropna=True, observed=True)
    out = grouped.agg(t=("t", "first"), x=("x", "mean"), y=("y", "mean")).reset_index()
    return out.dropna(subset=["x", "y"]).sort_values(["frame", "track_id"])


def detect_swaps(keypoints: pd.DataFrame, *, max_gap_s: float) -> list[SwapEvent]:
    """Large simultaneous jumps in opposite directions are the signature of a swap."""
    centroids = _centroids(keypoints)
    if centroids.empty:
        return []

    last: dict[int, tuple[float, float, float]] = {}
    events: list[SwapEvent] = []

    for frame_number, group in centroids.groupby("frame", sort=True):
        present = {
            int(row.track_id): (float(row.t), float(row.x), float(row.y))
            for row in group.itertuples(index=False)
        }
        candidates: list[SwapEvent] = []
        tracks = sorted(present)
        for index, track_a in enumerate(tracks):
            for track_b in tracks[index + 1:]:
                if track_a not in last or track_b not in last:
                    continue
                t_a, xa0, ya0 = last[track_a]
                t_b, xb0, yb0 = last[track_b]
                t_now, xa1, ya1 = present[track_a]
                _, xb1, yb1 = present[track_b]
                # a track that was gone for longer than the gap is a new track, not the same
                # person reappearing, so there is nothing to have swapped with
                if (t_now - t_a) > max_gap_s or (t_now - t_b) > max_gap_s:
                    continue
                # two people standing in the same place cannot be told apart by position, and
                # calling that a swap would relabel noise
                if np.hypot(xa0 - xb0, ya0 - yb0) <= 0:
                    continue

                stay = np.hypot(xa1 - xa0, ya1 - ya0) + np.hypot(xb1 - xb0, yb1 - yb0)
                swap = np.hypot(xa1 - xb0, ya1 - yb0) + np.hypot(xb1 - xa0, yb1 - ya0)
                total = stay + swap
                if total <= 0:
                    continue
                score = float((stay - swap) / total)
                if score > SWAP_MARGIN:
                    candidates.append(
                        SwapEvent(
                            t=float(t_now),
                            frame=int(frame_number),
                            track_a=int(track_a),
                            track_b=int(track_b),
                            evidence="centroid_crossover",
                            score=score,
                        )
                    )

        # one track cannot swap with two partners in the same frame; keep the clearest
        claimed: set[int] = set()
        for event in sorted(candidates, key=lambda item: item.score, reverse=True):
            if event.track_a in claimed or event.track_b in claimed:
                continue
            claimed.update({event.track_a, event.track_b})
            events.append(event)

        last.update(present)

    return events


def repair_tracks(keypoints: pd.DataFrame, swaps: list[SwapEvent]) -> pd.DataFrame:
    """Relabel from each swap onward. Detection reads raw ids throughout, so the mapping is
    rebuilt from the raw column every time rather than applied on top of itself."""
    out = keypoints.copy()
    if not swaps:
        return out

    raw = out["track_id"]
    mapping = {track: track for track in raw.dropna().unique()}
    frames = out["frame"].to_numpy()

    for event in sorted(swaps, key=lambda item: item.frame):
        if event.track_a not in mapping or event.track_b not in mapping:
            continue
        mapping[event.track_a], mapping[event.track_b] = (
            mapping[event.track_b],
            mapping[event.track_a],
        )
        selected = frames >= event.frame
        out.loc[selected, "track_id"] = raw[selected].map(mapping)

    return out


def _track_order(keypoints: pd.DataFrame) -> pd.DataFrame:
    """Tracks ranked by where they first appear on screen, longest-lived first."""
    standard = keypoints[keypoints["joint"].isin(STANDARD_JOINTS)]
    standard = standard[standard["track_id"].notna() & standard["x"].notna()]
    if standard.empty:
        return pd.DataFrame(columns=["track_id", "first_x", "n_frames"])

    first_frame = standard.groupby("track_id", observed=True)["frame"].transform("min")
    opening = standard[standard["frame"] == first_frame]
    return (
        pd.DataFrame(
            {
                "first_x": opening.groupby("track_id", observed=True)["x"].mean(),
                "n_frames": standard.groupby("track_id", observed=True)["frame"].nunique(),
            }
        )
        .reset_index()
        .sort_values("n_frames", ascending=False)
    )


def assign_subjects(
    keypoints: pd.DataFrame,
    subject_ids: list[str],
    *,
    identity_anchors: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Map repaired tracks onto stable subject_ids, anchored by initial screen position.

    Anchors are read as an ordering, not as a distance: they are configured as a fraction of
    frame width while x is in pixels, and a fraction cannot be compared to a pixel without
    knowing a width this function was never given. Ordering is the part that survives.
    """
    out = keypoints.copy()
    out["subject_id"] = pd.NA

    order = _track_order(keypoints)
    if order.empty or not subject_ids:
        return out

    anchors = identity_anchors or {}
    # without anchors the configured subject order is the only stated intent there is, and
    # inventing an alphabetical one would silently disagree with it
    ranked = (
        sorted(subject_ids, key=lambda subject: (anchors.get(subject, 0.0), subject_ids.index(subject)))
        if anchors
        else list(subject_ids)
    )
    # more tracks than people means the extras are spurious detections; the longest-lived
    # ones are the people, and what is left over stays unassigned rather than being forced
    kept = order.head(len(ranked)).sort_values("first_x")

    pairing = dict(zip(kept["track_id"].tolist(), ranked))
    out["subject_id"] = out["track_id"].map(pairing).astype("object")
    return out


def track_quality(keypoints: pd.DataFrame) -> pd.DataFrame:
    """Continuity, mean confidence, missing fraction. Bad tracks warn instead of emitting NaN."""
    columns = [
        "track_id",
        "n_frames",
        "t_start",
        "t_end",
        "mean_confidence",
        "missing_fraction",
        "continuity",
    ]
    standard = keypoints[keypoints["joint"].isin(STANDARD_JOINTS)]
    standard = standard[standard["track_id"].notna()]
    if standard.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for track, group in standard.groupby("track_id", observed=True):
        frames = np.sort(group["frame"].unique())
        span = int(frames[-1] - frames[0]) + 1
        rows.append(
            {
                "track_id": int(track),
                "n_frames": int(frames.size),
                "t_start": float(group["t"].min()),
                "t_end": float(group["t"].max()),
                "mean_confidence": float(group["confidence"].mean()),
                # a joint that was detected but below threshold is missing here too, which is
                # why x is what gets counted rather than the presence of a row
                "missing_fraction": float(group["x"].isna().mean()),
                # frames actually seen over frames the track spans: a track that vanishes for
                # half its life scores 0.5 even though every frame it does have is perfect
                "continuity": float(frames.size / span) if span > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("track_id").reset_index(drop=True)
