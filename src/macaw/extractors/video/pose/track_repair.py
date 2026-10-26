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


# a later fragment of the owner may overlap the chain this much at a tracker hand-over
HANDOVER_OVERLAP = 0.1
# and must sit within this many owner body sizes of the owner, at a similar size
SAME_PLACE = 0.5
SAME_SIZE = 2.0
# the choice is in doubt when someone else is this present relative to the owner
RIVAL_SHARE = 0.5
MIN_OWNER_COVERAGE = 0.5
# frames closer than this are one span when pointing a person at where to look
SPAN_GAP_S = 1.0


@dataclass
class OwnerChoice:
    tracks: list[int]
    doubt: str | None = None
    spans: list[tuple[float, float]] | None = None


def several_people(keypoints: pd.DataFrame) -> bool:
    tracked = keypoints[keypoints["track_id"].notna() & keypoints["x"].notna()]
    return bool((tracked.groupby("frame")["track_id"].nunique() > 1).any())


def _track_boxes(keypoints: pd.DataFrame) -> pd.DataFrame:
    """Per track and frame: centre and diagonal of the standard joints that were seen."""
    standard = keypoints[keypoints["joint"].isin(STANDARD_JOINTS)]
    standard = standard[standard["track_id"].notna() & standard["x"].notna()]
    grouped = standard.groupby(["track_id", "frame"], observed=True)
    boxes = grouped.agg(
        t=("t", "first"), x0=("x", "min"), x1=("x", "max"), y0=("y", "min"), y1=("y", "max")
    ).reset_index()
    boxes["cx"] = (boxes["x0"] + boxes["x1"]) / 2
    boxes["cy"] = (boxes["y0"] + boxes["y1"]) / 2
    boxes["size"] = np.hypot(boxes["x1"] - boxes["x0"], boxes["y1"] - boxes["y0"])
    return boxes


def _spans(times: pd.Series) -> list[tuple[float, float]]:
    values = np.sort(times.to_numpy(dtype=float))
    if values.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) > SPAN_GAP_S)
    starts = np.r_[values[0], values[breaks + 1]]
    ends = np.r_[values[breaks], values[-1]]
    return [(round(float(a), 2), round(float(b), 2)) for a, b in zip(starts, ends)]


def choose_camera_owner(keypoints: pd.DataFrame) -> OwnerChoice:
    """The person a per-person camera is pointed at: the most present and largest track, plus
    the fragments the tracker split them into after an occlusion."""
    boxes = _track_boxes(keypoints)
    if boxes.empty:
        return OwnerChoice(tracks=[])
    stats = boxes.groupby("track_id").agg(
        n=("frame", "size"), size=("size", "median"), cx=("cx", "median"), cy=("cy", "median")
    )
    stats["score"] = stats["n"] * stats["size"]
    stats = stats.sort_values("score", ascending=False)
    frames = boxes.groupby("track_id")["frame"].apply(set)

    seed = stats.iloc[0]
    chain = [int(stats.index[0])]
    owned = set(frames[stats.index[0]])
    for track, row in stats.iloc[1:].iterrows():
        mine = frames[track]
        if len(mine & owned) > HANDOVER_OVERLAP * len(mine):
            continue
        if np.hypot(row["cx"] - seed["cx"], row["cy"] - seed["cy"]) > SAME_PLACE * seed["size"]:
            continue
        sizes = sorted([row["size"], seed["size"]])
        if sizes[0] <= 0 or sizes[1] / sizes[0] > SAME_SIZE:
            continue
        chain.append(int(track))
        owned |= mine

    others = stats.drop(index=chain)
    owner_score = float(stats.loc[chain, "score"].sum())
    if not others.empty and others["score"].iloc[0] >= RIVAL_SHARE * owner_score:
        rival = int(others.index[0])
        return OwnerChoice(
            tracks=chain,
            doubt=f"track {rival} is in view nearly as much as the chosen tracks {chain}",
            spans=_spans(boxes.loc[boxes["track_id"] == rival, "t"]),
        )
    coverage = len(owned) / boxes["frame"].nunique()
    if coverage < MIN_OWNER_COVERAGE:
        elsewhere = boxes[~boxes["frame"].isin(owned)].drop_duplicates("frame")
        return OwnerChoice(
            tracks=chain,
            doubt=f"the chosen tracks {chain} are in only {coverage:.0%} of the frames with anyone in them",
            spans=_spans(elsewhere["t"]),
        )
    return OwnerChoice(tracks=chain)


def owner_rows(keypoints: pd.DataFrame, tracks: list[int]) -> np.ndarray:
    """Rows of the owner, one track per frame: where two overlap at a hand-over, the earlier listed wins."""
    rank = keypoints["track_id"].map({track: index for index, track in enumerate(tracks)})
    rank = rank.astype("float64").to_numpy()
    best = pd.Series(rank).groupby(keypoints["frame"].to_numpy()).transform("min").to_numpy()
    return ~np.isnan(rank) & (rank == best)
