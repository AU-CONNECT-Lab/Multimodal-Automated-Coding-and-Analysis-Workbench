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


# a fragment may overlap the chain it joins this much at a tracker hand-over
HANDOVER_OVERLAP = 0.1
# and must sit within this many body sizes of where the chain was, at a similar size
SAME_PLACE = 0.5
SAME_SIZE = 2.0
# detections either side of a break that say where the person was
EDGE_DETECTIONS = 10
# naming is in doubt when someone nobody clicked on is this present relative to those named
RIVAL_SHARE = 0.5
# a click falls on a person whose box, widened by this share of its size, holds it
CLICK_REACH = 0.15
# on two people at once, it names the one with a point nearer than this share of the other's distance
NEAREST_SHARE = 0.5
# frames closer than this are one span when pointing a person at where to look
SPAN_GAP_S = 1.0
# the one a camera is pointed at is in doubt when in fewer than this share of the frames with anyone
MIN_OWNER_COVERAGE = 0.5


@dataclass
class Chain:
    """One person's detections, possibly from several tracker ids, in frame order."""

    tracks: list[int]
    frames: np.ndarray
    t: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    size: float
    score: float

    @property
    def x(self) -> float:
        return float(np.median(self.cx))

    @property
    def present(self) -> int:
        return int(np.unique(self.frames).size)


@dataclass
class Click:
    """A person said who is where: at a moment of one camera's file, on a point of its picture."""

    person: str
    at: float
    t: float
    x: float
    y: float


@dataclass
class PeopleChoice:
    people: dict[str, list[int]]
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


def _spans(times) -> list[tuple[float, float]]:
    values = np.unique(np.asarray(times, dtype=float))
    if values.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) > SPAN_GAP_S)
    starts = np.r_[values[0], values[breaks + 1]]
    ends = np.r_[values[breaks], values[-1]]
    return [(round(float(a), 2), round(float(b), 2)) for a, b in zip(starts, ends)]


def _where(chain: Chain, rows) -> np.ndarray:
    return np.array([np.median(chain.cx[rows]), np.median(chain.cy[rows])])


def _gap_distance(chain: Chain, piece: Chain) -> float:
    """How far the piece starts from where the chain last was, or ends from where it is next, or
    sits from where the chain usually is: a box often jumps at the moment the tracker hands over."""
    distances = [np.hypot(*(_where(chain, slice(None)) - _where(piece, slice(None))))]
    before = np.flatnonzero(chain.frames < piece.frames[0])
    if before.size:
        start = _where(piece, slice(None, EDGE_DETECTIONS))
        distances.append(np.hypot(*(_where(chain, before[-EDGE_DETECTIONS:]) - start)))
    after = np.flatnonzero(chain.frames > piece.frames[-1])
    if after.size:
        end = _where(piece, slice(-EDGE_DETECTIONS, None))
        distances.append(np.hypot(*(_where(chain, after[:EDGE_DETECTIONS]) - end)))
    return float(min(distances))


def _conflicts(chain: Chain, piece: Chain, reach: float) -> int:
    """Frames both are in at different places; a second box on the same person is not one."""
    _, mine, theirs = np.intersect1d(piece.frames, chain.frames, return_indices=True)
    apart = np.hypot(piece.cx[mine] - chain.cx[theirs], piece.cy[mine] - chain.cy[theirs])
    return int((apart > reach).sum())


def _extend(chain: Chain, piece: Chain) -> None:
    order = np.argsort(np.r_[chain.frames, piece.frames], kind="stable")
    for name in ("frames", "t", "cx", "cy"):
        setattr(chain, name, np.r_[getattr(chain, name), getattr(piece, name)][order])
    chain.tracks.extend(piece.tracks)
    chain.score += piece.score


def join_fragments(keypoints: pd.DataFrame) -> list[Chain]:
    """People the tracker lost and found again under new ids, strongest first. Tracks are taken
    by presence times size, and each joins the nearest chain it fits."""
    boxes = _track_boxes(keypoints).sort_values(["track_id", "frame"])
    pieces = []
    for track, group in boxes.groupby("track_id"):
        size = float(group["size"].median())
        pieces.append(
            Chain(
                tracks=[int(track)],
                frames=group["frame"].to_numpy(),
                t=group["t"].to_numpy(dtype=float),
                cx=group["cx"].to_numpy(dtype=float),
                cy=group["cy"].to_numpy(dtype=float),
                size=size,
                score=len(group) * size,
            )
        )
    chains: list[Chain] = []
    for piece in sorted(pieces, key=lambda item: item.score, reverse=True):
        best, nearest = None, np.inf
        for chain in chains:
            small, large = sorted([piece.size, chain.size])
            if small <= 0 or large / small > SAME_SIZE:
                continue
            reach = SAME_PLACE * small
            if _conflicts(chain, piece, reach) > HANDOVER_OVERLAP * piece.frames.size:
                continue
            distance = _gap_distance(chain, piece)
            if distance <= reach and distance < nearest:
                best, nearest = chain, distance
        if best is None:
            chains.append(piece)
        else:
            _extend(best, piece)
    return sorted(chains, key=lambda chain: chain.score, reverse=True)


def named_by_clicks(keypoints: pd.DataFrame, clicks: list[Click], frame_period_s: float) -> PeopleChoice:
    """Who a person pointed at: each click names the chain whose body it falls on, in the frame nearest it.
    Nothing is placed by position or presence; what a click cannot settle is said, for a person to click again."""
    chains = join_fragments(keypoints)
    chain_of = {track: index for index, chain in enumerate(chains) for track in chain.tracks}
    boxes = _track_boxes(keypoints)
    people: dict[str, list[int]] = {}
    owner: dict[int, str] = {}
    doubts: list[str] = []
    for click in clicks:
        where = f"{click.person} clicked at {click.at:g} s"
        if boxes.empty or (gap := (boxes["t"] - click.t).abs()).min() > frame_period_s:
            doubts.append(f"nobody was detected in the frame where {where}")
            continue
        frame = boxes.loc[gap.idxmin(), "frame"]
        here = boxes[boxes["frame"] == frame]
        pad = CLICK_REACH * here["size"]
        inside = here[(here["x0"] - pad <= click.x) & (click.x <= here["x1"] + pad) & (here["y0"] - pad <= click.y) & (click.y <= here["y1"] + pad)]
        if inside.empty:
            doubts.append(f"nobody was detected where {where}")
            continue
        track = _clicked_track(keypoints, frame, inside["track_id"].astype(int).tolist(), click)
        if track is None:
            doubts.append(f"{where} falls on two people; click where they do not overlap")
            continue
        index = chain_of[track]
        if index in owner:
            if owner[index] != click.person:
                doubts.append(f"{owner[index]} and {click.person} were clicked on the same person")
            continue
        owner[index] = click.person
        people.setdefault(click.person, []).extend(chains[index].tracks)
    spans: list[tuple[float, float]] = []
    named = [chains[index] for index in owner]
    left_out = [chain for index, chain in enumerate(chains) if index not in owner]
    if named and left_out and left_out[0].present >= RIVAL_SHARE * min(chain.present for chain in named):
        doubts.append(f"tracks {left_out[0].tracks[:5]}, which nobody was clicked on, are in view nearly as much as the people named")
        spans = _spans(left_out[0].t)
    return PeopleChoice(people=people, doubt="; ".join(doubts) or None, spans=spans)


def suggested_people(
    keypoints: pd.DataFrame,
    subject_ids: list[str],
    *,
    references: dict[str, np.ndarray] | None = None,
    voices: dict[str, np.ndarray] | None = None,
    edges: np.ndarray | None = None,
) -> PeopleChoice:
    """Who is who on a camera nobody clicked, from evidence alone: each person's head motion where they are already
    named (references), else their talking on their own mic (voices: share of each bin between edges they were
    talking), matched against head motion here. Nobody is placed by position or presence. Pairings with
    references place even when weak, since the same head seen twice has no listener confound; a voice places
    only when clear. Anything short of clear pairings for everyone is said, for a person to check."""
    from macaw.extractors.video.pose.voice_motion import head_motion, match_voices

    evidence = references or voices
    by = "head motion where they are already named" if references else "their own microphone"
    chains = sorted(join_fragments(keypoints), key=lambda chain: chain.present, reverse=True)
    wanted = [subject for subject in subject_ids if evidence and subject in evidence]
    if not wanted or edges is None or not chains:
        return PeopleChoice(people={}, doubt="nothing to suggest who is who from: no one is named on another camera or has a microphone of their own")
    top = chains[: len(wanted)]
    # someone in view nearly as long as the people sought may be the one the evidence belongs to
    pool = sorted(
        top + [chain for chain in chains[len(wanted):] if chain.present >= RIVAL_SHARE * top[-1].present],
        key=lambda chain: chain.x,
    )
    motion = [head_motion(keypoints[owner_rows(keypoints, chain.tracks)], edges) for chain in pool]
    match = match_voices({subject: evidence[subject] for subject in wanted}, motion)
    sure = dict(match.people) if references else {subject: match.people[subject] for subject in match.clear}
    people = {subject: pool[index].tracks for subject, index in sure.items()}
    parts = [
        f"{subject} at x {pool[index].x:.0f}, z {match.z[subject]:.1f} against {match.runner_up[subject]:.1f}"
        + ("" if subject in match.clear else ", weak")
        for subject, index in sorted(sure.items(), key=lambda item: pool[item[1]].x)
    ]
    doubts = []
    if set(sure) - set(match.clear) or len(sure) < len(subject_ids):
        unplaced = [subject for subject in subject_ids if subject not in sure]
        doubts.append(
            f"suggested by {by} ({'; '.join(parts) or 'no pairing'})"
            + (f", {', '.join(unplaced)} not placed" if unplaced else "") + "; not yet checked by a person"
        )
    spans: list[tuple[float, float]] = []
    left_out = [chain for chain in chains if not any(chain.tracks == tracks for tracks in people.values())]
    placed = [chain for chain in chains if any(chain.tracks == tracks for tracks in people.values())]
    if placed and left_out and left_out[0].present >= RIVAL_SHARE * min(chain.present for chain in placed):
        doubts.append(f"tracks {left_out[0].tracks[:5]} (no subject) are in view nearly as much as the people suggested")
        spans = _spans(left_out[0].t)
    return PeopleChoice(people=people, doubt="; ".join(doubts) or None, spans=spans)


def most_in_view(keypoints: pd.DataFrame, person: str) -> PeopleChoice:
    """On a camera read for one person alone: the person most in view and largest. In doubt when someone else is in view
    at least half as long, or the one chosen is in fewer than half the frames with anyone."""
    chains = join_fragments(keypoints)
    if not chains:
        return PeopleChoice(people={}, doubt="nobody was detected")
    owner, others = chains[0], chains[1:]
    people = {person: owner.tracks}
    chosen = f"tracks {owner.tracks[:5]} suggested as {person} for being most in view and largest"
    rival = max(others, key=lambda chain: chain.present, default=None)
    if rival is not None and rival.present >= RIVAL_SHARE * owner.present:
        return PeopleChoice(people, doubt=f"{chosen}, but tracks {rival.tracks[:5]} are in view nearly as long", spans=_spans(rival.t))
    seen = np.unique(np.concatenate([chain.frames for chain in chains]))
    coverage = owner.present / seen.size
    if coverage < MIN_OWNER_COVERAGE:
        elsewhere = np.concatenate([chain.t[~np.isin(chain.frames, owner.frames)] for chain in others])
        return PeopleChoice(people, doubt=f"{chosen}, but in only {coverage:.0%} of the frames with anyone in them", spans=_spans(elsewhere))
    return PeopleChoice(people)


def _clicked_track(keypoints: pd.DataFrame, frame, candidates: list[int], click: Click) -> int | None:
    """Of the people a click falls on, the one with a point clearly nearest it."""
    if len(candidates) == 1:
        return candidates[0]
    here = keypoints[(keypoints["frame"] == frame) & keypoints["track_id"].isin(candidates) & keypoints["x"].notna()]
    nearest = (
        np.hypot(here["x"] - click.x, here["y"] - click.y).groupby(here["track_id"].astype(int).to_numpy()).min().sort_values()
    )
    if len(nearest) > 1 and nearest.iloc[0] >= NEAREST_SHARE * nearest.iloc[1]:
        return None
    return int(nearest.index[0])


def owner_rows(keypoints: pd.DataFrame, tracks: list[int]) -> np.ndarray:
    """Rows of one person, one track per frame: where two overlap at a hand-over, the earlier listed wins."""
    rank = keypoints["track_id"].map({track: index for index, track in enumerate(tracks)})
    rank = rank.astype("float64").to_numpy()
    best = pd.Series(rank).groupby(keypoints["frame"].to_numpy()).transform("min").to_numpy()
    return ~np.isnan(rank) & (rank == best)


def subject_column(keypoints: pd.DataFrame, people: dict[str, list[int]]) -> pd.Series:
    out = pd.Series(pd.NA, index=keypoints.index, dtype="object")
    for subject, tracks in people.items():
        out[owner_rows(keypoints, tracks)] = subject
    return out
