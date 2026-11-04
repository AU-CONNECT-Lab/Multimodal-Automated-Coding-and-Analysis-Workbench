"""Who is who on a camera nobody clicked, as a model that reads pictures sees each person in view against what
qc.look.people says each person looks like."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.extractors.video.pose.track_repair import RIVAL_SHARE, Chain, PeopleChoice, join_fragments, owner_rows

# frames asked about for each person in view, spread over the time they are in view
LOOK_FRAMES = 5
UNSURE = "unsure"


def people_to_ask(keypoints: pd.DataFrame, wanted: int) -> list[Chain]:
    """The people most in view, as many as are sought, and anyone else in view nearly as long."""
    chains = sorted(join_fragments(keypoints), key=lambda chain: chain.present, reverse=True)
    top = chains[:wanted]
    return top + [chain for chain in chains[wanted:] if top and chain.present >= RIVAL_SHARE * top[-1].present]


def frames_to_ask(chain: Chain) -> list[int]:
    frames = np.unique(chain.frames)
    picks = np.unique(np.linspace(0, frames.size - 1, min(LOOK_FRAMES, frames.size)).round().astype(int))
    return [int(frames[index]) for index in picks]


def picture_of(image: np.ndarray, keypoints: pd.DataFrame, chain: Chain, frame: int) -> bytes | None:
    """The person cut out of the frame with their own skeleton and no name on it, as qc.look shows people."""
    from macaw.qc.look import person_picture

    rows = keypoints[owner_rows(keypoints, chain.tracks) & (keypoints["frame"] == frame).to_numpy()]
    seen = rows[rows["joint"].isin(STANDARD_JOINTS) & np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
    if seen.empty:
        return None
    person = {
        "rows": rows.assign(subject_id=pd.NA),
        "bounds": (float(seen["x"].min()), float(seen["y"].min()), float(seen["x"].max()), float(seen["y"].max())),
    }
    return person_picture(image, person, [])


def named_by_look(
    keypoints: pd.DataFrame,
    people: list[str],
    read: Callable[[Iterable[int]], Iterator[tuple[int, np.ndarray]]],
    ask: Callable[[bytes], dict | None],
    by: str,
) -> tuple[PeopleChoice, list[dict]]:
    """Each person in view is shown on a few frames and the model says which of the people sought it is. A person
    takes a name the model gave on more than half their frames and no other name; a name given to two people goes
    to neither. Whoever it does not tell clearly waits for a click. Also returns what it said about each person."""
    pool = people_to_ask(keypoints, len(people))
    if not pool:
        return PeopleChoice(people={}, doubt="nobody was detected"), []
    asked = {index: frames_to_ask(chain) for index, chain in enumerate(pool)}
    answers: dict[int, list[dict]] = {index: [] for index in asked}
    for frame, image in read(sorted(set().union(*asked.values()))):
        for index, chain in enumerate(pool):
            if frame not in asked[index] or (picture := picture_of(image, keypoints, chain, frame)) is None:
                continue
            said = ask(picture) or {}
            who = str(said.get("who") or UNSURE).strip()
            answers[index].append({"frame": frame, "person": said.get("person"), "who": who if who in people else UNSURE})

    named: dict[int, str] = {}
    for index, given in answers.items():
        counts = Counter(answer["who"] for answer in given if answer["person"] is not False and answer["who"] != UNSURE)
        if len(counts) == 1 and next(iter(counts.values())) > len(given) / 2:
            named[index] = next(iter(counts))
    twice = Counter(named.values())
    placed = {name: pool[index].tracks for index, name in named.items() if twice[name] == 1}

    def told(index: int) -> str:
        said = Counter(answer["who"] for answer in answers[index])
        return f"tracks {pool[index].tracks[:5]}: " + ", ".join(f"{who} {count}" for who, count in said.most_common())

    doubts = []
    shared = sorted(name for name, count in twice.items() if count > 1)
    if shared:
        doubts.append(f"{by} took two people for {', '.join(shared)}")
    unplaced = [person for person in people if person not in placed]
    if unplaced:
        doubts.append(
            f"{by} did not clearly tell {', '.join(unplaced)} ({'; '.join(told(index) for index in answers) or 'no answers'}); "
            "click them on a frame"
        )
    record = [
        {"tracks": pool[index].tracks, "answers": answers[index], "named": named.get(index) if twice.get(named.get(index), 0) == 1 else None}
        for index in answers
    ]
    return PeopleChoice(people=placed, doubt="; ".join(doubts) or None), record


def answer_key(picture: bytes, model: str, described: dict[str, str]) -> str:
    return hashlib.sha1(picture + json.dumps([model, described], sort_keys=True).encode("utf-8")).hexdigest()[:16]
