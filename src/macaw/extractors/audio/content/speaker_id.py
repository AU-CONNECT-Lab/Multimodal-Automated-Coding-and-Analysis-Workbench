"""Whose voice on a shared microphone is whom: a suggestion a person confirms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macaw.project.recording import Recording


def suggest_voices(config, recording: Recording) -> dict | None:
    """Which voice on the shared microphone each person is: by their own microphone's loudness where
    they have one, else by their mouth movement, and a last person left by elimination."""
    from macaw.exceptions import InputNotFoundError, UnreadableMediaError
    from macaw.extractors.audio.content.diarization import read_diarization
    from macaw.extractors.prep import load_prep
    from macaw.extractors.video.pose.voice_motion import ENERGY_FRAME_S, frame_levels
    from macaw.project.layout import layout_for
    from macaw.qc.checks import MOUTH_SOURCE, RecordingResults

    rid = recording.recording_id
    try:
        turns = read_diarization(config, rid)
        prep = load_prep(config, rid)
    except (InputNotFoundError, UnreadableMediaError):
        return None
    if turns.empty:
        return None
    signals = {}
    for subject_id in prep.tracks:
        level = frame_levels(prep.track_path(subject_id))
        signals[subject_id] = (prep.t_origin_s + np.arange(level.size) * ENERGY_FRAME_S, level, OWN_MIC)
    mouth = RecordingResults(layout_for(config, create=False), rid).long_table(MOUTH_SOURCE, features=["mouth_movement"])
    for subject_id, rows in mouth.groupby("subject_id", dropna=True, observed=True):
        if str(subject_id) not in signals:
            signals[str(subject_id)] = (rows["t_start"].to_numpy(dtype=float), rows["value"].to_numpy(dtype=float), MOUTH)
    return match_voices(turns[turns["exclusive"].astype(bool)], signals, [subject.subject_id for subject in recording.subjects])


OWN_MIC = "own microphone"
MOUTH = "mouth movement"
ELIMINATION = "elimination"


def match_voices(turns: pd.DataFrame, signals: dict[str, tuple], subjects: list[str] | None = None) -> dict | None:
    """One voice per person, the assignment with the highest summed rank correlation of each person's
    signal (t, value, source) with that voice talking; one person and one voice left over are paired."""
    from scipy.optimize import linear_sum_assignment

    from macaw.extractors.audio.content.diarization import NO_ONE
    from macaw.qc.checks import _MOUTH_BIN_S, _rank_contrast, _within

    labels = sorted(str(label) for label in turns["label"].unique())
    scores: dict[str, dict[str, tuple[float, float]]] = {}
    sources: dict[str, str] = {}
    for subject_id, (t, value, source) in signals.items():
        bins = (np.asarray(t, dtype=float) // _MOUTH_BIN_S).astype(int)
        binned = pd.Series(np.asarray(value, dtype=float)).groupby(bins).mean().dropna()
        if binned.size < 10:
            continue
        centres = (binned.index.to_numpy() + 0.5) * _MOUTH_BIN_S
        level = binned.rank().to_numpy()
        sources[subject_id] = source
        scores[subject_id] = {}
        for label in labels:
            own = turns[turns["label"] == label]
            mask = _within(centres, own["t_start"].to_numpy(dtype=float), own["t_end"].to_numpy(dtype=float))
            scores[subject_id][label] = _rank_contrast(level, mask) if mask.any() else (float("nan"), float("nan"))
    people = {label: NO_ONE for label in labels}
    evidence = {}
    measured = sorted(scores)
    if measured:
        rho = np.array([[scores[subject][label][0] for label in labels] for subject in measured])
        rows, cols = linear_sum_assignment(np.nan_to_num(rho, nan=-1.0), maximize=True)
        for row, col in zip(rows, cols):
            subject, label = measured[row], labels[col]
            people[label] = subject
            others = {other: scores[subject][other][0] for other in labels if other != label and np.isfinite(scores[subject][other][0])}
            rival = max(others, key=others.get) if others else None
            evidence[subject] = {
                "label": label, "source": sources[subject], "rho": scores[subject][label][0], "z": scores[subject][label][1],
                **({"rival": rival, "rival_rho": others[rival]} if rival else {}),
            }
    unheard = [subject for subject in subjects or () if subject not in evidence]
    left = [label for label, who in people.items() if who == NO_ONE]
    if len(unheard) == 1 and len(left) == 1:
        people[left[0]] = unheard[0]
        evidence[unheard[0]] = {"label": left[0], "source": ELIMINATION, "rho": float("nan"), "z": float("nan")}
        unheard = []
    if unheard:
        # a voice left over could be anyone left over; leaving it out here would drop a person
        people.update({label: None for label in left})
    if not evidence:
        return None
    return {"people": people, "evidence": evidence, "unheard": unheard}


def describe_suggestion(suggestion: dict) -> str:
    from macaw.qc.checks import _MOUTH_Z

    shown = []
    for label, who in sorted(suggestion["people"].items()):
        if who is None:
            shown.append(f"{label} is one of {', '.join(suggestion.get('unheard', []))} (listen to it)")
            continue
        evidence = suggestion["evidence"].get(who)
        if evidence is None:
            shown.append(f"{label} is {who}")
            continue
        if evidence.get("source") == ELIMINATION:
            shown.append(f"{label} is {who} (the one voice and person left: listen to it)")
            continue
        by = f"{evidence['source']}, " if evidence.get("source") else ""
        if not np.isfinite(evidence["z"]):
            # too few stretches of signal to shift against, as when a face is seen only now and then
            shown.append(f"{label} is {who} ({by}rho {evidence['rho']:.2f}, too little of it to test: listen to it)")
            continue
        # a signal that barely follows the voice was matched by elimination, so it needs listening to
        weak = "" if evidence["z"] >= _MOUTH_Z else ", weak: listen to it"
        shown.append(f"{label} is {who} ({by}rho {evidence['rho']:.2f}, z {evidence['z']:.1f}{weak})")
    return "; ".join(shown)
