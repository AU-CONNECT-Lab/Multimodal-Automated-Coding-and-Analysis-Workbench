"""Whose voice on a shared microphone is whom: a suggestion a person confirms."""

from __future__ import annotations

import json
import re

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
    return match_voices(
        turns[turns["exclusive"].astype(bool)], signals, [subject.subject_id for subject in recording.subjects],
        hints=voice_hints(config, recording),
    )


OWN_MIC = "own microphone"
MOUTH = "mouth movement"
ELIMINATION = "elimination"
WORDS = "words"
PITCH = "pitch"

# two voices this far apart in median f0 are told apart by pitch
PITCH_GAP_HZ = 40.0


def read_voice_material(config, recording_id: str) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Each voice's lines, and its speech time and median pitch, as text.shared_asr left them."""
    from macaw.extractors.audio.content.asr import SHARED_ASR, TRANSCRIPT_SCHEMA, VOICE_LINES_NAME, VOICE_PITCH_NAME, read_transcript
    from macaw.project.layout import layout_for
    from macaw.store.schema import empty_frame

    root = layout_for(config, create=False).artifacts_dir(recording_id, SHARED_ASR)
    lines_path, pitch_path = root / VOICE_LINES_NAME, root / VOICE_PITCH_NAME
    lines = read_transcript(lines_path) if lines_path.is_file() else empty_frame(TRANSCRIPT_SCHEMA)
    pitch = json.loads(pitch_path.read_text(encoding="utf-8")) if pitch_path.is_file() else {}
    return lines, pitch


def telling_lines(lines: pd.DataFrame, label: str, n: int = 3) -> list[str]:
    """A voice's n longest lines in the order spoken: short ones like "yeah" say nothing about who it is."""
    own = lines[lines["subject_id"].astype(str) == label]
    longest = own.assign(length=own["text"].fillna("").astype(str).str.len()).nlargest(n, "length")
    return longest.sort_values("t_start")["text"].astype(str).tolist()


def voice_hints(config, recording: Recording) -> dict[str, dict]:
    """Per person, the voice identity.voice_hints and identity.pitch_order point to; words win over pitch."""
    identity = config.identity
    if not identity.voice_hints and not identity.pitch_order:
        return {}
    roles = {subject.subject_id: subject.role for subject in recording.subjects}
    lines, pitch = read_voice_material(config, recording.recording_id)
    found = by_words(lines, identity.voice_hints, roles)
    taken = {hint["label"] for hint in found.values()}
    for subject_id, hint in by_pitch(pitch, identity.pitch_order, roles).items():
        if subject_id in found:
            if found[subject_id]["label"] != hint["label"]:
                found[subject_id]["disputed"] = f"pitch points to {hint['label']}"
        elif hint["label"] not in taken:
            found[subject_id] = hint
    return found


def by_words(lines: pd.DataFrame, hints, roles: dict[str, str | None]) -> dict[str, dict]:
    """A voice goes to a person when more of its lines match that person's role than any other role."""
    from scipy.optimize import linear_sum_assignment

    matched: dict[str, dict[str, list[str]]] = {}
    for hint in hints:
        pattern = re.compile(hint.says, re.IGNORECASE)
        for label, text in zip(lines["subject_id"].astype(str), lines["text"].fillna("").astype(str)):
            if pattern.search(text):
                matched.setdefault(label, {}).setdefault(hint.role, []).append(text)
    people = [subject_id for subject_id, role in roles.items() if role is not None]
    labels = sorted(matched)
    if not people or not labels:
        return {}
    score = np.zeros((len(people), len(labels)))
    for row, subject_id in enumerate(people):
        for col, label in enumerate(labels):
            counts = {role: len(said) for role, said in matched[label].items()}
            mine = counts.get(roles[subject_id], 0)
            # a voice matching two roles equally says nothing about either
            score[row, col] = mine if mine > max((n for role, n in counts.items() if role != roles[subject_id]), default=0) else 0
    found = {}
    for row, col in zip(*linear_sum_assignment(score, maximize=True)):
        if score[row, col] > 0:
            said = matched[labels[col]][roles[people[row]]]
            found[people[row]] = {"label": labels[col], "source": WORDS, "n": len(said), "lines": said[:2]}
    return found


def by_pitch(pitch: dict[str, dict], order: list[str], roles: dict[str, str | None]) -> dict[str, dict]:
    """The voices that speak most, as many as order names, matched lowest to highest when far enough apart."""
    if not order:
        return {}
    loudest = sorted(pitch.items(), key=lambda item: item[1]["speech_s"], reverse=True)[: len(order)]
    voiced = sorted((voice["median_hz"], label) for label, voice in loudest if voice["median_hz"] is not None)
    if len(voiced) != len(order) or (np.diff([median for median, _ in voiced]) < PITCH_GAP_HZ).any():
        return {}
    found = {}
    for role, (median, label) in zip(order, voiced):
        holders = [subject_id for subject_id, held in roles.items() if held == role]
        if len(holders) != 1:
            return {}
        found[holders[0]] = {"label": label, "source": PITCH, "median_hz": median}
    return found


def match_voices(
    turns: pd.DataFrame, signals: dict[str, tuple], subjects: list[str] | None = None, hints: dict[str, dict] | None = None
) -> dict | None:
    """One voice per person, the assignment with the highest summed rank correlation of each person's
    signal (t, value, source) with that voice talking; hints place people no signal placed, and one person
    and one voice left over are paired."""
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
    for subject, hint in (hints or {}).items():
        if subject in evidence:
            if evidence[subject]["label"] != hint["label"]:
                evidence[subject]["disputed"] = f"{hint['source']} point to {hint['label']}"
        elif people.get(hint["label"]) == NO_ONE:
            people[hint["label"]] = subject
            evidence[subject] = {"rho": float("nan"), "z": float("nan"), **hint}
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
    return "; ".join(f"{label} is {said}" for label, said in sorted(suggested_people(suggestion).items()))


def suggested_people(suggestion: dict) -> dict[str, str]:
    """Per voice, whom the suggestion gives it to and on what evidence."""
    from macaw.qc.checks import _MOUTH_Z

    shown = {}
    for label, who in suggestion["people"].items():
        if who is None:
            shown[label] = f"one of {', '.join(suggestion.get('unheard', []))} (listen to it)"
            continue
        evidence = suggestion["evidence"].get(who)
        if evidence is None:
            shown[label] = who
            continue
        if evidence.get("source") == ELIMINATION:
            shown[label] = f"{who} (the one voice and person left: listen to it)"
            continue
        disputed = f", but {evidence['disputed']}: listen to it" if evidence.get("disputed") else ""
        if evidence.get("source") == WORDS:
            shown[label] = f"{who} ({evidence['n']} line(s) such as \"{evidence['lines'][0]}\"{disputed})"
            continue
        if evidence.get("source") == PITCH:
            shown[label] = f"{who} (pitch, median {evidence['median_hz']:.0f} Hz{disputed})"
            continue
        if disputed:
            shown[label] = f"{who} ({evidence['source']}{disputed})"
            continue
        by = f"{evidence['source']}, " if evidence.get("source") else ""
        if not np.isfinite(evidence["z"]):
            # too few stretches of signal to shift against, as when a face is seen only now and then
            shown[label] = f"{who} ({by}rho {evidence['rho']:.2f}, too little of it to test: listen to it)"
            continue
        # a signal that barely follows the voice was matched by elimination, so it needs listening to
        weak = "" if evidence["z"] >= _MOUTH_Z else ", weak: listen to it"
        shown[label] = f"{who} ({by}rho {evidence['rho']:.2f}, z {evidence['z']:.1f}{weak})"
    return shown
