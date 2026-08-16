"""Utterances become turns, turns become who held the floor and for how long."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.config.schema import TurnsSection
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import (
    ASRExtractor,
    TRANSCRIPT_NAME,
    load_transcript,
)
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale, empty_frame

TURNS_NAME = "turns.parquet"

TURN_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "turn_index": "int64",
    "t_start": "float64",
    "t_end": "float64",
    "n_utterances": "int64",
    "n_words": "int64",
    "text": "string",
}

# a conversation this short has no stable turn statistics, whatever the numbers say
_MIN_TURNS = 10
_DOMINANCE_WARN = 0.9


def _other_speaker_in(
    speakers: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    speaker: str,
    low: float,
    high: float,
) -> bool:
    if high <= low:
        return False
    return bool(((speakers != speaker) & (ends > low) & (starts < high)).any())


def utterances_to_turns(transcript: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Merge consecutive same-speaker utterances separated by less than max_gap_s.

    The gap also has to be uninterrupted: once somebody else speaks the floor has changed, so
    the turn ends there even if the same person picks up again straight away. Without that
    condition a turn is only a speaking spurt, and a floor transfer stops being a well defined
    event for turns.transitions to measure."""
    columns = ["subject_id", "turn_index", "t_start", "t_end", "n_utterances", "n_words", "text"]
    if transcript.empty:
        return empty_frame(TURN_SCHEMA).loc[:, columns]

    frame = transcript.sort_values("t_start", kind="stable").reset_index(drop=True)
    starts = frame["t_start"].to_numpy(dtype=float)
    ends = frame["t_end"].to_numpy(dtype=float)
    speakers = frame["subject_id"].astype("object").to_numpy()

    assigned = np.empty(len(frame), dtype=np.int64)
    open_turn: dict[object, int] = {}
    open_end: dict[object, float] = {}
    n_turns = 0

    for index in range(len(frame)):
        speaker = speakers[index]
        previous_end = open_end.get(speaker)
        continues = (
            previous_end is not None
            and starts[index] - previous_end < cfg.max_gap_s
            and not _other_speaker_in(
                speakers, starts, ends, speaker, previous_end, starts[index]
            )
        )
        if continues:
            assigned[index] = open_turn[speaker]
            open_end[speaker] = max(previous_end, ends[index])
        else:
            assigned[index] = n_turns
            open_turn[speaker] = n_turns
            open_end[speaker] = ends[index]
            n_turns += 1

    frame = frame.assign(turn_index=assigned)
    turns = (
        frame.groupby("turn_index", sort=True)
        .agg(
            subject_id=("subject_id", "first"),
            t_start=("t_start", "min"),
            t_end=("t_end", "max"),
            n_utterances=("t_start", "size"),
            n_words=("n_words", "sum"),
            text=("text", _join_text),
        )
        .reset_index()
    )
    return turns.loc[:, columns]


def _join_text(values) -> str:
    return " ".join(str(value).strip() for value in values if value is not None and str(value).strip())


def turn_durations(turns: pd.DataFrame) -> pd.DataFrame:
    out = turns.copy()
    out["duration_s"] = out["t_end"] - out["t_start"]
    return out


def dominance(turns: pd.DataFrame) -> dict[str, float]:
    """Share of floor time per speaker. Floor time counts the pauses inside a turn, which is
    what makes it different from total_speech_s over in text.asr."""
    if turns.empty:
        return {}
    held = turn_durations(turns).groupby("subject_id", observed=True)["duration_s"].sum()
    total = float(held.sum())
    return {str(name): float(value) / total if total > 0 else np.nan for name, value in held.items()}


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> list[tuple[float, float]]:
    order = np.argsort(starts, kind="stable")
    merged: list[tuple[float, float]] = []
    for index in order:
        low, high = float(starts[index]), float(ends[index])
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def silences(turns: pd.DataFrame, *, min_s: float) -> pd.DataFrame:
    """Joint silence: gaps after merging all speakers' talk into one timeline.

    Fed the utterances rather than the turns, because a pause inside a turn is still nobody
    talking. Only gaps between talk are returned; whatever sits before the first word or
    after the last is setup, not silence in the conversation."""
    if turns.empty:
        return pd.DataFrame({"t_start": [], "t_end": [], "duration_s": []}, dtype=float)

    merged = _merge_intervals(
        turns["t_start"].to_numpy(dtype=float), turns["t_end"].to_numpy(dtype=float)
    )
    gaps = [
        (previous[1], current[0])
        for previous, current in zip(merged, merged[1:])
        if current[0] - previous[1] >= min_s
    ]
    return pd.DataFrame(
        {
            "t_start": [gap[0] for gap in gaps],
            "t_end": [gap[1] for gap in gaps],
            "duration_s": [gap[1] - gap[0] for gap in gaps],
        },
        dtype=float,
    )


def read_turns(path: Path) -> pd.DataFrame:
    return _io.read_file(Path(path), TURN_SCHEMA, "turns")


def load_turns(config, recording_id: str) -> pd.DataFrame:
    """Where every downstream turn extractor starts, the way audio ones start at load_prep."""
    root = layout_for(config, create=False).artifacts_dir(
        recording_id, TurnStructureExtractor.name
    )
    return read_turns(root / TURNS_NAME)


_TURN_SPECS = {
    "turn_duration_s": ("Length of one turn, from its first word to its last.", "s"),
    "n_utterances_in_turn": ("Utterances merged into this turn.", "count"),
    "turn_n_words": ("Words in the turn.", "count"),
    "turn_speech_rate_wps": (
        "Words per second across the whole turn, pauses inside it included.",
        "word/s",
    ),
}

_SILENCE_SPECS = {
    "joint_silence_s": ("Length of a stretch where nobody was talking.", "s"),
}

_SUBJECT_SPECS = {
    "n_turns": ("Turns this speaker took.", "count"),
    "talk_time_s": ("Seconds this speaker held the floor, pauses inside turns included.", "s"),
    "talk_ratio": ("Share of all floor time held by this speaker; 0.5 is an even split.", None),
    "mean_turn_s": ("Mean turn length.", "s"),
    "median_turn_s": ("Median turn length, which a single long turn cannot move.", "s"),
    "turns_per_minute": ("Turns per minute of the task window.", "1/min"),
}

_RECORDING_SPECS = {
    "n_turns_total": ("Turns taken by everyone.", "count"),
    "n_joint_silences": ("Stretches of at least silence_min_s where nobody talked.", "count"),
    "joint_silence_total_s": ("Total time nobody was talking.", "s"),
    "joint_silence_fraction": ("Share of the task window with nobody talking.", None),
}


@register_extractor
class TurnStructureExtractor(Extractor):
    name = "turns.structure"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "turns"
    config_sections = ("turns", "text")
    requires = (ASRExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        root = layout_for(self.config, create=False).artifacts_dir(
            recording.recording_id, ASRExtractor.name
        )
        return [root / TRANSCRIPT_NAME]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.turns
        transcript = load_transcript(self.config, recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)

        turns = utterances_to_turns(transcript, cfg)
        turns["recording_id"] = recording.recording_id
        quiet = silences(transcript, min_s=cfg.silence_min_s)

        artifacts = {
            TURNS_NAME: _io.write_file(turns, workspace / TURNS_NAME, TURN_SCHEMA, "turns")
        }
        return ExtractorResult(
            rows=self._rows(recording, turns, quiet, prep),
            specs=self._specs(),
            artifacts=artifacts,
            warnings=self._warnings(recording, turns, prep),
        )

    def _rows(self, recording: Recording, turns, quiet, prep) -> pd.DataFrame:
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        minutes = prep.duration_s / 60.0
        shares = dominance(turns)
        frames = []

        for subject_id, group in turns.groupby("subject_id", dropna=False, observed=True):
            duration = (group["t_end"] - group["t_start"]).to_numpy(dtype=float)
            words = group["n_words"].to_numpy(dtype=float)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": group["t_start"].to_numpy(dtype=float),
                            "t_end": group["t_end"].to_numpy(dtype=float),
                            "turn_duration_s": duration,
                            "n_utterances_in_turn": group["n_utterances"].to_numpy(dtype=float),
                            "turn_n_words": words,
                            "turn_speech_rate_wps": np.divide(
                                words,
                                duration,
                                out=np.full_like(words, np.nan),
                                where=duration > 0,
                            ),
                        }
                    ),
                    scale=Scale.EVENT,
                    subject_id=str(subject_id),
                    **common,
                )
            )
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [window[0]],
                            "t_end": [window[1]],
                            "n_turns": [float(len(group))],
                            "talk_time_s": [float(duration.sum())],
                            "talk_ratio": [shares.get(str(subject_id), np.nan)],
                            "mean_turn_s": [float(duration.mean())],
                            "median_turn_s": [float(np.median(duration))],
                            "turns_per_minute": [len(group) / minutes if minutes > 0 else np.nan],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=str(subject_id),
                    **common,
                )
            )

        if not quiet.empty:
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": quiet["t_start"].to_numpy(dtype=float),
                            "t_end": quiet["t_end"].to_numpy(dtype=float),
                            "joint_silence_s": quiet["duration_s"].to_numpy(dtype=float),
                        }
                    ),
                    scale=Scale.EVENT,
                    **common,
                )
            )

        silent = float(quiet["duration_s"].sum()) if not quiet.empty else 0.0
        frames.append(
            build_long_rows(
                values=pd.DataFrame(
                    {
                        "t_start": [window[0]],
                        "t_end": [window[1]],
                        "n_turns_total": [float(len(turns))],
                        "n_joint_silences": [float(len(quiet))],
                        "joint_silence_total_s": [silent],
                        "joint_silence_fraction": [
                            silent / prep.duration_s if prep.duration_s > 0 else np.nan
                        ],
                    }
                ),
                scale=Scale.RECORDING,
                **common,
            )
        )
        return concat_long_tables(frames)

    def _warnings(self, recording: Recording, turns, prep) -> list[QCWarning]:
        warnings = []
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)

        if len(turns) < _MIN_TURNS:
            warnings.append(
                QCWarning(
                    check="turns_too_few",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"Only {len(turns)} turns in the task window. Turn statistics from this "
                        "recording are not stable enough to interpret."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )

        taken = set(turns["subject_id"].astype(str)) if not turns.empty else set()
        for subject in recording.subjects:
            if subject.subject_id not in taken:
                warnings.append(
                    QCWarning(
                        check="turns_silent_speaker",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message="This speaker never took a turn.",
                        t_start=window[0],
                        t_end=window[1],
                    )
                )

        for subject_id, share in dominance(turns).items():
            if share > _DOMINANCE_WARN:
                warnings.append(
                    QCWarning(
                        check="turns_dominance",
                        recording_id=recording.recording_id,
                        subject_id=subject_id,
                        severity="warn",
                        message=(
                            f"This speaker holds {share:.0%} of the floor. Check that the other "
                            "tracks were transcribed and attributed to the right people."
                        ),
                        t_start=window[0],
                        t_end=window[1],
                    )
                )
        return warnings

    def _specs(self) -> list[FeatureSpec]:
        cfg = self.config.turns
        params = {"max_gap_s": cfg.max_gap_s, "silence_min_s": cfg.silence_min_s}
        groups = (
            (_TURN_SPECS, Scale.EVENT),
            (_SILENCE_SPECS, Scale.EVENT),
            (_SUBJECT_SPECS, Scale.RECORDING),
            (_RECORDING_SPECS, Scale.RECORDING),
        )
        return [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=scale,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation="mean",
                validation="none",
            )
            for specs, scale in groups
            for feature, (description, unit) in specs.items()
        ]
