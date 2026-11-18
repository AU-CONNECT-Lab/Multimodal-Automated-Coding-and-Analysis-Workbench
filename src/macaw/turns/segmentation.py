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
    WORDS_NAME,
    load_transcript,
    load_words,
    transcript_between,
)
from macaw.extractors.audio.content.diarization import OTHER
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.segments import Segment
from macaw.store import _io
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale, empty_frame
from macaw.turns.talk import clip_talk, said_units, talk_inputs, talk_params, talk_spans, voices_time_talk

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
    "within": "bool",
}
_COLUMNS = ["subject_id", "turn_index", "t_start", "t_end", "n_utterances", "n_words", "text", "within"]

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


def _turn_index(frame: pd.DataFrame, cfg: TurnsSection) -> np.ndarray:
    """The turn of each row of frame, in time order: one speaker's rows separated by less than max_gap_s.

    The gap also has to be uninterrupted: once somebody else speaks the floor has changed, so
    the turn ends there even if the same person picks up again straight away. Without that
    condition a turn is only a speaking spurt, and a floor transfer stops being a well defined
    event for turns.transitions to measure."""
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
    return assigned


def utterances_to_turns(transcript: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Merge consecutive same-speaker utterances separated by less than max_gap_s, nobody else speaking in between."""
    if transcript.empty:
        return empty_frame(TURN_SCHEMA).loc[:, _COLUMNS]

    frame = transcript.sort_values("t_start", kind="stable").reset_index(drop=True)
    frame = frame.assign(turn_index=_turn_index(frame, cfg))
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
    return mark_within(turns.loc[:, _COLUMNS[:-1]])


def talk_to_turns(talk: pd.DataFrame, said: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Turns timed by the stretches of talk, merged as utterances are, each holding what its speaker said in it: a word
    or utterance goes to its speaker's turn holding its middle, else to their nearest. A turn in which nothing was
    recognised, such as a laugh or a murmur of assent, has no words."""
    if talk.empty:
        return empty_frame(TURN_SCHEMA).loc[:, _COLUMNS]

    frame = talk.sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    turns = (
        frame.assign(turn_index=_turn_index(frame, cfg))
        .groupby("turn_index", sort=True)
        .agg(subject_id=("subject_id", "first"), t_start=("t_start", "min"), t_end=("t_end", "max"))
        .reset_index()
    )
    owner = np.full(len(said), -1, dtype=np.int64)
    middle = (said["t_start"].to_numpy(dtype=float) + said["t_end"].to_numpy(dtype=float)) / 2.0
    speakers = said["subject_id"].astype(str).to_numpy()
    for speaker, group in turns.groupby(turns["subject_id"].astype(str)):
        mine = speakers == speaker
        low, high = group["t_start"].to_numpy(dtype=float), group["t_end"].to_numpy(dtype=float)
        distance = np.maximum(low[:, None] - middle[None, mine], middle[None, mine] - high[:, None]).clip(min=0.0)
        owner[mine] = group["turn_index"].to_numpy()[np.argmin(distance, axis=0)]
    placed = said.assign(turn_index=owner)
    placed = placed[placed["turn_index"] >= 0]
    held = placed.groupby("turn_index", sort=True).agg(
        n_utterances=("utterance_index", "nunique"), n_words=("n_words", "sum"), text=("text", _join_text)
    )
    turns = turns.join(held, on="turn_index")
    turns = turns.fillna({"n_utterances": 0, "n_words": 0, "text": ""}).astype({"n_utterances": "int64", "n_words": "int64"})
    return mark_within(turns.loc[:, _COLUMNS[:-1]])


def mark_within(turns: pd.DataFrame) -> pd.DataFrame:
    """Adds within: the turn starts inside a turn of somebody else's who is still talking when it ends, so the floor
    stayed where it was."""
    starts = turns["t_start"].to_numpy(dtype=float)
    ends = turns["t_end"].to_numpy(dtype=float)
    speakers = turns["subject_id"].astype(str).to_numpy()
    holds = (speakers[None, :] != speakers[:, None]) & (starts[None, :] <= starts[:, None]) & (ends[:, None] < ends[None, :])
    return turns.assign(within=holds.any(axis=1))


def floor_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """The turns that held the floor, without talk inside somebody else's turn."""
    if "within" not in turns.columns or turns.empty:
        return turns
    return turns[~turns["within"].astype(bool).to_numpy()]


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


def load_turns(config, recording_id: str, *, others: bool = False) -> pd.DataFrame:
    """Where every downstream turn extractor starts, the way audio ones start at load_prep.
    OTHER's turns only with others, for turn-taking."""
    root = layout_for(config, create=False).artifacts_dir(
        recording_id, TurnStructureExtractor.name
    )
    turns = read_turns(root / TURNS_NAME)
    return turns if others else turns[turns["subject_id"] != OTHER].reset_index(drop=True)


def turns_saved(config, recording_id: str) -> bool:
    return (layout_for(config, create=False).artifacts_dir(recording_id, TurnStructureExtractor.name) / TURNS_NAME).is_file()


def build_turns(config, recording_id: str, transcript: pd.DataFrame, words: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(talk, turns, voice stretches left out as noise) of a transcript and its words, OTHER included."""
    talk, left_out = talk_spans(config, recording_id, transcript, words)
    if voices_time_talk(config, recording_id):
        turns = talk_to_turns(talk, said_units(transcript, words), config.turns)
    else:
        turns = utterances_to_turns(transcript, config.turns)
    turns["recording_id"] = recording_id
    return talk, turns, left_out


def talk_between(config, recording_id: str, start_s: float, end_s: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The transcript cut at both edges, OTHER included, and the talk and turns as from a recording cut there."""
    transcript = transcript_between(config, recording_id, start_s, end_s, others=True)
    if not voices_time_talk(config, recording_id):
        turns = utterances_to_turns(transcript, config.turns)
        turns["recording_id"] = recording_id
        return transcript, talk_spans(config, recording_id, transcript, None)[0], turns
    words = load_words(config, recording_id)
    talk, _ = talk_spans(config, recording_id, load_transcript(config, recording_id, others=True), words)
    talk = clip_talk(talk, start_s, end_s)
    middle = (words["t_start"].to_numpy(dtype=float) + words["t_end"].to_numpy(dtype=float)) / 2.0
    words = words[(middle >= start_s) & (middle < end_s)]
    turns = talk_to_turns(talk, said_units(transcript, words), config.turns)
    turns["recording_id"] = recording_id
    return transcript, talk, turns


def turns_between(config, recording_id: str, start_s: float, end_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The transcript cut at both edges, OTHER included, and the turns made from it as from a recording cut there."""
    transcript, _, turns = talk_between(config, recording_id, start_s, end_s)
    return transcript, turns


_TURN_SPECS = {
    "turn_duration_s": (
        "Length of one turn, from when its speaker started to when they stopped: by the words, or on a shared microphone by "
        "the voice, which also hears what was not recognised as words.", "s",
    ),
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
    "n_turns": (
        "Turns this speaker took. Talk inside somebody else's turn leaves the floor where it was and is not a turn here; "
        "turns.transitions counts it as a backchannel or an attempt to take the floor.", "count",
    ),
    "talk_time_s": ("Seconds this speaker held the floor, pauses inside turns included.", "s"),
    "talk_ratio": ("Share of all floor time held by this speaker; 0.5 is an even split.", None),
    "mean_turn_s": ("Mean turn length.", "s"),
    "median_turn_s": ("Median turn length, which a single long turn cannot move.", "s"),
    "turns_per_minute": ("Turns per minute of the analysed part, or the segment.", "1/min"),
}

_RECORDING_SPECS = {
    "n_turns_total": ("Turns taken by everyone.", "count"),
    "n_joint_silences": ("Stretches of at least silence_min_s where nobody talked.", "count"),
    "joint_silence_total_s": (
        "Total length of those stretches; the time before the first word and after the last is not counted.", "s",
    ),
    "joint_silence_fraction": ("joint_silence_total_s as a share of the analysed part, or of the segment.", None),
}

_NO_WORDS = "0 means the transcript gave the turn no words, as for a laugh or a murmur heard on a shared microphone."
_NO_TURN = "Empty for a person who took no turn in the span: they get no row, not 0."
_EVERY_SPAN = "Empty only outside the rows of the recording as a whole: every span gets a value."
_NO_SILENCE = (
    "0 means no gap of at least turns.silence_min_s between stretches of talk, as in a span with fewer than two stretches of "
    "talk; time before the first word and after the last is not counted."
)
_MISSING = {
    "turn_duration_s": "Empty only in rows that are not this person's turns: every turn has a value.",
    "n_utterances_in_turn": f"Empty only in rows that are not this person's turns: every turn has a value. {_NO_WORDS}",
    "turn_n_words": f"Empty only in rows that are not this person's turns: every turn has a value. {_NO_WORDS}",
    "turn_speech_rate_wps": f"Empty for a turn that lasts no time, its first and last word at the same instant. {_NO_WORDS}",
    "joint_silence_s": (
        "Empty only in rows of other events: a row is written for each gap of at least turns.silence_min_s between stretches "
        "of talk; time before the first word and after the last is not counted."
    ),
    "talk_ratio": f"{_NO_TURN} Also empty when no turn in the span lasts any time.",
    "n_turns_total": f"{_EVERY_SPAN} 0 means nobody took a turn in the span.",
    "n_joint_silences": f"{_EVERY_SPAN} {_NO_SILENCE}",
    "joint_silence_total_s": f"{_EVERY_SPAN} {_NO_SILENCE}",
    "joint_silence_fraction": f"{_EVERY_SPAN} {_NO_SILENCE}",
}


@register_extractor
class TurnStructureExtractor(Extractor):
    name = "turns.structure"
    version = "0.2.0"
    scale = Scale.EVENT
    group = "turns"
    config_sections = ("turns",)
    requires = (ASRExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        rid = recording.recording_id
        root = layout_for(self.config, create=False).artifacts_dir(rid, ASRExtractor.name)
        return [root / TRANSCRIPT_NAME, *([root / WORDS_NAME] if voices_time_talk(self.config, rid) else []), *talk_inputs(self.config, rid)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.turns
        rid = recording.recording_id
        transcript = load_transcript(self.config, rid, others=True)
        prep = load_prep(self.config, rid)

        talk, turns, left_out = build_turns(self.config, rid, transcript, load_words(self.config, rid))
        quiet = silences(talk, min_s=cfg.silence_min_s)

        artifacts = {
            TURNS_NAME: _io.write_file(turns, workspace / TURNS_NAME, TURN_SCHEMA, "turns")
        }
        return ExtractorResult(
            rows=self._rows(recording, turns, quiet, prep),
            specs=self._specs(recording),
            artifacts=artifacts,
            warnings=self._warnings(recording, turns, prep) + self._noise(recording, left_out, prep),
        )

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        if not turns_saved(self.config, recording.recording_id):
            return None
        _, talk, turns = talk_between(self.config, recording.recording_id, start_s, end_s)
        return self._summary(recording, turns, silences(talk, min_s=self.config.turns.silence_min_s), (start_s, end_s))

    def segment_warnings(self, recording: Recording, segment: Segment) -> list[QCWarning]:
        if not turns_saved(self.config, recording.recording_id):
            return []
        _, turns = turns_between(self.config, recording.recording_id, segment.start_s, segment.end_s)
        return self._warnings_over(recording, turns, (segment.start_s, segment.end_s), f"segment {segment.name}")

    def _rows(self, recording: Recording, turns, quiet, prep) -> pd.DataFrame:
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        return concat_long_tables([self._events(recording, turns, quiet), self._summary(recording, turns, quiet, window)])

    def _events(self, recording: Recording, turns, quiet) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        frames = []
        for subject_id, group in floor_turns(turns).groupby("subject_id", dropna=False, observed=True):
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
        return concat_long_tables(frames)

    def _summary(self, recording: Recording, turns, quiet, window: tuple[float, float]) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        length = window[1] - window[0]
        minutes = length / 60.0
        turns = floor_turns(turns)
        shares = dominance(turns)
        frames = []
        for subject_id, group in turns.groupby("subject_id", dropna=False, observed=True):
            duration = (group["t_end"] - group["t_start"]).to_numpy(dtype=float)
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
                        "joint_silence_fraction": [silent / length if length > 0 else np.nan],
                    }
                ),
                scale=Scale.RECORDING,
                **common,
            )
        )
        return concat_long_tables(frames)

    def _warnings(self, recording: Recording, turns, prep) -> list[QCWarning]:
        return self._warnings_over(recording, turns, (prep.t_origin_s, prep.t_origin_s + prep.duration_s), "the task window")

    def _noise(self, recording: Recording, left_out: int, prep) -> list[QCWarning]:
        if not left_out:
            return []
        return [
            QCWarning(
                check="turns_voice_noise", recording_id=recording.recording_id, subject_id=None, severity="info",
                message=(
                    f"{left_out} stretch(es) of voice shorter than turns.min_voice_s ({self.config.turns.min_voice_s:g} s) with no "
                    "word recognised in them were taken for noise and left out of the turns; the diarization keeps them."
                ),
                t_start=prep.t_origin_s, t_end=prep.t_origin_s + prep.duration_s,
            )
        ]

    def _warnings_over(self, recording: Recording, turns, window: tuple[float, float], scope: str) -> list[QCWarning]:
        warnings = []
        turns = floor_turns(turns)
        if len(turns) < _MIN_TURNS:
            warnings.append(
                QCWarning(
                    check="turns_too_few",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"Only {len(turns)} turns in {scope}. Turn statistics from so few are not "
                        "stable enough to interpret."
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
                        message=f"This speaker never took a turn in {scope}.",
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
                            f"This speaker holds {share:.0%} of the floor in {scope}. Check that the other "
                            "tracks were transcribed and attributed to the right people."
                        ),
                        t_start=window[0],
                        t_end=window[1],
                    )
                )
        return warnings

    def _specs(self, recording: Recording) -> list[FeatureSpec]:
        cfg = self.config.turns
        params = {"max_gap_s": cfg.max_gap_s, "silence_min_s": cfg.silence_min_s, **talk_params(self.config, recording.recording_id)}
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
                missing=_MISSING.get(feature, _NO_TURN),
                validation="none",
            )
            for specs, scale in groups
            for feature, (description, unit) in specs.items()
        ]
