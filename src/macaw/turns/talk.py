"""When each person talks. The transcript's utterances say it, except on a shared microphone: one recogniser's words never
overlap there, so the voices the diarizer heard, overlaps included, give the times and the words only what was said."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors.audio.content.diarization import SHARED_MODE, diarization_path, read_diarization, speaker_names
from macaw.extractors.prep import load_prep
from macaw.project.layout import layout_for

TALK_COLUMNS = ["subject_id", "t_start", "t_end"]
# every stretch talk_spans looked at: where it came from, and whether the talk keeps it
TALK_RECORD_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "t_start": "float64",
    "t_end": "float64",
    "source": "string",
    "kept": "bool",
    "reason": "string",
}
SAID_COLUMNS = ["subject_id", "t_start", "t_end", "text", "n_words", "utterance_index"]


def voices_time_talk(config, recording_id: str) -> bool:
    """On a shared microphone, unless the transcript in use is one a person corrected, whose times are theirs."""
    return config.text.diarization == SHARED_MODE and not layout_for(config, create=False).corrected_transcript_path(recording_id).is_file()


def talk_inputs(config, recording_id: str, *, by_voices: bool | None = None) -> list[Path]:
    """The files talk_spans reads besides the transcript and its words, for a cache key."""
    if not (voices_time_talk(config, recording_id) if by_voices is None else by_voices):
        return []
    layout = layout_for(config, create=False)
    lines = layout.line_speakers_path(recording_id)
    return [diarization_path(config, recording_id), layout.voice_owners_path(recording_id), *([lines] if lines.is_file() else [])]


def talk_params(config, recording_id: str) -> dict:
    """How talk was timed, for the settings a measure states."""
    if not voices_time_talk(config, recording_id):
        return {"talk_timed_by": "utterances"}
    return {"talk_timed_by": "voices", "min_voice_s": config.turns.min_voice_s}


def said_units(transcript: pd.DataFrame, words: pd.DataFrame) -> pd.DataFrame:
    """What was said, word by word where the recogniser timed words, else utterance by utterance."""
    if words is not None and not words.empty:
        frame = words.assign(text=words["word"].astype(str), n_words=1)
    else:
        frame = transcript.assign(n_words=transcript["text"].fillna("").astype(str).str.split().str.len())
    frame = frame.assign(subject_id=frame["subject_id"].astype(str), text=frame["text"].fillna("").astype(str).str.strip())
    return frame.loc[:, SAID_COLUMNS].sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)


def talk_spans(
    config, recording_id: str, transcript: pd.DataFrame, words: pd.DataFrame, *, by_voices: bool | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every stretch of talk on the recording clock, and the record of every stretch looked at: its source (utterance,
    voice or word), whether it was kept and why not. by_voices defaults to voices_time_talk; the recogniser's transcript
    kept beside a correction is timed by voices all the same.

    By voices: each named voice's stretches in the diarization that lets voices overlap. One shorter than turns.min_voice_s
    stays only where a word of its person was recognised in it, and a word that lies in none of its person's stretches
    stands as one of its own. Whoever has a microphone of their own is timed by their utterances from it."""
    if not (voices_time_talk(config, recording_id) if by_voices is None else by_voices):
        talk = _talk(transcript)
        return talk, _record(talk, "utterance")
    own = set(load_prep(config, recording_id).tracks)
    names = speaker_names(config, recording_id)
    regular = read_diarization(config, recording_id, changed=True)
    regular = regular[~regular["exclusive"].astype(bool)]
    voices = pd.DataFrame(
        {
            "subject_id": regular["label"].astype(str).map(names),
            "t_start": regular["t_start"].to_numpy(dtype=float),
            "t_end": regular["t_end"].to_numpy(dtype=float),
        }
    )
    voices = voices[voices["subject_id"].notna() & ~voices["subject_id"].isin(own)].reset_index(drop=True)
    voices["subject_id"] = voices["subject_id"].astype(str)
    said = said_units(transcript, words)
    said = said[~said["subject_id"].isin(own)]
    long_enough = (voices["t_end"] - voices["t_start"]).to_numpy(dtype=float) >= config.turns.min_voice_s
    keep = long_enough | _holding(voices, said).any(axis=1)
    kept = voices[keep]
    orphans = said[~_holding(kept, said).any(axis=0)]
    spoken = transcript[transcript["subject_id"].astype(str).isin(own)]
    noise = f"shorter than turns.min_voice_s ({config.turns.min_voice_s:g} s) with no word recognised in it"
    record = pd.concat(
        [
            _record(_talk(spoken), "utterance"),
            _record(voices, "voice").assign(kept=keep, reason=np.where(keep, None, noise)),
            _record(_talk(orphans), "word"),
        ],
        ignore_index=True,
    )
    record = record.sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    return _talk(record[record["kept"].astype(bool)]), record


def _record(talk: pd.DataFrame, source: str) -> pd.DataFrame:
    return talk.loc[:, TALK_COLUMNS].assign(source=source, kept=True, reason=None)


def _talk(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, TALK_COLUMNS].astype({"t_start": float, "t_end": float})
    out["subject_id"] = out["subject_id"].astype(str)
    return out.sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)


def _holding(spans: pd.DataFrame, said: pd.DataFrame) -> np.ndarray:
    """spans x said: the span is its person's and holds the middle of what was said."""
    if spans.empty or said.empty:
        return np.zeros((len(spans), len(said)), dtype=bool)
    middle = (said["t_start"].to_numpy(dtype=float) + said["t_end"].to_numpy(dtype=float)) / 2.0
    same = spans["subject_id"].astype(str).to_numpy()[:, None] == said["subject_id"].astype(str).to_numpy()[None, :]
    return same & (spans["t_start"].to_numpy(dtype=float)[:, None] <= middle) & (middle <= spans["t_end"].to_numpy(dtype=float)[:, None])


def clip_talk(talk: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
    """The talk cut at both edges, as from a recording cut there."""
    out = talk[(talk["t_end"] > start_s) & (talk["t_start"] < end_s)].copy()
    out["t_start"] = out["t_start"].clip(lower=start_s)
    out["t_end"] = out["t_end"].clip(upper=end_s)
    return out.reset_index(drop=True)


def voice_gaps(lines: pd.DataFrame, talk: pd.DataFrame) -> np.ndarray:
    """The pause before each line, negative where it overlaps the talk before, with each line widened to the voice it is
    said in: back to where that voice set in and on to where it stopped, but not past its speaker's line before or after.
    lines carry subject_id, t_start and t_end in time order, on the clock of talk."""
    if lines.empty:
        return np.empty(0, dtype=float)
    starts = lines["t_start"].to_numpy(dtype=float).copy()
    ends = lines["t_end"].to_numpy(dtype=float).copy()
    speakers = lines["subject_id"].astype(str).to_numpy()
    for speaker in np.unique(speakers):
        mine = np.flatnonzero(speakers == speaker)
        spans = talk[talk["subject_id"].astype(str) == speaker]
        low, high = spans["t_start"].to_numpy(dtype=float), spans["t_end"].to_numpy(dtype=float)
        for position, index in enumerate(mine):
            floor = ends[mine[position - 1]] if position else -np.inf
            ceiling = lines["t_start"].to_numpy(dtype=float)[mine[position + 1]] if position + 1 < len(mine) else np.inf
            holding = (low <= starts[index]) & (high >= starts[index])
            if holding.any():
                starts[index] = max(float(low[holding].min()), floor)
            holding = (low <= ends[index]) & (high >= ends[index])
            if holding.any():
                ends[index] = min(float(high[holding].max()), ceiling)
    reached = np.maximum.accumulate(ends)
    gaps = np.full(len(lines), np.nan)
    gaps[1:] = starts[1:] - reached[:-1]
    return gaps
