"""The tables steps keep beside their measures, gathered across recordings for export, each with a data dictionary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from macaw.project.dictionary import FIXED, PACKAGE, fixed_entry, write_dictionary
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

TABLES = ("transcript", "turns", "voices", "judge", "warnings")
DICTIONARY_COLUMNS = ["column", "meaning", "unit", "missing", "step"]
_ENTITIES = ("group_id", "session_id", "task", "run")
_NOT_SHARED = "Empty unless the speech is on a shared microphone."
_OWN_MIC = "Empty except for an utterance of a person's own microphone checked against the other microphones"

_TRANSCRIPT = {
    "subject_id": ("The person who said it.", None, "Never empty."),
    "utterance_index": ("Number of the utterance among the person's, from 0; the words of an utterance carry the same number.", None, "Never empty."),
    "text": ("What was said, as recognised, or as a person corrected it.", None, "Empty for an utterance without text."),
    "n_words": ("Words in the utterance, counted on whitespace.", "count", "Never empty."),
    "confidence": ("The recogniser's confidence in the utterance, 0 to 1, defined differently by each recogniser.", None,
                   "Empty for a transcript a person corrected and where the recogniser gave none."),
    "avg_logprob": ("faster-whisper's mean log probability of the utterance's tokens.", None, "Empty with any other recogniser."),
    "no_speech_prob": ("faster-whisper's probability that the stretch holds no speech.", None, "Empty with any other recogniser."),
    "language": ("The language the utterance was recognised in.", None, "Empty where the recogniser gave none."),
    "louder_on": ("The other person's own microphone that was loudest over the utterance.", None, f"{_OWN_MIC}, and where there is one microphone."),
    "louder_by_db": (
        "How much louder the utterance was on that microphone than on the speaker's own, each measured against its own loud level; "
        "above text.crosstalk_margin_db it was dropped as bleed, so negative values are the speaker's own speech.", "dB",
        f"{_OWN_MIC}.",
    ),
    "gain_db": (
        "Loud level of the speaker's own microphone over the whole recording (99th percentile of its frame power), which louder_by_db "
        "measures each microphone against.", "dB", f"{_OWN_MIC}.",
    ),
    "gain_raised": (
        "True where that level lay more than 30 dB below the loudest microphone, so it was taken as a microphone nobody spoke into "
        "and raised to 30 dB below the loudest; a person whose speech was dropped as bleed throughout often has this.", None,
        f"{_OWN_MIC}.",
    ),
    "room_voice": ("The person heard only on the shared microphone whose voice covers most of the utterance.", None,
                   "Empty where no such person's voice overlaps it, and unless people with and without a microphone of their own share the room."),
    "room_voice_share": ("Share of the utterance in the voices of people heard only on the shared microphone; at 0.5 or more, with the "
                         "speaker's own voice under 0.2, it was dropped as their bleed.", None,
                         "Empty unless people with and without a microphone of their own share the room."),
    "own_voice_share": ("Share of the utterance in the speaker's own voice on the shared microphone.", None,
                        "Empty unless people with and without a microphone of their own share the room."),
}
_WORDS = {
    "subject_id": ("The person who said it.", None, "Never empty."),
    "utterance_index": ("The utterance the word belongs to, as numbered in utterances.csv for the same person.", None, "Never empty."),
    "word": ("The word as recognised.", None, "Never empty."),
    "score": ("The recogniser's score for the word, 0 to 1.", None, "Empty where the recogniser gave none."),
}
_DROPPED = {
    "subject_id": ("The person the utterance was first given to.", None, "Empty for words that fell in no named voice."),
    "text": ("What was said.", None, "Empty for an utterance without text."),
    "reason": (
        "Why it was left out: crosstalk (louder on another person's own microphone), room bleed (in the voice of someone heard only "
        "on the shared microphone), no voice (words in none of the named voices on a shared microphone), outside track (times "
        "beyond the prepared track).", None, "Never empty.",
    ),
    "detail": ("The reason in words, with its numbers.", None, "Empty where there is nothing to add."),
    **{name: _TRANSCRIPT[name] for name in ("louder_on", "louder_by_db", "gain_db", "gain_raised", "room_voice", "room_voice_share", "own_voice_share")},
}
_TURNS = {
    "subject_id": ("The person whose turn it is.", None, "Never empty."),
    "turn_index": ("Number of the turn among the person's, from 0.", None, "Never empty."),
    "n_utterances": ("Utterances merged into the turn.", "count", "Never empty."),
    "n_words": ("Words in the turn.", "count", "Never empty. 0 for a turn of voice without recognised words."),
    "text": ("What was said in the turn.", None, "Empty for a turn without recognised words."),
    "within": ("True for talk inside someone else's turn that does not take the floor: a backchannel or an attempt to cut in.", None, "Never empty."),
}
_TRANSITIONS = {
    "transition_index": ("Number of the transition in time order, from 0.", None, "Never empty."),
    "from_subject": ("The person whose turn ends, or whose turn the talk lies in.", None, "Never empty."),
    "to_subject": ("The person whose turn starts.", None, "Never empty."),
    "from_turn_index": ("The turn_index in turns.csv of the turn that ends.", None, "Never empty."),
    "to_turn_index": ("The turn_index in turns.csv of the turn that starts.", None, "Never empty."),
    "is_switch": ("True where the floor passes to another person.", None, "Never empty."),
    "prev_end": ("End of the turn before.", "s", "Never empty."),
    "next_start": ("Start of the turn after.", "s", "Never empty."),
    "fto_s": ("Floor transfer offset: the gap between the two turns, negative where they overlap.", "s", "Never empty."),
    "transition_type": (
        "What kind of transition: gap, overlap, interruption, hold (the same person again), backchannel, attempt (a longer cut-in "
        "that does not take the floor) or within.", None, "Never empty.",
    ),
    "within": ("True for talk inside someone else's turn, which leaves the floor where it was.", None, "Never empty."),
}
_TALK = {
    "subject_id": ("The person talking.", None, "Never empty."),
    "source": (
        "Where the stretch comes from: an utterance of the transcript, a voice the diarizer heard on a shared microphone, or a "
        "word heard outside every stretch of its speaker's voice.", None, "Never empty.",
    ),
    "kept": ("True where the talk and the turns rest on the stretch.", None, "Never empty."),
    "reason": ("Why a stretch was left out.", None, "Empty for a stretch that was kept."),
}
_VOICES = {
    "track": ("The shared microphone's track the voice was heard on.", None, "Never empty."),
    "label": ("The diarizer's name for the voice, the same across the recording.", None, "Never empty."),
    "person": ("The person the voice was given to.", None, "Empty for a voice given to no one, and before anyone names the voices."),
    "exclusive": (
        "True for the diarizer's version in which no two voices overlap, which the words are given out by; false for the one in "
        "which they may, which times the talk.", None, "Never empty.",
    ),
}
_SAMPLES = {
    "judge": ("The version of the language model judge that answered: judge, or a name under coding.judges.", None, "Never empty."),
    "unit_id": ("The unit coded, the same in every sample.", None, "Never empty."),
    "subject_id": ("The person whose unit it is.", None, "Never empty."),
    "sample": ("Which of the repeated answers, from 0.", None, "Never empty."),
    "label": ("The code the answer gave, ABSTAIN where the model said it cannot tell.", None, "Empty where the answer gave no usable code."),
    "reason": ("The model's reason for the code.", None, "Empty where it gave none."),
    "problem": ("Why the answer gave no usable code: missing, unreadable, not in the codebook, or cut off.", None, "Empty for a usable answer."),
    "given": ("The label as the model wrote it, kept where it was not in the codebook.", None, "Empty where it gave none."),
}
_UNITS = {
    "judge": _SAMPLES["judge"],
    "subject_id": _SAMPLES["subject_id"],
    "label": ("The code the samples settled on.", None, "Empty where the model abstained or no answer was usable."),
    "confidence": ("Share of the samples that gave the settled code.", None, "Empty where no answer was usable."),
    "abstained": ("True where the model said it cannot tell.", None, "Never empty."),
    "needs_review": ("True where the unit is in the review queue.", None, "Never empty."),
    "votes": ("Each code the samples gave and how often, most first.", None, "Empty where no answer was usable."),
}
_WARNINGS = {
    "check": ("The step or check that gave the warning.", None, "Never empty."),
    "subject_id": ("The person it is about.", None, "Empty for a warning about the recording as a whole."),
    "severity": ("info, warn or fail.", None, "Never empty."),
    "message": ("The warning as the report gives it.", None, "Never empty."),
}
_CANDIDATES = {
    "subject_id": ("The person the moment is about.", None, "Empty for a moment that involves no one person."),
    "check": ("The check that pointed to the moment.", None, "Never empty."),
    "reason": ("What the check saw.", None, "Never empty."),
    "t": ("The single instant to look at, the worst in the span.", "s", "Never empty."),
    "score": ("How suspicious the moment is; comparable within one reason only.", None, "Never empty."),
}


def table_dictionary(table: pd.DataFrame, said: dict, step: str) -> list[dict]:
    entries = []
    for column in table.columns:
        if column in said:
            meaning, unit, missing = said[column]
            entries.append({"column": column, "meaning": meaning, "unit": unit, "missing": missing, "step": step})
        elif column in FIXED:
            entries.append({**fixed_entry(column), "step": step if column in ("t_start", "t_end") else PACKAGE})
        else:
            entries.append({"column": column, "step": step})
    return entries


def write_table(folder: Path, name: str, table: pd.DataFrame, said: dict, step: str) -> list[Path]:
    path = folder / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8")
    return [path, *write_dictionary(table_dictionary(table, said, step), path, DICTIONARY_COLUMNS)]


def _with_entities(config, recording_id: str, table: pd.DataFrame) -> pd.DataFrame:
    """recording_id and the recording's entities first, as in the measures' table."""
    entities = build_recording(recording_id, config).entities()
    table = table.drop(columns=[column for column in ("recording_id", *_ENTITIES) if column in table.columns])
    front = pd.DataFrame({"recording_id": recording_id, **{name: entities.get(name) for name in _ENTITIES}}, index=table.index)
    return pd.concat([front, table], axis=1)


def _gathered(config, recording_ids: list[str], read) -> pd.DataFrame:
    pieces = []
    for recording_id in recording_ids:
        table = read(recording_id)
        if table is not None and len(table):
            pieces.append(_with_entities(config, recording_id, table))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def transcript_tables(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    from macaw.extractors.audio.content.asr import (
        BLEED_NAME, BLEED_SCHEMA, DROPPED_NAME, DROPPED_SCHEMA, TRANSCRIPT_NAME, WORD_SCHEMA, WORDS_NAME, ASRExtractor, read_transcript,
    )
    from macaw.store import _io

    def folder(recording_id):
        return layout_for(config, create=False).artifacts_dir(recording_id, ASRExtractor.name)

    def utterances(recording_id):
        if not (folder(recording_id) / TRANSCRIPT_NAME).is_file():
            return None
        said = read_transcript(folder(recording_id) / TRANSCRIPT_NAME)
        bleed = folder(recording_id) / BLEED_NAME
        if bleed.is_file():
            checked = _io.read_file(bleed, BLEED_SCHEMA, "bleed checks")
            checked = checked[checked["dropped_as"].isna()].drop(columns=["recording_id", "t_start", "t_end", "dropped_as"])
            said = said.merge(checked, on=["subject_id", "utterance_index"], how="left")
        return said.sort_values(["t_start", "subject_id"], kind="stable")

    def read(name, schema, label):
        return lambda recording_id: (
            _io.read_file(folder(recording_id) / name, schema, label).sort_values("t_start", kind="stable")
            if (folder(recording_id) / name).is_file() else None
        )

    step = ASRExtractor.name
    return {
        "utterances": (_gathered(config, recording_ids, utterances), _TRANSCRIPT, step),
        "words": (_gathered(config, recording_ids, read(WORDS_NAME, WORD_SCHEMA, "words")), _WORDS, step),
        "dropped": (_gathered(config, recording_ids, read(DROPPED_NAME, DROPPED_SCHEMA, "dropped utterances")), _DROPPED, step),
        **embedding_table(config, recording_ids),
    }


def embedding_table(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    """Each utterance's sentence embedding, one column per dimension, where text.features made them."""
    from macaw.extractors.audio.content.features import EMBEDDINGS_NAME, TextFeatureExtractor

    def one(recording_id):
        path = layout_for(config, create=False).artifacts_dir(recording_id, TextFeatureExtractor.name) / EMBEDDINGS_NAME
        return pd.read_parquet(path).sort_values(["t_start", "subject_id"], kind="stable") if path.is_file() else None

    found = _gathered(config, recording_ids, one)
    model = config.text.embedding_model or "text.embedding_model"
    said = {
        "subject_id": _TRANSCRIPT["subject_id"],
        "utterance_index": ("The utterance, as numbered in utterances.csv for the same person.", None, "Never empty."),
        **{
            column: (f"Dimension {column[1:]} of the utterance's sentence embedding from {model}; only distances between "
                     "vectors of the same model mean anything.", None, "Never empty.")
            for column in found.columns if column.startswith("e") and column[1:].isdigit()
        },
    }
    return {"embeddings": (found, said, TextFeatureExtractor.name)}


def turn_tables(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    from macaw.store import _io
    from macaw.turns.segmentation import TALK_NAME, TURN_SCHEMA, TURNS_NAME, TurnStructureExtractor
    from macaw.turns.talk import TALK_RECORD_SCHEMA
    from macaw.turns.transitions import TRANSITION_SCHEMA, TRANSITIONS_NAME, TransitionExtractor

    def read(step, name, schema, order):
        def one(recording_id):
            path = layout_for(config, create=False).artifacts_dir(recording_id, step) / name
            return _io.read_file(path, schema, name).sort_values(order, kind="stable") if path.is_file() else None
        return one

    structure, transitions = TurnStructureExtractor.name, TransitionExtractor.name
    return {
        "turns": (_gathered(config, recording_ids, read(structure, TURNS_NAME, TURN_SCHEMA, ["t_start", "subject_id"])), _TURNS, structure),
        "transitions": (
            _gathered(config, recording_ids, read(transitions, TRANSITIONS_NAME, TRANSITION_SCHEMA, ["transition_index"])), _TRANSITIONS, transitions,
        ),
        "talk": (_gathered(config, recording_ids, read(structure, TALK_NAME, TALK_RECORD_SCHEMA, ["t_start", "t_end"])), _TALK, structure),
    }


def voice_tables(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    from macaw.extractors.audio.content.diarization import DiarizationExtractor, diarization_path, speaker_names

    def one(recording_id):
        path = diarization_path(config, recording_id)
        if not path.is_file():
            return None
        heard = pd.read_parquet(path)
        names = speaker_names(config, recording_id)
        heard.insert(2, "person", heard["label"].astype(str).map(names))
        return heard.sort_values(["exclusive", "t_start"], ascending=[False, True], kind="stable")

    return {"voices": (_gathered(config, recording_ids, one), _VOICES, DiarizationExtractor.name)}


def judge_tables(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    from macaw.coding.llm.judge import ABSTAIN, SAMPLES_NAME, JudgementExtractor, version_steps
    from macaw.coding.review_queue import unit_answers
    from macaw.extractors import get_extractor
    from macaw.store import _io
    from macaw.store.schema import ANNOTATION_SCHEMA

    layout = layout_for(config, create=False)
    steps = [JudgementExtractor.name, *version_steps(config)]

    def judge_name(step):
        return get_extractor(step).judge or "judge"

    def samples(recording_id):
        pieces = []
        for step in steps:
            path = layout.artifacts_dir(recording_id, step) / SAMPLES_NAME
            if path.is_file():
                pieces.append(pd.read_parquet(path).assign(judge=judge_name(step)))
        if not pieces:
            return None
        found = pd.concat(pieces, ignore_index=True)
        first = ["judge", "unit_id", "subject_id", "t_start", "t_end", "sample", "label", "reason"]
        return found.reindex(columns=[*first, *[column for column in found.columns if column not in first]]).sort_values(
            ["judge", "t_start", "unit_id", "sample"], kind="stable"
        )

    def units(recording_id):
        pieces = []
        for step in steps:
            coded, path = layout.annotations_path(recording_id) / f"{get_extractor(step).layer()}.parquet", layout.artifacts_dir(recording_id, step) / SAMPLES_NAME
            if not coded.is_file() or not path.is_file():
                continue
            mine = _io.read_file(coded, ANNOTATION_SCHEMA, "annotations")
            keys = ["subject_id", "t_start", "t_end"]
            votes = unit_answers(pd.read_parquet(path))[[*keys, "votes"]]
            mine["t_start"], mine["t_end"] = mine["t_start"].astype(float).round(6), mine["t_end"].astype(float).round(6)
            mine = mine.assign(subject_id=mine["subject_id"].astype(str)).merge(votes.assign(subject_id=votes["subject_id"].astype(str)), on=keys, how="left")
            pieces.append(mine.assign(judge=judge_name(step), label=mine["label"].where(mine["label"] != ABSTAIN)))
        if not pieces:
            return None
        found = pd.concat(pieces, ignore_index=True)
        return found.loc[:, ["judge", "subject_id", "t_start", "t_end", "label", "confidence", "abstained", "needs_review", "votes"]].sort_values(
            ["judge", "t_start"], kind="stable"
        )

    return {
        "samples": (_gathered(config, recording_ids, samples), _SAMPLES, JudgementExtractor.name),
        "units": (_gathered(config, recording_ids, units), _UNITS, JudgementExtractor.name),
    }


def warning_tables(config, recording_ids: list[str]) -> dict[str, tuple[pd.DataFrame, dict, str]]:
    """The warnings of the last quality check, or the steps' own before one ran; each detail a column of its own."""
    from macaw.qc.checks import RecordingResults, read_candidates

    layout = layout_for(config, create=False)
    keys: dict[str, set[str]] = {}

    def warnings(recording_id):
        stored = layout.qc_dir(recording_id) / "warnings.json"
        if stored.is_file():
            found = json.loads(stored.read_text(encoding="utf-8"))
        else:
            from dataclasses import asdict

            found = [asdict(warning) for warning in RecordingResults(layout, recording_id).extractor_warnings()]
        rows = []
        for warning in found:
            details = warning.get("details") or {}
            row = {name: warning.get(name) for name in ("check", "subject_id", "severity", "message", "t_start", "t_end")}
            for key, value in details.items():
                keys.setdefault(f"detail_{key}", set()).add(str(warning.get("check")))
                row[f"detail_{key}"] = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value
            rows.append(row)
        return pd.DataFrame(rows) if rows else None

    def candidates(recording_id):
        path = layout.qc_dir(recording_id) / "candidates.parquet"
        return read_candidates(path).sort_values("t_start", kind="stable") if path.is_file() else None

    found = _gathered(config, recording_ids, warnings)
    said = dict(_WARNINGS)
    for key, checks in keys.items():
        said[key] = (f"A value the warning gave with its message, from: {', '.join(sorted(checks))}.", None, "Empty for a warning that gives no such value.")
    return {
        "warnings": (found, said, "macaw qc"),
        "candidates": (_gathered(config, recording_ids, candidates), _CANDIDATES, "macaw qc"),
    }


BUILDERS = {
    "transcript": transcript_tables,
    "turns": turn_tables,
    "voices": voice_tables,
    "judge": judge_tables,
    "warnings": warning_tables,
}
