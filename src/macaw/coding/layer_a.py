"""Layer A: physical facts defined by rules. Near-perfect, fully automatic."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.config.schema import CodingSection, TurnsSection
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import ASRExtractor, load_transcript
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import ANNOTATION_SCHEMA, Scale, coerce_frame, empty_frame
from macaw.turns.segmentation import TurnStructureExtractor, load_turns
from macaw.turns.transitions import TransitionExtractor, detect_transitions, load_transitions

LAYER = "A"
BACKCHANNEL_TIER = "backchannel"
PAUSE_TIER = "pause"
INTERRUPTION_TIER = "interruption"

_INTERVAL_COLUMNS = ["subject_id", "t_start", "t_end", "label"]
_TOKEN = re.compile(r"[\w'-]+", re.UNICODE)


def _empty_intervals() -> pd.DataFrame:
    return pd.DataFrame({column: [] for column in _INTERVAL_COLUMNS})


def _tokens(text) -> set[str]:
    return set(_TOKEN.findall(str(text).lower())) if text is not None else set()


def code_backchannels(turns: pd.DataFrame, cfg: CodingSection) -> pd.DataFrame:
    """Lexicon match, short duration, does not take the floor. Three rules is enough."""
    lexicon = {token.strip().lower() for token in cfg.backchannel_lexicon if str(token).strip()}
    if turns.empty or not lexicon:
        return _empty_intervals()

    frame = turns.sort_values("t_start", kind="stable").reset_index(drop=True)
    speakers = frame["subject_id"].astype(str).to_numpy()
    duration = (frame["t_end"] - frame["t_start"]).to_numpy(dtype=float)

    # the floor never moved when the same other speaker holds it on both sides. The first and
    # last turn therefore cannot qualify: nothing outside them can show the floor came back
    surrounded = np.zeros(len(frame), dtype=bool)
    if len(frame) > 2:
        surrounded[1:-1] = (speakers[:-2] == speakers[2:]) & (speakers[1:-1] != speakers[:-2])

    # every token has to be in the lexicon, so a sentence that merely contains "yeah" is out
    spoken = [_tokens(text) for text in frame["text"]]
    matched = np.array([bool(tokens) and tokens <= lexicon for tokens in spoken], dtype=bool)

    keep = surrounded & matched & (duration <= cfg.backchannel_max_s)
    out = frame.loc[keep, ["subject_id", "t_start", "t_end"]].copy()
    out["label"] = BACKCHANNEL_TIER
    return out.reset_index(drop=True).loc[:, _INTERVAL_COLUMNS]


def code_pause_types(
    turns: pd.DataFrame, cfg: TurnsSection, transcript: pd.DataFrame, *, min_internal_s: float = 0.0
) -> pd.DataFrame:
    """turn-internal vs switching pause; well established and highly reliable.

    Every positive gap between turns is emitted. How long a pause has to be before it means
    something is the analyst's assumption, not ours; the duration is on every row. Inside a turn,
    gaps under min_internal_s are left out: they are where the recogniser cut one stretch of
    speech into segments, not silence."""
    parts = [_between_turn_pauses(turns, cfg), _within_turn_pauses(turns, transcript, min_internal_s)]
    collected = [part for part in parts if not part.empty]
    if not collected:
        return _empty_intervals()
    return (
        pd.concat(collected, ignore_index=True)
        .sort_values("t_start", kind="stable")
        .reset_index(drop=True)
    )


def _between_turn_pauses(turns: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    transitions = detect_transitions(turns, cfg)
    if transitions.empty:
        return _empty_intervals()

    gaps = transitions[transitions["fto_s"].to_numpy(dtype=float) > 0.0]
    if gaps.empty:
        return _empty_intervals()

    # the pause belongs to whoever ends it, matching how turns.transitions attributes an offset
    return pd.DataFrame(
        {
            "subject_id": gaps["to_subject"].astype(str).to_numpy(),
            "t_start": gaps["prev_end"].to_numpy(dtype=float),
            "t_end": gaps["next_start"].to_numpy(dtype=float),
            "label": np.where(gaps["is_switch"].astype(bool).to_numpy(), "switching", "hold"),
        }
    )


def _within_turn_pauses(turns: pd.DataFrame, transcript: pd.DataFrame, min_s: float) -> pd.DataFrame:
    if turns.empty or transcript.empty:
        return _empty_intervals()

    rows = []
    for turn in turns.itertuples():
        inside = transcript[
            (transcript["subject_id"].astype(str) == str(turn.subject_id))
            & (transcript["t_start"].to_numpy(dtype=float) >= turn.t_start)
            & (transcript["t_start"].to_numpy(dtype=float) <= turn.t_end)
        ].sort_values("t_start", kind="stable")
        if len(inside) < 2:
            continue

        ends = inside["t_end"].to_numpy(dtype=float)[:-1]
        starts = inside["t_start"].to_numpy(dtype=float)[1:]
        # an utterance can overlap the previous one after crosstalk suppression trims things
        for low, high in zip(ends, starts):
            if high > low and high - low >= min_s:
                rows.append((str(turn.subject_id), low, high, "turn_internal"))

    if not rows:
        return _empty_intervals()
    return pd.DataFrame(rows, columns=_INTERVAL_COLUMNS)


def code_interruptions(transitions: pd.DataFrame, cfg: CodingSection) -> pd.DataFrame:
    if transitions.empty:
        return _empty_intervals()

    taken = transitions[transitions["transition_type"].astype(str).to_numpy() == "interruption"]
    if taken.empty:
        return _empty_intervals()

    # the annotated span is the overlap itself, which runs from the cut-in to the old turn ending
    return pd.DataFrame(
        {
            "subject_id": taken["to_subject"].astype(str).to_numpy(),
            "t_start": taken["next_start"].to_numpy(dtype=float),
            "t_end": taken["prev_end"].to_numpy(dtype=float),
            "label": INTERRUPTION_TIER,
        }
    )


def code_gesture_speech_alignment(strokes: pd.DataFrame, prominence: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


_SUBJECT_SPECS = {
    "n_backchannels_coded": (
        "Short turns by subject_id whose every token is in the configured lexicon and after "
        "which the floor went straight back. Zero whenever no lexicon is configured.",
        "count",
        "sum",
    ),
    "n_turn_internal_pauses": (
        "Silences inside one of subject_id's own turns. Nothing else in the pipeline can see "
        "these; turn boundaries hide them by construction.",
        "count",
        "sum",
    ),
    "median_turn_internal_pause_s": (
        "Median length of those within-turn silences.",
        "s",
        "mean",
    ),
}


@register_extractor
class LayerACodingExtractor(Extractor):
    name = "coding.layer_a"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    # only the coding keys it reads, so a language model setting leaves it alone
    config_sections = ("coding.backchannel_lexicon", "coding.backchannel_max_s", "coding.min_turn_internal_pause_s", "turns")
    requires = (TransitionExtractor.name, TurnStructureExtractor.name, ASRExtractor.name)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        return TransitionExtractor(self.config).inputs_for(recording)

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        turns = load_turns(self.config, recording.recording_id, others=True)
        transitions = load_transitions(self.config, recording.recording_id)
        transcript = load_transcript(self.config, recording.recording_id, others=True)

        coded = {
            BACKCHANNEL_TIER: code_backchannels(turns, self.config.coding),
            PAUSE_TIER: code_pause_types(
                turns, self.config.turns, transcript, min_internal_s=self.config.coding.min_turn_internal_pause_s
            ),
            INTERRUPTION_TIER: code_interruptions(transitions, self.config.coding),
        }
        return ExtractorResult(
            rows=self._rows(recording, turns, coded),
            specs=self._specs(),
            annotations=self._annotations(recording, coded),
            warnings=self._warnings(recording, turns, coded),
        )

    def _annotations(self, recording: Recording, coded: dict[str, pd.DataFrame]) -> pd.DataFrame:
        entities = recording.entities()
        frames = []
        for tier, intervals in coded.items():
            if intervals.empty:
                continue
            for subject_id, group in intervals.groupby("subject_id", observed=True):
                frames.append(
                    build_annotations(
                        recording_id=recording.recording_id,
                        tier=tier,
                        layer=LAYER,
                        annotator=self.name,
                        intervals=group.loc[:, ["t_start", "t_end", "label"]].reset_index(
                            drop=True
                        ),
                        entities=entities,
                        subject_id=str(subject_id),
                    )
                )
        if not frames:
            return empty_frame(ANNOTATION_SCHEMA)
        return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")

    def _rows(
        self, recording: Recording, turns: pd.DataFrame, coded: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        # spanned by the conversation rather than the task window, which would drag prep.media
        # in as a dependency for nothing but two column values
        span = (
            (float(turns["t_start"].min()), float(turns["t_end"].max()))
            if not turns.empty
            else (np.nan, np.nan)
        )
        frames = []

        pauses = coded[PAUSE_TIER]
        internal = (
            pauses[pauses["label"].astype(str).to_numpy() == "turn_internal"]
            if not pauses.empty
            else pauses
        )
        backchannels = coded[BACKCHANNEL_TIER]

        for subject in recording.subjects:
            mine = (
                internal[internal["subject_id"].astype(str) == subject.subject_id]
                if not internal.empty
                else internal
            )
            lengths = (
                (mine["t_end"] - mine["t_start"]).to_numpy(dtype=float)
                if not mine.empty
                else np.empty(0, dtype=float)
            )
            counted = (
                int((backchannels["subject_id"].astype(str) == subject.subject_id).sum())
                if not backchannels.empty
                else 0
            )
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [span[0]],
                            "t_end": [span[1]],
                            "n_backchannels_coded": [float(counted)],
                            "n_turn_internal_pauses": [float(lengths.size)],
                            "median_turn_internal_pause_s": [
                                float(np.median(lengths)) if lengths.size else np.nan
                            ],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=subject.subject_id,
                    **common,
                )
            )
        return concat_long_tables(frames)

    def _warnings(
        self, recording: Recording, turns: pd.DataFrame, coded: dict[str, pd.DataFrame]
    ) -> list[QCWarning]:
        cfg = self.config.coding
        warnings = []

        if not [token for token in cfg.backchannel_lexicon if str(token).strip()]:
            warnings.append(
                QCWarning(
                    check="coding_no_backchannel_lexicon",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="info",
                    message=(
                        "coding.backchannel_lexicon is empty, so no backchannels were coded. "
                        "The list ships empty because the right tokens depend on the language "
                        "being spoken; fill it in to turn this code on."
                    ),
                )
            )
        elif coded[BACKCHANNEL_TIER].empty and _has_backchannel_candidates(turns, cfg):
            warnings.append(
                QCWarning(
                    check="coding_backchannel_lexicon_unmatched",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        "Short turns that never took the floor are present, but not one matched "
                        "coding.backchannel_lexicon. Check the list against the language actually "
                        "spoken in this recording."
                    ),
                )
            )
        return warnings

    def _specs(self) -> list[FeatureSpec]:
        cfg = self.config.coding
        params = {
            "backchannel_lexicon": sorted(cfg.backchannel_lexicon),
            "backchannel_max_s": cfg.backchannel_max_s,
            "min_turn_internal_pause_s": cfg.min_turn_internal_pause_s,
        }
        return [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation=aggregation,
                validation="none",
            )
            for feature, (description, unit, aggregation) in _SUBJECT_SPECS.items()
        ]


def _has_backchannel_candidates(turns: pd.DataFrame, cfg: CodingSection) -> bool:
    if len(turns) < 3:
        return False
    frame = turns.sort_values("t_start", kind="stable").reset_index(drop=True)
    speakers = frame["subject_id"].astype(str).to_numpy()
    duration = (frame["t_end"] - frame["t_start"]).to_numpy(dtype=float)
    surrounded = np.zeros(len(frame), dtype=bool)
    surrounded[1:-1] = (speakers[:-2] == speakers[2:]) & (speakers[1:-1] != speakers[:-2])
    return bool((surrounded & (duration <= cfg.backchannel_max_s)).any())
