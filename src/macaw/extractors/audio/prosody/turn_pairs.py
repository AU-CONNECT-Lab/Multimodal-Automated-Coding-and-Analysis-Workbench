"""At each change of speaker, the turn handed over and the turn that took the floor, measure by measure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors import register_extractor
from macaw.extractors.audio.prosody.voice_report import _SPECS as VOICE_SPECS
from macaw.extractors.audio.prosody.voice_report import VoiceReportExtractor, load_turns
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

TRANSITIONS_STEP = "turns.transitions"

# description, unit, comparable across subjects
PAIRED = {
    **VOICE_SPECS,
    "turn_speech_rate_wps": ("Words per second over the turn.", "word/s", True),
}


@register_extractor
class TurnPairsExtractor(Extractor):
    name = "prosody.turn_pairs"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "prosody"
    config_sections = ("turns",)
    requires = (VoiceReportExtractor.name, TRANSITIONS_STEP)

    def check_available(self) -> None:
        pass

    def inputs_for(self, recording: Recording) -> list[Path]:
        from macaw.turns.segmentation import TURNS_NAME
        from macaw.turns.transitions import TRANSITIONS_NAME

        layout, rid = layout_for(self.config, create=False), recording.recording_id
        return [
            _voice_rows_file(self.config, rid),
            layout.artifacts_dir(rid, "turns.structure") / TURNS_NAME,
            layout.artifacts_dir(rid, TRANSITIONS_STEP) / TRANSITIONS_NAME,
        ]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        from macaw.turns.transitions import load_transitions

        rid = recording.recording_id
        voice = read_long_table(_voice_rows_file(self.config, rid), recording_id=rid, features=list(VOICE_SPECS))
        measures = per_turn(load_turns(self.config, rid), voice)
        people = {subject.subject_id for subject in recording.subjects}
        rows = pair_rows(load_transitions(self.config, rid), measures, people)
        common = dict(recording_id=rid, source=self.name, entities=recording.entities())
        frames = [
            build_long_rows(values=values, scale=Scale.EVENT, subject_id=taker, subject_id_b=giver, **common)
            for (taker, giver), values in rows
        ]
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs())

    def _specs(self) -> list[FeatureSpec]:
        specs = []
        for name, (description, unit, comparable) in PAIRED.items():
            for prefix, whose in (
                ("prior", "Measured on subject_id_b's turn just before subject_id took the floor."),
                ("next", "Measured on subject_id's turn that took the floor from subject_id_b."),
            ):
                specs.append(
                    FeatureSpec(
                        name=f"{prefix}_{name}", description=f"{description} {whose}", unit=unit, scale=Scale.EVENT,
                        group=self.group, extractor=self.name, extractor_version=self.version, aggregation="mean",
                        comparable_across_subjects=comparable, comparable_across_backends=False, validation="none",
                    )
                )
        return specs


def per_turn(turns: pd.DataFrame, voice: pd.DataFrame) -> pd.DataFrame:
    """One row per turn_index with every paired measure, NaN where a turn was not measured."""
    keys = turns[["turn_index", "subject_id", "t_start", "t_end", "n_words"]].assign(subject_id=lambda f: f["subject_id"].astype(str))
    duration = (keys["t_end"] - keys["t_start"]).to_numpy(dtype=float)
    words = keys["n_words"].to_numpy(dtype=float)
    keys["turn_speech_rate_wps"] = np.divide(words, duration, out=np.full_like(words, np.nan), where=duration > 0)
    if not voice.empty:
        wide = voice.assign(subject_id=voice["subject_id"].astype(str)).pivot_table(
            index=["subject_id", "t_start"], columns="feature", values="value", aggfunc="first", dropna=False,
        )
        keys = keys.merge(wide.reset_index(), on=["subject_id", "t_start"], how="left")
    return keys.reindex(columns=["turn_index", "subject_id", "t_start", "t_end", *PAIRED]).set_index("turn_index")


def pair_rows(transitions: pd.DataFrame, measures: pd.DataFrame, people: set[str]) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    """((taker, giver), values) for each pair of speakers, a row per switch spanning the turn that took the floor."""
    if transitions.empty:
        return []
    switches = transitions[
        transitions["is_switch"].astype(bool)
        & transitions["from_subject"].astype(str).isin(people)
        & transitions["to_subject"].astype(str).isin(people)
    ]
    out = []
    for (giver, taker), group in switches.groupby(["from_subject", "to_subject"], observed=True, sort=True):
        before = measures.reindex(group["from_turn_index"].to_numpy())
        after = measures.reindex(group["to_turn_index"].to_numpy())
        values = pd.DataFrame(
            {
                "t_start": group["next_start"].to_numpy(dtype=float),
                "t_end": after["t_end"].to_numpy(dtype=float),
                **{f"prior_{name}": before[name].to_numpy(dtype=float) for name in PAIRED},
                **{f"next_{name}": after[name].to_numpy(dtype=float) for name in PAIRED},
            }
        )
        out.append(((str(taker), str(giver)), values))
    return out


def _voice_rows_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).long_table_path(recording_id) / f"{VoiceReportExtractor.name}.parquet"
