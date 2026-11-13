"""How much each person moves while another person talks alone: listening behaviour from the transcript and a movement series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, load_transcript
from macaw.extractors.base import Extractor, ExtractorResult, Phase
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, load_prep
from macaw.extractors.video.motion.mea import MEAExtractor
from macaw.extractors.video.pose.kinematics import _SUBJECT_SPECS, KinematicsExtractor
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec, read_manifest
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

LISTENING_S = "listening_s"


def producers() -> dict[str, str]:
    """Each movement series a person can be measured on while listening, and the step that gives it."""
    return {**{name: KinematicsExtractor.name for name in _SUBJECT_SPECS}, "motion_energy": MEAExtractor.name}


@register_extractor
class ListeningExtractor(Extractor):
    name = "pose.listening"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.listening", "segment")
    requires = (ASRExtractor.name, MediaPrepExtractor.name)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.kinematics.listening)

    @classmethod
    def dependencies(cls, config) -> tuple[str, ...]:
        found = producers()
        return cls.requires + tuple(sorted({found[name] for name in config.kinematics.listening if name in found}))

    def check_available(self) -> None:
        unknown = sorted(set(self.config.kinematics.listening) - set(producers()))
        if unknown:
            raise ConfigError(f"kinematics.listening names {', '.join(unknown)}; it takes {', '.join(sorted(producers()))}")

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        rid = recording.recording_id
        steps = sorted({producers()[name] for name in self.config.kinematics.listening})
        return [
            layout.artifacts_dir(rid, ASRExtractor.name) / TRANSCRIPT_NAME,
            layout.artifacts_dir(rid, MediaPrepExtractor.name) / PREP_INFO_NAME,
            *(layout.long_table_path(rid) / f"{step}.parquet" for step in steps),
        ]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        prep = load_prep(self.config, recording.recording_id)
        rows = self._rows(recording, prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        return ExtractorResult(rows=rows, specs=self._specs(recording))

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        if not (layout_for(self.config, create=False).artifacts_dir(recording.recording_id, ASRExtractor.name) / TRANSCRIPT_NAME).is_file():
            return None
        return self._rows(recording, start_s, end_s)

    def _rows(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame:
        people = [subject.subject_id for subject in recording.subjects]
        talk = load_transcript(self.config, recording.recording_id, others=True)
        spans = alone_spans(talk, start_s, end_s)
        series = self._series(recording)
        frames = []
        for person in people:
            listening = [span for speaker, span in spans if speaker != person and speaker in people]
            values = {"t_start": [start_s], "t_end": [end_s], LISTENING_S: [float(sum(end - start for start, end in listening))]}
            for feature in self.config.kinematics.listening:
                mine = series[(series["subject_id"].astype(str) == person) & (series["feature"].astype(str) == feature)]
                inside = mine[within(mine["t_start"].to_numpy(dtype=float), listening)]["value"].to_numpy(dtype=float)
                inside = inside[np.isfinite(inside)]
                values[f"{feature}_listening"] = [float(inside.mean()) if inside.size else np.nan]
                values[f"{feature}_listening_frames"] = [float(inside.size)]
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(values), scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=person,
                )
            )
        return concat_long_tables(frames)

    def _series(self, recording: Recording) -> pd.DataFrame:
        layout = layout_for(self.config, create=False)
        wanted = self.config.kinematics.listening
        frames = []
        for step in sorted({producers()[name] for name in wanted}):
            path = layout.long_table_path(recording.recording_id) / f"{step}.parquet"
            if path.is_file():
                frames.append(read_long_table(path, recording_id=recording.recording_id, features=list(wanted)))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["subject_id", "feature", "t_start", "value"])

    def _specs(self, recording: Recording) -> list[FeatureSpec]:
        layout = layout_for(self.config, create=False)
        units = {}
        for feature in self.config.kinematics.listening:
            path = layout.manifest_fragment_path(recording.recording_id, producers()[feature])
            spec = read_manifest(path).specs.get(feature) if path.is_file() else None
            units[feature] = spec.unit if spec is not None else None
        params = {"listening": list(self.config.kinematics.listening)}
        missing = {
            LISTENING_S: "Never empty. 0 means no other person of the recording talked alone in the span, by the transcript.",
            **{
                f"{feature}_listening": (
                    "Empty where no other person of the recording talked alone in the span, or where this person's "
                    f"{feature} has no value in any frame of those times (its own entry says when)."
                )
                for feature in self.config.kinematics.listening
            },
            **{
                f"{feature}_listening_frames": (
                    f"Never empty. 0 means this person's {feature} has no value in any frame while another person of the "
                    "recording talked alone, or that nobody else talked alone in the span."
                )
                for feature in self.config.kinematics.listening
            },
        }
        described = {
            LISTENING_S: ("Seconds of the analysed part in which another person talked and nobody else did, this person included.", "s"),
            **{
                f"{feature}_listening": (
                    f"Mean {feature} of this person over the frames in which another person talked and nobody else did, by the "
                    "transcript's utterances; empty where they were not measured then.",
                    units[feature],
                )
                for feature in self.config.kinematics.listening
            },
            **{
                f"{feature}_listening_frames": (
                    f"Frames {feature}_listening averages; a mean over few frames is easily moved by a single jump of the tracker.",
                    "count",
                )
                for feature in self.config.kinematics.listening
            },
        }
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.RECORDING, group=self.group, extractor=self.name,
                extractor_version=self.version, params=params, aggregation="none", missing=missing[name], validation="none",
            )
            for name, (description, unit) in described.items()
        ]


def alone_spans(talk: pd.DataFrame, start_s: float, end_s: float) -> list[tuple[str, tuple[float, float]]]:
    """(speaker, (start, end)) for each stretch inside start_s to end_s where exactly one voice talks, by the utterances."""
    talk = talk[(talk["t_end"] > start_s) & (talk["t_start"] < end_s)]
    edges = sorted({start_s, end_s, *np.clip(talk["t_start"].to_numpy(dtype=float), start_s, end_s), *np.clip(talk["t_end"].to_numpy(dtype=float), start_s, end_s)})
    starts, ends = talk["t_start"].to_numpy(dtype=float), talk["t_end"].to_numpy(dtype=float)
    speakers = talk["subject_id"].astype(str).to_numpy()
    spans: list[tuple[str, tuple[float, float]]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        middle = (low + high) / 2.0
        talking = set(speakers[(starts <= middle) & (ends > middle)])
        if len(talking) == 1:
            speaker = talking.pop()
            if spans and spans[-1][0] == speaker and spans[-1][1][1] == low:
                spans[-1] = (speaker, (spans[-1][1][0], high))
            else:
                spans.append((speaker, (low, high)))
    return spans


def within(times: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Which times fall in one of the spans, start included and end not."""
    inside = np.zeros(len(times), dtype=bool)
    for low, high in spans:
        inside |= (times >= low) & (times < high)
    return inside
