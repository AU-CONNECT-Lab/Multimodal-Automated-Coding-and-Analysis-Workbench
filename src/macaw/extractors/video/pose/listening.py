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
from macaw.extractors.video.pose.kinematics import (
    _SUBJECT_SPECS, BODY_UNIT, KinematicsExtractor, _body_size, _keypoints_file, keypoints_by_subject,
)
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec, read_manifest
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

LISTENING_S = "listening_s"


def producers() -> dict[str, str]:
    """Each movement series a person can be measured on while listening, and the step that gives it."""
    return {**{name: KinematicsExtractor.name for name in _SUBJECT_SPECS}, "motion_energy": MEAExtractor.name}


def body_lengths() -> set[str]:
    """The series that are a length or a rate of one, which listening_per_body_size divides by the shoulder width."""
    return {name for name, (_, unit, _) in _SUBJECT_SPECS.items() if unit and "{length}" in unit}


@register_extractor
class ListeningExtractor(Extractor):
    name = "pose.listening"
    version = "0.2.0"
    scale = Scale.RECORDING
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("kinematics.listening", "kinematics.listening_per_body_size", "segment")
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
            *([_keypoints_file(self.config, rid)] if self.divided() else []),
        ]

    def divided(self) -> list[str]:
        """The listed series divided by each person's shoulder width."""
        kinematics = self.config.kinematics
        return [name for name in kinematics.listening if name in body_lengths()] if kinematics.listening_per_body_size else []

    def body_sizes(self, recording: Recording) -> dict[str, float]:
        """Each person's shoulder width over the recording, on the camera their body is measured from, as per_body_size of
        kinematics.measures takes it."""
        _, by_subject = keypoints_by_subject(self.config, recording)
        return {person: _body_size(rows) for person, rows in by_subject.items()}

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
        divided = self.divided()
        sizes = self.body_sizes(recording) if divided else {}
        frames = []
        for person in people:
            listening = [span for speaker, span in spans if speaker != person and speaker in people]
            values = {"t_start": [start_s], "t_end": [end_s], LISTENING_S: [float(sum(end - start for start, end in listening))]}
            for feature in self.config.kinematics.listening:
                mine = series[(series["subject_id"].astype(str) == person) & (series["feature"].astype(str) == feature)]
                inside = mine[within(mine["t_start"].to_numpy(dtype=float), listening)]["value"].to_numpy(dtype=float)
                inside = inside[np.isfinite(inside)]
                mean = float(inside.mean()) if inside.size else np.nan
                values[f"{feature}_listening"] = [mean / sizes.get(person, np.nan) if feature in divided else mean]
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
        divided = self.divided()
        units, comparable = {}, {}
        for feature in self.config.kinematics.listening:
            path = layout.manifest_fragment_path(recording.recording_id, producers()[feature])
            spec = read_manifest(path).specs.get(feature) if path.is_file() else None
            units[feature] = spec.unit if spec is not None else None
            comparable[feature] = spec.comparable_across_subjects if spec is not None else True
            if feature in divided:
                units[feature], comparable[feature] = _SUBJECT_SPECS[feature][1].format(length=BODY_UNIT), True
        params = {"listening": list(self.config.kinematics.listening), "per_body_size": divided}
        sized = {
            feature: (
                f" Divided by this person's shoulder width over the recording, so in {units[feature]}; comparable between "
                "recordings where the person stays at about the same distance from the camera and faces it for part of the "
                "recording." if feature in divided
                else " Not divided by the shoulder width, which applies to lengths only." if self.config.kinematics.listening_per_body_size
                else ""
            )
            for feature in self.config.kinematics.listening
        }
        missing = {
            LISTENING_S: "Never empty. 0 means no other person of the recording talked alone in the span, by the transcript.",
            **{
                f"{feature}_listening": (
                    "Empty where no other person of the recording talked alone in the span, or where this person's "
                    f"{feature} has no value in any frame of those times (its own entry says when)"
                    + (
                        ", and throughout when this person's two shoulders are never found in the same frame on the camera their "
                        "body is measured from." if feature in divided else "."
                    )
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
                    f"transcript's utterances; empty where they were not measured then.{sized[feature]}",
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
                comparable_across_subjects=comparable.get(name.removesuffix("_listening"), True),
            )
            for name, (description, unit) in described.items()
        ]


def body_size_notice(config) -> str | None:
    """For the run plan: which listed series listening_per_body_size divides, and which it leaves as they are."""
    kinematics = config.kinematics
    if not kinematics.listening_per_body_size:
        return None
    divided = [name for name in kinematics.listening if name in body_lengths()]
    kept = [name for name in kinematics.listening if name not in body_lengths()]
    said = f"pose.listening divides {', '.join(divided)} by each person's shoulder width" if divided else "pose.listening divides nothing by the shoulder width"
    return said + (f" and leaves {', '.join(kept)} unchanged, since {'it is' if len(kept) == 1 else 'they are'} not a length" if kept else "")


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
