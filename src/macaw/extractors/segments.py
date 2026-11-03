"""Every step's recording-level summaries again over each named segment of a recording, from what the steps saved."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from macaw.config.hashing import canonical
from macaw.extractors import get_extractor, list_extractors, register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.project.segments import SUMMARY_STEP, read_segments, segments_of
from macaw.store.longtable import read_long_table
from macaw.store.schema import SEGMENT_TABLE_SCHEMA, Scale, coerce_frame, empty_frame

# a segment set by hand at the window's own edge should not miss it by rounding
_EDGE_S = 1e-3


def summarising_steps() -> list[str]:
    return [name for name in list_extractors() if get_extractor(name).summarise is not Extractor.summarise]


@register_extractor
class SegmentSummaryExtractor(Extractor):
    name = SUMMARY_STEP
    version = "0.2.0"
    scale = Scale.RECORDING
    group = "segments"
    phase = Phase.DERIVE

    @classmethod
    def ordered_after(cls, config) -> tuple[str, ...]:
        return tuple(summarising_steps())

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(read_segments(config))

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        # a step's completion stamp is the key its output was made under, so it changes whenever that output does
        done = layout_for(self.config, create=False).completion_path(recording.recording_id, self.name).parent
        return sorted(path for path in done.glob("*.key") if path.stem != self.name) if done.is_dir() else []

    def extra_key(self, recording: Recording) -> str:
        return canonical([[s.name, s.start_s, s.end_s] for s in segments_of(self.config, recording.recording_id)])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        layout = layout_for(self.config, create=False)
        segments, warnings = self._inside(recording)
        frames, without = [], []
        for name in sorted(list_extractors()):
            if name == self.name or not layout.completion_path(recording.recording_id, name).is_file():
                continue
            step = get_extractor(name)(self.config)
            if type(step).summarise is Extractor.summarise:
                if self._has_summaries(recording, name):
                    without.append(name)
                continue
            for segment in segments:
                rows = step.summarise(recording, segment.start_s, segment.end_s)
                if rows is None:
                    break
                if not rows.empty:
                    frames.append(rows.assign(segment=segment.name, t_start=segment.start_s, t_end=segment.end_s))
                warnings += [
                    replace(warning, details={**warning.details, "segment": segment.name})
                    for warning in step.segment_warnings(recording, segment)
                ]
        if without and segments:
            warnings.append(
                QCWarning(
                    check="segment_no_summary", recording_id=recording.recording_id, subject_id=None, severity="info",
                    message=f"{', '.join(without)} gave results for the whole recording but has no per-segment version of them",
                )
            )
        table = (
            coerce_frame(pd.concat(frames, ignore_index=True), SEGMENT_TABLE_SCHEMA, "segment table")
            if frames else empty_frame(SEGMENT_TABLE_SCHEMA)
        )
        return ExtractorResult(rows=pd.DataFrame(), segment_rows=table, warnings=warnings)

    def _inside(self, recording: Recording):
        """The segments within the part of the recording that was analysed; the rest are left out with a warning."""
        segments = segments_of(self.config, recording.recording_id)
        prepared = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, MediaPrepExtractor.name)
        if not segments or not (prepared / PREP_INFO_NAME).is_file():
            return [], []
        prep = load_prep(self.config, recording.recording_id)
        low, high = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        kept, warnings = [], []
        for segment in segments:
            if segment.start_s >= low - _EDGE_S and segment.end_s <= high + _EDGE_S:
                kept.append(segment)
                continue
            warnings.append(
                QCWarning(
                    check="segment_outside", recording_id=recording.recording_id, subject_id=None, severity="warn",
                    message=(
                        f"Segment {segment.name} ({segment.start_s:g} to {segment.end_s:g} s) reaches outside the part of the "
                        f"recording that was analysed ({low:g} to {high:g} s), so it was left out. Move it inside, or widen "
                        "segment.task_start_s and segment.task_end_s"
                    ),
                    t_start=segment.start_s, t_end=segment.end_s,
                )
            )
        return kept, warnings

    def _has_summaries(self, recording: Recording, name: str) -> bool:
        path = layout_for(self.config, create=False).long_table_path(recording.recording_id) / f"{name}.parquet"
        return path.is_file() and not read_long_table(path, scale=Scale.RECORDING).empty
