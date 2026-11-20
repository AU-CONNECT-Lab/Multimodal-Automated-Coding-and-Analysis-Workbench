"""Where each thing followed is in each camera's picture, frame by frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.prep import apply_prep_timing
from macaw.extractors.video.objects.review import ObjectReviewGate
from macaw.extractors.video.objects.touch import slug
from macaw.extractors.video.objects.track import frame_span, load_objects, load_starts, object_cameras, objects_file, starts_file
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# measure: (description, unit)
MEASURES = {
    "in_view": ("1 while {thing} is in view on camera {camera}, 0 while it is out of view.", None),
    "x": ("Left-right place of the middle of {thing} in the picture of camera {camera}, in pixels from the left edge.", "px"),
    "y": ("Up-down place of the middle of {thing} in the picture of camera {camera}, in pixels from the top edge.", "px"),
    "area": ("How much of the picture of camera {camera} the outline of {thing} covers, in square pixels.", "px^2"),
}


def feature(thing: str, camera: str, measure: str) -> str:
    return f"{slug(thing)}_{slug(camera)}_{measure}"


@register_extractor
class ObjectPositionExtractor(Extractor):
    name = "objects.position"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "objects"
    phase = Phase.DERIVE
    config_sections = ("objects.names", "segment", "video", "setup.cameras")
    requires = (ObjectReviewGate.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.objects.names)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [objects_file(self.config, recording.recording_id), starts_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        things, starts = load_objects(self.config, recording.recording_id), load_starts(self.config, recording.recording_id)
        prep = apply_prep_timing(self.config, recording)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities(), subject_id=None)
        frames, named = [], []
        for item in object_cameras(recording):
            followed = [name for name in self.config.objects.names if "frame" in starts.get(item.name, {}).get(name, {})]
            first, stop = frame_span(item, prep.t_origin_s, prep.t_origin_s + prep.duration_s)
            if not followed or stop <= first:
                continue
            index = np.arange(first, stop)
            values = pd.DataFrame({"t_start": index / float(item.fps) + item.offset_s})
            mine = things[things["video"] == item.name]
            for name in followed:
                shown = mine[mine["label"] == name].drop_duplicates("frame").set_index("frame").reindex(index)
                values[feature(name, item.name, "in_view")] = shown["area"].notna().astype(float).to_numpy()
                values[feature(name, item.name, "x")] = shown["cx"].to_numpy(dtype=float)
                values[feature(name, item.name, "y")] = shown["cy"].to_numpy(dtype=float)
                values[feature(name, item.name, "area")] = shown["area"].to_numpy(dtype=float)
                named.append((name, item.name))
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, **common))
        warnings = [] if named else [QCWarning(
            check=self.name, recording_id=recording.recording_id, subject_id=None, severity="info",
            message="no thing was followed on any camera, so there are no places to give",
        )]
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(named), warnings=warnings)

    def _specs(self, named: list[tuple[str, str]]) -> list[FeatureSpec]:
        params = {"names": list(self.config.objects.names), "tracker": self.config.objects.tracker}
        return [
            FeatureSpec(
                name=feature(thing, camera, measure),
                description=description.format(thing=thing, camera=camera) + (
                    " Taken from the outline the tracker followed, so it holds as long as the outline stays on the thing."
                ),
                unit=unit,
                scale=Scale.FRAME,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                interpolation="none",
                aggregation="mean",
                missing=(
                    "Never empty in the frames of the recording."
                    if measure == "in_view" else f"Empty in frames where {thing} is out of view on this camera."
                ),
                comparable_across_subjects=False,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for thing, camera in named
            for measure, (description, unit) in MEASURES.items()
        ]
