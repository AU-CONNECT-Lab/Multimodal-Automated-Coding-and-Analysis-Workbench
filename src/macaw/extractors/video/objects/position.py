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
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

# measure: (description, unit)
MEASURES = {
    "in_view": ("1 while {thing} is in view on camera {camera}, 0 while it is out of view.", None),
    "x": ("Left-right place of the middle of {thing} in the picture of camera {camera}, in pixels from the left edge.", "px"),
    "y": ("Up-down place of the middle of {thing} in the picture of camera {camera}, in pixels from the top edge.", "px"),
    "area": ("How much of the picture of camera {camera} the outline of {thing} covers, in square pixels.", "px^2"),
    "score": (
        "How sure the tracker is that {thing} is in view on camera {camera}, 0 to 1; in_view is 1 where it reaches 0.5 and "
        "an outline was found.", None,
    ),
}
# measure: (description, unit), one value for the recording or a segment
SPAN_MEASURES = {
    "in_view_share": ("Share of the frames in which {thing} is in view on camera {camera}.", None),
    "longest_gap_s": ("Longest stretch in which {thing} is out of view on camera {camera}.", "s"),
}


def feature(thing: str, camera: str, measure: str) -> str:
    return f"{slug(thing)}_{slug(camera)}_{measure}"


@register_extractor
class ObjectPositionExtractor(Extractor):
    name = "objects.position"
    version = "0.3.0"
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
        everything, starts = load_objects(self.config, recording.recording_id, everything=True), load_starts(self.config, recording.recording_id)
        things = everything[everything["in_view"].astype(bool)]
        prep = apply_prep_timing(self.config, recording)
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities(), subject_id=None)
        frames, named, spans = [], [], {}
        for item in object_cameras(recording):
            followed = [name for name in self.config.objects.names if "frame" in starts.get(item.name, {}).get(name, {})]
            first, stop = frame_span(item, prep.t_origin_s, prep.t_origin_s + prep.duration_s)
            if not followed or stop <= first:
                continue
            index = np.arange(first, stop)
            values = pd.DataFrame({"t_start": index / float(item.fps) + item.offset_s})
            mine, scored = things[things["video"] == item.name], everything[everything["video"] == item.name]
            for name in followed:
                shown = mine[mine["label"] == name].drop_duplicates("frame").set_index("frame").reindex(index)
                score = scored[scored["label"] == name].drop_duplicates("frame").set_index("frame").reindex(index)
                values[feature(name, item.name, "in_view")] = shown["area"].notna().astype(float).to_numpy()
                values[feature(name, item.name, "x")] = shown["cx"].to_numpy(dtype=float)
                values[feature(name, item.name, "y")] = shown["cy"].to_numpy(dtype=float)
                values[feature(name, item.name, "area")] = shown["area"].to_numpy(dtype=float)
                values[feature(name, item.name, "score")] = score["score"].to_numpy(dtype=float)
                named.append((name, item.name))
                spans[feature(name, item.name, "in_view")] = values[feature(name, item.name, "in_view")]
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, **common))
            frames.append(_coverage(values["t_start"], {key: spans[key] for key in spans if key in values}, common, (prep.t_origin_s, prep.t_origin_s + prep.duration_s)))
        warnings = [] if named else [QCWarning(
            check=self.name, recording_id=recording.recording_id, subject_id=None, severity="info",
            message="no thing was followed on any camera, so there are no places to give",
        )]
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(named, starts), warnings=warnings)

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        path = layout_for(self.config, create=False).long_table_path(recording.recording_id) / f"{self.name}.parquet"
        if not path.is_file():
            return None
        saved = read_long_table(path)
        saved = saved[saved["feature"].astype(str).str.endswith("_in_view") & (saved["t_start"] >= start_s) & (saved["t_start"] < end_s)]
        if saved.empty:
            return None
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities(), subject_id=None)
        wide = saved.pivot_table(index="t_start", columns="feature", values="value", observed=True).sort_index()
        return _coverage(pd.Series(wide.index.to_numpy(dtype=float)), {str(key): wide[key].to_numpy() for key in wide.columns}, common, (start_s, end_s))

    def _specs(self, named: list[tuple[str, str]], starts: dict) -> list[FeatureSpec]:
        cfg = self.config.objects
        # the models the outlines came from, which are the open ones where SAM 3 gave way
        used = next((entry for entries in starts.values() for entry in entries.values() if "tracker" in entry), {})
        params = {"names": list(cfg.names), "finder": used.get("finder", cfg.finder), "tracker": used.get("tracker", cfg.tracker)}
        spans = [
            FeatureSpec(
                name=feature(thing, camera, measure),
                description=description.format(thing=thing, camera=camera) + " Over the frames of the span the row covers.",
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation="mean",
                missing="Never empty for a thing followed on this camera.",
                comparable_across_subjects=False,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for thing, camera in named
            for measure, (description, unit) in SPAN_MEASURES.items()
        ]
        return spans + [
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
                    "Never empty in the frames of the recording." if measure == "in_view"
                    else "Empty in frames the tracker gave no score for." if measure == "score"
                    else f"Empty in frames where {thing} is out of view on this camera."
                ),
                comparable_across_subjects=False,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for thing, camera in named
            for measure, (description, unit) in MEASURES.items()
        ]


def _coverage(times: pd.Series, in_view: dict[str, np.ndarray], common: dict, window: tuple[float, float]) -> pd.DataFrame:
    """Each thing's share of frames in view and its longest time out of view, from its 0/1 in_view per frame."""
    times = np.asarray(times, dtype=float)
    period = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    values = {"t_start": [window[0]], "t_end": [window[1]]}
    for name, seen in in_view.items():
        seen = np.asarray(seen, dtype=float) > 0.5
        stem = name[: -len("_in_view")]
        values[f"{stem}_in_view_share"] = [float(seen.mean()) if len(seen) else np.nan]
        gaps = np.diff(np.flatnonzero(np.r_[True, seen, True])) - 1
        values[f"{stem}_longest_gap_s"] = [float(gaps.max(initial=0)) * period]
    return build_long_rows(values=pd.DataFrame(values), scale=Scale.RECORDING, **common)
