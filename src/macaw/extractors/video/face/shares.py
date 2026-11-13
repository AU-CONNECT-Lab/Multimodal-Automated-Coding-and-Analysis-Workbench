"""Per person, how often a face was found, and how often each chosen action unit stood above its threshold where it was."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, apply_prep_timing
from macaw.extractors.video.face.au import ACTION_UNITS, FaceExtractor, _face_rows_file
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import MediaFile, Recording
from macaw.project.segments import clip_points
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

_FOUND = "face_found_share"


@register_extractor
class FaceShareExtractor(Extractor):
    name = "face.shares"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "face"
    phase = Phase.DERIVE
    config_sections = ("face.au_above", "segment")
    requires = (FaceExtractor.name,)

    def check_available(self) -> None:
        known = {code.lower() for code in ACTION_UNITS}
        unknown = sorted(set(self.config.face.au_above) - known)
        if unknown:
            raise ConfigError(f"face.au_above names {', '.join(unknown)}; the action units are {', '.join(sorted(known))}")

    def inputs_for(self, recording: Recording) -> list[Path]:
        prepared = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, MediaPrepExtractor.name)
        return [_face_rows_file(self.config, recording.recording_id), prepared / PREP_INFO_NAME]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        prep = apply_prep_timing(self.config, recording)
        faces = read_long_table(_face_rows_file(self.config, recording.recording_id), recording_id=recording.recording_id)
        rows = self._rows(recording, faces, prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        given = set(faces["feature"].astype(str))
        missing = sorted(set(self.config.face.au_above) - given)
        warnings = [
            QCWarning(
                check=self.name, recording_id=recording.recording_id, subject_id=None, severity="warn",
                message=f"face.au gave no {', '.join(missing)} here (its backend gives none, or face.action_units is off), so their shares are empty",
            )
        ] if missing and not faces.empty else []
        return ExtractorResult(rows=rows, specs=self._specs(), warnings=warnings)

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        path = _face_rows_file(self.config, recording.recording_id)
        if not path.is_file():
            return None
        apply_prep_timing(self.config, recording)
        return self._rows(recording, read_long_table(path, recording_id=recording.recording_id), start_s, end_s)

    def _rows(self, recording: Recording, faces: pd.DataFrame, start_s: float, end_s: float) -> pd.DataFrame:
        faces = clip_points(faces, start_s, end_s, column="t_start")
        thresholds = self.config.face.au_above
        frames = []
        for person, item in sorted(recording.views("face").items()):
            mine = faces[faces["subject_id"].astype(str) == person]
            total = frames_between(item, start_s, end_s)
            values = {"t_start": [start_s], "t_end": [end_s], _FOUND: [mine["t_start"].nunique() / total if total else np.nan]}
            for unit, threshold in thresholds.items():
                measured = mine.loc[mine["feature"].astype(str) == unit, "value"].to_numpy(dtype=float)
                measured = measured[np.isfinite(measured)]
                values[f"{unit}_above_share"] = [float((measured > threshold).mean()) if measured.size else np.nan]
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(values), scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=person,
                )
            )
        return concat_long_tables(frames)

    def _specs(self) -> list[FeatureSpec]:
        described = {
            _FOUND: "Share of the frames of the analysed part, on the camera this person's face is read from, in which a face was found and given to them.",
            **{
                f"{unit}_above_share": (
                    f"Share of the frames with this person's face in which action unit {int(unit[2:])} ({ACTION_UNITS[unit.upper()]}) "
                    f"was above {threshold:g} on the face model's scale (face.au_above); empty where no face was found."
                )
                for unit, threshold in self.config.face.au_above.items()
            },
        }
        params = {"au_above": dict(self.config.face.au_above), "face_backend": self.config.face.backend, "face_model": self.config.face.model}
        return [
            FeatureSpec(
                name=name, description=description, unit=None, scale=Scale.RECORDING, group=self.group, extractor=self.name,
                extractor_version=self.version, params=params, aggregation="none", validation="none",
            )
            for name, description in described.items()
        ]


def frames_between(item: MediaFile, start_s: float, end_s: float) -> int:
    """Frames of a video whose time on the recording clock falls in start_s to end_s, end excluded."""
    if not item.fps:
        return 0
    first = max(math.ceil((start_s - item.offset_s) * item.fps - 1e-9), 0)
    stop = math.ceil((end_s - item.offset_s) * item.fps - 1e-9)
    if item.span_s is not None:
        stop = min(stop, math.floor(item.span_s * item.fps + 1e-9))
    return max(stop - first, 0)
