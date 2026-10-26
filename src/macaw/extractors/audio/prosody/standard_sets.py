"""eGeMAPSv02 and ComParE_2016 via openSMILE; citeable, reviewers accept them, near-free."""

from __future__ import annotations

import warnings as python_warnings
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_extractor
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, load_turns

PREFIXES = {"eGeMAPSv02": "egemaps", "ComParE_2016": "compare"}
# openSMILE writes 0 (or -201) for a voiced-only measure of a turn with no voiced frame; that is a
# missing value. The UV measures are over unvoiced frames and stay.
VOICED_ONLY = "_sma3nz_"
UNVOICED = "UV"

_INSTALL = (
    "prosody.standard_sets needs openSMILE, which pymacaw does not install: its licence allows "
    "research and teaching use only, commercial use needs a licence from audEERING. "
    "If that fits your use: pip install opensmile"
)


@register_extractor
class StandardSetExtractor(Extractor):
    name = "prosody.standard_sets"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "prosody"
    config_sections = ("prosody.standard_sets", "turns", "segment")
    requires = (TurnStructureExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        try:
            import opensmile  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(_INSTALL) from error

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        prep = load_prep(self.config, recording.recording_id)
        turns = layout.artifacts_dir(recording.recording_id, TurnStructureExtractor.name) / TURNS_NAME
        return [turns] + [prep.track_path(subject) for subject in sorted(prep.tracks)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        import opensmile

        prep = load_prep(self.config, recording.recording_id)
        turns = load_turns(self.config, recording.recording_id)
        warnings: list[QCWarning] = []
        frames, names = [], {}
        for subject_id, spoken in turns.groupby("subject_id", observed=True):
            subject_id = str(subject_id)
            if subject_id not in prep.tracks:
                continue
            audio, rate = load_audio(prep.track_path(subject_id))
            starts = spoken["t_start"].to_numpy(dtype=float) - prep.t_origin_s
            ends = spoken["t_end"].to_numpy(dtype=float) - prep.t_origin_s
            for feature_set in self.config.prosody.standard_sets:
                values, too_short = functionals(audio, rate, starts, ends, feature_set)
                names[feature_set] = list(values.columns)
                values.insert(0, "t_start", spoken["t_start"].to_numpy(dtype=float))
                values.insert(1, "t_end", spoken["t_end"].to_numpy(dtype=float))
                frames.append(
                    build_long_rows(
                        values=values, scale=Scale.EVENT, recording_id=recording.recording_id, source=self.name,
                        entities=recording.entities(), subject_id=subject_id,
                    )
                )
                if too_short:
                    warnings.append(
                        QCWarning(
                            check=self.name, recording_id=recording.recording_id, subject_id=subject_id,
                            severity="info",
                            message=f"{too_short} turn(s) too short for {feature_set}; their values are missing",
                        )
                    )
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(names, opensmile.__version__), warnings=warnings)

    def _specs(self, names: dict[str, list[str]], version: str) -> list[FeatureSpec]:
        return [
            FeatureSpec(
                name=name,
                description=f"openSMILE {feature_set} functional {name.split('_', 1)[1]}, over the turn on the speaker's own track.",
                unit=None,
                scale=Scale.EVENT,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend="opensmile",
                backend_version=version,
                params={"feature_set": feature_set},
                aggregation="mean",
                comparable_across_backends=False,
                validation="reference",
            )
            for feature_set, columns in names.items()
            for name in columns
        ]


def functionals(
    audio: np.ndarray, rate: float, starts: np.ndarray, ends: np.ndarray, feature_set: str
) -> tuple[pd.DataFrame, int]:
    """(one row of functionals per segment, how many were too short to measure); times in seconds into audio."""
    import opensmile

    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set), feature_level=opensmile.FeatureLevel.Functionals
    )
    rows, too_short = [], 0
    for start, end in zip(starts, ends):
        piece = audio[max(int(start * rate), 0) : max(int(end * rate), 0)].astype(np.float32)
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            row = smile.process_signal(piece, rate) if piece.size else None
        if row is None or any("too short" in str(item.message) for item in caught):
            too_short += 1
            rows.append(None)
            continue
        rows.append(row.iloc[0])
    columns = list(smile.feature_names)
    table = pd.DataFrame([row if row is not None else pd.Series(np.nan, index=columns) for row in rows], columns=columns)
    table = table.reset_index(drop=True).astype(float)
    voiced_only = [column for column in columns if VOICED_ONLY in column and UNVOICED not in column.split("_")[0]]
    f0 = [column for column in voiced_only if column.startswith("F0")]
    unvoiced = table[f0].eq(0).all(axis=1) if f0 else pd.Series(False, index=table.index)
    table.loc[unvoiced, voiced_only] = np.nan
    table.columns = [f"{PREFIXES[feature_set]}_{column}" for column in columns]
    return table, too_short
