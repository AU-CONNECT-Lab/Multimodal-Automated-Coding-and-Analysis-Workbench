from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors import register_extractor
from macaw.extractors.audio.prosody.backends import (
    ProsodyBackend,
    frame_times,
    load_audio,
    resolve_backend,
)
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording, Subject
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# Pitch is perceived logarithmically, so a fixed Hz step means different things to a low and
# a high voice: 12*log2(f0/ref) makes one semitone the same interval for everyone. The
# reference only sets the zero point; differences and spreads come out identical whatever it
# is. 100 Hz keeps the zero shared, so absolute pitch level stays comparable across people.
SEMITONE_REF_HZ = 100.0

N_FORMANTS = 4


def to_semitones(f0_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    f0_hz = np.asarray(f0_hz, dtype=float)
    if not np.isfinite(reference_hz) or reference_hz <= 0:
        return np.full(f0_hz.shape, np.nan)
    return np.where(f0_hz > 0, 12.0 * np.log2(np.maximum(f0_hz, 1e-9) / reference_hz), np.nan)


_FRAME_SPECS = {
    "f0_hz": ("Fundamental frequency; NaN where the frame is unvoiced.", "Hz"),
    "f0_semitone": (
        f"Fundamental frequency in semitones above {SEMITONE_REF_HZ:g} Hz, a fixed reference "
        "shared by everyone.",
        "semitone",
    ),
    "f0_semitone_rel": (
        "Fundamental frequency in semitones relative to this speaker's own median in this "
        "recording; zero means their usual pitch.",
        "semitone",
    ),
    "intensity_db": ("Frame energy in dB. Only differences within one track are meaningful.", "dB"),
    "voiced": ("1 where pitch was found, 0 where not.", None),
    "hnr_db": ("Harmonics-to-noise ratio; lower means a breathier or noisier voice.", "dB"),
}

_SUMMARY_SPECS = {
    "f0_median_hz": (
        "Median voiced pitch, the reference f0_semitone_rel is measured against.",
        "Hz",
    ),
    "voiced_fraction": ("Share of frames in the task window that carried pitch.", None),
}


@register_extractor
class FrameProsodyExtractor(Extractor):
    name = "prosody.frame"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "prosody"
    config_sections = ("prosody", "segment")
    requires = (MediaPrepExtractor.name,)

    def __init__(self, config):
        super().__init__(config)
        self._backend: ProsodyBackend | None = None

    def backend(self) -> ProsodyBackend:
        if self._backend is None:
            self._backend = resolve_backend(self._chain())
        return self._backend

    def _chain(self) -> list[str]:
        cfg = self.config.prosody
        # the named backend is only a preference; the chain says what to fall back to
        return [cfg.backend] + [name for name in cfg.fallback_chain if name != cfg.backend]

    def check_available(self) -> None:
        self.backend()

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, str(backend.version))

    def inputs_for(self, recording: Recording) -> list[Path]:
        prep = load_prep(self.config, recording.recording_id)
        return [prep.track_path(subject_id) for subject_id in sorted(prep.tracks)]

    def _f0_range(self, subject: Subject) -> tuple[float, float]:
        cfg = self.config.prosody
        # a child and an adult do not share a plausible pitch range, and searching the wrong
        # one is what produces octave errors
        for key in (subject.role, subject.subject_id):
            if key is not None and key in cfg.f0_range_by_role:
                low, high = cfg.f0_range_by_role[key]
                return float(low), float(high)
        return float(cfg.f0_min_hz), float(cfg.f0_max_hz)

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.prosody
        prep = load_prep(self.config, recording.recording_id)
        backend = self.backend()

        frames = []
        warnings: list[QCWarning] = []
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)

        for subject in recording.subjects:
            if subject.subject_id not in prep.tracks:
                continue
            audio, sr = load_audio(prep.track_path(subject.subject_id))
            f0_min, f0_max = self._f0_range(subject)
            contours = self._contours(backend, audio, sr, cfg.frame_step_s, f0_min, f0_max)

            voiced = contours["f0_hz"].to_numpy(dtype=float)
            median = float(np.nanmedian(voiced)) if np.any(np.isfinite(voiced)) else np.nan
            contours["f0_semitone"] = to_semitones(voiced, SEMITONE_REF_HZ)
            contours["f0_semitone_rel"] = to_semitones(voiced, median)

            # the grid is relative to the prepared track, whose sample 0 is t_origin_s
            contours["t_start"] = contours["t_start"] + prep.t_origin_s
            contours["t_end"] = contours["t_start"] + cfg.frame_step_s

            frames.append(
                _drop_missing(
                    build_long_rows(
                        values=contours,
                        scale=Scale.FRAME,
                        subject_id=subject.subject_id,
                        **common,
                    )
                )
            )

            fraction = float(np.mean(np.isfinite(voiced))) if voiced.size else 0.0
            frames.append(
                _drop_missing(
                    build_long_rows(
                        values=pd.DataFrame(
                            {
                                "t_start": [prep.t_origin_s],
                                "t_end": [prep.t_origin_s + prep.duration_s],
                                "f0_median_hz": [median],
                                "voiced_fraction": [fraction],
                            }
                        ),
                        scale=Scale.RECORDING,
                        subject_id=subject.subject_id,
                        **common,
                    )
                )
            )

            if fraction < cfg.min_voiced_fraction:
                warnings.append(
                    QCWarning(
                        check="voiced_fraction",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message=(
                            f"Only {fraction:.1%} of frames are voiced, below the expected "
                            f"{cfg.min_voiced_fraction:.1%}. The track may be silent, "
                            "mis-attributed, or the pitch range may be wrong."
                        ),
                        t_start=prep.t_origin_s,
                        t_end=prep.t_origin_s + prep.duration_s,
                        details={"f0_min_hz": f0_min, "f0_max_hz": f0_max},
                    )
                )

        return ExtractorResult(
            rows=concat_long_tables(frames),
            specs=self._specs(backend),
            warnings=warnings,
        )

    def _contours(
        self,
        backend: ProsodyBackend,
        audio: np.ndarray,
        sr: float,
        step_s: float,
        f0_min: float,
        f0_max: float,
    ) -> pd.DataFrame:
        times = frame_times(audio.size, sr, step_s)
        f0 = backend.f0(audio, sr, f0_min, f0_max, step_s)

        contours = pd.DataFrame(
            {
                "t_start": times,
                "f0_hz": f0,
                "intensity_db": backend.intensity(audio, sr, step_s),
                "voiced": np.isfinite(f0).astype(float),
            }
        )
        if backend.supports("hnr"):
            contours["hnr_db"] = backend.hnr(audio, sr, step_s)
        if backend.supports("formants"):
            formants = backend.formants(audio, sr, step_s, N_FORMANTS)
            for index in range(N_FORMANTS):
                contours[f"f{index + 1}_hz"] = formants[:, index]
        return contours

    def _specs(self, backend: ProsodyBackend) -> list[FeatureSpec]:
        name, version = backend.name, str(backend.version)
        specs = [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.FRAME,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=name,
                backend_version=version,
                params={"frame_step_s": self.config.prosody.frame_step_s},
                interpolation="none",
                aggregation="mean",
                comparable_across_subjects=feature != "intensity_db",
                comparable_across_backends=feature not in {"hnr_db"},
                validation="none",
            )
            for feature, (description, unit) in _FRAME_SPECS.items()
            if feature != "hnr_db" or backend.supports("hnr")
        ]
        if backend.supports("formants"):
            specs += [
                FeatureSpec(
                    name=f"f{index + 1}_hz",
                    description=f"Formant F{index + 1} centre frequency.",
                    unit="Hz",
                    scale=Scale.FRAME,
                    group=self.group,
                    extractor=self.name,
                    extractor_version=self.version,
                    backend=name,
                    backend_version=version,
                    interpolation="none",
                    aggregation="mean",
                    comparable_across_backends=False,
                    validation="none",
                )
                for index in range(N_FORMANTS)
            ]
        specs += [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=name,
                backend_version=version,
                aggregation="mean",
                validation="none",
            )
            for feature, (description, unit) in _SUMMARY_SPECS.items()
        ]
        return specs


def _drop_missing(rows: pd.DataFrame) -> pd.DataFrame:
    """An unvoiced frame has no pitch, so it gets no row. Roughly halves the table and keeps
    'missing' from being confused with a measured value."""
    return rows[rows["value"].notna()].reset_index(drop=True)


def f0_contour(audio_path: Path, cfg, *, f0_min: float, f0_max: float) -> pd.DataFrame:
    audio, sr = load_audio(Path(audio_path))
    f0 = resolve_backend(list(cfg.fallback_chain)).f0(audio, sr, f0_min, f0_max, cfg.frame_step_s)
    return pd.DataFrame(
        {
            "t_start": frame_times(audio.size, sr, cfg.frame_step_s),
            "f0_hz": f0,
            "f0_semitone": to_semitones(f0, SEMITONE_REF_HZ),
            "voiced": np.isfinite(f0).astype(float),
        }
    )


def intensity_contour(audio_path: Path, cfg) -> pd.DataFrame:
    """Intensity is not comparable across speakers; the manifest must say so."""
    audio, sr = load_audio(Path(audio_path))
    return pd.DataFrame(
        {
            "t_start": frame_times(audio.size, sr, cfg.frame_step_s),
            "intensity_db": resolve_backend(list(cfg.fallback_chain)).intensity(
                audio, sr, cfg.frame_step_s
            ),
        }
    )


def formant_contours(audio_path: Path, cfg, n_formants: int = N_FORMANTS) -> pd.DataFrame:
    audio, sr = load_audio(Path(audio_path))
    backend = resolve_backend(list(cfg.fallback_chain))
    values = backend.formants(audio, sr, cfg.frame_step_s, n_formants)
    out = pd.DataFrame({"t_start": frame_times(audio.size, sr, cfg.frame_step_s)})
    for index in range(n_formants):
        out[f"f{index + 1}_hz"] = values[:, index]
    return out
