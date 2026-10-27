"""Praat voice report per turn: pitch statistics and slope, jitter, shimmer, HNR, CPPS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.diarization import turn_pieces, voice_inputs, voice_sources
from macaw.extractors.audio.prosody.backends import load_audio
from macaw.extractors.audio.prosody.frame import SEMITONE_REF_HZ, f0_range, to_semitones
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, load_turns

# Praat's voice report defaults: shortest and longest period, largest period ratio, largest amplitude ratio
_PERIOD_ARGS = (0.0001, 0.02, 1.3)
_AMPLITUDE_FACTOR = 1.6
_MIN_VOICED_FRAMES = 3
# on a shared microphone a turn measured on less than this share of itself gets a note
_SOLO_NOTE = 0.5

JITTER = {
    "jitter_local": "local",
    "jitter_local_abs_s": "local, absolute",
    "jitter_rap": "rap",
    "jitter_ppq5": "ppq5",
    "jitter_ddp": "ddp",
}
SHIMMER = {
    "shimmer_local": "local",
    "shimmer_local_db": "local_dB",
    "shimmer_apq3": "apq3",
    "shimmer_apq5": "apq5",
    "shimmer_apq11": "apq11",
    "shimmer_dda": "dda",
}

# (description, unit, comparable across subjects); noise and mic distance move the amplitude measures
_SPECS = {
    "turn_f0_mean_hz": ("Mean pitch over the voiced frames of the turn.", "Hz", True),
    "turn_f0_median_hz": ("Median pitch over the voiced frames of the turn.", "Hz", True),
    "turn_f0_sd_hz": ("Standard deviation of pitch over the voiced frames of the turn.", "Hz", True),
    "turn_f0_min_hz": ("Lowest pitch in the turn.", "Hz", True),
    "turn_f0_max_hz": ("Highest pitch in the turn.", "Hz", True),
    "turn_f0_iqr_hz": ("Interquartile range of pitch in the turn.", "Hz", True),
    "turn_f0_range_st": ("Highest over lowest pitch in the turn, in semitones.", "semitone", True),
    "turn_f0_slope_st_per_s": (
        "Least-squares slope of pitch in semitones over the turn; rising reads as a question, falling as a statement.",
        "semitone/s",
        True,
    ),
    "turn_voiced_fraction": ("Share of the turn's frames that carried pitch.", None, True),
    "jitter_local": ("Mean absolute difference of consecutive periods over the mean period.", None, True),
    "jitter_local_abs_s": ("Mean absolute difference of consecutive periods.", "s", True),
    "jitter_rap": ("Relative average perturbation: each period against the mean of it and its two neighbours.", None, True),
    "jitter_ppq5": ("Five-point period perturbation quotient.", None, True),
    "jitter_ddp": ("Mean absolute difference of consecutive period differences over the mean period; three times RAP.", None, True),
    "shimmer_local": ("Mean absolute difference of consecutive peak amplitudes over the mean amplitude.", None, False),
    "shimmer_local_db": ("Mean absolute base-10 log ratio of consecutive peak amplitudes.", "dB", False),
    "shimmer_apq3": ("Three-point amplitude perturbation quotient.", None, False),
    "shimmer_apq5": ("Five-point amplitude perturbation quotient.", None, False),
    "shimmer_apq11": ("Eleven-point amplitude perturbation quotient.", None, False),
    "shimmer_dda": ("Mean absolute difference of consecutive amplitude differences over the mean amplitude; three times APQ3.", None, False),
    "turn_hnr_db": ("Mean harmonics-to-noise ratio over the voiced part of the turn.", "dB", False),
    "cpps_db": ("Smoothed cepstral peak prominence; lower means a breathier or less periodic voice.", "dB", False),
}

_INSTALL = 'prosody.voice_report needs praat-parselmouth: pip install "pymacaw[audio]"'


@register_extractor
class VoiceReportExtractor(Extractor):
    name = "prosody.voice_report"
    version = "0.3.0"
    scale = Scale.EVENT
    group = "prosody"
    config_sections = ("prosody", "turns", "segment", "text.diarization")
    requires = (TurnStructureExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        try:
            import parselmouth  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(_INSTALL) from error

    def backend_info(self) -> tuple[str | None, str | None]:
        import parselmouth

        return ("parselmouth", parselmouth.VERSION)

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        prep = load_prep(self.config, recording.recording_id)
        turns = layout.artifacts_dir(recording.recording_id, TurnStructureExtractor.name) / TURNS_NAME
        return [turns, *voice_inputs(self.config, recording, prep)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.prosody
        prep = load_prep(self.config, recording.recording_id)
        turns = load_turns(self.config, recording.recording_id)
        frames, warnings = [], []
        sources = voice_sources(self.config, recording, prep)
        for subject in recording.subjects:
            spoken = turns[turns["subject_id"].astype(str) == subject.subject_id]
            if spoken.empty or subject.subject_id not in sources:
                continue
            path, spans = sources[subject.subject_id]
            audio, rate = load_audio(path)
            f0_min, f0_max = f0_range(cfg, subject)
            pieces, kept = turn_pieces(
                audio, rate, prep.t_origin_s, spoken["t_start"].to_numpy(dtype=float), spoken["t_end"].to_numpy(dtype=float), spans
            )
            reports = [voice_report(piece, rate, f0_min=f0_min, f0_max=f0_max, step_s=cfg.frame_step_s) for piece in pieces]
            if spans is not None and (partial := sum(share < _SOLO_NOTE for share in kept)):
                warnings.append(
                    QCWarning(
                        check=self.name, recording_id=recording.recording_id, subject_id=subject.subject_id, severity="info",
                        message=f"{partial} of {len(kept)} turn(s) had someone else talking over more than half of them; only their solo part was measured",
                    )
                )
            values = pd.DataFrame(reports, columns=list(_SPECS)).astype(float)
            values.insert(0, "t_start", spoken["t_start"].to_numpy(dtype=float))
            values.insert(1, "t_end", spoken["t_end"].to_numpy(dtype=float))
            frames.append(
                build_long_rows(
                    values=values, scale=Scale.EVENT, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=subject.subject_id,
                )
            )
            unmeasured = int(values["jitter_local"].isna().sum())
            if unmeasured:
                warnings.append(
                    QCWarning(
                        check=self.name, recording_id=recording.recording_id, subject_id=subject.subject_id,
                        severity="info",
                        message=f"{unmeasured} of {len(values)} turn(s) had too few voice periods for jitter and shimmer; their values are missing",
                    )
                )
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(), warnings=warnings)

    def _specs(self) -> list[FeatureSpec]:
        import parselmouth

        backend, version = self.backend_info()
        params = {"frame_step_s": self.config.prosody.frame_step_s, "praat_version": parselmouth.PRAAT_VERSION}
        return [
            FeatureSpec(
                name=name,
                description=description,
                unit=unit,
                scale=Scale.EVENT,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=backend,
                backend_version=version,
                params=params,
                aggregation="mean",
                comparable_across_subjects=comparable,
                comparable_across_backends=False,
                validation="synthetic",
            )
            for name, (description, unit, comparable) in _SPECS.items()
        ]


def voice_report(audio: np.ndarray, rate: float, *, f0_min: float, f0_max: float, step_s: float) -> dict[str, float]:
    """Every measure of one stretch of audio; NaN where Praat cannot measure it, e.g. too few periods."""
    import parselmouth
    from parselmouth.praat import call

    report = dict.fromkeys(_SPECS, np.nan)
    try:
        sound = parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=rate)
        pitch = sound.to_pitch_ac(time_step=step_s, pitch_floor=f0_min, pitch_ceiling=f0_max)
    except parselmouth.PraatError:
        return report                  # shorter than one pitch analysis window
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=float)
    times = np.asarray(pitch.xs())
    report.update(pitch_statistics(f0, times))
    if report["turn_voiced_fraction"] == 0:
        return report

    points = call([sound, pitch], "To PointProcess (cc)")
    for name, kind in JITTER.items():
        report[name] = call(points, f"Get jitter ({kind})", 0, 0, *_PERIOD_ARGS)
    for name, kind in SHIMMER.items():
        report[name] = call([sound, points], f"Get shimmer ({kind})", 0, 0, *_PERIOD_ARGS, _AMPLITUDE_FACTOR)
    harmonicity = sound.to_harmonicity_cc(time_step=step_s, minimum_pitch=f0_min, silence_threshold=0.1, periods_per_window=1.0)
    report["turn_hnr_db"] = call(harmonicity, "Get mean", 0, 0)
    try:
        cepstrogram = call(sound, "To PowerCepstrogram", f0_min, 0.002, 5000, 50)
        # a least-squares straight trend line, as CPPS was first defined; the robust fits are 70 times slower
        report["cpps_db"] = call(
            cepstrogram, "Get CPPS", False, 0.02, 0.0005, f0_min, f0_max, 0.05, "Parabolic", 0.001, 0.05,
            "Straight", "Least squares",
        )
    except parselmouth.PraatError:
        pass
    return {name: float(value) if value is not None else np.nan for name, value in report.items()}


def pitch_statistics(f0: np.ndarray, times: np.ndarray) -> dict[str, float]:
    """f0 in Hz with 0 or NaN for unvoiced frames."""
    f0 = np.asarray(f0, dtype=float)
    voiced = np.isfinite(f0) & (f0 > 0)
    stats = {name: np.nan for name in _SPECS if name.startswith("turn_")}
    stats["turn_voiced_fraction"] = float(voiced.mean()) if f0.size else np.nan
    if not voiced.any():
        return stats
    hz = f0[voiced]
    low, high = np.percentile(hz, [25, 75])
    stats.update(
        turn_f0_mean_hz=float(hz.mean()),
        turn_f0_median_hz=float(np.median(hz)),
        turn_f0_sd_hz=float(hz.std(ddof=1)) if hz.size > 1 else np.nan,
        turn_f0_min_hz=float(hz.min()),
        turn_f0_max_hz=float(hz.max()),
        turn_f0_iqr_hz=float(high - low),
        turn_f0_range_st=float(12 * np.log2(hz.max() / hz.min())),
        turn_f0_slope_st_per_s=f0_slope(f0, times),
    )
    return stats


def f0_slope(f0: np.ndarray, times: np.ndarray) -> float:
    """Semitones per second over the voiced frames; in semitones so the slope means the same at any pitch."""
    f0 = np.asarray(f0, dtype=float)
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.sum() < _MIN_VOICED_FRAMES:
        return np.nan
    t = np.asarray(times, dtype=float)[voiced]
    if np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, to_semitones(f0[voiced], SEMITONE_REF_HZ), 1)[0])
