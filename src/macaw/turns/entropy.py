"""Transfer, sample and multiscale entropy are excluded: all are biased on short series."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.extractors import register_extractor
from macaw.extractors.audio.content.diarization import turn_takers
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale
from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, load_turns
from macaw.turns.transitions import (
    TRANSITIONS_NAME,
    TransitionExtractor,
    load_transitions,
    transition_matrix,
)

# five observations per state is the usual floor for a multinomial entropy estimate
_MIN_PER_STATE = 5
_MIN_WINDOWS = 10


def _shannon(counts, *, correct: bool = True) -> float:
    values = np.asarray(list(counts), dtype=float)
    values = values[values > 0]
    total = float(values.sum())
    if total <= 0:
        return np.nan

    p = values / total
    h = float(-(p * np.log2(p)).sum())
    if not correct:
        return h
    # the plug-in estimate is biased low, and worse on shorter recordings, so the bias would
    # masquerade as a real effect. Miller-Madow removes the leading term at no free parameter
    return h + (len(values) - 1) / (2.0 * total * math.log(2.0))


def _normalise(h: float, ceiling: float) -> float:
    if not np.isfinite(h) or ceiling <= 0:
        return np.nan
    # the correction can overshoot the ceiling when the series is tiny, which is exactly when
    # a QC warning is already firing
    return min(h / ceiling, 1.0)


def transition_entropy(matrix: pd.DataFrame, *, correct: bool = True) -> float:
    """Shannon entropy over the N x N state distribution; max is 2 bits when N=2."""
    return _shannon(np.asarray(matrix, dtype=float).ravel(), correct=correct)


def permutation_entropy(
    series: np.ndarray, *, order: int = 3, normalise: bool = True, correct: bool = True
) -> float:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if order < 2 or len(values) - order + 1 < 1:
        return np.nan

    windows = np.lib.stride_tricks.sliding_window_view(values, order)
    patterns = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)

    h = _shannon(counts, correct=correct)
    return _normalise(h, math.log2(math.factorial(order))) if normalise else h


def speaker_entropy(talk_time: dict[str, float]) -> float:
    """No bias correction: talk time is a measured duration, not a sample of counts."""
    return _shannon(talk_time.values(), correct=False)


def conditional_entropy(x: np.ndarray, y: np.ndarray, *, correct: bool = True) -> float:
    """H(partner response | my utterance); quantifies direction of influence."""
    left = list(np.asarray(x, dtype=object))
    right = list(np.asarray(y, dtype=object))
    if not left or len(left) != len(right):
        return np.nan

    joint = pd.Series(list(zip(left, right))).value_counts().to_numpy()
    marginal = pd.Series(left).value_counts().to_numpy()
    return _shannon(joint, correct=correct) - _shannon(marginal, correct=correct)


def _talk_time(turns: pd.DataFrame) -> dict[str, float]:
    if turns.empty:
        return {}
    held = (turns["t_end"] - turns["t_start"]).groupby(turns["subject_id"].astype(str)).sum()
    return {str(name): float(value) for name, value in held.items()}


def _speaker_sequence(turns: pd.DataFrame) -> np.ndarray:
    if turns.empty:
        return np.empty(0, dtype=object)
    return turns.sort_values("t_start", kind="stable")["subject_id"].astype(str).to_numpy()


_RECORDING_SPECS = {
    "transition_entropy_bits": (
        "Shannon entropy of the who-followed-whom distribution over all N x N speaker pairs. "
        "Miller-Madow corrected.",
        "bit",
        "mean",
    ),
    "transition_entropy_norm": (
        "The same quantity over its ceiling of log2(N^2), so recordings with different numbers "
        "of speakers can be compared.",
        None,
        "mean",
    ),
    "conditional_entropy_bits": (
        "How unpredictable the next speaker is once the current one is known. Emitted for the "
        "whole recording, and once per subject_id conditioned on that speaker holding the "
        "floor; the gap between the per-speaker values is the asymmetry of influence.",
        "bit",
        "mean",
    ),
    "conditional_entropy_norm": (
        "The same quantity over its ceiling of log2(N).",
        None,
        "mean",
    ),
    "permutation_entropy_fto_bits": (
        "Ordinal-pattern entropy of the floor transfer offset series, which measures how "
        "regular the rhythm of handovers is rather than how long they are.",
        "bit",
        "mean",
    ),
    "permutation_entropy_fto_norm": (
        "The same quantity over its ceiling of log2(order!).",
        None,
        "mean",
    ),
    "permutation_entropy_n": (
        "Ordinal patterns the estimate was built from. The sample size for the other entropies "
        "is n_transitions over in turns.transitions.",
        "count",
        "sum",
    ),
    "speaker_entropy_bits": (
        "Evenness of the floor-time split. Computed from durations rather than counts, so no "
        "bias correction applies.",
        "bit",
        "mean",
    ),
    "speaker_entropy_norm": (
        "The same quantity over its ceiling of log2(N); 1 is a perfectly even split.",
        None,
        "mean",
    ),
}


def window_hop(cfg) -> float:
    """turns.window_hop_s, or half the window when it is not set."""
    return cfg.window_hop_s or cfg.window_s / 2.0


@register_extractor
class EntropyExtractor(Extractor):
    name = "turns.entropy"
    version = "0.2.0"
    scale = Scale.WINDOW
    group = "turns"
    config_sections = ("turns", "text.diarization", "text.extra_voices")
    requires = (TransitionExtractor.name, TurnStructureExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        return [
            layout.artifacts_dir(recording.recording_id, TurnStructureExtractor.name) / TURNS_NAME,
            layout.artifacts_dir(recording.recording_id, TransitionExtractor.name)
            / TRANSITIONS_NAME,
        ]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Warns when the task duration yields too few windows to interpret."""
        turns = load_turns(self.config, recording.recording_id, others=True)
        transitions = load_transitions(self.config, recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)

        return ExtractorResult(
            rows=self._rows(recording, turns, transitions, prep),
            specs=self._specs(),
            warnings=self._warnings(recording, turns, transitions, prep),
        )

    def _fto_series(self, transitions: pd.DataFrame) -> np.ndarray:
        if transitions.empty:
            return np.empty(0, dtype=float)
        switches = transitions[transitions["is_switch"].astype(bool).to_numpy()]
        return switches["fto_s"].to_numpy(dtype=float)

    def _window_edges(self, prep) -> list[tuple[float, float]]:
        cfg = self.config.turns
        if not cfg.window_s:
            return []
        hop = window_hop(cfg)
        # entropy cannot be averaged, so windows are recomputed from the turns they contain
        # rather than rolled up out of the recording-level rows
        span = prep.duration_s - cfg.window_s
        if span < 0:
            return []
        count = int(math.floor(span / hop)) + 1
        return [
            (
                prep.t_origin_s + index * hop,
                prep.t_origin_s + index * hop + cfg.window_s,
            )
            for index in range(count)
        ]

    def _state_values(self, turns: pd.DataFrame, subject_ids: list[str]) -> dict[str, float]:
        matrix = transition_matrix(turns, subject_ids)
        speakers = _speaker_sequence(turns)
        n_states = max(len(subject_ids), 1)

        joint = transition_entropy(matrix)
        conditional = conditional_entropy(speakers[:-1], speakers[1:])
        even = speaker_entropy(_talk_time(turns))
        return {
            "transition_entropy_bits": joint,
            "transition_entropy_norm": _normalise(joint, 2.0 * math.log2(n_states)),
            "conditional_entropy_bits": conditional,
            "conditional_entropy_norm": _normalise(conditional, math.log2(n_states)),
            "speaker_entropy_bits": even,
            "speaker_entropy_norm": _normalise(even, math.log2(n_states)),
        }

    def _rows(self, recording: Recording, turns, transitions, prep) -> pd.DataFrame:
        cfg = self.config.turns
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        subject_ids = turn_takers(self.config, recording)
        n_states = max(len(subject_ids), 1)
        frames = []

        fto = self._fto_series(transitions)
        overall = self._state_values(turns, subject_ids)
        overall["permutation_entropy_fto_bits"] = permutation_entropy(
            fto, order=cfg.permutation_entropy_order, normalise=False
        )
        overall["permutation_entropy_fto_norm"] = permutation_entropy(
            fto, order=cfg.permutation_entropy_order, normalise=True
        )
        overall["permutation_entropy_n"] = float(
            max(len(fto) - cfg.permutation_entropy_order + 1, 0)
        )
        frames.append(
            build_long_rows(
                values=pd.DataFrame(
                    {
                        "t_start": [window[0]],
                        "t_end": [window[1]],
                        **{feature: [value] for feature, value in overall.items()},
                    }
                ),
                scale=Scale.RECORDING,
                **common,
            )
        )

        # the per-speaker row is the entropy of that speaker's own row of the matrix, which is
        # what makes the two directions comparable
        matrix = transition_matrix(turns, subject_ids)
        for subject_id in subject_ids:
            given = _shannon(matrix.loc[subject_id])
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [window[0]],
                            "t_end": [window[1]],
                            "conditional_entropy_bits": [given],
                            "conditional_entropy_norm": [_normalise(given, math.log2(n_states))],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=str(subject_id),
                    **common,
                )
            )

        for low, high in self._window_edges(prep):
            starts = turns["t_start"].to_numpy(dtype=float) if not turns.empty else np.empty(0)
            inside = turns[(starts >= low) & (starts < high)] if not turns.empty else turns
            values = self._state_values(inside, subject_ids)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [low],
                            "t_end": [high],
                            **{
                                feature: [value]
                                for feature, value in values.items()
                                if not feature.startswith("conditional_")
                            },
                        }
                    ),
                    scale=Scale.WINDOW,
                    **common,
                )
            )
        return concat_long_tables(frames)

    def _warnings(self, recording: Recording, turns, transitions, prep) -> list[QCWarning]:
        cfg = self.config.turns
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        n_states = max(len(turn_takers(self.config, recording)), 1) ** 2
        n_transitions = max(len(turns) - 1, 0)
        fto = self._fto_series(transitions)
        patterns = max(len(fto) - cfg.permutation_entropy_order + 1, 0)
        needed_patterns = _MIN_PER_STATE * math.factorial(cfg.permutation_entropy_order)
        warnings = []

        if n_transitions < _MIN_PER_STATE * n_states:
            warnings.append(
                QCWarning(
                    check="entropy_few_transitions",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"{n_transitions} turn pairs across {n_states} states, below the "
                        f"{_MIN_PER_STATE} per state these estimates want. The transition and "
                        "conditional entropies are bias corrected but still unstable here."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )

        if patterns < needed_patterns:
            warnings.append(
                QCWarning(
                    check="entropy_short_fto_series",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"{patterns} ordinal patterns from the floor transfer offsets, below "
                        f"the {needed_patterns} that order {cfg.permutation_entropy_order} "
                        "wants. Read permutation_entropy_fto with that in mind."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )

        edges = self._window_edges(prep)
        if cfg.window_s and len(edges) < _MIN_WINDOWS:
            warnings.append(
                QCWarning(
                    check="entropy_too_few_windows",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"A {cfg.window_s} s window at {window_hop(cfg)} s hop yields "
                        f"{len(edges)} windows over {prep.duration_s:.0f} s. Nothing about "
                        "change over time can be read off that few."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )
        return warnings

    def _specs(self) -> list[FeatureSpec]:
        cfg = self.config.turns
        params = {
            "permutation_entropy_order": cfg.permutation_entropy_order,
            "window_s": cfg.window_s,
            "window_hop_s": window_hop(cfg) if cfg.window_s else None,
            "estimator": "miller-madow",
        }
        return [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation=aggregation,
                validation="synthetic",
                notes=(
                    "Always emitted at recording scale, and at window scale too once window_s "
                    "is set. Entropy is not averageable, so window rows are recomputed from "
                    "the turns they contain rather than rolled up."
                ),
            )
            for feature, (description, unit, aggregation) in _RECORDING_SPECS.items()
        ]
