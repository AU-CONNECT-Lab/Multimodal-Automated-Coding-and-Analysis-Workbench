"""Turns become floor transfers: who handed over to whom, how fast, and who cut in."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.config.schema import TurnsSection
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.diarization import turn_takers
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale, empty_frame
from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, load_turns

TRANSITIONS_NAME = "transitions.parquet"

TRANSITION_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "transition_index": "int64",
    "from_subject": "string",
    "to_subject": "string",
    "from_turn_index": "int64",
    "to_turn_index": "int64",
    "is_switch": "bool",
    "prev_end": "float64",
    "next_start": "float64",
    "fto_s": "float64",
    "transition_type": "string",
}

_TRANSITION_COLUMNS = [column for column in TRANSITION_SCHEMA if column != "recording_id"]

# below this a transition rate or an FTO distribution is noise, whatever the numbers say
_MIN_SWITCHES = 10
_OVERLAP_WARN = 0.5


def detect_transitions(turns: pd.DataFrame, cfg: TurnsSection) -> pd.DataFrame:
    """Gap is positive, overlap negative. Stored per speaker pair, never as A/B.

    Consecutive turns are not always by different people: a speaker who pauses for longer than
    max_gap_s with nobody cutting in starts a new turn of their own. Those rows are kept as
    holds so the transition matrix has a diagonal, and every floor transfer statistic filters
    on is_switch."""
    if len(turns) < 2:
        return empty_frame(TRANSITION_SCHEMA).loc[:, _TRANSITION_COLUMNS]

    frame = turns.sort_values("t_start", kind="stable").reset_index(drop=True)
    previous = frame.iloc[:-1].reset_index(drop=True)
    following = frame.iloc[1:].reset_index(drop=True)

    from_values = previous["subject_id"].astype(str).to_numpy()
    to_values = following["subject_id"].astype(str).to_numpy()
    prev_end = previous["t_end"].to_numpy(dtype=float)
    next_start = following["t_start"].to_numpy(dtype=float)
    fto = next_start - prev_end
    is_switch = from_values != to_values

    kind = np.where(~is_switch, "hold", np.where(fto >= 0.0, "gap", "overlap"))

    out = pd.DataFrame(
        {
            "transition_index": np.arange(len(fto), dtype=np.int64),
            "from_subject": from_values,
            "to_subject": to_values,
            "from_turn_index": previous["turn_index"].to_numpy(dtype=np.int64),
            "to_turn_index": following["turn_index"].to_numpy(dtype=np.int64),
            "is_switch": is_switch,
            "prev_end": prev_end,
            "next_start": next_start,
            "fto_s": fto,
            "transition_type": kind,
        }
    )
    return out.loc[:, _TRANSITION_COLUMNS]


def classify_overlaps(transitions: pd.DataFrame, *, interruption_s: float) -> pd.DataFrame:
    """Overlap beyond interruption_s takes the floor; shorter is a backchannel.

    Length alone is not enough: a short overlap that does take the floor is not a backchannel.
    The second condition is that the floor came straight back to whoever was talking, which is
    directly observable here, so nothing has to be inferred from duration."""
    if transitions.empty:
        return transitions

    out = transitions.copy()
    kind = out["transition_type"].astype(str).to_numpy().copy()
    fto = out["fto_s"].to_numpy(dtype=float)
    from_values = out["from_subject"].astype(str).to_numpy()
    to_values = out["to_subject"].astype(str).to_numpy()

    # the floor came back when the next transition hands it to whoever just lost it
    returned = np.zeros(len(out), dtype=bool)
    returned[:-1] = to_values[1:] == from_values[:-1]

    is_overlap = kind == "overlap"
    long_enough = is_overlap & (-fto > interruption_s)
    kind[long_enough] = "interruption"
    kind[is_overlap & ~long_enough & returned] = "backchannel"

    out["transition_type"] = kind
    return out


def floor_transfer_efficiency(transitions: pd.DataFrame, cfg: TurnsSection) -> dict[str, float]:
    is_switch = _switch_mask(transitions)
    fto = (
        transitions["fto_s"].to_numpy(dtype=float)
        if not transitions.empty
        else np.empty(0, dtype=float)
    )
    switches = fto[is_switch]
    holds = fto[~is_switch]
    gaps = switches[switches >= 0.0]
    n_switches = len(switches)

    # overlaps make the mean sit near zero, so the ratio is only meaningful on gaps
    gap_mean = float(gaps.mean()) if gaps.size else np.nan
    gap_cv = (
        float(np.std(gaps, ddof=1)) / gap_mean if gaps.size > 1 and gap_mean > 0 else np.nan
    )

    return {
        "n_transitions": float(len(transitions)),
        "n_switches": float(n_switches),
        "n_holds": float(len(holds)),
        "median_fto_s": float(np.median(switches)) if n_switches else np.nan,
        "sd_fto_s": float(np.std(switches, ddof=1)) if n_switches > 1 else np.nan,
        "gap_cv": gap_cv,
        "smooth_transfer_ratio": (
            float(np.mean(np.abs(switches) <= cfg.smooth_transfer_gap_s))
            if n_switches
            else np.nan
        ),
        "gap_ratio": float(gaps.size) / n_switches if n_switches else np.nan,
        "overlap_ratio": 1.0 - float(gaps.size) / n_switches if n_switches else np.nan,
        "median_hold_s": float(np.median(holds)) if holds.size else np.nan,
    }


def response_latency(transitions: pd.DataFrame) -> pd.DataFrame:
    """Per direction, how long the responder waited. Overlaps drop out: somebody who started
    before the other finished was not waiting at all."""
    columns = ["from_subject", "to_subject", "n", "median_latency_s", "mean_latency_s"]
    if transitions.empty:
        return pd.DataFrame({column: [] for column in columns})

    is_gap = transitions["fto_s"].to_numpy(dtype=float) >= 0.0
    gaps = transitions[_switch_mask(transitions) & is_gap]
    if gaps.empty:
        return pd.DataFrame({column: [] for column in columns})

    out = (
        gaps.groupby(["from_subject", "to_subject"], observed=True)["fto_s"]
        .agg(["size", "median", "mean"])
        .reset_index()
    )
    out.columns = columns
    return out


def transition_matrix(turns: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """N x N, not the hard-coded AA/AB/BA/BB four-state version."""
    matrix = pd.DataFrame(
        np.zeros((len(subject_ids), len(subject_ids)), dtype=np.int64),
        index=pd.Index(list(subject_ids), name="from_subject"),
        columns=pd.Index(list(subject_ids), name="to_subject"),
    )
    if len(turns) < 2:
        return matrix

    speakers = turns.sort_values("t_start", kind="stable")["subject_id"].astype(str).to_numpy()
    for source, target in zip(speakers, speakers[1:]):
        if source in matrix.index and target in matrix.columns:
            matrix.loc[source, target] += 1
    return matrix


def _switch_mask(transitions: pd.DataFrame) -> np.ndarray:
    if transitions.empty:
        return np.zeros(0, dtype=bool)
    return transitions["is_switch"].astype(bool).to_numpy()


def read_transitions(path: Path) -> pd.DataFrame:
    return _io.read_file(Path(path), TRANSITION_SCHEMA, "transitions")


def load_transitions(config, recording_id: str) -> pd.DataFrame:
    root = layout_for(config, create=False).artifacts_dir(
        recording_id, TransitionExtractor.name
    )
    return read_transitions(root / TRANSITIONS_NAME)


_EVENT_SPECS = {
    "fto_s": (
        "Floor transfer offset: seconds between one speaker stopping and subject_id starting. "
        "Positive is a gap, negative an overlap. subject_id_b held the floor before.",
        "s",
        "mean",
    ),
    "response_latency_s": (
        "The same offset restricted to gaps, so it reads as how long subject_id waited. "
        "Absent on overlaps.",
        "s",
        "mean",
    ),
}

_SUBJECT_SPECS = {
    "n_turns_taken_from_other": ("Times subject_id took the floor from somebody else.", "count", "sum"),
    "median_response_latency_s": (
        "Median wait before subject_id picks up, gaps only.",
        "s",
        "mean",
    ),
    "mean_fto_s": ("Mean floor transfer offset before subject_id picks up.", "s", "mean"),
    "n_interruptions_made": (
        "Times subject_id took the floor with an overlap longer than interruption_overlap_s.",
        "count",
        "sum",
    ),
    "n_backchannels_made": (
        "Short overlaps by subject_id after which the floor went straight back.",
        "count",
        "sum",
    ),
    "interruptions_per_minute": ("Interruptions by subject_id per minute of the task window.", "1/min", "mean"),
}

_PAIR_SPECS = {
    "transition_count": (
        "Turns where subject_id handed the floor to subject_id_b. The two are equal on the "
        "diagonal, where the same speaker restarted after a long pause.",
        "count",
        "sum",
    ),
    "pair_median_fto_s": (
        "Median floor transfer offset for that same direction.",
        "s",
        "mean",
    ),
}

_RECORDING_SPECS = {
    "n_transitions": ("Adjacent turn pairs, switches and holds together.", "count", "sum"),
    "n_switches": ("Adjacent turn pairs where the floor changed hands.", "count", "sum"),
    "n_holds": ("Adjacent turn pairs where the same speaker restarted.", "count", "sum"),
    "median_fto_s": ("Median floor transfer offset across all switches.", "s", "mean"),
    "sd_fto_s": (
        "Spread of the offsets with their signs kept. Emitted once for the whole recording "
        "and once per subject_id over the transfers that speaker received.",
        "s",
        "mean",
    ),
    "gap_cv": (
        "Coefficient of variation of the positive gaps. Overlaps are left out because a mean "
        "near zero makes the ratio meaningless.",
        None,
        "mean",
    ),
    "smooth_transfer_ratio": (
        "Share of switches within smooth_transfer_gap_s of the previous turn ending.",
        None,
        "mean",
    ),
    "gap_ratio": ("Share of switches that left a gap.", None, "mean"),
    "overlap_ratio": ("Share of switches that overlapped.", None, "mean"),
    "median_hold_s": ("Median pause when a speaker restarted instead of handing over.", "s", "mean"),
}


@register_extractor
class TransitionExtractor(Extractor):
    name = "turns.transitions"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "turns"
    config_sections = ("turns", "text.diarization", "text.extra_voices")
    requires = (TurnStructureExtractor.name, MediaPrepExtractor.name)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        root = layout_for(self.config, create=False).artifacts_dir(
            recording.recording_id, TurnStructureExtractor.name
        )
        return [root / TURNS_NAME]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.turns
        turns = load_turns(self.config, recording.recording_id, others=True)
        prep = load_prep(self.config, recording.recording_id)

        transitions = classify_overlaps(
            detect_transitions(turns, cfg), interruption_s=cfg.interruption_overlap_s
        )
        transitions["recording_id"] = recording.recording_id

        artifacts = {
            TRANSITIONS_NAME: _io.write_file(
                transitions, workspace / TRANSITIONS_NAME, TRANSITION_SCHEMA, "transitions"
            )
        }
        return ExtractorResult(
            rows=self._rows(recording, turns, transitions, prep),
            specs=self._specs(),
            artifacts=artifacts,
            warnings=self._warnings(recording, transitions, prep),
        )

    def _rows(self, recording: Recording, turns, transitions, prep) -> pd.DataFrame:
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        minutes = prep.duration_s / 60.0
        subject_ids = turn_takers(self.config, recording)
        switches = transitions[_switch_mask(transitions)]
        frames = []

        for (giver, taker), group in switches.groupby(
            ["from_subject", "to_subject"], observed=True
        ):
            fto = group["fto_s"].to_numpy(dtype=float)
            prev_end = group["prev_end"].to_numpy(dtype=float)
            next_start = group["next_start"].to_numpy(dtype=float)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            # an overlap runs backwards in time, so the span is the crossing
                            # itself and the sign lives in the value
                            "t_start": np.minimum(prev_end, next_start),
                            "t_end": np.maximum(prev_end, next_start),
                            "fto_s": fto,
                            "response_latency_s": np.where(fto >= 0.0, fto, np.nan),
                        }
                    ),
                    scale=Scale.EVENT,
                    subject_id=str(taker),
                    subject_id_b=str(giver),
                    **common,
                )
            )

        for taker, group in switches.groupby("to_subject", observed=True):
            fto = group["fto_s"].to_numpy(dtype=float)
            gaps = fto[fto >= 0.0]
            kinds = group["transition_type"].astype(str).to_numpy()
            interruptions = float((kinds == "interruption").sum())
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [window[0]],
                            "t_end": [window[1]],
                            "n_turns_taken_from_other": [float(len(group))],
                            "median_response_latency_s": [
                                float(np.median(gaps)) if gaps.size else np.nan
                            ],
                            "mean_fto_s": [float(fto.mean())],
                            "sd_fto_s": [
                                float(np.std(fto, ddof=1)) if fto.size > 1 else np.nan
                            ],
                            "n_interruptions_made": [interruptions],
                            "n_backchannels_made": [float((kinds == "backchannel").sum())],
                            "interruptions_per_minute": [
                                interruptions / minutes if minutes > 0 else np.nan
                            ],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=str(taker),
                    **common,
                )
            )

        matrix = transition_matrix(turns, subject_ids)
        pair_median = {
            (str(giver), str(taker)): float(value)
            for (giver, taker), value in switches.groupby(
                ["from_subject", "to_subject"], observed=True
            )["fto_s"]
            .median()
            .items()
        }
        for giver in matrix.index:
            for taker in matrix.columns:
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame(
                            {
                                "t_start": [window[0]],
                                "t_end": [window[1]],
                                "transition_count": [float(matrix.at[giver, taker])],
                                "pair_median_fto_s": [
                                    pair_median.get((str(giver), str(taker)), np.nan)
                                ],
                            }
                        ),
                        scale=Scale.RECORDING,
                        subject_id=str(giver),
                        subject_id_b=str(taker),
                        **common,
                    )
                )

        summary = floor_transfer_efficiency(transitions, self.config.turns)
        frames.append(
            build_long_rows(
                values=pd.DataFrame(
                    {
                        "t_start": [window[0]],
                        "t_end": [window[1]],
                        **{feature: [value] for feature, value in summary.items()},
                    }
                ),
                scale=Scale.RECORDING,
                **common,
            )
        )
        return concat_long_tables(frames)

    def _warnings(self, recording: Recording, transitions, prep) -> list[QCWarning]:
        warnings = []
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        switches = transitions[_switch_mask(transitions)]

        if len(switches) < _MIN_SWITCHES:
            warnings.append(
                QCWarning(
                    check="transitions_too_few",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"Only {len(switches)} floor transfers in the task window. Transition "
                        "statistics from this recording are not stable enough to interpret."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )

        responders = set(switches["to_subject"].astype(str)) if not switches.empty else set()
        for subject in recording.subjects:
            if subject.subject_id not in responders:
                warnings.append(
                    QCWarning(
                        check="transitions_never_responds",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message="This speaker never took the floor from anybody else.",
                        t_start=window[0],
                        t_end=window[1],
                    )
                )

        overlap_ratio = floor_transfer_efficiency(transitions, self.config.turns)["overlap_ratio"]
        if pd.notna(overlap_ratio) and overlap_ratio > _OVERLAP_WARN:
            warnings.append(
                QCWarning(
                    check="transitions_overlap_heavy",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="warn",
                    message=(
                        f"{overlap_ratio:.0%} of floor transfers overlap. Check that crosstalk "
                        "suppression removed the leaked copies of each utterance."
                    ),
                    t_start=window[0],
                    t_end=window[1],
                )
            )
        return warnings

    def _specs(self) -> list[FeatureSpec]:
        cfg = self.config.turns
        params = {
            "max_gap_s": cfg.max_gap_s,
            "interruption_overlap_s": cfg.interruption_overlap_s,
            "smooth_transfer_gap_s": cfg.smooth_transfer_gap_s,
        }
        groups = (
            (_EVENT_SPECS, Scale.EVENT),
            (_SUBJECT_SPECS, Scale.RECORDING),
            (_PAIR_SPECS, Scale.RECORDING),
            (_RECORDING_SPECS, Scale.RECORDING),
        )
        return [
            FeatureSpec(
                name=feature,
                description=description,
                unit=unit,
                scale=scale,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params=params,
                aggregation=aggregation,
                validation="none",
            )
            for specs, scale in groups
            for feature, (description, unit, aggregation) in specs.items()
        ]
