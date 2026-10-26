"""The quality checks. This is the part no existing tool does.

A check returns warnings, which are statements, and candidates, which are places worth looking
at. Candidates are what the overlay renderer and any reviewer read, so a check that cannot point
at a moment is only half a check."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from macaw.config.schema import MacawConfig
from macaw.exceptions import ConfigError
from macaw.extractors.base import QCWarning
from macaw.extractors.video.motion.mea import PICTURE_NAME, WHOLE_PICTURE_CHANGE
from macaw.extractors.video.pose.skeleton import SKELETON_EDGES, STANDARD_JOINTS
from macaw.project.recording import Recording
from macaw.store import _io
from macaw.store import keypoints as keypoints_store
from macaw.store import longtable as longtable_store
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, coerce_frame, empty_frame

POSE_SOURCE = "pose.keypoints"
PROSODY_SOURCE = "prosody.frame"
MEA_SOURCE = "motion.mea"

CANDIDATE_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",   # null when the moment involves no one person, e.g. a track swap
    "check": "string",
    "reason": "string",
    "t_start": "float64",     # seconds, recording clock
    "t_end": "float64",
    "t": "float64",           # the single instant to look at, the worst one in the span
    "score": "float64",       # larger is more suspicious; comparable within one reason only
}

# confidence is judged against the person's own distribution, because what counts as low
# differs by backend, the lesson of the backend-blind ASR threshold
_CONFIDENCE_MADS = 4.0
# a segment cannot be longer than its own upper envelope by this much without the keypoints
# being wrong; shorter is just foreshortening
_BONE_STRETCH = 1.5
_BONE_REFERENCE_QUANTILE = 0.95
_MISSING_GAP_S = 0.5
# 0.8 octave between two 10 ms frames is beyond any voice, so it is the tracker jumping
_OCTAVE_JUMP = 0.8
_OCTAVE_WARN_FRACTION = 0.01
# an identical picture for this long is a stalled recording, not a still room
_FROZEN_S = 1.0
# nothing at all moving in a person's region for this long: they left it, or it is drawn on a wall
_STILL_S = 10.0

_CHECKS: dict[str, Callable] = {}


@dataclass
class CheckResult:
    warnings: list[QCWarning] = field(default_factory=list)
    candidates: pd.DataFrame = field(default_factory=lambda: empty_frame(CANDIDATE_SCHEMA))
    skipped: str | None = None    # why this check could not run on this recording


class RecordingResults:
    """What one recording's run left on disk, read on first use so a check pays only for what
    it reads. A missing table reads as empty, which a check reports as skipped."""

    def __init__(self, layout, recording_id: str):
        self.layout = layout
        self.recording_id = recording_id

    def long_table(self, source: str, features: Sequence[str] | None = None) -> pd.DataFrame:
        path = self.layout.long_table_path(self.recording_id) / f"{source}.parquet"
        if not path.is_file():
            return empty_frame(LONG_TABLE_SCHEMA)
        return longtable_store.read_long_table(path, recording_id=self.recording_id, features=features)

    def keypoints(self) -> pd.DataFrame:
        path = self.layout.keypoints_path(self.recording_id) / f"{POSE_SOURCE}.parquet"
        if not path.is_file():
            return empty_frame(KEYPOINT_SCHEMA)
        return keypoints_store.read_keypoints(path, recording_id=self.recording_id)

    def artifact(self, extractor: str, name: str) -> Path | None:
        path = self.layout.artifacts_dir(self.recording_id, extractor) / name
        return path if path.exists() else None

    def extractor_warnings(self) -> list[QCWarning]:
        directory = self.layout.warnings_path(self.recording_id, "x").parent
        warnings: list[QCWarning] = []
        for path in sorted(directory.glob("*.json")):
            warnings += [QCWarning(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
        return warnings


def register_check(name: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        _CHECKS[name] = function
        return function

    return decorator


def available_checks() -> list[str]:
    return sorted(_CHECKS)


def candidate_spans(
    recording_id: str,
    subject_id: str | None,
    check: str,
    reason: str,
    t: np.ndarray,
    flagged: np.ndarray,
    score: np.ndarray,
) -> pd.DataFrame:
    """Consecutive flagged samples become one candidate, pointing at its worst sample."""
    t, flagged, score = np.asarray(t, float), np.asarray(flagged, bool), np.asarray(score, float)
    if not flagged.any():
        return empty_frame(CANDIDATE_SCHEMA)
    starts = flagged & ~np.concatenate([[False], flagged[:-1]])
    run = np.cumsum(starts)
    rows = []
    for run_id in np.unique(run[flagged]):
        index = np.flatnonzero(flagged & (run == run_id))
        worst = index[np.nanargmax(score[index])] if np.isfinite(score[index]).any() else index[0]
        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "check": check,
                "reason": reason,
                "t_start": t[index[0]],
                "t_end": t[index[-1]],
                "t": t[worst],
                "score": score[worst],
            }
        )
    return coerce_frame(pd.DataFrame(rows), CANDIDATE_SCHEMA, "candidates")


def _concat_candidates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return empty_frame(CANDIDATE_SCHEMA)
    return coerce_frame(pd.concat(frames, ignore_index=True), CANDIDATE_SCHEMA, "candidates")


def _summary(check: str, recording_id: str, candidates: pd.DataFrame) -> list[QCWarning]:
    """One line per reason, so the report shows the check ran and what it pointed at."""
    warnings = []
    for (subject_id, reason), group in candidates.groupby(["subject_id", "reason"], dropna=False):
        subject = None if pd.isna(subject_id) else str(subject_id)
        warnings.append(
            QCWarning(
                check=check,
                recording_id=recording_id,
                subject_id=subject,
                severity="info",
                message=f"{len(group)} span(s) flagged for review: {reason}",
                details={"reason": reason, "n_spans": len(group)},
            )
        )
    return warnings


@register_check("alignment_confidence")
def check_alignment_confidence(recording: Recording, config: MacawConfig, results) -> CheckResult:
    raise NotImplementedError


@register_check("clock_drift")
def check_clock_drift(recording: Recording, config: MacawConfig, results) -> CheckResult:
    raise NotImplementedError


@register_check("transcript_match")
def check_transcript_match(recording: Recording, config: MacawConfig, results) -> CheckResult:
    """Energy-ratio test: does this transcript belong to this recording at all."""
    raise NotImplementedError


@register_check("track_quality")
def check_track_quality(recording: Recording, config: MacawConfig, results) -> CheckResult:
    keypoints = results.keypoints()
    if keypoints.empty:
        return CheckResult(skipped=f"no {POSE_SOURCE} output")

    rid = recording.recording_id
    standard = keypoints[keypoints["joint"].isin(STANDARD_JOINTS)]
    pieces = [_unassigned(rid, standard), _swaps(rid, results)]
    span = (float(keypoints["t"].min()), float(keypoints["t"].max()))
    # frame numbers are each camera's own, so each person is checked per camera they are on
    for (subject_id, _), person in standard.groupby(["subject_id", "video"], dropna=True, observed=True):
        subject_id = str(subject_id)
        pieces += [
            _missing(rid, subject_id, person, span),
            _low_confidence(rid, subject_id, person),
            _bone_stretch(rid, subject_id, person),
        ]
    candidates = _concat_candidates(pieces)
    return CheckResult(warnings=_summary("track_quality", rid, candidates), candidates=candidates)


def _unassigned(rid: str, keypoints: pd.DataFrame) -> pd.DataFrame:
    """A detection no subject was matched to: a third person, a poster, a coat on a chair."""
    pieces = []
    for _, view in keypoints.groupby("video", observed=True):
        per_frame = (
            view.assign(stray=view["subject_id"].isna()).groupby("frame").agg(t=("t", "first"), stray=("stray", "any"))
        )
        pieces.append(
            candidate_spans(
                rid, None, "track_quality", "unassigned_detection",
                per_frame["t"], per_frame["stray"], np.ones(len(per_frame)),
            )
        )
    return _concat_candidates(pieces)


def _swaps(rid: str, results) -> pd.DataFrame:
    path = results.artifact(POSE_SOURCE, "swaps.json")
    if path is None:
        return empty_frame(CANDIDATE_SCHEMA)
    events = [event for file_events in json.loads(path.read_text(encoding="utf-8")).values() for event in file_events]
    rows = [
        {
            "recording_id": rid,
            "subject_id": None,
            "check": "track_quality",
            "reason": "track_swap",
            "t_start": event["t"],
            "t_end": event["t"],
            "t": event["t"],
            "score": event["score"],
        }
        for event in events
    ]
    return coerce_frame(pd.DataFrame(rows), CANDIDATE_SCHEMA, "candidates") if rows else empty_frame(CANDIDATE_SCHEMA)


def _missing(rid: str, subject_id: str, person: pd.DataFrame, span: tuple[float, float]) -> pd.DataFrame:
    """Absences long enough to look at: the person left, or is there and was not detected."""
    seen = np.sort(person.loc[person["x"].notna(), "t"].unique())
    edges = np.concatenate([[span[0]], seen, [span[1]]]) if len(seen) else np.array(span)
    gaps = np.diff(edges)
    rows = [
        {
            "recording_id": rid,
            "subject_id": subject_id,
            "check": "track_quality",
            "reason": "missing_subject",
            "t_start": edges[index],
            "t_end": edges[index + 1],
            "t": (edges[index] + edges[index + 1]) / 2.0,
            "score": gaps[index],
        }
        for index in np.flatnonzero(gaps > _MISSING_GAP_S)
    ]
    return coerce_frame(pd.DataFrame(rows), CANDIDATE_SCHEMA, "candidates") if rows else empty_frame(CANDIDATE_SCHEMA)


def _low_confidence(rid: str, subject_id: str, person: pd.DataFrame) -> pd.DataFrame:
    per_frame = person.groupby("frame").agg(t=("t", "first"), confidence=("confidence", "mean"))
    confidence = per_frame["confidence"].to_numpy(dtype=float)
    centre = np.nanmedian(confidence)
    spread = 1.4826 * np.nanmedian(np.abs(confidence - centre))
    if not spread > 0:
        return empty_frame(CANDIDATE_SCHEMA)
    depth = (centre - confidence) / spread
    return candidate_spans(
        rid, subject_id, "track_quality", "low_confidence",
        per_frame["t"], depth > _CONFIDENCE_MADS, depth,
    )


def _bone_stretch(rid: str, subject_id: str, person: pd.DataFrame) -> pd.DataFrame:
    wide = person.pivot(index="frame", columns="joint", values=["x", "y"])
    times = person.groupby("frame")["t"].first().reindex(wide.index)
    worst = np.zeros(len(wide))
    for a, b in SKELETON_EDGES:
        if ("x", a) not in wide or ("x", b) not in wide:
            continue
        length = np.hypot(wide[("x", a)] - wide[("x", b)], wide[("y", a)] - wide[("y", b)]).to_numpy()
        if not np.isfinite(length).any():
            continue
        reference = np.nanquantile(length, _BONE_REFERENCE_QUANTILE)
        if reference > 0:
            worst = np.fmax(worst, length / reference)
    return candidate_spans(
        rid, subject_id, "track_quality", "bone_stretch", times, worst > _BONE_STRETCH, worst
    )


@register_check("au_child_reliability")
def check_au_child_reliability(recording: Recording, config: MacawConfig, results) -> CheckResult:
    """AU models degrade badly on children; always warn, never silently pass."""
    raise NotImplementedError


@register_check("octave_error")
def check_octave_error(recording: Recording, config: MacawConfig, results) -> CheckResult:
    """Doubled or halved f0 relative to the speaker's own distribution."""
    f0 = results.long_table(PROSODY_SOURCE, features=["f0_hz"])
    if f0.empty:
        return CheckResult(skipped=f"no {PROSODY_SOURCE} output")

    rid = recording.recording_id
    pieces, warnings = [], []
    for subject_id, contour in f0.groupby("subject_id", dropna=True, observed=True):
        contour = contour.sort_values("t_start")
        spans, n_error, n_voiced = octave_jumps(
            contour["t_start"].to_numpy(dtype=float), contour["value"].to_numpy(dtype=float)
        )
        if not n_error:
            continue
        pieces.append(
            coerce_frame(
                pd.DataFrame(
                    [
                        {
                            "recording_id": rid,
                            "subject_id": str(subject_id),
                            "check": "octave_error",
                            "reason": "octave_jump",
                            **span,
                        }
                        for span in spans
                    ]
                ),
                CANDIDATE_SCHEMA,
                "candidates",
            )
        )
        fraction = n_error / n_voiced
        warnings.append(
            QCWarning(
                check="octave_error",
                recording_id=rid,
                subject_id=str(subject_id),
                severity="warn" if fraction > _OCTAVE_WARN_FRACTION else "info",
                message=f"{fraction:.1%} of voiced frames sit an octave away after a jump",
                details={"n_frames": n_error, "n_voiced": n_voiced},
            )
        )
    return CheckResult(warnings=warnings, candidates=_concat_candidates(pieces))


def octave_jumps(t: np.ndarray, f0_hz: np.ndarray) -> tuple[list[dict], int, int]:
    """(spans, error frames, voiced frames). A stretch counts only when a jump leads into or out
    of it: a whole utterance pitched high is a voice, not a tracker error."""
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    if voiced.sum() < 2:
        return [], 0, int(voiced.sum())
    pitch = np.where(voiced, np.log2(np.where(voiced, f0_hz, 1.0)), np.nan)
    step = float(np.median(np.diff(t)))
    reference = float(np.nanmedian(pitch))

    joined = voiced[1:] & voiced[:-1] & (np.diff(t) <= 1.5 * step)
    jump = joined & (np.abs(np.diff(pitch)) >= _OCTAVE_JUMP)
    starts = np.ones(len(t), dtype=bool)
    starts[1:] = ~joined | jump
    segment = np.cumsum(starts)

    spans, n_error = [], 0
    for segment_id in np.unique(segment[voiced]):
        index = np.flatnonzero((segment == segment_id) & voiced)
        first, last = index[0], index[-1]
        led_in = first > 0 and jump[first - 1]
        led_out = last < len(t) - 1 and jump[last]
        deviation = float(np.median(pitch[index]) - reference)
        if (led_in or led_out) and abs(deviation) >= _OCTAVE_JUMP:
            n_error += len(index)
            spans.append(
                {
                    "t_start": t[first],
                    "t_end": t[last],
                    "t": t[index[len(index) // 2]],
                    "score": abs(deviation),
                }
            )
    return spans, n_error, int(voiced.sum())


@register_check("motion_energy")
def check_motion_energy(recording: Recording, config: MacawConfig, results) -> CheckResult:
    """Moments where frame differencing measured something other than a person moving."""
    energy = results.long_table(MEA_SOURCE, features=["motion_energy"])
    path = results.artifact(MEA_SOURCE, PICTURE_NAME)
    if energy.empty and path is None:
        return CheckResult(skipped=f"no {MEA_SOURCE} output")

    rid = recording.recording_id
    pieces = []
    if path is not None:
        for _, view in pd.read_parquet(path).groupby("video"):
            view = view.sort_values("t")
            t, share = view["t"].to_numpy(dtype=float), view["whole_picture"].to_numpy(dtype=float)
            pieces.append(candidate_spans(rid, None, "motion_energy", "picture_change", t, share > WHOLE_PICTURE_CHANGE, share))
            # a still room still flickers below the threshold; a stalled recording repeats the picture exactly
            frozen = candidate_spans(rid, None, "motion_energy", "frozen_video", t, view["identical"].to_numpy(dtype=bool), np.ones(t.size))
            pieces.append(_lasting(frozen, _FROZEN_S))
    for subject_id, series in energy.groupby("subject_id", dropna=True, observed=True):
        series = series.sort_values("t_start")
        t, value = series["t_start"].to_numpy(dtype=float), series["value"].to_numpy(dtype=float)
        still = candidate_spans(rid, str(subject_id), "motion_energy", "no_motion", t, value == 0, np.ones(t.size))
        pieces.append(_lasting(still, _STILL_S))
    candidates = _concat_candidates(pieces)
    return CheckResult(warnings=_summary("motion_energy", rid, candidates), candidates=candidates)


def _lasting(candidates: pd.DataFrame, at_least_s: float) -> pd.DataFrame:
    """Spans at least this long, scored by how long they last."""
    kept = candidates[candidates["t_end"] - candidates["t_start"] >= at_least_s].copy()
    kept["score"] = kept["t_end"] - kept["t_start"]
    return kept


def run_checks(
    recording: Recording,
    config: MacawConfig,
    results,
    *,
    only: Sequence[str] | None = None,
) -> CheckResult:
    """The configured checks plus what the extractors already warned about, so one result holds
    everything the report and the reviewer need."""
    names = list(only) if only else list(config.qc.checks)
    unknown = [name for name in names if name not in _CHECKS]
    if unknown:
        raise ConfigError(
            f"Unknown QC check(s): {', '.join(unknown)}. Available: {', '.join(available_checks())}"
        )

    warnings = list(results.extractor_warnings())
    pieces = []
    for name in names:
        try:
            outcome = _CHECKS[name](recording, config, results)
        except NotImplementedError:
            outcome = CheckResult(skipped="not implemented yet")
        if outcome.skipped:
            warnings.append(
                QCWarning(
                    check=name,
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="info",
                    message=f"not run: {outcome.skipped}",
                )
            )
        warnings += outcome.warnings
        pieces.append(outcome.candidates)
    return CheckResult(warnings=warnings, candidates=_concat_candidates(pieces))


def warnings_to_frame(warnings: list[QCWarning]) -> pd.DataFrame:
    columns = [item.name for item in fields(QCWarning)]
    if not warnings:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([asdict(warning) for warning in warnings], columns=columns)
    frame["details"] = frame["details"].map(lambda value: json.dumps(value, ensure_ascii=False, default=str))
    return frame


def write_candidates(candidates: pd.DataFrame, path: Path) -> Path:
    return _io.write_file(candidates, path, CANDIDATE_SCHEMA, "candidates")


def read_candidates(path: Path) -> pd.DataFrame:
    return _io.read_file(Path(path), CANDIDATE_SCHEMA, "candidates")
