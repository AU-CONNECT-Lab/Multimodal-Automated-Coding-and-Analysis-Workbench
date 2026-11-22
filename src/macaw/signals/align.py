"""Each signal file put on the recording's clock, by its person's IMU where there is one, and checked by a person."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import PERSON, SOFTWARE, Extractor, ExtractorResult, Gate, Phase, QCWarning
from macaw.media.align import MANUAL, UNALIGNED, AlignResult, read_offset_table, write_offset_table
from macaw.media.drift import drift_is_real
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.signals.imu import AGREEMENT_S, FACE_ANGLES, POSE_ANGLES, gyro_speed, head_speed, match
from macaw.signals.read import SignalReadExtractor, aux_name, info_name, load_aux, load_info, read_dir
from macaw.signals.table import EVERYONE, SignalFile, read_signals, signal_files
from macaw.store.longtable import read_long_table
from macaw.store.schema import Scale

IMU = "imu"
# what every signal file's offset is measured against
RECORDING_CLOCK = "recording"
COMPUTED_NAME = "offsets.json"
# each person's figure of the match, beside the table
FIGURE_SUFFIX = "_imu.png"
# the least head movement on the video, in s of frames with angles, worth matching against
MIN_HEAD_S = 10.0
# s; markers further apart than this in two files of one person say the files do not share a clock
MARKER_TOLERANCE_S = 0.1
# (step, angles, what the angles are) in the order they are tried
HEAD_SOURCES = (
    ("face.au", FACE_ANGLES, "the face's angles"),
    ("pose.kinematics", POSE_ANGLES, "the head's tilt from body points"),
)


def computed_path(config, recording_id: str) -> Path:
    return layout_for(config, create=False).artifacts_dir(recording_id, SignalAlignExtractor.name) / COMPUTED_NAME


def computed_offsets(config, recording_id: str) -> list[AlignResult]:
    path = computed_path(config, recording_id)
    return read_offset_table(path) if path.is_file() else []


def decided_offsets(config, recording_id: str) -> list[AlignResult]:
    path = layout_for(config, create=False).signal_offsets_path(recording_id)
    return read_offset_table(path) if path.is_file() else []


def signal_offsets(config, recording_id: str) -> dict[str, AlignResult]:
    """Each file's offset as it stands: one a person set, else the one found, kept by a person while it has not changed."""
    found = {result.target: result for result in computed_offsets(config, recording_id)}
    for decision in decided_offsets(config, recording_id):
        current = found.get(decision.target)
        if decision.method == MANUAL:
            found[decision.target] = decision
        elif decision.accepted and current is not None and same_result(current, decision):
            current.accepted = decision.accepted
    return found


def same_result(a: AlignResult, b: AlignResult) -> bool:
    return a.method == b.method and (a.method == UNALIGNED or abs(a.offset_s - b.offset_s) <= AGREEMENT_S)


def drift_corrected(result: AlignResult) -> bool:
    return result.method == IMU and result.drift_intercept_s is not None and drift_is_real(
        result.drift_ppm, result.drift_r2, result.overlap_s, AGREEMENT_S
    )


def signal_clock(result: AlignResult) -> tuple[float, float]:
    """(offset_s, clock_scale) with recording time = offset_s + the file's own time x clock_scale."""
    if not drift_corrected(result):
        return result.offset_s, 1.0
    scale = 1.0 / (1.0 - result.drift_ppm * 1e-6)
    return result.drift_intercept_s * scale, scale


def review_reason(result: AlignResult, cfg) -> str | None:
    """Why a person should look at this file's offset; None for one a person set or kept."""
    if result.method == MANUAL or result.accepted:
        return None
    if result.method == UNALIGNED:
        return f"not aligned: {result.note}" if result.note else "not aligned"
    if result.confidence is not None and result.confidence < cfg.min_confidence and (
        result.agreement is None or result.agreement < cfg.min_agreement
    ):
        return "low confidence"
    if drift_corrected(result):
        return "drift, corrected along the fitted line"
    return None


def needs_person(result: AlignResult, cfg) -> str | None:
    reason = review_reason(result, cfg)
    if reason is None and cfg.review == "always" and not (result.method == MANUAL or result.accepted):
        return "not looked at yet"
    return reason


def describe(result: AlignResult) -> str:
    if result.method == UNALIGNED:
        return f"not aligned ({result.note})" if result.note else "not aligned"
    offset, scale = signal_clock(result)
    text = f"{offset:+.2f} s"
    if result.method == MANUAL:
        return text + f", {result.note or 'set by hand'}"
    text += f", confidence {result.confidence:.2f}" if result.confidence is not None else ""
    if result.agreement is not None:
        total = len(result.segment_offsets or ())
        text += f", same offset in {round(result.agreement * total)} of {total} stretches"
    if drift_corrected(result):
        text += f", drift {result.drift_ppm:+.0f} ppm corrected"
    if result.metadata_offset_s is not None:
        text += f", start times say {result.metadata_offset_s:+.1f} s"
    return text + (f"; {result.note}" if result.note else "")


def timestamp_offset(signal_started: str | None, video_started: str | None, video_offset_s: float = 0.0) -> float | None:
    """Recording time of the file's first moment by the two devices' own clocks, both read as written; a hint, never applied."""
    if not signal_started or not video_started:
        return None
    try:
        signal = datetime.fromisoformat(signal_started.replace("Z", "+00:00")).replace(tzinfo=None)
        video = datetime.fromisoformat(video_started.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
    return (signal - video).total_seconds() + video_offset_s


def whole_hours(offset_s: float | None) -> int | None:
    """The hours an offset is off by when it is a whole number of hours give or take a minute, as a time zone written wrong is."""
    if offset_s is None:
        return None
    hours = round(offset_s / 3600)
    return hours if hours and abs(offset_s - hours * 3600) <= 60 else None


def video_started(recording: Recording) -> tuple[str | None, float]:
    """The creation time a video of the recording carries, as its device wrote it, and that video's offset on the recording clock."""
    from macaw.media.probe import probe

    for item in recording.media:
        if item.kind != "video":
            continue
        try:
            info = probe(item.path)
        except MacawError:
            continue
        if info.creation_time:
            return info.creation_time, item.offset_s
    return None, 0.0


def head_turning(config, recording_id: str, person: str) -> tuple[np.ndarray, np.ndarray, str] | None:
    """(recording times, deg/s, what the angles are) of the person's head on the video, from the first step that has
    enough of it; None when neither has."""
    features = layout_for(config, create=False).long_table_path(recording_id)
    for step, angles, said in HEAD_SOURCES:
        path = features / f"{step}.parquet"
        if not path.is_file():
            continue
        rows = read_long_table(path, subject_id=person, scale=Scale.FRAME, features=list(angles))
        speed = head_speed(rows, angles) if not rows.empty else None
        if speed is None:
            continue
        period = float(np.median(np.diff(speed[0]))) if speed[0].size > 1 else 0.0
        if np.isfinite(speed[1]).sum() * period >= MIN_HEAD_S:
            return speed[0], speed[1], said
    return None


@register_extractor
class SignalAlignExtractor(Extractor):
    name = "signals.align"
    version = "0.3.0"
    scale = Scale.RECORDING
    group = "signals"
    phase = Phase.DERIVE
    config_sections = ("signals",)
    requires = (SignalReadExtractor.name,)
    runs_after = tuple(step for step, _, _ in HEAD_SOURCES)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(read_signals(config))

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        rid = recording.recording_id
        folder = read_dir(self.config, rid)
        paths = [folder / name for item in signal_files(self.config, rid) for name in (aux_name(item), info_name(item))]
        features = layout_for(self.config, create=False).long_table_path(rid)
        paths += [features / f"{step}.parquet" for step, _, _ in HEAD_SOURCES]
        return [path for path in paths if path.is_file()]

    def extra_key(self, recording: Recording) -> str:
        return json.dumps([[item.person, item.stage, item.path.name] for item in signal_files(self.config, recording.recording_id)])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        rid = recording.recording_id
        started, video_offset = video_started(recording)
        people: dict[str | None, list[SignalFile]] = {}
        for item in signal_files(self.config, rid):
            people.setdefault(item.person, []).append(item)
        results, warnings = [], []
        artifacts = {}
        for person, items in people.items():
            found, more = self._person(rid, person, items, started, video_offset, workspace)
            results.extend(found)
            warnings.extend(more)
        artifacts.update({figure.name: figure for figure in workspace.glob(f"*{FIGURE_SUFFIX}")})
        path = workspace / COMPUTED_NAME
        write_offset_table(results, path)
        warnings.extend(self._notes(rid, results))
        return ExtractorResult(rows=pd.DataFrame(), artifacts={COMPUTED_NAME: path, **artifacts}, warnings=warnings)

    def _person(self, rid, person, items, started, video_offset, workspace: Path) -> tuple[list[AlignResult], list[QCWarning]]:
        cfg = self.config.signals

        def unaligned(note: str) -> list[AlignResult]:
            hint = timestamp_offset(load_info(self.config, rid, items[0]).get("started"), started, video_offset)
            return [AlignResult(reference=RECORDING_CLOCK, target=item.key, offset_s=0.0, confidence=None, method=UNALIGNED,
                                metadata_offset_s=hint, note=note) for item in items]

        if person is None:
            return unaligned("a file of the whole recording has no head to match; set its offset by hand"), []
        anchor = imu = None
        for item in items:
            imu = gyro_speed(load_aux(self.config, rid, item), cfg.gyro)
            if imu is not None:
                anchor = item
                break
        if anchor is None:
            return unaligned(f"no channel named like {cfg.gyro!r} in {person}'s files; set the offset by hand"), []
        head = head_turning(self.config, rid, person)
        if head is None:
            return unaligned(f"no head angles on the video for {person}: run face.au or pose.kinematics, or set the offset by hand"), []
        info = load_info(self.config, rid, anchor)
        hint = timestamp_offset(info.get("started"), started, video_offset)
        found = match(head[0], head[1], imu[0], imu[1], hint_s=hint, window_s=cfg.search_window_s,
                      segment_s=cfg.segment_s, min_confidence=cfg.min_confidence)
        if found is None:
            return unaligned("the head on the video and the IMU never overlap long enough to match"), []
        results, warnings = [], []
        for item in items:
            results.append(AlignResult(
                reference=RECORDING_CLOCK, target=item.key, offset_s=found.offset_s, confidence=found.confidence,
                agreement=found.agreement, overlap_s=found.overlap_s, drift_ppm=found.drift_ppm, drift_r2=found.drift_r2,
                drift_intercept_s=found.drift_intercept_s, segment_offsets=found.segments, method=IMU, metadata_offset_s=hint,
                note=f"matched by {head[2]}" if item is anchor else f"same clock as {anchor.key}, matched by {head[2]}",
            ))
            if item is not anchor and not _same_markers(info, load_info(self.config, rid, item)):
                warnings.append(QCWarning(
                    check=self.name, recording_id=rid, subject_id=person, severity="warn",
                    message=f"{item.key} has its markers at other times than {anchor.key}, so it may not share its clock "
                            "(a cut copy?); set its offset by hand if so",
                ))
        try:
            from macaw.qc.figures import imu_figure

            imu_figure(head[:2], imu, results[items.index(anchor)], workspace / f"{person}{FIGURE_SUFFIX}")
        except BackendNotAvailableError:   # without matplotlib the table and the page still show the match
            pass
        return results, warnings

    def _notes(self, rid: str, results: list[AlignResult]) -> list[QCWarning]:
        out = []
        for result in results:
            person = result.target.rsplit("_", 1)[0]
            severity = "warn" if review_reason(result, self.config.signals) else "info"
            message = f"{result.target}: {describe(result)}"
            hours = whole_hours(result.metadata_offset_s if result.method != UNALIGNED else None)
            if hours is not None and abs(result.offset_s - result.metadata_offset_s) > 1800:
                message += f"; the start times differ by a whole {hours:+d} h more than the match, as a time zone written wrong would"
            out.append(QCWarning(check=self.name, recording_id=rid, subject_id=None if person == EVERYONE else person,
                                 severity=severity, message=message))
        return out


def _same_markers(first: dict, second: dict) -> bool:
    """Whether two files of one person carry their shared markers at the same times; True when there is nothing to compare."""
    def onsets(info: dict) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for marker in info.get("markers", []):
            out.setdefault(marker["name"], []).append(float(marker["onset"]))
        return out

    a, b = onsets(first), onsets(second)
    shared = [name for name in a if name in b and len(a[name]) == len(b[name])]
    return all(np.allclose(a[name], b[name], atol=MARKER_TOLERANCE_S) for name in shared)


@register_extractor
class SignalReviewGate(Gate):
    """Signal offsets a person has to look at before the signals are used: not found, in doubt, or not looked at yet."""

    name = "signals.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "signals"
    config_sections = ("signals",)
    requires = (SignalAlignExtractor.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(read_signals(config))

    def inputs_for(self, recording: Recording) -> list[Path]:
        rid = recording.recording_id
        paths = [computed_path(self.config, rid), layout_for(self.config, create=False).signal_offsets_path(rid)]
        return [path for path in paths if path.is_file()]

    def pending(self, recording: Recording) -> str | None:
        cfg, rid = self.config.signals, recording.recording_id
        files = signal_files(self.config, rid)
        if cfg.review == "skip" or not files:
            return None
        current = signal_offsets(self.config, rid)
        waiting = []
        for item in files:
            result = current.get(item.key)
            reason = "not aligned yet" if result is None else needs_person(result, cfg)
            if reason is not None:
                waiting.append(f"{item.key} ({reason})")
        if not waiting:
            return None
        return (
            f"{len(waiting)} signal file(s) to look at: {'; '.join(waiting)}. Look at them on the Signals page and keep or set "
            f"each offset there, or with `macaw signal-offsets -r {rid}` and `macaw accept-signal-offset -r {rid}`"
        )

    def decided_by(self, recording: Recording) -> str | None:
        rid = recording.recording_id
        if not signal_files(self.config, rid):
            return None
        current = signal_offsets(self.config, rid).values()
        return PERSON if any(result.method == MANUAL or result.accepted for result in current) else SOFTWARE
