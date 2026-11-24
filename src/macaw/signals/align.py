"""Each signal file put on the recording's clock, by its events marked on the video or by its person's IMU, and checked by a person."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ConfigError, MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import PERSON, SOFTWARE, Extractor, ExtractorResult, Gate, Phase, QCWarning
from macaw.media.align import MANUAL, UNALIGNED, AlignResult, read_offset_table, write_offset_table
from macaw.media.drift import drift_is_real
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.signals.imu import AGREEMENT_S, FACE_ANGLES, POSE_ANGLES, gyro_speed, head_speed, match
from macaw.signals.markers import RESIDUAL_S, MarkerMatch, match_markers
from macaw.signals.marks import marks_of, marks_path
from macaw.signals.read import COHERENCE, SNIRF, XDF, SignalReadExtractor, aux_name, info_name, load_aux, load_info, read_dir
from macaw.signals.table import SignalFile, read_signals, signal_files
from macaw.store.longtable import read_long_table
from macaw.store.schema import Scale

IMU = "imu"
MARKERS = "markers"
# a file computed from a member's recording, such as the coherence between two people, on that member's clock
FOLLOWS = "follows"
# what every signal file's offset is measured against
RECORDING_CLOCK = "recording"
TABLE_CLOCK = "table"
# why a file of a kind the IMU does not place is not placed yet
UNPLACED = {
    TABLE_CLOCK: "a table keeps its own clock: mark its events on the video or set its offset by hand",
    XDF: "an XDF file's streams share the file's clock: mark one of its events on the video or set its offset by hand",
}
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
    """Each file's offset as it stands: one a person set, else the one found, kept by a person while it has not changed;
    a file that follows a member's takes that member's offset as it stands."""
    found = {result.target: result for result in computed_offsets(config, recording_id)}
    for decision in decided_offsets(config, recording_id):
        current = found.get(decision.target)
        if decision.method == MANUAL:
            found[decision.target] = decision
        elif decision.accepted and current is not None and same_result(current, decision):
            current.accepted = decision.accepted
    for key, result in list(found.items()):
        if result.method == FOLLOWS:
            found[key] = _following(result, found, config.signals)
    return found


def _following(result: AlignResult, found: dict[str, AlignResult], cfg) -> AlignResult:
    """The file on the clock of the member it follows (reference, offset_s s into that member's file), checked against
    the other members it could follow."""
    member = found.get(result.reference)
    if member is None or member.method in (UNALIGNED, FOLLOWS):
        return AlignResult(reference=RECORDING_CLOCK, target=result.target, offset_s=0.0, confidence=None, method=UNALIGNED,
                           note=f"follows {result.reference}, which is not placed yet")
    offset, scale = signal_clock(member)
    placed = offset + result.offset_s * scale
    doubts = []
    waiting = needs_person(member, cfg)
    if waiting:
        doubts.append(f"follows {result.reference}, which waits: {waiting}")
    for other, into in result.follows or ():
        if other == result.reference or found.get(other) is None or found[other].method in (UNALIGNED, FOLLOWS):
            continue
        other_offset, other_scale = signal_clock(found[other])
        gap = other_offset + into * other_scale - placed
        if abs(gap) > AGREEMENT_S:
            doubts.append(f"{other}'s offset puts it {gap:+.2f} s from where {result.reference}'s does")
    out = AlignResult(reference=result.reference, target=result.target, offset_s=placed, confidence=None, method=FOLLOWS,
                      note=result.note, doubt="; ".join(doubts) or None)
    if scale != 1.0:
        out.drift_ppm, out.drift_r2, out.overlap_s = member.drift_ppm, member.drift_r2, member.overlap_s
        out.drift_intercept_s = placed / scale
    return out


def same_result(a: AlignResult, b: AlignResult) -> bool:
    return a.method == b.method and (a.method == UNALIGNED or abs(a.offset_s - b.offset_s) <= AGREEMENT_S)


def drift_corrected(result: AlignResult) -> bool:
    return result.method in (IMU, MARKERS, FOLLOWS) and result.drift_intercept_s is not None and drift_is_real(
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
    if result.doubt:
        return result.doubt
    if result.method == FOLLOWS:   # its member's offset is the one a person looks at
        return None
    if result.confidence is not None and result.confidence < cfg.min_confidence and (
        result.agreement is None or result.agreement < cfg.min_agreement
    ):
        return "low confidence"
    if drift_corrected(result):
        return "drift, corrected along the fitted line"
    return None


def needs_person(result: AlignResult, cfg) -> str | None:
    reason = review_reason(result, cfg)
    if reason is None and cfg.review == "always" and not (result.method in (MANUAL, FOLLOWS) or result.accepted):
        return "not looked at yet"
    return reason


def describe(result: AlignResult) -> str:
    if result.method == UNALIGNED:
        return f"not aligned ({result.note})" if result.note else "not aligned"
    offset, scale = signal_clock(result)
    text = f"{offset:+.2f} s"
    if result.method == MANUAL:
        return text + f", {result.note or 'set by hand'}"
    if result.method == FOLLOWS:
        return text + f", on {result.reference}'s clock" + (f"; {result.note}" if result.note else "")
    if result.method == MARKERS:
        text += f", from {len(result.segment_offsets or ())} marked event(s)"
    text += f", confidence {result.confidence:.2f}" if result.confidence is not None else ""
    if result.agreement is not None:
        total = len(result.segment_offsets or ())
        text += f", same offset in {round(result.agreement * total)} of {total} stretches"
    if drift_corrected(result):
        text += f", drift {result.drift_ppm:+.0f} ppm corrected"
    if result.cross_check_s is not None:
        text += f", the IMU finds {result.cross_check_s:+.2f} s"
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
    version = "0.6.0"
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
        paths.append(marks_path(self.config))
        return [path for path in paths if path.is_file()]

    def extra_key(self, recording: Recording) -> str:
        return json.dumps([[item.person, item.stage, item.path.name, item.stream]
                           for item in signal_files(self.config, recording.recording_id)])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        rid = recording.recording_id
        started, video_offset = video_started(recording)
        files = signal_files(self.config, rid)
        formats = {item.key: load_info(self.config, rid, item).get("format", SNIRF) for item in files}
        people: dict[str | None, list[SignalFile]] = {}
        for item in files:
            people.setdefault(item.person, []).append(item)
        results, warnings = [], []
        artifacts = {}
        marked = [(mark.name, mark.start_s) for mark in marks_of(self.config, rid)]
        for person, items in people.items():
            # a person's SNIRF files share one device's clock; a table from another device keeps its own
            snirfs = [item for item in items if formats[item.key] == SNIRF]
            found, more = self._person(rid, person, snirfs, started, video_offset, workspace) if snirfs else ([], [])
            by_key = {result.target: result for result in found}
            mine = []
            for item in items:
                if formats[item.key] == COHERENCE:
                    mine.append(self._follower(rid, item, files, formats))
                    continue
                imu = by_key.get(item.key) or AlignResult(
                    reference=RECORDING_CLOCK, target=item.key, offset_s=0.0, confidence=None, method=UNALIGNED,
                    note=UNPLACED.get(formats[item.key], UNPLACED[TABLE_CLOCK]),
                    metadata_offset_s=timestamp_offset(load_info(self.config, rid, item).get("started"), started, video_offset),
                )
                sure = imu.method == IMU and review_reason(imu, self.config.signals) is None
                paired = match_markers(marked, [(m["name"], float(m["onset"])) for m in load_info(self.config, rid, item).get("markers", [])],
                                       prefer=imu.offset_s if sure else imu.metadata_offset_s)
                if paired is None:
                    mine.append(imu)
                    continue
                mine.append(marker_result(paired, item.key, imu, self.config.signals))
                # a file placed by its own markers needs no shared clock
                more = [warning for warning in more if not warning.message.startswith(f"{item.key} has its markers")]
            results.extend(_sharing_marked_clock(mine, {item.key for item in snirfs}))
            warnings.extend(more)
        artifacts.update({figure.name: figure for figure in workspace.glob(f"*{FIGURE_SUFFIX}")})
        path = workspace / COMPUTED_NAME
        write_offset_table(results, path)
        warnings.extend(self._notes(rid, results, {item.key: item.person for item in files}))
        return ExtractorResult(rows=pd.DataFrame(), artifacts={COMPUTED_NAME: path, **artifacts}, warnings=warnings)

    def _follower(self, rid: str, item: SignalFile, files: list[SignalFile], formats: dict[str, str]) -> AlignResult:
        """A coherence series placed through the SNIRF file of a member it was computed from."""
        about = load_info(self.config, rid, item)["coherence"]
        members = []
        for subject, into in about["members"].items():
            label = subject.removeprefix("sub-")
            for other in files:
                if formats[other.key] == SNIRF and other.person and _subject(other.path) == label:
                    members.append((other.key, float(load_info(self.config, rid, other).get("start_s") or 0.0) + float(into)))
                    break
        low, high = about["band_hz"]
        said = f"the coherence of {' and '.join(about['members'])} at {low:g} to {high:g} Hz"
        if not members:
            return AlignResult(reference=RECORDING_CLOCK, target=item.key, offset_s=0.0, confidence=None, method=UNALIGNED,
                               note=f"{said}: none of them has a SNIRF file in the signals table to take a clock from; list one or set its offset by hand")
        return AlignResult(reference=members[0][0], target=item.key, offset_s=members[0][1], confidence=None, method=FOLLOWS,
                           follows=members, note=said)

    def _person(self, rid, person, items, started, video_offset, workspace: Path) -> tuple[list[AlignResult], list[QCWarning]]:
        cfg = self.config.signals

        def unaligned(note: str) -> list[AlignResult]:
            hint = timestamp_offset(load_info(self.config, rid, items[0]).get("started"), started, video_offset)
            return [AlignResult(reference=RECORDING_CLOCK, target=item.key, offset_s=0.0, confidence=None, method=UNALIGNED,
                                metadata_offset_s=hint, note=note) for item in items]

        if person is None:
            return unaligned("a file of the whole recording has no head to match; mark its events on the video or set its offset by hand"), []
        anchor = imu = None
        for item in items:
            try:
                imu = gyro_speed(load_aux(self.config, rid, item), cfg.gyro)
            except ConfigError as error:   # several gyroscopes: the file's marked events may still place it
                return unaligned(str(error)), []
            if imu is not None:
                anchor = item
                break
        if anchor is None:
            return unaligned(f"no channel named like {cfg.gyro!r} in {person}'s files; mark the file's events on the video or set the offset by hand"), []
        head = head_turning(self.config, rid, person)
        if head is None:
            return unaligned(f"no head angles on the video for {person}: run face.au or pose.kinematics, mark the file's events on the video, or set the offset by hand"), []
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

    def _notes(self, rid: str, results: list[AlignResult], whose: dict[str, str | None]) -> list[QCWarning]:
        out = []
        for result in results:
            person = whose.get(result.target)
            severity = "warn" if review_reason(result, self.config.signals) else "info"
            message = f"{result.target}: {describe(result)}"
            hours = whole_hours(result.metadata_offset_s if result.method != UNALIGNED else None)
            if hours is not None and abs(result.offset_s - result.metadata_offset_s) > 1800:
                message += f"; the start times differ by a whole {hours:+d} h more than the match, as a time zone written wrong would"
            out.append(QCWarning(check=self.name, recording_id=rid, subject_id=person, severity=severity, message=message))
        return out


def marker_result(paired: MarkerMatch, target: str, imu: AlignResult, cfg) -> AlignResult:
    """The file placed by its marked events, with the IMU's offset beside it when that match is sure."""
    result = AlignResult(
        reference=RECORDING_CLOCK, target=target, offset_s=paired.offset_s, confidence=None, overlap_s=paired.span_s,
        drift_ppm=paired.drift_ppm, drift_r2=paired.drift_r2, drift_intercept_s=paired.drift_intercept_s,
        segment_offsets=[(at, at - own, 1.0) for _, at, own in paired.pairs], method=MARKERS,
        metadata_offset_s=imu.metadata_offset_s, note="paired " + ", ".join(sorted({name for name, _, _ in paired.pairs})),
    )
    doubts = []
    if paired.unpaired:
        doubts.append(f"marked {', '.join(paired.unpaired)} found no event of that name near the others' offset")
    fitted = (lambda at: result.drift_intercept_s + result.drift_ppm * 1e-6 * at) if drift_corrected(result) else None
    worst = max((abs(gap) for gap in paired.residuals(fitted)), default=0.0)
    if worst > RESIDUAL_S:
        doubts.append(f"the marked events disagree by up to {worst:.2f} s")
    confirmed = False
    if imu.method == IMU and review_reason(imu, cfg) is None:
        result.cross_check_s = imu.offset_s
        confirmed = abs(imu.offset_s - result.offset_s) <= RESIDUAL_S
        if not confirmed:
            doubts.append(f"the IMU finds {imu.offset_s:+.2f} s, {abs(imu.offset_s - result.offset_s):.2f} s away")
    elif imu.method == IMU:
        result.note += "; the IMU's match was too unsure to check against"
    if paired.alternatives and not confirmed:
        doubts.append(f"the marks fit {len(paired.alternatives) + 1} places as well ({', '.join(sorted({n for n, _, _ in paired.pairs}))} "
                      "recurs in the file); a mark of another event, or a look at the video, tells them apart")
    result.doubt = "; ".join(doubts) or None
    return result


def _sharing_marked_clock(results: list[AlignResult], snirfs: set[str]) -> list[AlignResult]:
    """A person's SNIRF files with no event of their own paired take the clock of the first placed by its events, as
    files of one device share its clock; a copy whose events were renamed is the usual case."""
    marked = next((result for result in results if result.method == MARKERS and result.target in snirfs), None)
    if marked is None:
        return results
    return [
        replace(marked, target=result.target, note=f"same clock as {marked.target}, {marked.note}")
        if result.target in snirfs and result.method in (IMU, UNALIGNED) else result
        for result in results
    ]


def _subject(path: Path) -> str | None:
    found = re.search(r"(?:^|_)sub-([A-Za-z0-9]+)(?=_|$)", path.stem)
    return found.group(1) if found else None


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
