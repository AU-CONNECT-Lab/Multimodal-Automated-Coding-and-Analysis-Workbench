"""Where each person looks: at whom, at which named thing, elsewhere in the picture or out of it."""

from __future__ import annotations

import math
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError, UnreadableMediaError
from macaw.extractors import get_backend, register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import apply_prep_timing
from macaw.extractors.video.gaze import angles
from macaw.extractors.video.gaze.backends import BACKEND_GROUP, resolve_backend
from macaw.extractors.video.gaze.heads import head_boxes
from macaw.extractors.video.gaze.objects import EVERY_S, GazeObjectsExtractor, load_objects, objects_file
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate
from macaw.project.gaze_labels import labels_frame, read_labels
from macaw.project.room import seats as room_seats
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.segments import clip_points
from macaw.store import keypoints as keypoints_store
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import ANNOTATION_SCHEMA, Scale, coerce_frame, empty_frame

LAYER = "gaze"
TARGET_TIER = "gaze_target"
MUTUAL_TIER = "mutual_gaze"
JOINT_TIER = "joint_attention"
ELSEWHERE = "elsewhere"
OFF_SCREEN = "off_screen"
# the model's own cut between a target in the picture and one outside it
IN_FRAME_CUT = 0.5
POINTS_NAME = "gaze_points.parquet"
FACE_FEATURES = ("face_yaw_deg", "face_pitch_deg", "gaze_yaw_deg", "gaze_pitch_deg")

_LOOKER_SPECS = {
    "gaze_x": ("Horizontal position in the picture of the point the person looks at.", "px", "none"),
    "gaze_y": ("Vertical position in the picture of the point the person looks at.", "px", "none"),
    "gaze_in_frame": ("How sure the model is that what the person looks at is in the picture, 0 to 1.", None, "mean"),
}
_DISTANCE_SPECS = {
    "gaze_to_head": (
        "Distance from the point subject_id looks at to subject_id_b's head box, in widths of that box; 0 on the head.",
        None, "mean",
    ),
    "gaze_off_deg": (
        "Angle between where subject_id looks (their eyes' gaze where read, else their head) and where the room laid out "
        "in the setup or labelled frames put subject_id_b; 0 straight at them.",
        "deg", "mean",
    ),
}
_MUTUAL_SPEC = ("1 while subject_id and subject_id_b each look at the other, else 0.", None, "mean")
_JOINT_SPEC = ("1 while subject_id and subject_id_b look at the same named thing, else 0.", None, "mean")
_LAG_SPEC = (
    "For each stretch of joint attention, how long after subject_id_b began looking at the thing (or was first read "
    "looking at it) subject_id did; 0 when both began together, the pair then in name order.",
    "s", "mean",
)
# compared with hand labels of where people look
_CHECKED = ("looks_at", "looks_at_share")
_SUMMARY_SPECS = {
    "looks_at_share": "Share of the frames subject_id's gaze was read in during which they looked at subject_id_b.",
    "looks_at_n": "Looks from subject_id to subject_id_b: runs of frames with looks_at 1.",
    "mutual_gaze_share": "Share of the frames both gazes were read in during which the two looked at each other.",
    "joint_attention_share": "Share of the frames both gazes were read in during which the two looked at the same named thing.",
    "joint_attention_n": "Stretches of joint attention: unbroken runs of frames with both looking at one named thing.",
    "joint_attention_led_n": "Stretches of joint attention in which subject_id looked at the thing first and subject_id_b joined.",
}
_ONCE = "Each pair is written once, in name order, so its rows with the later name first hold none."
_LAG_MISSING = (
    "Never empty: each row is one stretch of joint attention, under the person who joined second, or under the two in name "
    "order when both began together."
)


def _missing(points: bool) -> dict[str, str]:
    if points:
        unread = "no camera is set to read it, or pose found fewer than two of their nose, eyes and ears in its picture"
        both, nothing = unread, "gaze.targets names no things"
        unseen = "either one's head never appears in the picture the other's gaze is read from"
        sideways = ", or could not be seen looking at the other because the other's head was not in their picture then"
        looker = f"Empty in frames where subject_id's gaze was not read ({unread}),"
        own = {
            name: f"Empty in frames where this person's gaze was not read: {unread}." for name in ("gaze_x", "gaze_y", "gaze_in_frame")
        }
        pair = {
            "gaze_to_head": f"{looker} and in frames where subject_id_b's head is not in that picture.",
            "looks_at": (
                f"{looker} and throughout the span when subject_id_b's head never appears in that picture. 0 means subject_id "
                "looked elsewhere, which includes frames in which subject_id_b's head is not in the picture."
            ),
        }
        span = (
            f"Empty where subject_id's gaze was never read in the span ({unread}), or where subject_id_b's head never appears in "
            "that picture in it."
        )
    else:
        unread = "no camera is set to read it, or face.au found no face of theirs"
        both = (
            "no camera is set to read it, face.au found no face of theirs, or neither the room laid out in the setup nor labelled "
            "frames say where anyone or anything is for them"
        )
        unseen, nothing = "neither the room nor labelled frames say where one of the two is", "setup.room.things and the labelled frames name no things"
        sideways = ""
        placed = (
            "neither the room laid out in the setup (with the camera that reads subject_id's face and both people placed) nor at least "
            f"{angles.MIN_LABELS} labelled frames of subject_id looking at subject_id_b say where subject_id_b is"
        )
        looker = f"Empty in frames where subject_id's gaze was not read ({unread}), and throughout for a pair where {placed}."
        own = {}
        pair = {
            "gaze_off_deg": looker,
            "looks_at": (
                f"{looker} 0 means subject_id looked elsewhere: not within gaze.tolerance_deg of subject_id_b, or nearer someone or "
                "something else."
            ),
        }
        span = f"Empty where subject_id's gaze was never read in the span ({unread}), or where {placed}."
    never = f"Empty where the two gazes were never read at the same moment in the span (for either person: {both})"
    joint = f"{never}, and when {nothing}. {_ONCE} 0 means they were never seen looking at the same named thing."
    return {
        **own, **pair,
        "looks_at_share": f"{span} 0 means the gaze was read but never rested on subject_id_b.",
        "looks_at_n": f"{span} 0 means the gaze was read but never rested on subject_id_b.",
        "mutual_gaze": (
            f"Empty at moments when either person's gaze was not read ({both}), and throughout for a pair where {unseen}. {_ONCE} "
            f"0 means at least one of them looked elsewhere{sideways}."
        ),
        "mutual_gaze_share": (
            f"{never}, and for a pair where {unseen}. {_ONCE} 0 means both gazes were read but the two were never seen looking at "
            "each other."
        ),
        "joint_attention": (
            f"Empty at moments when either person's gaze was not read ({both}), and throughout when {nothing}. {_ONCE} 0 means "
            "the two were not seen looking at the same named thing."
        ),
        "joint_attention_share": joint,
        "joint_attention_n": joint,
        "joint_attention_led_n": (
            f"{never}, and when {nothing}. 0 means subject_id led no stretch: there was none, subject_id_b looked first, or both "
            "began together."
        ),
        "joint_attention_lag_s": _LAG_MISSING,
    }


def finds_points(config) -> bool:
    return get_backend(BACKEND_GROUP, config.gaze.backend).finds_points


@register_extractor
class GazeTargetExtractor(Extractor):
    name = "gaze.target"
    version = "0.4.1"
    scale = Scale.FRAME
    group = "gaze"
    config_sections = ("gaze", "segment", "video", "setup.cameras", "setup.room")
    requires = (PoseReviewGate.name,)
    resource_profile = "gpu_standard"

    @classmethod
    def dependencies(cls, config) -> tuple[str, ...]:
        from macaw.extractors.video.face.au import FaceExtractor

        if not finds_points(config):
            return cls.requires + (FaceExtractor.name,)
        return cls.requires + ((GazeObjectsExtractor.name,) if config.gaze.targets else ())

    def backend(self):
        return resolve_backend(self.config.gaze.backend)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, backend.version)

    def check_available(self) -> None:
        backend = self.backend()
        if not backend.finds_points and self.config.gaze.tolerance_deg is None:
            raise ConfigError(
                f"gaze.backend {backend.name} needs gaze.tolerance_deg: how many degrees off the direction to someone "
                "a look may be and still count as a look at them"
            )

    def prefetch(self) -> None:
        self.backend().prefetch(model=self.config.gaze.model)

    def inputs_for(self, recording: Recording) -> list[Path]:
        rid = recording.recording_id
        files = [_keypoints_file(self.config, rid)]
        if not finds_points(self.config):
            labels = layout_for(self.config, create=False).gaze_labels_path(rid)
            return files + [_face_rows_file(self.config, rid)] + ([labels] if labels.is_file() else [])
        return files + ([objects_file(self.config, rid)] if self.config.gaze.targets else [])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        span = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        keypoints = keypoints_store.read_keypoints(
            _keypoints_file(self.config, recording.recording_id), recording_id=recording.recording_id
        )
        views = {person: item.name for person, item in recording.views("gaze").items()}
        by_name = {item.name: item for item in recording.media if item.kind == "video"}
        if backend.finds_points:
            points, warnings = self._points(recording, backend, keypoints, views, by_name, span)
        else:
            points, warnings = self._angles(recording, views, by_name, span)
        points = settle_looks(points, self.config.gaze.min_look_s)
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        points.to_parquet(directory / POINTS_NAME, index=False)
        subjects = [subject.subject_id for subject in recording.subjects]
        things = self._things(recording)
        rows = self._rows(recording, points, subjects, things)
        return ExtractorResult(
            rows=rows,
            specs=self._specs(set(rows["feature"]) if not rows.empty else set(), backend, things),
            annotations=self._annotations(recording, points, subjects, things),
            artifacts={POINTS_NAME: directory / POINTS_NAME},
            warnings=warnings,
        )

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        path = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, self.name) / POINTS_NAME
        if not path.is_file():
            return None
        points = clip_points(pd.read_parquet(path), start_s, end_s)
        rows = self._rows(recording, points, [subject.subject_id for subject in recording.subjects], self._things(recording))
        return rows[rows["scale"] == Scale.RECORDING.value].reset_index(drop=True)

    def _things(self, recording: Recording) -> list[str]:
        """The named things a look can land on, which joint attention is about."""
        if finds_points(self.config):
            return list(self.config.gaze.targets)
        labels = labels_frame(read_labels(layout_for(self.config, create=False).gaze_labels_path(recording.recording_id)))
        roles = {subject.subject_id for subject in recording.subjects}
        named = set(labels["target"]) if not labels.empty else set()
        return sorted((named | set(self.config.setup.room.things)) - roles - {ELSEWHERE})

    def _points(self, recording, backend, keypoints, views, by_name, span) -> tuple[pd.DataFrame, list[QCWarning]]:
        cfg = self.config.gaze
        heads = head_boxes(keypoints[keypoints["subject_id"].notna()])
        things = load_objects(self.config, recording.recording_id) if cfg.targets else None
        points, warnings = [], []
        for video in sorted(set(views.values())):
            item, (first, stop) = by_name[video], _frame_span(by_name[video], span)
            shown = heads[(heads["video"] == video) & (heads["frame"] >= first) & (heads["frame"] < stop)]
            lookers = [subject for subject, view in views.items() if view == video]
            for subject in [subject for subject in lookers if set(shown["subject_id"]) <= {subject}]:
                warnings.append(
                    self._warning(
                        recording, subject, "info",
                        "no one else is ever in this person's picture, so whom they look at cannot be read off it",
                    )
                )
            found = backend.infer(
                item.frames_path, heads=shown[shown["subject_id"].isin(lookers)], device=cfg.device,
                model=cfg.model, batch_size=cfg.batch_size,
            )
            found["t"] = found["frame"] / item.fps + item.offset_s
            named = name_targets(found[(found["t"] >= span[0]) & (found["t"] <= span[1])], shown)
            if things is not None:
                named = name_objects(named, things[things["video"] == video], max(round(EVERY_S * item.fps), 1))
            points.append(named)
        return (pd.concat(points, ignore_index=True) if points else _no_points()), warnings

    def _angles(self, recording, views, by_name, span) -> tuple[pd.DataFrame, list[QCWarning]]:
        cfg, rid = self.config.gaze, recording.recording_id
        face_rows = read_long_table(_face_rows_file(self.config, rid), recording_id=rid, features=list(FACE_FEATURES))
        measured = angles.directions(face_rows)
        measured = measured[(measured["t"] >= span[0]) & (measured["t"] <= span[1])]
        labels = labels_frame(read_labels(layout_for(self.config, create=False).gaze_labels_path(rid)))
        roles = [subject.subject_id for subject in recording.subjects]
        points, warnings = [], []
        room = self.config.setup.room
        for subject, mine in measured.groupby("subject_id", sort=True):
            subject = str(subject)
            item = by_name.get(views.get(subject, ""))
            if item is None or not item.fps:
                continue
            # seen from the camera that reads this person's face, wherever the room places it
            seats = room_seats(room, item.name)
            labelled = angles.labelled_directions(measured, labels, 1.0 / item.fps)
            named_things = sorted((set(labels.loc[labels["subject"] == subject, "target"]) | set(room.things)) - set(roles) - {ELSEWHERE})
            expected = angles.expected_directions(subject, [role for role in roles if role != subject] + named_things, seats, labelled)
            if not expected:
                warnings.append(
                    self._warning(
                        recording, subject, "warn",
                        f"neither the room laid out in the setup (with {item.name} and this person placed) nor {angles.MIN_LABELS} or "
                        "more labelled frames say where anyone is for this person, so whom they look at is not read",
                    )
                )
                continue
            used = int((labelled["subject"] == subject).sum()) if not labelled.empty else 0
            warnings.append(
                self._warning(
                    recording, subject, "info",
                    f"looks read from {', '.join(f'{name} ({source})' for name, source in _sources(expected, labelled, subject))}"
                    f"; {used} labelled frame(s) of this person used",
                )
            )
            named = angles.name_by_angles(mine, expected, cfg.tolerance_deg, ELSEWHERE)
            named["frame"] = np.round((named["t"].to_numpy(dtype=float) - item.offset_s) * item.fps).astype(int)
            points.append(named.rename(columns={column: f"angle_{column[4:]}" for column in named.columns if column.startswith("off_")}))
        if not points:
            return _no_points(), warnings
        return pd.concat(points, ignore_index=True), warnings

    def _rows(self, recording: Recording, points: pd.DataFrame, subjects: list[str], things: list[str]) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        frames = []
        if "gaze_x" in points:
            for subject, mine in points.groupby("subject_id", sort=True):
                mine = mine.sort_values("t")
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame(
                            {
                                "t_start": mine["t"].to_numpy(),
                                "gaze_x": mine["gaze_x"].to_numpy(), "gaze_y": mine["gaze_y"].to_numpy(),
                                "gaze_in_frame": mine["in_frame"].to_numpy(),
                            }
                        ),
                        scale=Scale.FRAME, subject_id=str(subject), **common,
                    )
                )
        measured = set()
        for a, b in permutations(subjects, 2):
            mine = points[points["subject_id"] == a].sort_values("t")
            column, feature = (f"to_{b}", "gaze_to_head") if f"to_{b}" in mine else (f"angle_{b}", "gaze_off_deg")
            if mine.empty or column not in mine or mine[column].isna().all():
                continue
            measured.add((a, b))
            looks = (mine["target"] == b).astype(float)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame({"t_start": mine["t"].to_numpy(), feature: mine[column].to_numpy(), "looks_at": looks.to_numpy()}),
                    scale=Scale.FRAME, subject_id=a, subject_id_b=b, **common,
                )
            )
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [float(mine["t"].min())], "t_end": [float(mine["t"].max())],
                            "looks_at_share": [float(looks.mean())], "looks_at_n": [float(len(runs(mine["frame"], looks.to_numpy() > 0)))],
                        }
                    ),
                    scale=Scale.RECORDING, subject_id=a, subject_id_b=b, **common,
                )
            )
        for a, b in [(a, b) for a, b in permutations(subjects, 2) if a < b]:
            # never 0 for a pair where one of the two looks could not be seen at all
            mutual = mutual_gaze(points, a, b) if {(a, b), (b, a)} <= measured else pd.DataFrame()
            if not mutual.empty:
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame({"t_start": mutual["t"].to_numpy(), "mutual_gaze": mutual["mutual"].astype(float).to_numpy()}),
                        scale=Scale.FRAME, subject_id=a, subject_id_b=b, **common,
                    )
                )
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame(
                            {"t_start": [float(mutual["t"].min())], "t_end": [float(mutual["t"].max())], "mutual_gaze_share": [float(mutual["mutual"].mean())]}
                        ),
                        scale=Scale.RECORDING, subject_id=a, subject_id_b=b, **common,
                    )
                )
            if things:
                frames.extend(self._joint_rows(joint_attention(points, a, b, things), a, b, common))
        return concat_long_tables(frames)

    def _joint_rows(self, joint: pd.DataFrame, a: str, b: str, common: dict) -> list[pd.DataFrame]:
        if joint.empty:
            return []
        on = joint["thing"].notna().astype(float)
        episodes = joint_episodes(joint, a, b)
        span = {"t_start": [float(joint["t"].min())], "t_end": [float(joint["t"].max())]}
        frames = [
            build_long_rows(
                values=pd.DataFrame({"t_start": joint["t"].to_numpy(), "joint_attention": on.to_numpy()}),
                scale=Scale.FRAME, subject_id=a, subject_id_b=b, **common,
            ),
            build_long_rows(
                values=pd.DataFrame({**span, "joint_attention_share": [float(on.mean())], "joint_attention_n": [float(len(episodes))]}),
                scale=Scale.RECORDING, subject_id=a, subject_id_b=b, **common,
            ),
        ]
        for first, then in ((a, b), (b, a)):
            frames.append(
                build_long_rows(
                    values=pd.DataFrame({**span, "joint_attention_led_n": [float((episodes["first"] == first).sum())]}),
                    scale=Scale.RECORDING, subject_id=first, subject_id_b=then, **common,
                )
            )
        for (follower, leader), mine in episodes.groupby(["follower", "leader"], sort=True):
            frames.append(
                build_long_rows(
                    values=mine[["t_start", "t_end"]].assign(joint_attention_lag_s=mine["lag_s"].astype(float)).reset_index(drop=True),
                    scale=Scale.EVENT, subject_id=str(follower), subject_id_b=str(leader), **common,
                )
            )
        return frames

    def _annotations(self, recording: Recording, points: pd.DataFrame, subjects: list[str], things: list[str]) -> pd.DataFrame:
        common = dict(recording_id=recording.recording_id, layer=LAYER, annotator=self.name, entities=recording.entities())
        frames = []
        for subject, mine in points.groupby("subject_id", sort=True):
            intervals = label_runs(mine.sort_values("frame"), "target")
            if not intervals.empty:
                frames.append(build_annotations(tier=TARGET_TIER, intervals=intervals, subject_id=str(subject), **common))
        for a, b in [(a, b) for a, b in permutations(subjects, 2) if a < b]:
            mutual = mutual_gaze(points, a, b)
            if mutual.empty:
                continue
            mutual = mutual.assign(label=np.where(mutual["mutual"], b, None))
            intervals = label_runs(mutual, "label").dropna(subset=["label"])
            if not intervals.empty:
                frames.append(build_annotations(tier=MUTUAL_TIER, intervals=intervals.reset_index(drop=True), subject_id=a, **common))
            # a tier per pair: with three or more people one person's stretches with two others can overlap
            episodes = joint_episodes(joint_attention(points, a, b, things), a, b) if things else pd.DataFrame()
            if not episodes.empty:
                intervals = episodes[["t_start", "t_end"]].assign(label=episodes["thing"]).reset_index(drop=True)
                frames.append(build_annotations(tier=f"{JOINT_TIER}_with_{b}", intervals=intervals, subject_id=a, **common))
        if not frames:
            return empty_frame(ANNOTATION_SCHEMA)
        return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")

    def _specs(self, emitted: set[str], backend, things: list[str]) -> list[FeatureSpec]:
        cfg = self.config.gaze
        if backend.finds_points:
            rule = (
                "1 while the point subject_id looks at lies on subject_id_b's head box and the model puts it in the picture "
                f"(in-frame score at least {IN_FRAME_CUT}), else 0."
            )
            params = {"model": cfg.model or getattr(backend, "default_model", None), "in_frame_cut": IN_FRAME_CUT}
        else:
            rule = f"1 while subject_id looks within {cfg.tolerance_deg:g} deg of subject_id_b, nearer them than anyone else, else 0."
            params = {"tolerance_deg": cfg.tolerance_deg, "room": self.config.setup.room.model_dump(mode="json", exclude_none=True)}
        if cfg.min_look_s > 0:
            rule += f" A look at anyone or anything lasting less than {cfg.min_look_s:g} s joins the look before it."
        params["min_look_s"] = cfg.min_look_s
        params["things"] = things
        missing = _missing(backend.finds_points)
        described = {
            **_LOOKER_SPECS, **_DISTANCE_SPECS, "looks_at": (rule, None, "mean"), "mutual_gaze": _MUTUAL_SPEC,
            "joint_attention": _JOINT_SPEC, "joint_attention_lag_s": _LAG_SPEC,
            **{name: (text, None, "mean") for name, text in _SUMMARY_SPECS.items()},
        }
        return [
            FeatureSpec(
                name=name, description=description, unit=unit,
                scale=Scale.RECORDING if name in _SUMMARY_SPECS else Scale.EVENT if name == "joint_attention_lag_s" else Scale.FRAME,
                group=self.group,
                extractor=self.name, extractor_version=self.version, backend=backend.name, backend_version=backend.version,
                params=params, interpolation="none", aggregation=aggregation, missing=missing.get(name, missing["looks_at"]),
                comparable_across_backends=False,
                validation="against_reference" if backend.finds_points and name in _CHECKED else "none",
            )
            for name, (description, unit, aggregation) in described.items()
            if name in emitted
        ]

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message)


def name_targets(found: pd.DataFrame, heads: pd.DataFrame) -> pd.DataFrame:
    """Adds target (whose head the point lies on, else elsewhere or off_screen) and to_<subject>, the distance
    to each other head shown in that frame in widths of its box."""
    out = found.copy()
    out["target"] = np.where(out["in_frame"] >= IN_FRAME_CUT, ELSEWHERE, OFF_SCREEN)
    for other, boxes in heads.groupby("subject_id", sort=True):
        joined = out[["frame", "subject_id", "gaze_x", "gaze_y"]].merge(boxes[["frame", "x0", "y0", "x1", "y1"]], on="frame", how="left")
        width = (joined["x1"] - joined["x0"]).to_numpy()
        dx = np.fmax(np.fmax(joined["x0"] - joined["gaze_x"], joined["gaze_x"] - joined["x1"]), 0.0)
        dy = np.fmax(np.fmax(joined["y0"] - joined["gaze_y"], joined["gaze_y"] - joined["y1"]), 0.0)
        distance = np.array(np.hypot(dx, dy) / width, dtype=float)
        distance[(joined["subject_id"] == other).to_numpy()] = np.nan
        out[f"to_{other}"] = distance
        on_head = (distance == 0) & (out["target"] == ELSEWHERE).to_numpy()
        out.loc[on_head, "target"] = str(other)
    return out


def name_objects(points: pd.DataFrame, things: pd.DataFrame, within_frames: int) -> pd.DataFrame:
    """A point that is on no one's head but in a named thing's box, as found in the nearest frame read for things,
    takes that thing's name; the smallest box holding it wins."""
    out = points.copy()
    free = out[out["target"] == ELSEWHERE]
    if free.empty or things.empty:
        return out
    read = np.sort(things["frame"].unique())
    position = np.clip(np.searchsorted(read, free["frame"].to_numpy()), 1, len(read) - 1) if len(read) > 1 else np.zeros(len(free), int)
    nearest = read[position]
    if len(read) > 1:
        earlier = read[position - 1]
        nearest = np.where(np.abs(free["frame"].to_numpy() - earlier) <= np.abs(nearest - free["frame"].to_numpy()), earlier, nearest)
    candidates = free[["gaze_x", "gaze_y"]].assign(seen=nearest, gap=np.abs(nearest - free["frame"].to_numpy())).reset_index()
    candidates = candidates[candidates["gap"] <= within_frames].merge(things, left_on="seen", right_on="frame")
    inside = (
        (candidates["gaze_x"] >= candidates["x0"]) & (candidates["gaze_x"] <= candidates["x1"])
        & (candidates["gaze_y"] >= candidates["y0"]) & (candidates["gaze_y"] <= candidates["y1"])
    )
    candidates = candidates[inside].assign(area=lambda f: (f["x1"] - f["x0"]) * (f["y1"] - f["y0"]))
    chosen = candidates.sort_values("area").drop_duplicates("index")
    out.loc[chosen["index"].to_numpy(), "target"] = chosen["label"].to_numpy()
    return out


def settle_looks(points: pd.DataFrame, min_look_s: float) -> pd.DataFrame:
    """Within each person's unbroken frames, a run of one target shorter than min_look_s takes the target of the run
    before it, the first run that of the run after; target_read keeps what each frame read."""
    out = points.assign(target_read=points["target"])
    if min_look_s <= 0 or out.empty:
        return out
    for _, mine in out.groupby("subject_id", sort=False):
        mine = mine.sort_values("frame")
        period = float(np.median(np.diff(mine["t"]))) if len(mine) > 1 else 0.0
        if period <= 0:
            continue
        unbroken = (mine["frame"].diff() != 1).cumsum()
        for _, part in mine.groupby(unbroken, sort=False):
            # the small margin keeps a run of exactly min_look_s from counting as shorter through rounding
            out.loc[part.index, "target"] = _settled(part["target"].tolist(), min_look_s / period - 1e-6)
    return out


def _settled(labels: list, min_frames: float) -> list:
    pieces: list[list] = []
    for label in labels:
        if pieces and pieces[-1][0] == label:
            pieces[-1][1] += 1
        else:
            pieces.append([label, 1])
    kept: list[list] = []
    for label, length in pieces:
        if kept and (length < min_frames or kept[-1][0] == label):
            kept[-1][1] += length
        else:
            kept.append([label, length])
    if len(kept) > 1 and kept[0][1] < min_frames:
        first = kept.pop(0)
        kept[0][1] += first[1]
    return [label for label, length in kept for _ in range(length)]


def paired_looks(points: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """a's frame and t with each one's target and when their look at it began, for the moments both gazes were read in;
    matched by time, so people read from cameras of their own line up."""
    mine, theirs = _with_starts(points[points["subject_id"] == a]), _with_starts(points[points["subject_id"] == b])
    if mine.empty or theirs.empty:
        return pd.DataFrame(columns=["frame", "t", "target_a", "began_a", "target_b", "began_b"])
    half = 0.5 * max(_period(mine), _period(theirs))
    both = pd.merge_asof(
        mine[["frame", "t", "target", "began"]], theirs[["t", "target", "began"]].rename(columns={"t": "t_b"}),
        left_on="t", right_on="t_b", direction="nearest", tolerance=half + 1e-9, suffixes=("_a", "_b"),
    )
    return both.dropna(subset=["t_b"]).drop(columns="t_b").sort_values("frame").reset_index(drop=True)


def mutual_gaze(points: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """t and mutual for the moments both gazes were read in."""
    both = paired_looks(points, a, b)
    return pd.DataFrame({"frame": both["frame"], "t": both["t"], "mutual": (both["target_a"] == b) & (both["target_b"] == a)})


def joint_attention(points: pd.DataFrame, a: str, b: str, things: list[str]) -> pd.DataFrame:
    """frame, t and thing, the named thing both look at or None, for the moments both gazes were read in."""
    both = paired_looks(points, a, b)
    shared = (both["target_a"] == both["target_b"]) & both["target_a"].isin(things)
    return both.assign(thing=both["target_a"].where(shared, None))[["frame", "t", "thing", "began_a", "began_b"]]


def joint_episodes(joint: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """One row per unbroken run of frames on one thing, with who looked at it first and how long before the other."""
    columns = ["t_start", "t_end", "thing", "first", "follower", "leader", "lag_s"]
    if joint.empty or joint["thing"].isna().all():
        return pd.DataFrame(columns=columns)
    joint = joint.reset_index(drop=True)
    period = _period(joint)
    broken = (joint["frame"].diff() != 1) | (joint["thing"] != joint["thing"].shift())
    spans = joint.groupby(broken.cumsum()).agg(
        t_start=("t", "first"), t_end=("t", "last"), thing=("thing", "first"), began_a=("began_a", "first"), began_b=("began_b", "first"),
    )
    spans = spans.dropna(subset=["thing"]).reset_index(drop=True)
    spans["t_end"] = spans["t_end"] + period
    ahead = spans["began_b"].astype(float) - spans["began_a"].astype(float)
    # less than half a frame apart is the same moment
    a_first, b_first = (ahead > period / 2).to_numpy(), (ahead < -period / 2).to_numpy()
    spans["first"] = np.where(a_first, a, np.where(b_first, b, None))
    spans["follower"] = np.where(a_first, b, a)
    spans["leader"] = np.where(a_first, a, b)
    spans["lag_s"] = np.where(a_first | b_first, ahead.abs(), 0.0)
    return spans[columns]


def _with_starts(mine: pd.DataFrame) -> pd.DataFrame:
    mine = mine.sort_values("frame").assign(t=lambda f: f["t"].astype(float))
    broken = (mine["frame"].diff() != 1) | (mine["target"] != mine["target"].shift())
    return mine.assign(began=mine["t"].groupby(broken.cumsum()).transform("first")).sort_values("t")


def _period(frame: pd.DataFrame) -> float:
    return float(np.median(np.diff(np.sort(frame["t"].to_numpy(dtype=float))))) if len(frame) > 1 else 0.0


def runs(frames: pd.Series, on: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) positions of each run of consecutive frames with on true."""
    frames = frames.to_numpy()
    found, begin = [], None
    for index, value in enumerate(on):
        joined = begin is not None and frames[index] == frames[index - 1] + 1
        if value and not joined:
            if begin is not None:
                found.append((begin, index - 1))
            begin = index
        elif not value and begin is not None:
            found.append((begin, index - 1))
            begin = None
    if begin is not None:
        found.append((begin, len(on) - 1))
    return found


def label_runs(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Intervals of one label over consecutive frames; each runs to the next frame's time."""
    if frame.empty:
        return pd.DataFrame(columns=["t_start", "t_end", "label"])
    frame = frame.reset_index(drop=True)
    period = float(np.median(np.diff(frame["t"]))) if len(frame) > 1 else 0.0
    broken = (frame["frame"].diff() != 1) | (frame[column] != frame[column].shift())
    group = broken.cumsum()
    spans = frame.groupby(group).agg(t_start=("t", "first"), t_end=("t", "last"), label=(column, "first"))
    spans["t_end"] = spans["t_end"] + period
    return spans.reset_index(drop=True)


def _sources(expected: dict, labelled: pd.DataFrame, subject: str) -> list[tuple[str, str]]:
    counts = labelled.loc[labelled["subject"] == subject, "target"].value_counts() if not labelled.empty else pd.Series(dtype=int)
    return [(name, "labelled frames" if counts.get(name, 0) >= angles.MIN_LABELS else "the room") for name in expected]


def _frame_span(item, span: tuple[float, float]) -> tuple[int, int]:
    if not item.fps or item.fps <= 0:
        raise UnreadableMediaError(f"{item.path}: no frame rate to convert frames to time")
    return max(math.floor((span[0] - item.offset_s) * item.fps), 0), math.ceil((span[1] - item.offset_s) * item.fps) + 1


def _no_points() -> pd.DataFrame:
    return pd.DataFrame(columns=["frame", "subject_id", "t", "target"])


def _keypoints_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).keypoints_path(recording_id) / f"{PoseExtractor.name}.parquet"


def _face_rows_file(config, recording_id: str) -> Path:
    from macaw.extractors.video.face.au import FaceExtractor

    return layout_for(config, create=False).long_table_path(recording_id) / f"{FaceExtractor.name}.parquet"
