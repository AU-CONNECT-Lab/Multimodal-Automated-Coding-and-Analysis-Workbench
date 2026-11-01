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
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate, primary_views
from macaw.project.gaze_labels import labels_frame, read_labels
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import keypoints as keypoints_store
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import ANNOTATION_SCHEMA, Scale, coerce_frame, empty_frame

LAYER = "gaze"
TARGET_TIER = "gaze_target"
MUTUAL_TIER = "mutual_gaze"
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
        "Angle between where subject_id looks (their eyes' gaze where read, else their head) and where the seats or "
        "labelled frames put subject_id_b; 0 straight at them.",
        "deg", "mean",
    ),
}
_MUTUAL_SPEC = ("1 while subject_id and subject_id_b each look at the other, else 0.", None, "mean")
# compared with hand labels of where people look
_CHECKED = ("looks_at", "looks_at_share")
_SUMMARY_SPECS = {
    "looks_at_share": "Share of the frames subject_id's gaze was read in during which they looked at subject_id_b.",
    "looks_at_n": "Looks from subject_id to subject_id_b: runs of frames with looks_at 1.",
    "mutual_gaze_share": "Share of the frames both gazes were read in during which the two looked at each other.",
}


def finds_points(config) -> bool:
    return get_backend(BACKEND_GROUP, config.gaze.backend).finds_points


@register_extractor
class GazeTargetExtractor(Extractor):
    name = "gaze.target"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "gaze"
    config_sections = ("gaze", "segment", "video")
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
            return files + [_face_rows_file(self.config, rid), layout_for(self.config, create=False).gaze_labels_path(rid)]
        return files + ([objects_file(self.config, rid)] if self.config.gaze.targets else [])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        span = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        keypoints = keypoints_store.read_keypoints(
            _keypoints_file(self.config, recording.recording_id), recording_id=recording.recording_id
        )
        views = primary_views(keypoints, recording)
        by_name = {item.path.name: item for item in recording.media if item.kind == "video"}
        if backend.finds_points:
            points, warnings = self._points(recording, backend, keypoints, views, by_name, span)
        else:
            points, warnings = self._angles(recording, views, by_name, span)
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        points.to_parquet(directory / POINTS_NAME, index=False)
        subjects = [subject.subject_id for subject in recording.subjects]
        rows = self._rows(recording, points, subjects)
        return ExtractorResult(
            rows=rows,
            specs=self._specs(set(rows["feature"]) if not rows.empty else set(), backend),
            annotations=self._annotations(recording, points, subjects),
            artifacts={POINTS_NAME: directory / POINTS_NAME},
            warnings=warnings,
        )

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
        seats = cfg.seats
        if seats and len({views.get(str(subject)) for subject in measured["subject_id"].unique()}) > 1:
            seats = {}
            warnings.append(
                self._warning(
                    recording, None, "warn",
                    "gaze.seats describe one camera, and these people are measured on several, so only labelled frames say where anyone is",
                )
            )
        for subject, mine in measured.groupby("subject_id", sort=True):
            subject = str(subject)
            item = by_name.get(views.get(subject, ""))
            if item is None or not item.fps:
                continue
            labelled = angles.labelled_directions(measured, labels, 1.0 / item.fps)
            named_things = sorted(set(labels.loc[labels["subject"] == subject, "target"]) - set(roles) - {ELSEWHERE})
            expected = angles.expected_directions(subject, [role for role in roles if role != subject] + named_things, seats, labelled)
            if not expected:
                warnings.append(
                    self._warning(
                        recording, subject, "warn",
                        f"neither gaze.seats nor {angles.MIN_LABELS} or more labelled frames say where anyone is for this person, "
                        "so whom they look at is not read",
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

    def _rows(self, recording: Recording, points: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
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
        for a, b in permutations(subjects, 2):
            mine = points[points["subject_id"] == a].sort_values("t")
            column, feature = (f"to_{b}", "gaze_to_head") if f"to_{b}" in mine else (f"angle_{b}", "gaze_off_deg")
            if mine.empty or column not in mine or mine[column].isna().all():
                continue
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
            mutual = mutual_gaze(points, a, b)
            if mutual.empty:
                continue
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
        return concat_long_tables(frames)

    def _annotations(self, recording: Recording, points: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
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
        if not frames:
            return empty_frame(ANNOTATION_SCHEMA)
        return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")

    def _specs(self, emitted: set[str], backend) -> list[FeatureSpec]:
        cfg = self.config.gaze
        if backend.finds_points:
            rule = (
                "1 while the point subject_id looks at lies on subject_id_b's head box and the model puts it in the picture "
                f"(in-frame score at least {IN_FRAME_CUT}), else 0."
            )
            params = {"model": cfg.model or getattr(backend, "default_model", None), "in_frame_cut": IN_FRAME_CUT}
        else:
            rule = f"1 while subject_id looks within {cfg.tolerance_deg:g} deg of subject_id_b, nearer them than anyone else, else 0."
            params = {"tolerance_deg": cfg.tolerance_deg, "seats": {name: list(place) for name, place in cfg.seats.items()}}
        described = {
            **_LOOKER_SPECS, **_DISTANCE_SPECS, "looks_at": (rule, None, "mean"), "mutual_gaze": _MUTUAL_SPEC,
            **{name: (text, None, "mean") for name, text in _SUMMARY_SPECS.items()},
        }
        return [
            FeatureSpec(
                name=name, description=description, unit=unit,
                scale=Scale.RECORDING if name in _SUMMARY_SPECS else Scale.FRAME, group=self.group,
                extractor=self.name, extractor_version=self.version, backend=backend.name, backend_version=backend.version,
                params=params, interpolation="none", aggregation=aggregation, comparable_across_backends=False,
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


def mutual_gaze(points: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """t and mutual for the frames both gazes were read in."""
    mine = points[points["subject_id"] == a][["frame", "t", "target"]]
    theirs = points[points["subject_id"] == b][["frame", "target"]]
    both = mine.merge(theirs, on="frame", suffixes=("_a", "_b"))
    return pd.DataFrame({"frame": both["frame"], "t": both["t"], "mutual": (both["target_a"] == b) & (both["target_b"] == a)})


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
    return [(name, "labelled frames" if counts.get(name, 0) >= angles.MIN_LABELS else "seats") for name in expected]


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
