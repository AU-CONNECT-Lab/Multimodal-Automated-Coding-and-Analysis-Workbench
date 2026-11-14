from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
from pathlib import Path

import pandas as pd

from macaw.exceptions import InputNotFoundError, MacawError, UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Gate, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, apply_prep_timing, load_prep, load_video_prep
from macaw.extractors.video.pose.backends import BACKEND_KEYPOINT_SCHEMA, resolve_backend
from macaw.extractors.video.pose.look_names import answer_key, named_by_look
from macaw.extractors.video.pose.track_repair import (
    Click,
    PeopleChoice,
    detect_swaps,
    most_in_view,
    named_by_clicks,
    repair_tracks,
    several_people,
    subject_column,
    suggested_people,
    track_quality,
)
from macaw.extractors.video.pose.voice_motion import bin_edges, head_motion, speech_activity
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.names import read_names
from macaw.project.recording import Recording
from macaw.store.keypoints import read_keypoints
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import KEYPOINT_SCHEMA, Scale, coerce_frame, empty_frame

# A track this broken is worth telling the user about rather than quietly averaging over.
_LOW_CONTINUITY = 0.8
_LOW_DETECTION_RATE = 0.5

_SPECS = {
    "detection_rate": (
        "Of the frames in which anyone was found on the camera this person's body is measured from, the share with this person.", None,
    ),
    "mean_confidence": ("Mean keypoint confidence over detected frames.", None),
    "missing_fraction": ("Standard joints without a usable coordinate.", None),
    "track_continuity": ("Frames seen over frames spanned by the track.", None),
    "n_track_swaps": ("Identity crossovers detected between tracks.", "count"),
}
_MISSING = {
    "detection_rate": (
        "Empty for a person not named on any camera in the span, and where the camera their body is measured from found nobody "
        "at all. 0 means that camera found people but never this person."
    ),
    "missing_fraction": (
        "Empty for a person not named on any camera in the span, or never detected on the camera their body is measured from. "
        "0 means every standard joint had a usable coordinate in every frame they were detected in."
    ),
    "n_track_swaps": (
        "Empty where nobody was detected in the span. 0 means no crossover was found, which is also the count with "
        "pose.track_repair off and on a camera the setup reads for one person alone."
    ),
    "person": "Empty for a person not named on any camera in the span, or never detected on the camera their body is measured from.",
}


def read_owner_file(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def inside_frame(keypoints: pd.DataFrame, frame_size: tuple[int, int] | None) -> pd.DataFrame:
    """A point outside the picture was guessed, not seen: it becomes missing, whatever the backend."""
    if frame_size is None:
        return keypoints
    width, height = frame_size
    x, y = keypoints["x"], keypoints["y"]
    outside = (x < 0) | (x > width) | (y < 0) | (y > height)
    if not outside.any():
        return keypoints
    out = keypoints.copy()
    out.loc[outside, ["x", "y", "z"]] = float("nan")
    return out


def _warning(check: str, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
    return QCWarning(
        check=check, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message
    )


def camera_people(item) -> list[str]:
    """Everyone a camera is read for, as the setup says."""
    return sorted(set().union(*item.uses.values())) if item.uses else []


def measured_views(keypoints: pd.DataFrame, recording: Recording) -> dict[str, str]:
    """person -> the camera their body is measured on: the one the setup reads it from, else the first they were named on."""
    views = {person: item.name for person, item in recording.views("body").items()}
    order = [item.name for item in recording.cameras()]
    named = keypoints[keypoints["subject_id"].notna()]
    for (subject, video), _ in sorted(named.groupby(["subject_id", "video"], observed=True), key=lambda entry: order.index(entry[0][1]) if entry[0][1] in order else len(order)):
        views.setdefault(str(subject), str(video))
    return views


def _frames_between(item, start: float, end: float) -> int:
    """The frames of one video from start to end on the recording clock, both ends included, up to the video's own end."""
    fps = item.fps or 0.0
    last = math.floor((end - item.offset_s) * fps)
    if item.span_s is not None:
        last = min(last, math.floor(item.span_s * fps + 1e-9) - 1)
    return max(last - max(math.ceil((start - item.offset_s) * fps), 0) + 1, 0)


def _videos_with_frames(sources, start: float, end: float) -> int:
    return sum(1 for item in sources if item.fps and _frames_between(item, start, end) > 0)


DETECTIONS_NAME = "detections.parquet"
LOOK_NAME = "look_names.json"
LOOKER = "the picture-reading model"
SWAPS_NAME = "swaps.json"
BACKEND_NAME = "backend.json"

_DETECT_SPECS = {
    "n_videos_read": ("Videos with frames in the span, which the pose backend was run on.", "count"),
    "frames_with_anyone": ("Frames in which the backend found anyone, over frames processed.", None),
}
_DETECT_MISSING = {
    "n_videos_read": "Never empty. 0 means no camera recorded in the span.",
    "frames_with_anyone": (
        "Empty for a span in which no camera recorded, as one before every camera started or after every camera stopped. 0 means "
        "the backend found nobody in any frame of the span."
    ),
}


def load_detections(config, recording_id: str) -> tuple[pd.DataFrame, dict]:
    directory = layout_for(config, create=False).artifacts_dir(recording_id, PoseDetectExtractor.name)
    path = directory / DETECTIONS_NAME
    if not path.is_file():
        raise InputNotFoundError(f"{recording_id}: no pose detections at {path}; run pose.detect first")
    return pd.read_parquet(path), json.loads((directory / BACKEND_NAME).read_text(encoding="utf-8"))


@register_extractor
class PoseDetectExtractor(Extractor):
    """Who the backend sees in each frame, before anyone is named, so naming people again does
    not run the model again."""

    name = "pose.detect"
    version = "0.1.1"
    scale = Scale.FRAME
    group = "pose"
    config_sections = (
        "pose.backend", "pose.model", "pose.device", "pose.min_confidence", "pose.input_size", "setup.cameras",
        "segment", "video",
    )
    requires = (MediaPrepExtractor.name, VideoPrepExtractor.name)
    resource_profile = "gpu_standard"

    def backend(self):
        return resolve_backend(self.config.pose.backend)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, backend.version)

    def check_available(self) -> None:
        backend = self.backend()
        backend.check_available()
        backend.check_model(self.config.pose.model)

    def inputs_for(self, recording: Recording) -> list[Path]:
        """The offset table moves every frame on the recording clock, so it is an input too."""
        paths = sorted(item.path for item in recording.cameras())
        offsets = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
        return paths + ([offsets] if offsets.is_file() else [])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.pose
        if not recording.cameras():
            raise UnreadableMediaError(f"{recording.recording_id}: the setup reads no camera here for anyone's body, face or gaze")

        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        start, end = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        warnings: list[QCWarning] = []
        sources = recording.cameras()

        pieces: list[pd.DataFrame] = []
        n_processed = n_with_anyone = 0
        for item in sources:
            fps = item.fps or 0.0
            if fps <= 0:
                raise UnreadableMediaError(f"{item.path}: no frame rate to convert frames to time")
            # only the task window is decoded; a short task in a long recording would otherwise
            # spend most of the run on frames that are thrown away
            first = max(math.floor((start - item.offset_s) * fps), 0)
            stop = math.ceil((end - item.offset_s) * fps) + 1
            if stop <= first:
                continue
            n_processed += _frames_between(item, start, end)
            expected = max(len(camera_people(item)), 1)
            frames = backend.infer(
                item.frames_path,
                device=cfg.device,
                min_confidence=cfg.min_confidence,
                model=cfg.model,
                max_people=expected,
                frames=(first, stop),
                input_size=cfg.input_size,
            )
            piece = frames.keypoints
            if piece.empty:
                continue
            piece = inside_frame(piece, frames.frame_size)

            # the backend counts frames from the file's own start; offset_s is what puts it on
            # the recording clock, and it is applied once, here, exactly as prep does for audio
            piece = piece.copy()
            piece["video"] = item.name
            piece["t"] = piece["frame"] / fps + item.offset_s
            piece = piece[(piece["t"] >= start) & (piece["t"] <= end)]
            if piece.empty:
                continue
            n_with_anyone += int(piece["frame"].nunique())
            pieces.append(piece)

        detections = (
            pd.concat(pieces, ignore_index=True) if pieces
            else empty_frame(BACKEND_KEYPOINT_SCHEMA).assign(video=pd.Series(dtype="string"), t=pd.Series(dtype=float))
        )
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        detections.to_parquet(directory / DETECTIONS_NAME, index=False)
        (directory / BACKEND_NAME).write_text(
            json.dumps({"name": backend.name, "version": backend.version, "joint_set": backend.joint_set}),
            encoding="utf-8",
        )
        return ExtractorResult(
            rows=self._summary(recording, _videos_with_frames(sources, start, end), n_with_anyone, n_processed, (start, end)),
            specs=[
                FeatureSpec(
                    name=name, description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                    extractor=self.name, extractor_version=self.version, backend=backend.name,
                    backend_version=backend.version, missing=_DETECT_MISSING[name], validation="none",
                )
                for name, (description, unit) in _DETECT_SPECS.items()
            ],
            artifacts={DETECTIONS_NAME: directory / DETECTIONS_NAME, BACKEND_NAME: directory / BACKEND_NAME},
            warnings=warnings,
        )

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        try:
            detections, _ = load_detections(self.config, recording.recording_id)
        except InputNotFoundError:
            return None
        recording = copy.deepcopy(recording)
        apply_prep_timing(self.config, recording)
        sources = recording.cameras()
        times = detections["t"].to_numpy(dtype=float)
        inside = detections[(times >= start_s) & (times <= end_s)]
        n_with_anyone = int(inside.groupby("video", observed=True)["frame"].nunique().sum()) if not inside.empty else 0
        n_processed = sum(_frames_between(item, start_s, end_s) for item in sources if item.fps)
        return self._summary(recording, _videos_with_frames(sources, start_s, end_s), n_with_anyone, n_processed, (start_s, end_s))

    def _summary(self, recording: Recording, n_videos: int, n_with_anyone: int, n_processed: int, window) -> pd.DataFrame:
        values = {
            "t_start": [window[0]],
            "t_end": [window[1]],
            "n_videos_read": [float(n_videos)],
            "frames_with_anyone": [n_with_anyone / n_processed if n_processed else float("nan")],
        }
        return build_long_rows(
            values=pd.DataFrame(values), scale=Scale.RECORDING, recording_id=recording.recording_id,
            source=self.name, entities=recording.entities(),
        )


@register_extractor
class PoseExtractor(Extractor):
    """Names the people pose.detect found, as a person clicked them; a camera the setup reads for one person alone
    needs no click while it never sees two people at once."""

    name = "pose.keypoints"
    version = "0.9.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose", "setup.cameras")
    requires = (PoseDetectExtractor.name,)

    def check_available(self) -> None:
        return None

    def inputs_for(self, recording: Recording) -> list[Path]:
        layout = layout_for(self.config, create=False)
        named = layout.names_path(recording.recording_id)
        detections = layout.artifacts_dir(recording.recording_id, PoseDetectExtractor.name) / DETECTIONS_NAME
        inputs = [detections] + ([named] if named.is_file() else [])
        if self.config.pose.suggest_names:
            # a suggestion may be read off each person's own mic
            prep = load_prep(self.config, recording.recording_id)
            inputs += [prep.track_path(subject) for subject in sorted(prep.tracks)]
        return inputs

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        detections, backend = load_detections(self.config, recording.recording_id)
        self._start_looking(recording)
        apply_prep_timing(self.config, recording)
        videos = load_video_prep(self.config, recording.recording_id)
        names = read_names(layout_for(self.config, create=False).names_path(recording.recording_id))
        warnings: list[QCWarning] = []
        owners: dict[str, dict] = {}
        pieces: list[pd.DataFrame] = []
        swaps_by_file: dict[str, list] = {}
        cameras = [item for item in recording.cameras() if not detections[detections["video"] == item.name].empty]
        # cameras a person named first, so a suggestion elsewhere can follow the heads named there
        cameras.sort(key=lambda item: not (names.get(item.name) or item.subject_id is not None))
        for item in cameras:
            piece = detections[detections["video"] == item.name].reset_index(drop=True)
            clicks = [self._click(item, videos.get(item.name), click) for click in names.get(item.name, [])]
            piece, swaps, owner = self._identify(piece, item, clicks, names.get(item.name, []), pieces, recording)
            swaps_by_file[item.name] = swaps
            if owner is not None:
                owners[item.name] = owner
                if owner["doubt"]:
                    warnings.append(self._owner_warning(recording, item.name, owner))
            pieces.append(piece)

        if not pieces:
            warnings.append(
                self._warning(recording, None, "fail", "no person was detected in any video")
            )
            return ExtractorResult(
                rows=concat_long_tables([]),
                specs=self._specs(backend),
                keypoints=empty_frame(KEYPOINT_SCHEMA),
                warnings=warnings,
            )

        keypoints = pd.concat(pieces, ignore_index=True)
        keypoints["recording_id"] = recording.recording_id
        keypoints["source"] = self.name
        keypoints = coerce_frame(keypoints, KEYPOINT_SCHEMA, "keypoints")

        artifacts = {**self._write_swaps(workspace, swaps_by_file), **self._write_looks(workspace)}
        if owners:
            artifacts["owners"] = self._write_json(workspace, "owners.json", owners)
        rows = self._rows(recording, keypoints, swaps_by_file, warnings)
        return ExtractorResult(
            rows=rows,
            specs=self._specs(backend),
            keypoints=keypoints,
            artifacts=artifacts,
            warnings=warnings,
        )

    @staticmethod
    def _click(item, video: dict | None, click: dict) -> Click:
        """A click at a moment of the camera's file, from its first picture, on a share of its picture's width and
        height, placed on the recording clock and in the pixels the keypoints are in."""
        first = video["first_picture_s"] if video else 0.0
        start = video["frames_start_s"] if video else 0.0
        width, height = (video.get("prepared_size") or video["size"]) if video else (1.0, 1.0)
        return Click(
            person=click["person"], at=float(click["at"]), t=item.offset_s + (float(click["at"]) + first - start) * item.clock_scale,
            x=float(click["x"]) * width, y=float(click["y"]) * height,
        )

    def _identify(self, piece: pd.DataFrame, item, clicks: list[Click], written: list[dict], named: list[pd.DataFrame], recording: Recording):
        """Attach subject_id. The stored track_id stays the backend's raw numbering, so a
        disputed assignment can still be traced back to what the model actually returned."""
        alone = item.subject_id
        if alone is not None and not clicks and not several_people(piece):
            piece["subject_id"] = alone
            return piece, [], None

        swaps: list = []
        how = ""
        identified = piece
        if alone is None and self.config.pose.track_repair:
            swaps = detect_swaps(piece, max_gap_s=self.config.pose.track_repair_max_gap_s)
            identified = repair_tracks(piece, swaps)
        if clicks:
            period = 1.0 / item.fps if item.fps else 0.1
            choice = named_by_clicks(identified, clicks, period)
            missing = [person for person in camera_people(item) if person not in choice.people]
            doubts = [choice.doubt] if choice.doubt else []
            if missing:
                doubts.append(f"{', '.join(missing)} {'was' if len(missing) == 1 else 'were'} not clicked on {item.name}")
            choice.doubt = "; ".join(doubts) or None
        elif self.config.pose.suggest_names and alone is not None:
            choice, how = most_in_view(identified, alone), "for being most in view"
        elif self.config.pose.suggest_names:
            choice, how = self._suggest(identified, item, named, recording)
        elif alone is not None:
            choice = PeopleChoice(people={}, doubt=(
                f"more than one person is in view at once on {item.name}, which the setup reads for {alone} alone: "
                f"click {alone} on a frame of it"
            ))
        else:
            choice = PeopleChoice(people={}, doubt=f"nobody has said who is who on {item.name} yet: click each person on a frame of it")

        piece["subject_id"] = subject_column(identified, choice.people).to_numpy()
        by = {click["person"]: f"{click.get('by', '')} (clicked at {float(click['at']):g} s)".strip() for click in written}
        owner = {
            "people": choice.people,
            "chosen_by": {subject: by.get(subject, f"suggested {how}") for subject in choice.people},
            "doubt": choice.doubt,
            "spans": choice.spans or [],
        }
        return piece, swaps, owner

    def _suggest(self, identified: pd.DataFrame, item, named: list[pd.DataFrame], recording: Recording) -> tuple[PeopleChoice, str]:
        """Who is who on a camera read for several people, from the first evidence there is: head motion where they are
        named, their own microphones, then the picture-reading model where qc.look.people describes them."""
        people = camera_people(item)
        references, voices, edges = self._evidence(named, people, recording)
        if references is None and voices is None and self._describes(people):
            return self._looked(identified, people, item), f"by {LOOKER}"
        how = "by head motion where they are named" if references else "by their own microphone"
        return suggested_people(identified, people, references=references, voices=voices, edges=edges), how

    def _describes(self, people: list[str]) -> bool:
        look = self.config.qc.look
        return bool(look.model) and all(person in look.people for person in people)

    def _looked(self, identified: pd.DataFrame, people: list[str], item) -> PeopleChoice:
        from macaw.qc.look import backend_for, parsed_answer, question
        from macaw.qc.overlay import read_frames

        look, state = self.config.qc.look, self._look_state
        described = {person: look.people[person] for person in people}
        if state.get("backend") is None:
            try:
                state["backend"] = backend_for(look)
                state["model"] = state["backend"].model_version()
            except MacawError as error:
                return PeopleChoice(people={}, doubt=f"{LOOKER} could not be asked ({error}); click the people on a frame")

        def ask(picture: bytes) -> dict | None:
            key = answer_key(picture, state["model"], described)
            if key not in state["answers"]:
                reply = state["backend"].complete(question(described), "Check this person.", seed=0, images=[picture])
                state["answers"][key] = parsed_answer(reply.text)
            state["used"].add(key)
            return state["answers"][key]

        choice, record = named_by_look(identified, people, lambda frames: read_frames(item.frames_path, frames), ask, LOOKER)
        state["cameras"][item.name] = record
        return choice

    def _start_looking(self, recording: Recording) -> None:
        """Answers kept from the last run are used again for the same picture, model and descriptions."""
        earlier = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, self.name) / LOOK_NAME
        kept = json.loads(earlier.read_text(encoding="utf-8")).get("answers", {}) if earlier.is_file() else {}
        self._look_state = {"answers": kept, "used": set(), "cameras": {}}

    def _write_looks(self, workspace: Path) -> dict[str, Path]:
        state = self._look_state
        if not state["cameras"]:
            return {}
        look = self.config.qc.look
        payload = {
            "model": state.get("model"), "provider": look.provider, "described": look.people, "cameras": state["cameras"],
            "answers": {key: value for key, value in state["answers"].items() if key in state["used"]},
        }
        return {LOOK_NAME: self._write_json(workspace, LOOK_NAME, payload)}

    def extra_key(self, recording: Recording) -> str:
        if not self.config.pose.suggest_names:
            return ""
        look = self.config.qc.look
        return json.dumps([look.provider, look.model, look.people], sort_keys=True)

    def _evidence(self, named: list[pd.DataFrame], people: list[str], recording: Recording):
        """(references, voices, edges) to suggest who is who from: each person's head motion on the first camera
        they were named on when everyone sought is named somewhere, else their talking on their own mic."""
        prep = load_prep(self.config, recording.recording_id)
        edges = bin_edges(prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        first: dict[str, pd.DataFrame] = {}
        for piece in named:
            for subject, group in piece[piece["subject_id"].notna()].groupby("subject_id"):
                first.setdefault(str(subject), group)
        if people and all(person in first for person in people):
            return {person: head_motion(first[person], edges) for person in people}, None, edges
        tracks = {person: prep.track_path(person) for person in people if person in prep.tracks}
        if tracks:
            return None, speech_activity(tracks, prep.t_origin_s, edges), edges
        return None, None, None

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        layout = layout_for(self.config, create=False)
        path = layout.keypoints_path(recording.recording_id) / f"{self.name}.parquet"
        if not path.is_file():
            return None
        keypoints = read_keypoints(path, t_range=(start_s, end_s))
        if keypoints.empty:
            return concat_long_tables([])
        saved = layout.artifacts_dir(recording.recording_id, self.name) / SWAPS_NAME
        events = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else {}
        swaps = {name: [event for event in found if start_s <= event["t"] < end_s] for name, found in events.items()}
        return self._rows(recording, keypoints, swaps, [])

    def segment_warnings(self, recording: Recording, segment) -> list[QCWarning]:
        path = layout_for(self.config, create=False).keypoints_path(recording.recording_id) / f"{self.name}.parquet"
        if not path.is_file():
            return []
        keypoints = read_keypoints(path, t_range=(segment.start_s, segment.end_s))
        where = f" in segment {segment.name}"
        warnings: list[QCWarning] = []
        seen = set()
        if not keypoints.empty:
            for subject_id, detection_rate, quality in self._people(recording, keypoints):
                seen.add(subject_id)
                self._quality_warnings(recording, subject_id, detection_rate, quality, warnings, where)
        for person in sorted(set(recording.views("body")) - seen):
            warnings.append(self._warning(recording, person, "warn", f"not detected at all{where}"))
        return [replace(warning, t_start=segment.start_s, t_end=segment.end_s) for warning in warnings]

    def _owner_warning(self, recording: Recording, file_name: str, owner: dict) -> QCWarning:
        spans = ", ".join(f"{start:.1f}-{end:.1f} s" for start, end in owner["spans"][:5])
        where = f" (look at {spans})" if spans else ""
        subject = next(iter(owner["people"])) if len(owner["people"]) == 1 else None
        return self._warning(recording, subject, "warn", f"{file_name}: {owner['doubt']}{where}")

    def _write_swaps(self, workspace: Path, swaps_by_file: dict[str, list]) -> dict[str, Path]:
        """Evidence for the identity review step, not an input to anything computed."""
        if not any(swaps_by_file.values()):
            return {}
        payload = {
            name: [
                {
                    "t": event.t,
                    "frame": event.frame,
                    "track_a": event.track_a,
                    "track_b": event.track_b,
                    "evidence": event.evidence,
                    "score": event.score,
                }
                for event in events
            ]
            for name, events in swaps_by_file.items()
        }
        return {SWAPS_NAME: self._write_json(workspace, SWAPS_NAME, payload)}

    def _write_json(self, workspace: Path, file_name: str, payload) -> Path:
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / file_name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _rows(
        self,
        recording: Recording,
        keypoints: pd.DataFrame,
        swaps_by_file: dict[str, list],
        warnings: list[QCWarning],
    ) -> pd.DataFrame:
        entities = recording.entities()
        common = dict(recording_id=recording.recording_id, source=self.name, entities=entities)
        t_start = float(keypoints["t"].min())
        t_end = float(keypoints["t"].max())

        frames = []
        for subject_id, detection_rate, quality in self._people(recording, keypoints):
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [t_start],
                            "t_end": [t_end],
                            "detection_rate": [detection_rate],
                            "mean_confidence": [float(quality["mean_confidence"].mean())],
                            "missing_fraction": [float(quality["missing_fraction"].mean())],
                            "track_continuity": [float(quality["continuity"].min())],
                        }
                    ),
                    scale=Scale.RECORDING,
                    subject_id=subject_id,
                    **common,
                )
            )
            self._quality_warnings(recording, subject_id, detection_rate, quality, warnings)

        n_swaps = sum(len(events) for events in swaps_by_file.values())
        frames.append(
            build_long_rows(
                values=pd.DataFrame(
                    {"t_start": [t_start], "t_end": [t_end], "n_track_swaps": [float(n_swaps)]}
                ),
                scale=Scale.RECORDING,
                **common,
            )
        )
        if n_swaps:
            warnings.append(
                self._warning(
                    recording, None, "info",
                    f"{n_swaps} identity crossover(s) were detected and repaired; the evidence "
                    "is in swaps.json",
                )
            )
        if keypoints["subject_id"].isna().any():
            warnings.append(
                self._warning(
                    recording, None, "warn",
                    "some tracks could not be matched to a subject and carry no subject_id",
                )
            )
        return concat_long_tables(frames)

    def _people(self, recording: Recording, keypoints: pd.DataFrame):
        """Each named person's detection rate and track quality on the camera their body is measured on."""
        views = measured_views(keypoints, recording)
        for subject_id, group in keypoints.groupby("subject_id", dropna=True, observed=True):
            group = group[group["video"] == views[str(subject_id)]]
            n_frames = int(keypoints.loc[keypoints["video"] == views[str(subject_id)], "frame"].nunique())
            # one person may be several tracker ids joined together; their gaps are what count
            quality = track_quality(group.assign(track_id=0))
            seen = int(group.loc[group["x"].notna(), "frame"].nunique())
            yield str(subject_id), (seen / n_frames if n_frames else float("nan")), quality

    def _quality_warnings(self, recording, subject_id, detection_rate, quality, warnings, where: str = "") -> None:
        if detection_rate == detection_rate and detection_rate < _LOW_DETECTION_RATE:
            warnings.append(self._warning(recording, subject_id, "warn", f"detected in only {detection_rate:.0%} of frames{where}"))
        continuity = float(quality["continuity"].min()) if not quality.empty else float("nan")
        if continuity == continuity and continuity < _LOW_CONTINUITY:
            warnings.append(
                self._warning(recording, subject_id, "warn", f"track continuity is {continuity:.2f}{where}; the track disappears and returns")
            )

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return _warning(self.name, recording, subject_id, severity, message)

    def _specs(self, backend) -> list[FeatureSpec]:
        return [
            FeatureSpec(
                name=name,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                backend=backend["name"],
                backend_version=backend["version"],
                params={"joint_set": backend["joint_set"], "min_confidence": self.config.pose.min_confidence},
                aggregation="mean",
                missing=_MISSING.get(name, _MISSING["person"]),
                comparable_across_backends=False,
            )
            for name, (description, unit) in _SPECS.items()
        ]


def waiting_cameras(config, recording_id: str, *, every: bool) -> list[str]:
    """Cameras whose people no person has kept yet: those in doubt, or all of them with every."""
    layout = layout_for(config, create=False)
    owners = read_owner_file(layout.artifacts_dir(recording_id, PoseExtractor.name) / "owners")
    accepted = read_owner_file(layout.camera_owners_accepted_path(recording_id))
    return [
        file_name
        for file_name, owner in owners.items()
        if (every or owner["doubt"]) and accepted.get(file_name, {}).get("people") != owner["people"]
    ]


@register_extractor
class PoseReviewGate(Gate):
    """Held where nobody has said who is who on a camera, where a click did not settle it, or on every camera with review always."""

    name = "pose.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "pose"
    config_sections = ("pose", "setup.cameras")
    requires = (PoseExtractor.name,)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return []

    def _in_doubt(self, recording: Recording) -> list[str]:
        return waiting_cameras(self.config, recording.recording_id, every=False)

    def pending(self, recording: Recording) -> str | None:
        if self.config.pose.review == "skip":
            return None
        waiting = waiting_cameras(self.config, recording.recording_id, every=self.config.pose.review == "always")
        if not waiting:
            return None
        rid = recording.recording_id
        why = "(pose.review is always)" if self.config.pose.review == "always" else "(the warnings say why)"
        return (
            f"who is on {', '.join(waiting)} waits for a person {why}: click the people on a frame of each camera on the "
            f"Setup page (`macaw name-people -r {rid}`), look at `macaw qc -r {rid} --render`, and "
            f"`macaw accept-owner -r {rid}` keeps the naming as it stands"
        )

    def notes(self, recording: Recording) -> list[QCWarning]:
        if self.config.pose.review != "skip" or not self._in_doubt(recording):
            return []
        return [
            QCWarning(
                check="camera_owner_unreviewed",
                recording_id=recording.recording_id,
                subject_id=None,
                severity="warn",
                message="pose.review is skip: who is on camera was used as it stood, doubts and all, without a person looking",
            )
        ]
