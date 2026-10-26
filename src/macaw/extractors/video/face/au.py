from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import apply_prep_timing
from macaw.extractors.video.face.backends import resolve_backend
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate, primary_views
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store import keypoints as keypoints_store
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# a face is a person's when one of these of theirs lies inside its box
NAMING_JOINTS = ("nose", "left_eye", "right_eye")
LANDMARKS_NAME = "landmarks.parquet"
FACES_NAME = "faces.json"
_LOW_FACE_RATE = 0.5

ACTION_UNITS = {
    "AU01": "inner brow raiser", "AU02": "outer brow raiser", "AU04": "brow lowerer", "AU05": "upper lid raiser",
    "AU06": "cheek raiser", "AU07": "lid tightener", "AU09": "nose wrinkler", "AU10": "upper lip raiser",
    "AU11": "nasolabial deepener", "AU12": "lip corner puller", "AU14": "dimpler", "AU15": "lip corner depressor",
    "AU17": "chin raiser", "AU20": "lip stretcher", "AU23": "lip tightener", "AU24": "lip pressor",
    "AU25": "lips part", "AU26": "jaw drop", "AU28": "lip suck", "AU43": "eyes closed",
}
_POSE_SPECS = {
    "face_pitch_deg": "Head pitch from the face model.",
    "face_roll_deg": "Head roll from the face model.",
    "face_yaw_deg": "Head yaw from the face model.",
    "gaze_pitch_deg": "Gaze pitch from the face model, both eyes together; up is positive.",
    "gaze_yaw_deg": "Gaze yaw from the face model, both eyes together; toward the person's own right is positive.",
}


@register_extractor
class FaceExtractor(Extractor):
    name = "face.au"
    version = "0.2.0"
    scale = Scale.FRAME
    group = "face"
    config_sections = (
        "face.backend", "face.model", "face.device", "face.action_units", "face.landmarks", "face.gaze", "segment",
    )
    requires = (PoseReviewGate.name,)
    resource_profile = "gpu_standard"

    def backend(self):
        return resolve_backend(self.config.face.backend)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, backend.version)

    def check_available(self) -> None:
        backend = self.backend()
        if self.config.face.gaze and not backend.provides_gaze:
            raise BackendNotAvailableError(f"face.gaze is on but face.backend {backend.name} gives no gaze direction")

    def inputs_for(self, recording: Recording) -> list[Path]:
        """Faces are named through the people pose named, so the keypoints are the input that matters."""
        return [_keypoints_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Emits a warning on child data: AU models are trained on adults."""
        cfg = self.config.face
        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        start, end = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        keypoints = keypoints_store.read_keypoints(
            _keypoints_file(self.config, recording.recording_id), recording_id=recording.recording_id
        )
        views = primary_views(keypoints, recording)
        by_name = {item.path.name: item for item in recording.media if item.kind == "video"}
        warnings: list[QCWarning] = []
        per_subject: dict[str, pd.DataFrame] = {}
        seen_frames: dict[str, int] = {}
        for video in sorted(set(views.values())):
            item = by_name[video]
            fps = item.fps or 0.0
            if fps <= 0:
                raise UnreadableMediaError(f"{item.path}: no frame rate to convert frames to time")
            first = max(math.floor((start - item.offset_s) * fps), 0)
            stop = math.ceil((end - item.offset_s) * fps) + 1
            faces = backend.infer(
                item.path, frames=(first, stop), device=cfg.device, model=cfg.model,
                batch_size=cfg.batch_size, landmarks=cfg.landmarks, gaze=cfg.gaze,
            )
            subjects = [subject for subject, view in views.items() if view == video]
            named = name_faces(faces, keypoints[keypoints["video"] == video], subjects)
            named["t"] = named["frame"] / fps + item.offset_s
            named = named[(named["t"] >= start) & (named["t"] <= end)]
            on_camera = keypoints[(keypoints["video"] == video) & keypoints["x"].notna()]
            for subject in subjects:
                per_subject[subject] = named[named["subject_id"] == subject]
                seen_frames[subject] = int(on_camera.loc[on_camera["subject_id"] == subject, "frame"].nunique())

        frames, landmark_rows = [], []
        for subject, faces in per_subject.items():
            if seen_frames[subject] and len(faces) / seen_frames[subject] < _LOW_FACE_RATE:
                warnings.append(
                    self._warning(
                        recording, subject, "warn",
                        f"a face was found in {len(faces)} of the {seen_frames[subject]} frames this person was seen in",
                    )
                )
            if faces.empty:
                continue
            values = pd.DataFrame({"t_start": faces["t"].to_numpy()})
            if cfg.action_units:
                for column in [column for column in faces.columns if column in ACTION_UNITS]:
                    values[column.lower()] = faces[column].to_numpy(dtype=float)
            for name, source in (("face_pitch_deg", "pitch"), ("face_roll_deg", "roll"), ("face_yaw_deg", "yaw")):
                values[name] = faces[source].to_numpy(dtype=float)
            if cfg.gaze:
                for name, source in (("gaze_pitch_deg", "gaze_pitch"), ("gaze_yaw_deg", "gaze_yaw")):
                    values[name] = faces[source].to_numpy(dtype=float)
            frames.append(
                build_long_rows(
                    values=values, scale=Scale.FRAME, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=subject,
                )
            )
            if cfg.landmarks:
                points = [column for column in faces.columns if column.startswith(("x_", "y_"))]
                landmark_rows.append(faces[["subject_id", "frame", "t", *points]])
        warnings.append(
            self._warning(
                recording, None, "info",
                "face models are trained on adult faces; read action units of children with care",
            )
        )
        rows = concat_long_tables(frames)
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        if landmark_rows:
            pd.concat(landmark_rows, ignore_index=True).to_parquet(directory / LANDMARKS_NAME, index=False)
            artifacts[LANDMARKS_NAME] = directory / LANDMARKS_NAME
        (directory / FACES_NAME).write_text(json.dumps({"views": views, "model": cfg.model or "v1"}, indent=2), encoding="utf-8")
        artifacts[FACES_NAME] = directory / FACES_NAME
        return ExtractorResult(
            rows=rows, specs=self._specs(set(rows["feature"]), backend), artifacts=artifacts, warnings=warnings
        )

    def _specs(self, emitted: set[str], backend) -> list[FeatureSpec]:
        model = self.config.face.model or "v1"
        described = {
            **{
                code.lower(): (f"Action unit {int(code[2:])} ({meaning}), 0 to 1 as the {backend.name} {model} model scores it.", None)
                for code, meaning in ACTION_UNITS.items()
            },
            **{name: (text, "deg") for name, text in _POSE_SPECS.items()},
        }
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.FRAME, group=self.group,
                extractor=self.name, extractor_version=self.version, backend=backend.name,
                backend_version=backend.version, params={"model": model}, interpolation="none",
                aggregation="mean", comparable_across_backends=False, validation="none",
            )
            for name, (description, unit) in described.items()
            if name in emitted
        ]

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(
            check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message
        )


def name_faces(faces: pd.DataFrame, keypoints: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """Each face goes to the person whose nose or eyes lie inside its box; a person keeps one face
    per frame, the most confident, and a face no one's head is in stays out."""
    if faces.empty:
        return faces.assign(subject_id=pd.Series(dtype="object"))
    head = keypoints[keypoints["joint"].isin(NAMING_JOINTS) & keypoints["x"].notna() & keypoints["subject_id"].isin(subjects)]
    head = head.groupby(["frame", "subject_id"], observed=True)[["x", "y"]].median().reset_index()
    joined = faces.reset_index(names="face").merge(head, on="frame", how="inner")
    inside = (
        (joined["x"] >= joined["face_x"]) & (joined["x"] <= joined["face_x"] + joined["face_width"])
        & (joined["y"] >= joined["face_y"]) & (joined["y"] <= joined["face_y"] + joined["face_height"])
    )
    joined = joined[inside].sort_values("face_score", ascending=False)
    # one face per person and one person per face, most confident first
    joined = joined.drop_duplicates(["frame", "subject_id"]).drop_duplicates("face")
    out = faces.loc[joined["face"].to_numpy()].copy()
    out["subject_id"] = joined["subject_id"].astype(str).to_numpy()
    return out.sort_values(["subject_id", "frame"]).reset_index(drop=True)


def _keypoints_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).keypoints_path(recording_id) / f"{PoseExtractor.name}.parquet"


def blink_rate(landmarks: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def mouth_opening(landmarks: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def cross_backend_agreement(au_a: pd.DataFrame, au_b: pd.DataFrame) -> pd.DataFrame:
    """QC only. AU values are never merged across backends; different intensity calibration."""
    raise NotImplementedError
