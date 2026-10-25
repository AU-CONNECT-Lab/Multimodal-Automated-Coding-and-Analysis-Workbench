from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from macaw.exceptions import UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, apply_prep_timing
from macaw.extractors.video.pose.backends import resolve_backend
from macaw.extractors.video.pose.track_repair import (
    assign_subjects,
    detect_swaps,
    repair_tracks,
    track_quality,
)
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import KEYPOINT_SCHEMA, Scale, coerce_frame, empty_frame

# A track this broken is worth telling the user about rather than quietly averaging over.
_LOW_CONTINUITY = 0.8
_LOW_DETECTION_RATE = 0.5

_SPECS = {
    "detection_rate": ("Frames in which this person was detected, over frames processed.", None),
    "mean_confidence": ("Mean keypoint confidence over detected frames.", None),
    "missing_fraction": ("Standard joints without a usable coordinate.", None),
    "track_continuity": ("Frames seen over frames spanned by the track.", None),
    "n_track_swaps": ("Identity crossovers detected between tracks.", "count"),
}


def _video_files(recording: Recording) -> list:
    return [item for item in recording.media if item.kind == "video"]


def select_videos(recording: Recording) -> tuple[list, list]:
    """(read, skipped). Per-person cameras answer the identity question by themselves, so they
    win over a shared one, the same preference as per_track over a shared mic. Only one view is
    read per person or per room: a second would put two rows on one frame and joint, numbered
    on another file's clock."""
    videos = sorted(_video_files(recording), key=lambda item: str(item.path))
    per_subject = [item for item in videos if item.subject_id is not None]
    if per_subject:
        chosen: dict[str, object] = {}
        for item in per_subject:
            chosen.setdefault(item.subject_id, item)
        read = list(chosen.values())
    else:
        read = videos[:1]
    return read, [item for item in videos if item not in read]


@register_extractor
class PoseExtractor(Extractor):
    name = "pose.keypoints"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose", "segment")
    requires = (MediaPrepExtractor.name,)
    resource_profile = "gpu_standard"

    def backend(self):
        return resolve_backend(self.config.pose.backend)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, backend.version)

    def check_available(self) -> None:
        self.backend().check_available()

    def inputs_for(self, recording: Recording) -> list[Path]:
        """The offset table moves every frame on the recording clock, so it is an input too."""
        paths = sorted(item.path for item in select_videos(recording)[0])
        offsets = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
        return paths + [offsets] if offsets.is_file() else paths

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.pose
        if not _video_files(recording):
            raise UnreadableMediaError(f"{recording.recording_id}: no video to read")

        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        start, end = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        subjects = [subject.subject_id for subject in recording.subjects]
        warnings: list[QCWarning] = []

        sources, skipped = select_videos(recording)
        per_subject = any(item.subject_id is not None for item in sources)
        if per_subject and any(item.subject_id is None for item in skipped):
            warnings.append(
                self._warning(
                    recording, None, "info",
                    "both per-subject and shared cameras are present; the shared ones were "
                    "not read",
                )
            )
        repeated = [
            item.path.name for item in skipped if item.subject_id is not None or not per_subject
        ]
        if repeated:
            warnings.append(
                self._warning(
                    recording, None, "warn",
                    f"only one camera per person or per room is read; skipped {', '.join(repeated)}",
                )
            )

        pieces: list[pd.DataFrame] = []
        swaps_by_file: dict[str, list] = {}
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
            expected = 1 if item.subject_id is not None else max(len(subjects), 1)
            frames = backend.infer(
                item.path,
                device=cfg.device,
                min_confidence=cfg.min_confidence,
                model=cfg.model,
                max_people=expected,
                frames=(first, stop),
            )
            piece = frames.keypoints
            if piece.empty:
                continue

            # the backend counts frames from the file's own start; offset_s is what puts it on
            # the recording clock, and it is applied once, here, exactly as prep does for audio
            piece = piece.copy()
            piece["t"] = piece["frame"] / fps + item.offset_s
            piece = piece[(piece["t"] >= start) & (piece["t"] <= end)]
            if piece.empty:
                continue

            piece, swaps = self._identify(piece, item, subjects, cfg)
            swaps_by_file[item.path.name] = swaps
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

        artifacts = self._write_swaps(workspace, swaps_by_file)
        rows = self._rows(recording, keypoints, swaps_by_file, warnings)
        return ExtractorResult(
            rows=rows,
            specs=self._specs(backend),
            keypoints=keypoints,
            artifacts=artifacts,
            warnings=warnings,
        )

    def _identify(self, piece: pd.DataFrame, item, subjects: list[str], cfg):
        """Attach subject_id. The stored track_id stays the backend's raw numbering, so a
        disputed assignment can still be traced back to what the model actually returned."""
        if item.subject_id is not None:
            piece["subject_id"] = item.subject_id
            return piece, []

        swaps = detect_swaps(piece, max_gap_s=cfg.track_repair_max_gap_s) if cfg.track_repair else []
        repaired = repair_tracks(piece, swaps)
        assigned = assign_subjects(repaired, subjects, identity_anchors=cfg.identity_anchors)
        piece["subject_id"] = assigned["subject_id"].to_numpy()
        return piece, swaps

    def _write_swaps(self, workspace: Path, swaps_by_file: dict[str, list]) -> dict[str, Path]:
        """Evidence for the identity review step, not an input to anything computed."""
        if not any(swaps_by_file.values()):
            return {}
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "swaps.json"
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
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"swaps": path}

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
        n_frames = int(keypoints["frame"].nunique())

        frames = []
        for subject_id, group in keypoints.groupby("subject_id", dropna=True, observed=True):
            quality = track_quality(group)
            seen = int(group.loc[group["x"].notna(), "frame"].nunique())
            detection_rate = seen / n_frames if n_frames else float("nan")
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
                    subject_id=str(subject_id),
                    **common,
                )
            )
            self._quality_warnings(recording, str(subject_id), detection_rate, quality, warnings)

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

    def _quality_warnings(self, recording, subject_id, detection_rate, quality, warnings) -> None:
        if detection_rate == detection_rate and detection_rate < _LOW_DETECTION_RATE:
            warnings.append(
                self._warning(
                    recording, subject_id, "warn",
                    f"detected in only {detection_rate:.0%} of frames",
                )
            )
        continuity = float(quality["continuity"].min()) if not quality.empty else float("nan")
        if continuity == continuity and continuity < _LOW_CONTINUITY:
            warnings.append(
                self._warning(
                    recording, subject_id, "warn",
                    f"track continuity is {continuity:.2f}; the track disappears and returns",
                )
            )

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(
            check=self.name,
            recording_id=recording.recording_id,
            subject_id=subject_id,
            severity=severity,
            message=message,
        )

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
                backend=backend.name,
                backend_version=backend.version,
                params={"joint_set": backend.joint_set, "min_confidence": self.config.pose.min_confidence},
                aggregation="mean",
                comparable_across_backends=False,
            )
            for name, (description, unit) in _SPECS.items()
        ]
