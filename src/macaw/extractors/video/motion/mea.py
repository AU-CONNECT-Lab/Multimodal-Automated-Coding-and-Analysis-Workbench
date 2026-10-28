"""Motion Energy Analysis: frame differencing per ROI. Cheap, stable, independent of pose."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ConfigError, ExternalToolError, UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, apply_prep_timing
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# more of the whole picture than this changing at once is a light switch or a bumped camera
WHOLE_PICTURE_CHANGE = 0.5
PICTURE_NAME = "picture.parquet"

_DESCRIPTION = (
    "Share of the region's pixels whose grey level changed by more than motion.threshold since "
    "the previous frame (motion energy analysis, frame differencing)."
)


@register_extractor
class MEAExtractor(Extractor):
    name = "motion.mea"
    version = "0.2.0"
    scale = Scale.FRAME
    group = "motion"
    config_sections = ("motion", "segment", "video")
    requires = (MediaPrepExtractor.name, VideoPrepExtractor.name)

    def check_available(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError('motion.mea needs opencv: pip install "pymacaw[vision]"') from error

    def inputs_for(self, recording: Recording) -> list[Path]:
        paths = sorted({item.path for item, _ in self._regions(recording).values()})
        offsets = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
        return paths + ([offsets] if offsets.is_file() else [])

    def _regions(self, recording: Recording) -> dict[str, tuple[object, tuple[int, int, int, int]]]:
        """subject -> (the video their region is drawn on, the region): their own camera, else the room's."""
        videos = sorted((item for item in recording.media if item.kind == "video"), key=lambda item: str(item.path))
        if not videos:
            raise UnreadableMediaError(f"{recording.recording_id}: no video to read")
        own = {}
        for item in videos:
            if item.subject_id is not None:
                own.setdefault(item.subject_id, item)
        shared = [item for item in videos if item.subject_id is None]
        missing = [subject.subject_id for subject in recording.subjects if subject.roi is None]
        if missing:
            raise ConfigError(
                f"{recording.recording_id}: motion.rois has no region for {', '.join(missing)}; draw one per "
                "person as subject_id: [x, y, width, height] in pixels of their own camera, or of the shared "
                "camera when they have none"
            )
        regions = {}
        for subject in recording.subjects:
            video = own.get(subject.subject_id) or (shared[0] if shared else None)
            if video is None:
                raise ConfigError(f"{recording.recording_id}: no camera shows {subject.subject_id}")
            regions[subject.subject_id] = (video, tuple(int(value) for value in subject.roi))
        return regions

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.motion
        prep = apply_prep_timing(self.config, recording)
        start, end = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        regions = self._regions(recording)
        warnings: list[QCWarning] = []

        by_video: dict[Path, tuple[object, dict]] = {}
        for subject_id, (item, roi) in regions.items():
            by_video.setdefault(item.path, (item, {}))[1][subject_id] = roi

        frames, pictures = [], []
        for path, (item, rois) in by_video.items():
            fps = item.fps or 0.0
            if fps <= 0:
                raise UnreadableMediaError(f"{path}: no frame rate to convert frames to time")
            first = max(math.floor((start - item.offset_s) * fps), 0)
            stop = math.ceil((end - item.offset_s) * fps) + 1
            if stop <= first:
                continue
            energy, size = frame_difference_energy(item.frames_path, rois, threshold=cfg.threshold, frames=(first, stop))
            energy["t"] = energy["frame"] / fps + item.offset_s
            energy = energy[(energy["t"] >= start) & (energy["t"] <= end)]
            pictures.append(energy[["frame", "t", "whole_picture", "identical"]].assign(video=path.name))
            warnings += self._clip_warnings(recording, path, rois, size)
            n_jumps = int((energy["whole_picture"] > WHOLE_PICTURE_CHANGE).sum())
            if n_jumps:
                warnings.append(
                    self._warning(
                        recording, None, "warn",
                        f"{path.name}: in {n_jumps} frame(s) more than half of the whole picture changed at once "
                        "(lighting or a moved camera); motion energy there is not movement",
                    )
                )
            for subject_id in rois:
                series = smooth(energy[subject_id], round(cfg.smoothing_s * fps))
                if not (series > 0).any():
                    warnings.append(
                        self._warning(recording, subject_id, "warn", f"{path.name}: nothing moved in this person's region")
                    )
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame({"t_start": energy["t"].to_numpy(), "motion_energy": series.to_numpy()}),
                        scale=Scale.FRAME,
                        recording_id=recording.recording_id,
                        source=self.name,
                        entities=recording.entities(),
                        subject_id=subject_id,
                    )
                )

        # for the checks: what changed across the whole picture, not only inside the regions
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        picture = pd.concat(pictures, ignore_index=True) if pictures else pd.DataFrame(
            columns=["frame", "t", "whole_picture", "identical", "video"]
        )
        picture.to_parquet(directory / PICTURE_NAME, index=False)
        return ExtractorResult(
            rows=concat_long_tables(frames),
            artifacts={PICTURE_NAME: directory / PICTURE_NAME},
            specs=[
                FeatureSpec(
                    name="motion_energy",
                    description=_DESCRIPTION,
                    unit=None,
                    scale=Scale.FRAME,
                    group=self.group,
                    extractor=self.name,
                    extractor_version=self.version,
                    params={"threshold": cfg.threshold, "smoothing_s": cfg.smoothing_s},
                    interpolation="none",
                    aggregation="mean",
                    # the share depends on how much of the region the person fills
                    comparable_across_subjects=False,
                    validation="synthetic",
                )
            ],
            warnings=warnings,
        )

    def _clip_warnings(self, recording, path: Path, rois: dict, size: tuple[int, int]) -> list[QCWarning]:
        width, height = size
        return [
            self._warning(
                recording, subject_id, "warn",
                f"{path.name}: the region {list(roi)} reaches outside the {width}x{height} picture and was cut to it",
            )
            for subject_id, roi in rois.items()
            if roi[0] < 0 or roi[1] < 0 or roi[0] + roi[2] > width or roi[1] + roi[3] > height
        ]

    def _warning(self, recording: Recording, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(
            check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity=severity, message=message
        )


def frame_difference_energy(
    video_path: Path,
    rois: dict[str, tuple[int, int, int, int]],
    *,
    threshold: int,
    frames: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, tuple[int, int]]:
    """(table, picture size). Per frame (file's own numbering, half-open frames range), the share of each region's pixels
    that changed by more than threshold grey levels, and of the whole picture, and whether the
    picture is identical to the previous one (a stalled recording); NaN on the first."""
    import cv2

    from macaw.extractors.video.pose.backends import _seek

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExternalToolError(f"could not open video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        boxes = {subject: _clip(roi, width, height) for subject, roi in rois.items()}
        empty = [subject for subject, box in boxes.items() if box is None]
        if empty:
            raise ConfigError(f"{video_path.name}: the region for {', '.join(empty)} lies outside the {width}x{height} picture")
        first, stop = frames if frames is not None else (0, None)
        index = _seek(capture, cv2, max(int(first), 0))
        rows, previous = [], None
        while stop is None or index < stop:
            ok, image = capture.read()
            if not ok:
                break
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int16)
            row = {"frame": index}
            if previous is None:
                row.update({subject: np.nan for subject in boxes}, whole_picture=np.nan, identical=False)
            else:
                change = np.abs(grey - previous)
                moved = change > threshold
                row["whole_picture"] = float(moved.mean())
                row["identical"] = not change.any()
                for subject, (x0, y0, x1, y1) in boxes.items():
                    row[subject] = float(moved[y0:y1, x0:x1].mean())
            rows.append(row)
            previous = grey
            index += 1
    finally:
        capture.release()
    return pd.DataFrame(rows, columns=["frame", "whole_picture", "identical", *boxes]), (width, height)


def _clip(roi: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = roi
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, width), min(y + h, height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def smooth(series: pd.Series, n_frames: int) -> pd.Series:
    """Centred moving average over n_frames; under two frames leaves the series as it is."""
    if n_frames < 2:
        return series
    return series.rolling(n_frames, center=True, min_periods=1).mean()
