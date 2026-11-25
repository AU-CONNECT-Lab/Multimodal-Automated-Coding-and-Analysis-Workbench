"""Frames of the video a rating shows between the lines of its transcript: when they are taken, from which camera, and
the one box each recording's frames are cropped to."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.config.schema import MacawConfig, RatingFrames
from macaw.exceptions import UnreadableMediaError
from macaw.media.ffmpeg import extract_frame
from macaw.project.layout import layout_for
from macaw.project.recording import MediaFile, Recording

FRAMES_NAME = "frames.parquet"
CROP_NAME = "frames_crop.json"
PREVIEW_NAME = "frames_crop.jpg"
PEOPLE, WHOLE_PICTURE, GIVEN, NO_POSE = "people", "whole", "given", "no pose"
# share of the boxes left out at each edge, so a stray detection does not widen the crop
_OUTLIER = 2.0
_PREVIEW_SIDE = 960
_TIME_EPSILON = 1e-6
FRAME_COLUMNS = ["segment", "scale", "frame", "t", "video", "video_t", "shown"]


@dataclass
class Crop:
    """The box a recording's frames are cut to, in the video's own pixels, and how it was found."""

    box: tuple[int, int, int, int]
    source: str
    width: int
    height: int
    people: list[str] = field(default_factory=list)

    def as_dict(self, camera: str) -> dict:
        return {"camera": camera, "box": list(self.box), "source": self.source, "width": self.width, "height": self.height, "people": self.people}


def frame_times(start: float, end: float, every_s: float, spans: list[tuple[float, float]] | None = None) -> list[float]:
    """Moments on the recording clock from start up to end, every_s apart; with spans, from the start of each span that
    lies in start to end, within it, a span shorter than every_s still giving one."""
    stretches = [(start, end)] if spans is None else [(max(low, start), min(high, end)) for low, high in spans]
    found: list[float] = []
    for low, high in stretches:
        if high < low:
            continue
        count = max(int(np.ceil((high - low) / every_s - _TIME_EPSILON)), 1)
        found += [low + index * every_s for index in range(count)]
    found.sort()
    return [moment for index, moment in enumerate(found) if index == 0 or moment - found[index - 1] > _TIME_EPSILON]


def frame_camera(recording: Recording) -> MediaFile:
    """The camera read for most people's bodies, the first in the setup among equals, else the recording's main video."""
    videos = [item for item in recording.media if item.kind == "video"]
    if not videos:
        raise UnreadableMediaError(f"{recording.recording_id}: the rating shows frames, but the recording has no video")
    reading = [item for item in recording.cameras() if item.uses.get("body")]
    if reading:
        return max(reading, key=lambda item: len(item.uses["body"]))
    return recording.reference_media(videos)


def crop_for(
    config: MacawConfig, recording: Recording, camera: MediaFile, settings: RatingFrames, start: float, end: float,
    size: tuple[int, int],
) -> Crop:
    """The box given in the settings, the whole picture, or the named people's bodies in start to end with margin."""
    width, height = size
    if isinstance(settings.crop, tuple):
        x0, y0, x1, y1 = settings.crop
        return Crop(_clipped(x0, y0, x1, y1, width, height), GIVEN, width, height)
    if settings.crop == WHOLE_PICTURE:
        return Crop((0, 0, width, height), WHOLE_PICTURE, width, height)
    boxes = body_boxes(config, recording, camera, start, end)
    if boxes.empty:
        return Crop((0, 0, width, height), NO_POSE, width, height)
    return Crop(people_box(boxes, settings.margin, width, height), PEOPLE, width, height, sorted(boxes["subject_id"].unique()))


def body_boxes(config: MacawConfig, recording: Recording, camera: MediaFile, start: float, end: float) -> pd.DataFrame:
    """A row per named person and frame of camera in start to end: x0, y0, x1, y1 round their confident joints."""
    from macaw.extractors.video.pose.keypoints import PoseExtractor
    from macaw.store import keypoints as keypoints_store

    path = layout_for(config, create=False).keypoints_path(recording.recording_id) / f"{PoseExtractor.name}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["subject_id", "frame", "x0", "y0", "x1", "y1"])
    points = keypoints_store.read_keypoints(path, recording_id=recording.recording_id, t_range=(start, end))
    points = points[
        (points["video"] == camera.name) & points["subject_id"].notna() & (points["confidence"] >= config.pose.min_confidence)
        & points["x"].notna() & points["y"].notna()
    ]
    grouped = points.groupby(["subject_id", "frame"], sort=True)
    return pd.DataFrame({
        "x0": grouped["x"].min(), "y0": grouped["y"].min(), "x1": grouped["x"].max(), "y1": grouped["y"].max(),
    }).reset_index()


def people_box(boxes: pd.DataFrame, margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """The union of the bodies' boxes with the outer 2% of each edge left out, widened by margin on each side."""
    x0, y0 = np.percentile(boxes["x0"], _OUTLIER), np.percentile(boxes["y0"], _OUTLIER)
    x1, y1 = np.percentile(boxes["x1"], 100 - _OUTLIER), np.percentile(boxes["y1"], 100 - _OUTLIER)
    wide, high = (x1 - x0) * margin, (y1 - y0) * margin
    return _clipped(x0 - wide, y0 - high, x1 + wide, y1 + high, width, height)


def _clipped(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int]:
    # even sides keep the encoder's chroma planes whole
    left, top = int(np.clip(np.floor(x0), 0, width - 2)), int(np.clip(np.floor(y0), 0, height - 2))
    right, bottom = int(np.clip(np.ceil(x1), left + 2, width)), int(np.clip(np.ceil(y1), top + 2, height))
    right -= (right - left) % 2
    bottom -= (bottom - top) % 2
    return left, top, right, bottom


def picture_size(path: Path) -> tuple[int, int]:
    from macaw.media.probe import probe

    stream = probe(path).video_streams()[0]
    return int(stream.width), int(stream.height)


def take_frames(
    camera: MediaFile, times: list[float], crop: Crop, long_side: int, directory: Path, length_s: float,
) -> dict[float, Path | None]:
    """Each moment's frame, cropped and scaled, by moment; None where the video does not reach it."""
    taken: dict[float, Path | None] = {}
    for index, moment in enumerate(times, 1):
        at = camera.own_time(moment)
        if not 0 <= at < length_s:
            taken[moment] = None
            continue
        taken[moment] = extract_frame(camera.frames_path, at, directory / f"f{index:05d}.jpg", long_side=long_side, crop=crop.box)
    return taken


def write_preview(camera: MediaFile, moment: float, crop: Crop, dest: Path, length_s: float) -> Path | None:
    """The whole picture at moment with the crop drawn on it, for a person to check; None where the video does not reach."""
    from macaw.media.images import read_image, write_image

    at = min(max(camera.own_time(moment), 0.0), max(length_s, 0.0))
    picture = extract_frame(camera.frames_path, at, dest.with_name(f"whole_{dest.name}"), long_side=_PREVIEW_SIDE)
    image = read_image(picture)
    picture.unlink(missing_ok=True)
    if image is None:
        return None
    import cv2

    scale = image.shape[1] / crop.width
    x0, y0, x1, y1 = (int(round(value * scale)) for value in crop.box)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (40, 200, 255), max(2, image.shape[1] // 320))
    write_image(dest, image)
    return dest

