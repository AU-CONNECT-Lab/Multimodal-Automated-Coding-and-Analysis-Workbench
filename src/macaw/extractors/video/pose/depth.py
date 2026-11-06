"""A depth camera records how far each of its pixels is. Read beside the camera's own video, the depth places each body
point the video finds at the nearest surface there, in metres in the colour camera's frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from macaw.exceptions import ConfigError, InputNotFoundError, MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.extractors.prep import VideoPrepExtractor, load_video_prep
from macaw.extractors.video.pose.cameras import Camera
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.store import keypoints as keypoints_store
from macaw.store.schema import KEYPOINT_SCHEMA, LONG_TABLE_SCHEMA, Scale, coerce_frame, empty_frame

DEPTH_FOLDER = "depth"          # in a depth camera's device folder, beside its video
LENS_FILE = "depth.yaml"
INFO_NAME = "depth.json"
QUALITY_NAME = "depth.csv"
_CHUNK = 300                    # depth frames per stored file
_UNITS_METRE = "metre"
# around a body point, cells this far behind the nearest one are the background, not the body
_SURFACE_M = 0.3
# a depth image is taken for a frame of the video when it lies within this share of a frame's time
_HALF_FRAME = 0.5
_BONE_SPREAD = 0.15
_FEW_PLACED = 0.5


@dataclass
class DepthLens:
    depth: Camera       # its rotation and translation take points of the colour camera's frame into its own
    colour: Camera      # the colour camera, at the origin of its frame
    unit_m: float       # one step of a stored depth value, in metres


def depth_cameras(config) -> list[str]:
    return sorted(name for name, camera in config.setup.cameras.items() if camera.depth)


def read_lens(path: Path) -> DepthLens:
    """depth.yaml: both cameras' lenses, where the depth camera sits in the colour camera's frame, and the depth unit."""
    if not path.is_file():
        raise InputNotFoundError(f"{path} is missing: it gives the depth camera's lens and where it sits beside the colour camera")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        rotation = np.asarray(data.get("rotation", np.eye(3)), float).reshape(3, 3)
        translation = np.asarray(data.get("translation", [0.0, 0.0, 0.0]), float).reshape(3)
        # depth.yaml says where a depth-camera point lands in the colour frame; Camera wants the way back
        back = Rotation.from_matrix(rotation).inv()
        depth = Camera("depth", tuple(data["depth_size"]), np.asarray(data["depth_matrix"], float),
                       _five(data.get("depth_distortions")), back.as_rotvec(), -back.apply(translation))
        colour = Camera("colour", tuple(data["colour_size"]), np.asarray(data["colour_matrix"], float),
                        _five(data.get("colour_distortions")), np.zeros(3), np.zeros(3))
        return DepthLens(depth, colour, float(data.get("unit_mm", 1.0)) / 1000.0)
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"{path}: {error}; it needs depth_size, depth_matrix, colour_size and colour_matrix") from None


def write_lens(lens: DepthLens, path: Path) -> Path:
    to_colour = Rotation.from_rotvec(lens.depth.rotation).inv()
    data = {
        "depth_size": list(lens.depth.size), "depth_matrix": lens.depth.matrix.tolist(), "depth_distortions": lens.depth.distortions.tolist(),
        "rotation": to_colour.as_matrix().tolist(), "translation": (-to_colour.apply(lens.depth.translation)).tolist(),
        "colour_size": list(lens.colour.size), "colour_matrix": lens.colour.matrix.tolist(),
        "colour_distortions": lens.colour.distortions.tolist(), "unit_mm": lens.unit_m * 1000.0,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _five(values) -> np.ndarray:
    """k1 k2 p1 p2 k3, or eight with a rational lens's k4 k5 k6, as OpenCV orders them."""
    given = np.asarray(values if values is not None else [], float).ravel()[:8]
    out = np.zeros(8 if len(given) > 5 else 5)
    out[: len(given)] = given
    return out


def depth_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() == ".png") if folder.is_dir() else []


def read_png(path: Path) -> np.ndarray:
    from macaw.extractors.video.pose.board import _cv2

    image = _cv2().imread(str(path), -1)
    if image is None or image.dtype != np.uint16 or image.ndim != 2:
        raise MacawError(f"{path}: not a 16-bit single-channel PNG")
    return image


class Surfaces:
    """Where a depth image's pixels fall in the colour picture, nearest first, on a grid of cells."""

    def __init__(self, lens: DepthLens):
        self.lens = lens
        width, height = lens.depth.size
        grid = np.stack(np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float)), axis=-1).reshape(-1, 2)
        self.rays = lens.depth.undistort(grid)
        back = Rotation.from_rotvec(lens.depth.rotation).inv()
        self.turn, self.shift = back.as_matrix(), -back.apply(lens.depth.translation)
        # about one depth pixel lands in each cell of the colour picture
        self.cell = max(1, int(np.ceil(lens.colour.matrix[0, 0] / lens.depth.matrix[0, 0])))
        self.shape = (int(np.ceil(lens.colour.size[1] / self.cell)), int(np.ceil(lens.colour.size[0] / self.cell)))

    def nearest(self, image: np.ndarray) -> np.ndarray:
        """Per cell of the colour picture, how far the nearest surface is along the colour camera's axis; inf where none."""
        values = image.reshape(-1).astype(float)
        found = values > 0
        z = values[found] * self.lens.unit_m
        points = np.column_stack([self.rays[found] * z[:, None], z]) @ self.turn.T + self.shift
        ahead = points[:, 2] > 0
        points = points[ahead]
        pixels = self.lens.colour.distort(points[:, :2] / points[:, 2:3])
        rows, cols = np.floor(pixels[:, 1] / self.cell).astype(int), np.floor(pixels[:, 0] / self.cell).astype(int)
        inside = (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])
        out = np.full(self.shape, np.inf)
        np.minimum.at(out, (rows[inside], cols[inside]), points[inside, 2])
        return out

    def place(self, nearest: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Colour-picture points (n, 2) to 3D (n, 3) at the nearest surface around each; NaN where there is none."""
        out = np.full((len(pixels), 3), np.nan)
        rows, cols = np.floor(pixels[:, 1] / self.cell), np.floor(pixels[:, 0] / self.cell)
        for index, (row, col) in enumerate(zip(rows, cols)):
            if not (np.isfinite(row) and np.isfinite(col)):
                continue
            row, col = int(row), int(col)
            window = nearest[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2]
            window = window[np.isfinite(window)]
            if window.size == 0:
                continue
            z = float(np.median(window[window <= window.min() + _SURFACE_M]))
            normal = self.lens.colour.undistort(pixels[index])
            out[index] = [normal[0] * z, normal[1] * z, z]
        return out


def frame_pairs(video: dict, n_images: int) -> dict[int, int]:
    """Prepared video frame -> the depth image taken at it, depth image i being the source video's frame i."""
    pairs = {}
    source = float(video["source_fps"])
    for image in range(n_images):
        at = video["first_picture_s"] + image / source - video["frames_start_s"]
        frame = int(round(at * video["fps"]))
        if frame >= 0 and abs(at - frame / video["fps"]) <= _HALF_FRAME / source:
            pairs.setdefault(frame, image)
    return pairs


@register_extractor
class DepthPrepExtractor(Extractor):
    """Each depth camera's depth images, once, on its prepared video's frames."""

    name = "prep.depth"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.PREP
    config_sections = ("video", "setup.cameras")
    requires = (VideoPrepExtractor.name,)
    resource_profile = "light_short"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return False                    # pose.depth pulls it in

    def check_available(self) -> None:
        from macaw.extractors.video.pose.board import _cv2

        _cv2()

    def _sources(self, recording: Recording) -> dict[str, tuple[Path, Path]]:
        """camera -> (its folder of depth images, its depth.yaml)."""
        found = {}
        for item in recording.cameras():
            if item.device in depth_cameras(self.config):
                folder = item.path.parent / DEPTH_FOLDER
                found[item.device] = (folder, folder / LENS_FILE)
        return found

    def inputs_for(self, recording: Recording) -> list[Path]:
        paths = []
        for folder, lens in self._sources(recording).values():
            paths += [lens, *depth_images(folder)]
        return [path for path in paths if path.is_file()]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        videos = load_video_prep(self.config, recording.recording_id)
        info, warnings = {}, []
        for camera, (folder, lens_path) in self._sources(recording).items():
            images = depth_images(folder)
            if not images:
                raise InputNotFoundError(f"{recording.recording_id}: {camera} records depth, but {folder} holds no 16-bit PNG")
            lens = read_lens(lens_path)
            pairs = frame_pairs(videos[camera], len(images))
            out = workspace / f"{camera}.depth"
            out.mkdir(parents=True, exist_ok=True)
            frames = sorted(pairs)
            for start in range(0, len(frames), _CHUNK):
                chunk = frames[start : start + _CHUNK]
                stack = np.stack([read_png(images[pairs[frame]]) for frame in chunk])
                if stack.shape[1:] != (lens.depth.size[1], lens.depth.size[0]):
                    raise MacawError(f"{camera}: the depth images are {stack.shape[2]} x {stack.shape[1]}, but {LENS_FILE} says {lens.depth.size[0]} x {lens.depth.size[1]}")
                np.savez_compressed(out / f"{start // _CHUNK:05d}.npz", frames=np.asarray(chunk), depth=stack)
            write_lens(lens, out / LENS_FILE)
            info[camera] = {"folder": out.name, "images": len(images), "frames": len(frames)}
            if len(frames) < len(images):
                warnings.append(QCWarning(
                    check=self.name, recording_id=recording.recording_id, subject_id=None, severity="info",
                    message=f"{camera}: {len(images) - len(frames)} of {len(images)} depth images fall outside the prepared video",
                ))
        (workspace / INFO_NAME).write_text(json.dumps(info, indent=2), encoding="utf-8")
        artifacts = {INFO_NAME: workspace / INFO_NAME, **{entry["folder"]: workspace / entry["folder"] for entry in info.values()}}
        return ExtractorResult(rows=empty_frame(LONG_TABLE_SCHEMA), artifacts=artifacts, warnings=warnings)


def prepared_depth(config, recording_id: str) -> dict[str, Path]:
    """camera -> prep.depth's folder of its depth frames; empty before it has run."""
    root = layout_for(config, create=False).artifacts_dir(recording_id, DepthPrepExtractor.name)
    path = root / INFO_NAME
    if not path.is_file():
        return {}
    return {camera: root / entry["folder"] for camera, entry in json.loads(path.read_text(encoding="utf-8")).items()}


def depth_frames(folder: Path):
    """(frame, depth image) in frame order from a prepared folder."""
    for chunk in sorted(folder.glob("*.npz")):
        with np.load(chunk) as data:
            for frame, image in zip(data["frames"], data["depth"]):
                yield int(frame), image


@register_extractor
class DepthPoseExtractor(Extractor):
    """Each person's body points in 3D from the depth camera their body is measured on."""

    name = "pose.depth"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    phase = Phase.DERIVE
    config_sections = ("setup.cameras",)
    requires = ("pose.review", DepthPrepExtractor.name)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(depth_cameras(config))

    def check_available(self) -> None:
        from macaw.extractors.video.pose.triangulate import three_d_step

        if not depth_cameras(self.config):
            raise ConfigError("pose.depth needs a camera that records depth: set depth: true for it in the setup")
        three_d_step(self.config)

    def inputs_for(self, recording: Recording) -> list[Path]:
        from macaw.extractors.video.pose.kinematics import _keypoints_file

        return [_keypoints_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        from macaw.extractors.video.pose.kinematics import _keypoints_file
        from macaw.extractors.video.pose.triangulate import bone_spread

        rid = recording.recording_id
        keypoints = keypoints_store.read_keypoints(_keypoints_file(self.config, rid), recording_id=rid)
        keypoints = keypoints[keypoints["subject_id"].notna() & keypoints["joint"].isin(STANDARD_JOINTS) & keypoints["x"].notna()]
        prepared, videos = prepared_depth(self.config, rid), load_video_prep(self.config, rid)
        warnings: list[QCWarning] = []
        pieces, quality = [], []
        for person, item in sorted(recording.views("body").items()):
            camera = item.device
            if camera not in prepared:
                warnings.append(self._warning(rid, person, "warn", f"their body is measured on {camera}, which records no depth, so they are not placed in 3D"))
                continue
            own = keypoints[(keypoints["subject_id"] == person) & (keypoints["video"] == camera)]
            if own.empty:
                continue
            lens = read_lens(prepared[camera] / LENS_FILE)
            # points from one camera are in that camera's frame, which the video column names
            placed = self._place(own, Surfaces(lens), prepared[camera], videos.get(camera, {}), lens).assign(video=camera)
            pieces.append(placed)
            for joint, group in placed.groupby("joint"):
                quality.append({"subject_id": person, "camera": camera, "joint": joint, "seen": len(group), "placed": int(group["z"].notna().sum())})
        points = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["subject_id", "video", "frame", "t", "joint", "x", "y", "z", "confidence"])
        quality = pd.DataFrame(quality, columns=["subject_id", "camera", "joint", "seen", "placed"])
        for person, own in quality.groupby("subject_id"):
            share = own["placed"].sum() / max(own["seen"].sum(), 1)
            warnings.append(self._warning(
                rid, person, "warn" if share < _FEW_PLACED else "info",
                f"placed in 3D for {share:.0%} of the body points seen; the rest had no depth around them",
            ))
        for person, own in points.dropna(subset=["z"]).groupby("subject_id"):
            for (first, second), spread in bone_spread(own).items():
                if spread > _BONE_SPREAD:
                    bone = f"{first.replace('_', ' ')} to {second.replace('_', ' ')}"
                    warnings.append(self._warning(
                        rid, person, "warn",
                        f"the {bone} changes length by {spread:.0%}: in some frames an end lies on something in front of the body or behind it",
                    ))
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        quality.to_csv(directory / QUALITY_NAME, index=False)
        rows = coerce_frame(
            points.dropna(subset=["z"]).assign(
                recording_id=rid, source=self.name, track_id=pd.NA, units=_UNITS_METRE, metric_scale_valid=True,
            ),
            KEYPOINT_SCHEMA, "keypoints",
        )
        return ExtractorResult(rows=empty_frame(LONG_TABLE_SCHEMA), keypoints=rows, artifacts={QUALITY_NAME: directory / QUALITY_NAME}, warnings=warnings)

    def _place(self, own: pd.DataFrame, surfaces: Surfaces, folder: Path, video: dict, lens: DepthLens) -> pd.DataFrame:
        """One person's points on one camera, each at the depth there; z is NaN where there was none."""
        # the prepared picture may be scaled from the camera's own, which depth.yaml describes
        size = video.get("prepared_size") or video.get("size") or list(lens.colour.size)
        stretch = np.array([lens.colour.size[0] / size[0], lens.colour.size[1] / size[1]])
        by_frame = {frame: group for frame, group in own.groupby("frame")}
        out = []
        for frame, image in depth_frames(folder):
            group = by_frame.pop(frame, None)
            if group is None:
                continue
            xyz = surfaces.place(surfaces.nearest(image), group[["x", "y"]].to_numpy(float) * stretch)
            out.append(group[["subject_id", "frame", "t", "joint", "confidence"]].assign(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2]))
        # frames without a depth image keep their points unplaced
        for group in by_frame.values():
            out.append(group[["subject_id", "frame", "t", "joint", "confidence"]].assign(x=np.nan, y=np.nan, z=np.nan))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["subject_id", "frame", "t", "joint", "confidence", "x", "y", "z"])

    def _warning(self, rid: str, subject_id, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=rid, subject_id=subject_id, severity=severity, message=message)
