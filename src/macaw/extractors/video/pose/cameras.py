"""Cameras of a calibrated setup and the geometry between them: a point seen by two or more cameras
becomes one point in the room. The lens model is OpenCV's, and the file format Anipose's calibration.toml."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from macaw.exceptions import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

UNITS_TO_METRES = {"mm": 0.001, "cm": 0.01, "m": 1.0}
_UNDISTORT_ITERATIONS = 20


@dataclass
class Camera:
    name: str
    size: tuple[int, int]          # width, height in pixels
    matrix: np.ndarray             # 3 x 3: focal lengths and the picture's centre, in pixels
    distortions: np.ndarray        # k1, k2, p1, p2, k3, and for a rational lens k4, k5, k6, as OpenCV orders them
    rotation: np.ndarray           # axis-angle, room to camera
    translation: np.ndarray        # room to camera, in the setup's units

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return Rotation.from_rotvec(self.rotation).apply(points.reshape(-1, 3)).reshape(points.shape) + self.translation

    def project(self, points: np.ndarray) -> np.ndarray:
        """Room points (..., 3) to pixels (..., 2)."""
        inside = self.to_camera(points)
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.distort(inside[..., :2] / inside[..., 2:3])

    def distort(self, normal: np.ndarray) -> np.ndarray:
        return through_lens(normal, self.matrix, self.distortions)

    def undistort(self, pixels: np.ndarray) -> np.ndarray:
        """Pixels (..., 2) to normalised image coordinates, inverting the lens by fixed-point iteration."""
        p1, p2 = self.distortions[2], self.distortions[3]
        (fx, skew, cx), (_, fy, cy) = self.matrix[0], self.matrix[1]
        y0 = (pixels[..., 1] - cy) / fy
        x0 = (pixels[..., 0] - cx - skew * y0) / fx
        x, y = x0.copy(), y0.copy()
        for _ in range(_UNDISTORT_ITERATIONS):
            r2 = x * x + y * y
            radial = _radial(self.distortions, r2)
            x = (x0 - 2.0 * p1 * x * y - p2 * (r2 + 2.0 * x * x)) / radial
            y = (y0 - p1 * (r2 + 2.0 * y * y) - 2.0 * p2 * x * y) / radial
        return np.stack([x, y], axis=-1)

    def position(self) -> np.ndarray:
        """Where the camera stands in the room."""
        return -Rotation.from_rotvec(self.rotation).inv().apply(self.translation)


def through_lens(normal: np.ndarray, matrix: np.ndarray, distortions: np.ndarray) -> np.ndarray:
    """Normalised image coordinates (..., 2) to pixels; matrix (..., 3, 3) and distortions (..., 5) broadcast
    against the points, so each point may have its own camera."""
    p1, p2 = distortions[..., 2], distortions[..., 3]
    x, y = normal[..., 0], normal[..., 1]
    r2 = x * x + y * y
    radial = _radial(distortions, r2)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack(
        [matrix[..., 0, 0] * xd + matrix[..., 0, 1] * yd + matrix[..., 0, 2], matrix[..., 1, 1] * yd + matrix[..., 1, 2]], axis=-1
    )


def _radial(distortions: np.ndarray, r2):
    """The radial factor of a five-coefficient lens, or of a rational one with k4 to k6 dividing."""
    k1, k2, k3 = distortions[..., 0], distortions[..., 1], distortions[..., 4]
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    if distortions.shape[-1] >= 8:
        radial = radial / (1.0 + distortions[..., 5] * r2 + distortions[..., 6] * r2**2 + distortions[..., 7] * r2**3)
    return radial


@dataclass
class CameraSetup:
    cameras: list[Camera]
    units: str                     # of the translations: mm, cm or m
    metadata: dict = field(default_factory=dict)

    def named(self, name: str) -> Camera | None:
        return next((camera for camera in self.cameras if camera.name == name), None)


def read_setup(path: Path, default_units: str = "mm") -> CameraSetup:
    """Every table with a camera matrix is a camera; [metadata] units, when there, says what the translations are in."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: cannot read the calibration: {error}") from None
    metadata = data.get("metadata", {})
    cameras = []
    for key, table in data.items():
        if not isinstance(table, dict) or "matrix" not in table:
            continue
        try:
            cameras.append(Camera(
                name=str(table.get("name", key)),
                size=(int(table["size"][0]), int(table["size"][1])),
                matrix=np.asarray(table["matrix"], dtype=float).reshape(3, 3),
                distortions=np.resize(np.asarray(table.get("distortions", [0.0] * 5), dtype=float), 5),
                rotation=np.asarray(table["rotation"], dtype=float).reshape(3),
                translation=np.asarray(table["translation"], dtype=float).reshape(3),
            ))
        except (KeyError, ValueError, TypeError) as error:
            raise ConfigError(f"{path}: camera {key} is incomplete or malformed: {error}") from None
    units = str(metadata.get("units", default_units))
    if units not in UNITS_TO_METRES:
        raise ConfigError(f"{path}: units {units} is not one of {', '.join(UNITS_TO_METRES)}")
    if len(cameras) < 2:
        raise ConfigError(f"{path}: a calibration needs two cameras or more, found {len(cameras)}")
    return CameraSetup(cameras, units, metadata)


def write_setup(setup: CameraSetup, path: Path) -> Path:
    lines = ["[metadata]", f"units = {json.dumps(setup.units)}"]
    lines += [f"{key} = {_toml_value(value)}" for key, value in setup.metadata.items() if key != "units"]
    for index, camera in enumerate(setup.cameras):
        lines += [
            "", f"[cam_{index}]", f"name = {json.dumps(camera.name)}", f"size = {_toml_value(list(camera.size))}",
            f"matrix = {_toml_value(camera.matrix.tolist())}", f"distortions = {_toml_value(camera.distortions.tolist())}",
            f"rotation = {_toml_value(camera.rotation.tolist())}", f"translation = {_toml_value(camera.translation.tolist())}",
        ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toml_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(key))} = {_toml_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return repr(float(value)) if isinstance(value, (float, np.floating)) else str(int(value))
    return json.dumps(str(value))


def triangulate(
    cameras: list[Camera], pixels: np.ndarray, weights: np.ndarray, *, max_error_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pixels (N, V, 2) as seen by each camera, weights (N, V) with 0 where a camera has no point.
    Returns the room points (N, 3), the views kept (N, V) and the mean reprojection error in pixels (N,).
    With three or more views, the view furthest off is dropped while it misses by more than max_error_px;
    a point is placed only when two or more views remain within it and it lies in front of each."""
    normal = np.stack([camera.undistort(pixels[:, index]) for index, camera in enumerate(cameras)], axis=1)
    used = (weights > 0) & np.isfinite(normal).all(axis=-1)
    for _ in range(len(cameras) - 2):
        points = _dlt(cameras, normal, np.where(used, weights, 0.0))
        errors = np.nan_to_num(_reprojection_errors(cameras, points, pixels, used), nan=-1.0)
        far = (used.sum(axis=1) > 2) & (errors.max(axis=1) > max_error_px)
        if not far.any():
            break
        # one bad view spreads its miss over all of them, so drop the view without which the rest agree best
        best, drop = np.full(len(points), np.inf), np.full(len(points), -1)
        for left_out in range(len(cameras)):
            trial = used.copy()
            trial[:, left_out] = False
            rows = np.flatnonzero(far & used[:, left_out])
            if len(rows) == 0:
                continue
            kept = trial[rows]
            placed = _dlt(cameras, normal[rows], np.where(kept, weights[rows], 0.0))
            spread = np.nan_to_num(_reprojection_errors(cameras, placed, pixels[rows], kept), nan=-1.0).max(axis=1)
            better = spread < best[rows]
            best[rows[better]], drop[rows[better]] = spread[better], left_out
        chosen = np.flatnonzero(drop >= 0)
        used[chosen, drop[chosen]] = False
    points = _dlt(cameras, normal, np.where(used, weights, 0.0))
    errors = _reprojection_errors(cameras, points, pixels, used)
    depth = np.stack([camera.to_camera(points)[:, 2] for camera in cameras], axis=1)
    # two views that still miss each other cannot say which is wrong, and a point behind a camera that saw it is a mismatch
    with np.errstate(invalid="ignore"):
        agree = ~(np.where(used, errors, 0.0).max(axis=1) > max_error_px) & ~(used & ~(depth > 0)).any(axis=1)
    enough = (used.sum(axis=1) >= 2) & agree
    points[~enough] = np.nan
    used[~enough] = False
    error = np.where(enough, np.where(used, errors, 0.0).sum(axis=1) / np.maximum(used.sum(axis=1), 1), np.nan)
    return points, used, error


def _dlt(cameras: list[Camera], normal: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted linear triangulation in normalised coordinates, every point at once."""
    projections = np.stack(
        [np.hstack([Rotation.from_rotvec(camera.rotation).as_matrix(), camera.translation[:, None]]) for camera in cameras]
    )
    x = np.nan_to_num(normal[..., 0])[..., None]
    y = np.nan_to_num(normal[..., 1])[..., None]
    w = weights[..., None]
    rows = np.concatenate(
        [w * (x * projections[None, :, 2] - projections[None, :, 0]), w * (y * projections[None, :, 2] - projections[None, :, 1])],
        axis=1,
    )
    if len(rows) == 0:
        return np.empty((0, 3))
    homogeneous = np.linalg.svd(rows)[2][:, -1, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return homogeneous[:, :3] / homogeneous[:, 3:4]


def _reprojection_errors(cameras: list[Camera], points: np.ndarray, pixels: np.ndarray, used: np.ndarray) -> np.ndarray:
    errors = np.stack([np.linalg.norm(camera.project(points) - pixels[:, index], axis=-1) for index, camera in enumerate(cameras)], axis=1)
    return np.where(used, errors, np.nan)
