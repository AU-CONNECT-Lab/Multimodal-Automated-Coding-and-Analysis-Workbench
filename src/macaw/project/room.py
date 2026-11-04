"""Where people, devices and things are in the room, as the setup lays them out: x across and y along the room seen
from above, z up from the floor, in the project's unit. Facing is in degrees clockwise from +y, tilt in degrees below level."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from macaw.config.schema import PlacedCamera, RoomSection

CM_PER_UNIT = {"cm": 1.0, "m": 100.0, "in": 2.54}
UNIT_NAMES = {"cm": "centimetres", "m": "metres", "in": "inches"}


def to_unit(value_cm, units: str):
    return np.asarray(value_cm, dtype=float) / CM_PER_UNIT[units] if np.ndim(value_cm) else float(value_cm) / CM_PER_UNIT[units]


def from_unit(value, units: str):
    return np.asarray(value, dtype=float) * CM_PER_UNIT[units] if np.ndim(value) else float(value) * CM_PER_UNIT[units]


def heading(degrees: float) -> np.ndarray:
    """The level direction a facing of this many degrees points in."""
    angle = np.radians(degrees)
    return np.array([np.sin(angle), np.cos(angle), 0.0])


@dataclass
class View:
    """A camera's place and its axes in the room: right, ahead along its line of sight, and up in its picture."""

    origin: np.ndarray
    right: np.ndarray
    ahead: np.ndarray
    up: np.ndarray
    lens: float | None

    def local(self, points) -> np.ndarray:
        """Points in the room as (right, ahead, up) from the camera."""
        gap = np.atleast_2d(np.asarray(points, dtype=float)) - self.origin
        return np.stack([gap @ self.right, gap @ self.ahead, gap @ self.up], axis=-1)

    def pixels(self, points, size: tuple[int, int]) -> np.ndarray:
        """Where points land in a picture of this size; NaN behind the camera. Needs the lens angle."""
        width, height = size
        focal = (width / 2) / np.tan(np.radians(self.lens) / 2)
        right, ahead, up = self.local(points).T
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(ahead > 0, width / 2 + focal * right / ahead, np.nan)
            y = np.where(ahead > 0, height / 2 - focal * up / ahead, np.nan)
        return np.stack([x, y], axis=-1)


def view_of(camera: PlacedCamera) -> View | None:
    """None until the camera's place, height and facing are given."""
    if camera.at is None or camera.height is None or camera.facing is None:
        return None
    level = heading(camera.facing)
    right = np.array([level[1], -level[0], 0.0])
    tilt = np.radians(camera.tilt)
    ahead = np.cos(tilt) * level - np.sin(tilt) * np.array([0.0, 0.0, 1.0])
    up = np.cross(right, ahead)
    return View(np.array([*camera.at, camera.height], dtype=float), right, ahead, up, camera.lens)


def points(room: RoomSection) -> dict[str, np.ndarray]:
    """Each person's eyes and each thing's top that the room places."""
    found = {name: np.array([*place.at, place.eyes]) for name, place in room.people.items() if place.at is not None and place.eyes is not None}
    found.update({name: np.array([*place.at, place.height]) for name, place in room.things.items() if place.at is not None and place.height is not None})
    return found


def seats(room: RoomSection, camera: str) -> dict[str, tuple[float, float, float]]:
    """Each person and thing placed, and the camera, as [right, ahead, up] from that camera; empty until it is placed."""
    view = view_of(room.cameras[camera]) if camera in room.cameras else None
    if view is None:
        return {}
    placed = points(room)
    found = {name: tuple(float(value) for value in view.local(point)[0]) for name, point in placed.items()}
    return {"camera": (0.0, 0.0, 0.0), **found} if found else {}


def angle_to(camera: PlacedCamera, person) -> float | None:
    """Degrees between where a person faces and the direction to the camera: 0 when they face it, 180 when it is behind them."""
    if camera.at is None or person.at is None or person.facing is None:
        return None
    toward = np.subtract(camera.at, person.at)
    if not np.hypot(*toward):
        return None
    facing = heading(person.facing)[:2]
    cos = float(np.dot(facing, toward) / np.hypot(*toward))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def distance(camera: PlacedCamera, person) -> float | None:
    """From the camera to the person's eyes, in the room's unit."""
    if camera.at is None or camera.height is None or person.at is None or person.eyes is None:
        return None
    return float(np.linalg.norm(np.subtract([*person.at, person.eyes], [*camera.at, camera.height])))


def in_picture(camera: PlacedCamera, point) -> bool | None:
    """Whether a point is within the camera's angle of view across; None until the camera is placed with its lens."""
    view = view_of(camera)
    if view is None or camera.lens is None:
        return None
    right, ahead, _ = view.local(point)[0]
    return bool(ahead > 0 and abs(np.degrees(np.arctan2(right, ahead))) <= camera.lens / 2)
