"""Each calibrated camera's place in the room: the calibration says where the cameras stand from one another, and
everything measured in the room ties them to it, with what is missing named and what does not agree said."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from macaw.config.schema import RoomSection
from macaw.exceptions import MacawError
from macaw.extractors.video.pose.cameras import UNITS_TO_METRES, Camera, CameraSetup
from macaw.project.room import CM_PER_UNIT, DECIMALS

FLOOR_FILE = "floor.yaml"
# how closely a person measures, in centimetres and degrees; the board lying flat is far closer
_LENGTH_CM, _ANGLE_DEG = 2.0, 3.0
_FLOOR_CM, _FLOOR_DEG = 0.5, 0.5
# laying the board's marked corner on a point and its edge along a wall
_BOARD_CM, _BOARD_DEG = 1.0, 1.0
# beside the cameras in the fit's state; a device name cannot hold a colon
_BOARD, _CORNER = ":board", ":corner"
# a given number further than this from where everything else puts it does not agree
_OFF_CM, _OFF_DEG = 10.0, 5.0
# a camera turned about its line of sight by more than this is still drawn level in the room
_ROLL_DEG = 5.0
# misses beyond this many measuring errors count for less in the final fit
_ROBUST = 2.0
# singular values this much smaller than the largest leave a way to move the cameras that nothing measured stops
_LOOSE = 1e-6


@dataclass
class Floor:
    up: np.ndarray            # unit, in the calibration's frame, toward the cameras
    middle: np.ndarray        # the board's middle, in the calibration's frame and units
    on: str                   # "floor", or a thing of the room the board lay on
    seen_by: dict[str, int]   # camera -> corners found
    corner: np.ndarray | None = None     # the board's marked corner, in the calibration's frame and units
    arrow: np.ndarray | None = None      # unit, along the board's edge from the marked corner
    place: tuple[float, float] | None = None   # where a person said the marked corner lay, in the room
    toward: float | None = None                # and where the arrow pointed, as a camera's facing


@dataclass
class Placing:
    values: dict[str, dict[str, float]]   # camera -> x, y, height, facing, tilt, lens; only lens until the room is tied
    tied: bool
    missing: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)    # each number given, against where everything puts it
    rolled: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class _Term:
    who: str
    what: str
    given: float
    sigma: float
    angle: bool
    value: Callable[[dict], float]


def lens_angle(camera: Camera) -> float:
    """The picture's width as an angle, in degrees, from the calibrated focal length."""
    return float(np.degrees(2.0 * np.arctan(camera.size[0] / (2.0 * camera.matrix[0, 0]))))


def _axes(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where the camera stands, and its picture's right, its line of sight and its picture's up, in the calibration's frame."""
    back = Rotation.from_rotvec(camera.rotation).inv()
    return camera.position(), back.apply([1.0, 0.0, 0.0]), back.apply([0.0, 0.0, 1.0]), back.apply([0.0, -1.0, 0.0])


def _level_axes(facing: float, tilt: float) -> np.ndarray:
    """A level camera's right, line of sight and up in the room, as columns, as project.room.view_of draws it."""
    angle, down = np.radians(facing), np.radians(tilt)
    level = np.array([np.sin(angle), np.cos(angle), 0.0])
    right = np.array([np.cos(angle), -np.sin(angle), 0.0])
    ahead = np.cos(down) * level - np.sin(down) * np.array([0.0, 0.0, 1.0])
    return np.column_stack([right, ahead, np.cross(right, ahead)])


def angles_of(right: np.ndarray, ahead: np.ndarray) -> tuple[float, float, float]:
    """Facing (clockwise from +y), tilt below level and the turn about the line of sight, in degrees, of room axes."""
    facing = float(np.degrees(np.arctan2(ahead[0], ahead[1])) % 360.0)
    tilt = float(np.degrees(np.arcsin(np.clip(-ahead[2], -1.0, 1.0))))
    level = _level_axes(facing, 0.0)[:, 0]
    up = np.cross(level, ahead)
    roll = float(np.degrees(np.arctan2(np.dot(right, up), np.dot(right, level))))
    return facing, tilt, roll


def _wrapped(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def _scale(setup: CameraSetup, units: str) -> float:
    """The calibration's unit in the project's."""
    return UNITS_TO_METRES[setup.units] * 100.0 / CM_PER_UNIT[units]


def _minimal_turn(start: np.ndarray, end: np.ndarray) -> Rotation:
    axis = np.cross(start, end)
    sine, cosine = np.linalg.norm(axis), float(np.dot(start, end))
    if sine < 1e-12:
        return Rotation.identity() if cosine > 0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine))


def place_cameras(setup: CameraSetup, room: RoomSection, units: str, floor: Floor | None = None) -> Placing:
    """Each camera of the calibration that the room lists, placed from everything given; only its lens while that leaves them free."""
    per_cm = 1.0 / CM_PER_UNIT[units]
    scale = _scale(setup, units)
    cameras = [camera for camera in setup.cameras if camera.name in room.cameras]
    notes = []
    absent = [camera.name for camera in setup.cameras if camera.name not in room.cameras]
    if absent:
        notes.append(f"{', '.join(absent)} of the calibration {'is' if len(absent) == 1 else 'are'} not in this setup")
    uncalibrated = [name for name in room.cameras if setup.named(name) is None]
    if uncalibrated:
        notes.append(f"{', '.join(uncalibrated)} {'is' if len(uncalibrated) == 1 else 'are'} not in the calibration and stay as they are")
    lens = {camera.name: round(lens_angle(camera), 1) for camera in cameras}
    if not cameras:
        return Placing({}, False, ["none of the calibration's cameras is in this setup"], notes=notes)

    axes = {camera.name: _axes(camera) for camera in cameras}
    terms = _terms(room, axes, floor, per_cm)
    surface = _surface(room, floor)
    span = max(100.0 * per_cm, *(np.linalg.norm(scale * (axes[a][0] - axes[b][0])) for a in axes for b in axes))

    def state(rotation: Rotation, shift: np.ndarray) -> dict:
        placed = {name: (scale * rotation.apply(where) + shift, rotation.apply(right), rotation.apply(ahead)) for name, (where, right, ahead, _) in axes.items()}
        if floor is not None:
            placed[_BOARD] = (scale * rotation.apply(floor.middle) + shift, rotation.apply(floor.up))
            if _laid(floor):
                placed[_CORNER] = (scale * rotation.apply(floor.corner) + shift, rotation.apply(floor.arrow))
        return placed

    def residuals(rotation: Rotation, shift: np.ndarray) -> np.ndarray:
        now = state(rotation, shift)
        out = []
        for term in terms:
            gap = term.value(now) - term.given
            out.append((_wrapped(gap) if term.angle else gap) / term.sigma)
        return np.asarray(out, dtype=float)

    best = None
    for start in _starts(room, axes, floor, scale, surface):
        shift0 = _start_shift(start, room, axes, floor, scale, surface)
        result = least_squares(
            lambda p: residuals(Rotation.from_rotvec(p[:3]), p[3:]), np.concatenate([start.as_rotvec(), shift0]),
            method="trf", x_scale=np.array([1.0, 1.0, 1.0, span, span, span]),
        ) if terms else None
        cost = result.cost if result is not None else 0.0
        if best is None or cost < best[0]:
            best = (cost, result.x if result is not None else np.concatenate([start.as_rotvec(), shift0]))
    if terms:
        # one number measured wrong would tip every camera a little; this lets the rest outvote it
        best = (0.0, least_squares(
            lambda p: residuals(Rotation.from_rotvec(p[:3]), p[3:]), best[1], method="trf",
            x_scale=np.array([1.0, 1.0, 1.0, span, span, span]), loss="soft_l1", f_scale=_ROBUST,
        ).x)
    rotation, shift = Rotation.from_rotvec(best[1][:3]), best[1][3:]

    missing = _missing(lambda q: residuals(Rotation.from_rotvec(q[:3]) * rotation, shift + q[3:] * span), len(terms))
    if missing:
        return Placing({name: {"lens": value} for name, value in lens.items()}, False, missing, notes=notes)

    now = state(rotation, shift)
    values, rolled = {}, {}
    for name, (where, right, ahead) in ((name, now[name]) for name in axes):
        facing, tilt, roll = angles_of(right, ahead)
        places = DECIMALS[units]
        values[name] = {
            "x": round(float(where[0]), places), "y": round(float(where[1]), places), "height": round(float(where[2]), places),
            "facing": round(facing, 1) % 360.0, "tilt": round(tilt, 1), "lens": lens[name],
        }
        if abs(roll) > _ROLL_DEG:
            rolled[name] = round(roll, 1)
    checks = []
    for term in terms:
        fitted = term.value(now)
        off = _wrapped(fitted - term.given) if term.angle else fitted - term.given
        limit = _OFF_DEG if term.angle else _OFF_CM * per_cm
        checks.append({"who": term.who, "what": term.what, "given": term.given, "fitted": fitted, "off": off, "angle": term.angle, "agrees": abs(off) <= limit})
    off = [check for check in checks if not check["agrees"]]
    if len(off) > 1:
        notes.append("a number measured wrong tips the cameras and moves the others too, so the wrong one may be any of those that do not agree; "
                     "the board lying flat, or more numbers given, tell them apart")
    if floor is not None and surface is None:
        notes.append(f"the board lay on {floor.on}, which has no height in the room, so the floor's level is left to the rest")
    return Placing(values, True, [], checks, rolled, notes)


def _terms(room: RoomSection, axes: dict, floor: Floor | None, per_cm: float) -> list[_Term]:
    length, angle = _LENGTH_CM * per_cm, _ANGLE_DEG
    terms: list[_Term] = []
    for name in axes:
        placed = room.cameras[name]
        if placed.at is not None:
            terms.append(_Term(name, "across", float(placed.at[0]), length, False, lambda s, n=name: float(s[n][0][0])))
            terms.append(_Term(name, "along", float(placed.at[1]), length, False, lambda s, n=name: float(s[n][0][1])))
        if placed.height is not None:
            terms.append(_Term(name, "height", float(placed.height), length, False, lambda s, n=name: float(s[n][0][2])))
        if placed.facing is not None:
            terms.append(_Term(name, "facing", float(placed.facing), angle, True, lambda s, n=name: angles_of(s[n][1], s[n][2])[0]))
        if placed.tilt is not None:
            terms.append(_Term(name, "tilt", float(placed.tilt), angle, True, lambda s, n=name: angles_of(s[n][1], s[n][2])[1]))
        # a camera whose facing and tilt are given is laid out level, as the room draws every camera
        if placed.facing is not None and placed.tilt is not None:
            terms.append(_Term(name, "level", 0.0, angle, True, lambda s, n=name: angles_of(s[n][1], s[n][2])[2]))
    if floor is not None:
        # against up rather than level, so a room upside down is as far off as can be
        terms.append(_Term("board", "flat across", 0.0, _FLOOR_DEG, True, lambda s: float(np.degrees(np.arctan2(s[_BOARD][1][0], s[_BOARD][1][2])))))
        terms.append(_Term("board", "flat along", 0.0, _FLOOR_DEG, True, lambda s: float(np.degrees(np.arctan2(s[_BOARD][1][1], s[_BOARD][1][2])))))
        height = _surface(room, floor)
        if height is not None:
            terms.append(_Term("board", f"on {floor.on}", height, _FLOOR_CM * per_cm, False, lambda s: float(s[_BOARD][0][2])))
        if _laid(floor):
            corner = _BOARD_CM * per_cm
            terms.append(_Term("board", "corner across", floor.place[0], corner, False, lambda s: float(s[_CORNER][0][0])))
            terms.append(_Term("board", "corner along", floor.place[1], corner, False, lambda s: float(s[_CORNER][0][1])))
            terms.append(_Term("board", "arrow toward", floor.toward, _BOARD_DEG, True, lambda s: _heading(s[_CORNER][1])))
    return terms


def _laid(floor: Floor | None) -> bool:
    return floor is not None and floor.place is not None and floor.toward is not None and floor.corner is not None


def _heading(direction: np.ndarray) -> float:
    """A direction's facing on the floor, clockwise from +y as a camera's."""
    return float(np.degrees(np.arctan2(direction[0], direction[1])) % 360.0)


def _surface(room: RoomSection, floor: Floor | None) -> float | None:
    """How high the board lay above the floor, in project units; None for a thing without a height."""
    if floor is None:
        return None
    if floor.on == "floor":
        return 0.0
    thing = room.things.get(floor.on)
    return float(thing.height) if thing is not None and thing.height is not None else None


def _starts(room: RoomSection, axes: dict, floor: Floor | None, scale: float, surface: float | None) -> list[Rotation]:
    """Turns from the calibration's frame to the room to begin from; the best fit of them all is kept."""
    starts = []
    for name, (_, right, ahead, up) in axes.items():
        placed = room.cameras[name]
        if placed.facing is not None:
            own = np.column_stack([right, ahead, up])
            starts.append(Rotation.from_matrix(_level_axes(placed.facing, placed.tilt or 0.0) @ own.T))
    headings = [Rotation.from_euler("z", degrees, degrees=True) for degrees in range(0, 360, 45)]
    if floor is not None:
        level = _minimal_turn(floor.up, np.array([0.0, 0.0, 1.0]))
        starts += [heading * level for heading in headings]
        if _laid(floor):
            starts.append(Rotation.from_euler("z", _heading(level.apply(floor.arrow)) - floor.toward, degrees=True) * level)
    measured = [name for name in axes if room.cameras[name].at is not None and room.cameras[name].height is not None]
    if len(measured) >= 3:
        room_points = np.array([[*room.cameras[name].at, room.cameras[name].height] for name in measured], dtype=float)
        own_points = np.array([scale * axes[name][0] for name in measured])
        turn, _ = Rotation.align_vectors(room_points - room_points.mean(axis=0), own_points - own_points.mean(axis=0))
        starts.append(turn)
    if not starts:
        name = next(iter(axes))
        _, right, ahead, up = axes[name]
        starts = [heading * Rotation.from_matrix(_level_axes(0.0, 0.0) @ np.column_stack([right, ahead, up]).T) for heading in headings]
    return starts


def _start_shift(turn: Rotation, room: RoomSection, axes: dict, floor: Floor | None, scale: float, surface: float | None) -> np.ndarray:
    shift = np.zeros(3)
    across = [np.asarray(room.cameras[name].at, float) - (scale * turn.apply(axes[name][0]))[:2] for name in axes if room.cameras[name].at is not None]
    if _laid(floor):
        across.append(np.asarray(floor.place, float) - (scale * turn.apply(floor.corner))[:2])
    if across:
        shift[:2] = np.mean(across, axis=0)
    heights = [room.cameras[name].height - (scale * turn.apply(axes[name][0]))[2] for name in axes if room.cameras[name].height is not None]
    if heights:
        shift[2] = float(np.mean(heights))
    elif floor is not None and surface is not None:
        shift[2] = surface - (scale * turn.apply(floor.middle))[2]
    return shift


def _missing(residuals: Callable[[np.ndarray], np.ndarray], count: int) -> list[str]:
    """What still lets the cameras move without any measured number changing."""
    if count == 0:
        return [_UP, _ON_FLOOR]
    step = 1e-5
    jacobian = np.column_stack([(residuals(np.eye(6)[index] * step) - residuals(-np.eye(6)[index] * step)) / (2 * step) for index in range(6)])
    _, values, rows = np.linalg.svd(jacobian, full_matrices=True)
    kept = int(np.sum(values > _LOOSE * values.max())) if values.size and values.max() > 0 else 0
    said = []
    for free in rows[kept:]:
        upward = np.linalg.norm(free[:2]) + abs(free[5])
        across = abs(free[2]) + np.linalg.norm(free[3:5])
        if upward > 0.3 * (upward + across) and _UP not in said:
            said.append(_UP)
        if across > 0.3 * (upward + across) and _ON_FLOOR not in said:
            said.append(_ON_FLOOR)
    return said


_UP = "which way is up and how high the cameras are: film the board lying flat, or give one camera's height and tilt, or three cameras' places and heights"
_ON_FLOOR = (
    "where the cameras stand on the floor and which way they face: say where the flat board's marked corner lay and where its arrow "
    "pointed, or give two cameras' places, or one camera's place and facing"
)


def floor_said(folder: Path) -> dict | None:
    """The moment of a calibration's board clips at which the board lay flat, on what, and where if known, as a person said it."""
    path = Path(folder) / FLOOR_FILE
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if data.get("at_s") is None:
        return None
    corner = data.get("corner")
    return {
        "at_s": float(data["at_s"]), "on": str(data.get("on") or "floor"),
        "corner": (float(corner[0]), float(corner[1])) if corner else None,
        "toward": float(data["toward"]) if corner and data.get("toward") is not None else None,
    }


def write_floor(folder: Path, at_s: float, on: str, corner: tuple[float, float] | None = None, toward: float | None = None) -> Path:
    path = Path(folder) / FLOOR_FILE
    said = {"at_s": round(float(at_s), 3), "on": on}
    if corner is not None:
        said.update(corner=[float(corner[0]), float(corner[1])], toward=float(toward if toward is not None else 90.0) % 360.0)
    path.write_text(yaml.safe_dump(said, sort_keys=False), encoding="utf-8")
    return path


def find_floor(setup: CameraSetup, clips: dict[str, Path], board, at_s: float, on: str, *, normalise: bool) -> Floor | None:
    """The board lying flat at at_s seconds of the board clips' common clock, from every camera that finds it; None when none does."""
    from macaw.extractors.video.pose.board import board_in_frame, corners_on, frame_at, width_factor

    offsets = setup.metadata.get("offsets_s", {}) or {}
    seen, places = {}, []
    for camera in setup.cameras:
        clip = clips.get(camera.name)
        if clip is None:
            continue
        frame = frame_at(clip, at_s - float(offsets.get(camera.name, 0.0)), width_factor(clip, normalise))
        sighting = corners_on(frame, board) if frame is not None else None
        seen[camera.name] = 0 if sighting is None else len(sighting.ids)
        placed = board_in_frame(camera, sighting, board, setup.units) if sighting is not None else None
        if placed is not None:
            places.append((placed, camera))
    if not places:
        return None
    # the side the cameras see is up
    up = np.mean([place.up if np.dot(place.up, camera.position() - place.middle) > 0 else -place.up for place, camera in places], axis=0)
    arrow = np.mean([place.arrow for place, _ in places], axis=0)
    return Floor(
        up / np.linalg.norm(up), np.mean([place.middle for place, _ in places], axis=0), on, seen,
        np.mean([place.corner for place, _ in places], axis=0), arrow / np.linalg.norm(arrow),
    )


def floor_of(config, calibration: str, setup: CameraSetup) -> tuple[Floor | None, str | None]:
    """The floor a calibration's board clips show where a person said when the board lay flat; and what kept it out."""
    from macaw.extractors.video.pose.board import make_board
    from macaw.project.recording import camera_clips

    folder = Path(config.project.calibration_dir) / calibration
    said = floor_said(folder) if folder.is_dir() else None
    if said is None:
        return None, None
    board = make_board(config.pose3d.board.in_mm(config.project.units))
    try:
        floor = find_floor(setup, camera_clips(folder), board, said["at_s"], said["on"], normalise=config.video.normalise)
    except MacawError as error:
        return None, str(error)
    if floor is None:
        return None, f"no camera finds the board at {said['at_s']:g} s of the board clips, where it was said to lie flat"
    floor.place, floor.toward = said["corner"], said["toward"]
    return floor, None


def summary(placing: Placing, calibration: str) -> str:
    if placing.tied:
        return f"Placed {', '.join(placing.values)} from the calibration {calibration}."
    if not placing.values:
        return f"Nothing came from the calibration {calibration}: {'; '.join(placing.missing)}."
    return f"Only the lens angles came from the calibration {calibration}. Still missing: {'; '.join(placing.missing)}."


def check_rows(placing: Placing, units: str) -> list[list[str]]:
    """Each number given, where everything together puts it, and by how much it is off."""
    places = DECIMALS[units]
    rows = []
    for check in placing.checks:
        form = (lambda value: f"{value:.1f}°") if check["angle"] else (lambda value: f"{value:.{places}f} {units}")
        rows.append([check["who"], check["what"], form(check["given"]), form(check["fitted"]), form(check["off"]), "yes" if check["agrees"] else "no"])
    return rows


def remarks(placing: Placing) -> list[str]:
    said = list(placing.notes)
    said += [f"{name} is turned {abs(roll):.0f}° about its line of sight; the room draws it level" for name, roll in placing.rolled.items()]
    return said
