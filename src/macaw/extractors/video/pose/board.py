"""A ChArUco board filmed by every camera of a setup tells each camera's lens and where the cameras stand:
each camera's lens from its own views of the board, then the cameras' places from the moments two of them
saw it at once, refined together so the board's corners land where every camera saw them."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from macaw.exceptions import BackendNotAvailableError, MacawError
from macaw.extractors.video.pose.cameras import Camera, CameraSetup, through_lens

# landscape, in millimetres
PAPERS_MM = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594)}
PRINT_DPI = 300
_PAGE_MARGIN_MM = 10
_CAPTION_MM = 12
# fewer corners than this make a board pose that a slight error tips over
_MIN_CORNERS = 6
# calibrateCamera slows down a lot past this many views and gains little
_MAX_LENS_VIEWS = 60
# moments two cameras must share before one camera's place is taken from the other's
_MIN_SHARED = 5
# a corner missing by more than this many times the median miss, and by over _STRAY_MIN_PX, was misread
_STRAY_TIMES = 5.0
_STRAY_MIN_PX = 2.0
# a well-posed setup converges in tens of evaluations; this only stops a hopeless one
_MAX_EVALUATIONS = 500


@dataclass
class Sighting:
    step: int                      # index on the common sampling grid
    ids: np.ndarray                # board corner ids
    corners: np.ndarray            # (n, 2) pixels


@dataclass
class CameraClip:
    name: str
    size: tuple[int, int]
    sightings: list[Sighting] = field(default_factory=list)


@dataclass
class Fit:
    setup: CameraSetup
    lens_error_px: dict[str, float]       # each camera's own board views, after its lens was fitted
    error_px: dict[str, float]            # after all cameras were placed together
    views: dict[str, int]                 # board views each camera contributed
    shared: dict[tuple[str, str], int]    # moments each pair of cameras saw the board together
    notes: list[str]


def _cv2():
    try:
        import cv2
    except ImportError:
        raise BackendNotAvailableError("calibrating cameras needs OpenCV: pip install 'pymacaw[vision]'") from None
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise BackendNotAvailableError("calibrating cameras needs OpenCV 4.7 or later, with its aruco module")
    return cv2


def make_board(board_cfg):
    cv2 = _cv2()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, f"DICT_{board_cfg.dictionary}"))
    return cv2.aruco.CharucoBoard(
        (board_cfg.squares_x, board_cfg.squares_y), float(board_cfg.square_mm), float(board_cfg.marker_mm), dictionary
    )


def describe_board(board_cfg) -> str:
    """The board as a calibration's metadata names it, so a later change of pose3d.board shows."""
    return (
        f"ChArUco {board_cfg.squares_x}x{board_cfg.squares_y}, squares {board_cfg.square_mm:g} mm, "
        f"markers {board_cfg.marker_mm:g} mm, {board_cfg.dictionary}"
    )


def fits_on(board_cfg, paper: str) -> bool:
    page_w, page_h = PAPERS_MM[paper]
    board_w, board_h = board_cfg.squares_x * board_cfg.square_mm, board_cfg.squares_y * board_cfg.square_mm
    return board_w + 2 * _PAGE_MARGIN_MM <= page_w and board_h + 2 * _PAGE_MARGIN_MM + _CAPTION_MM <= page_h


def board_page(board_cfg, paper: str) -> np.ndarray:
    """The board at its true size on a page, to print at PRINT_DPI and 100%."""
    cv2 = _cv2()
    page_w, page_h = PAPERS_MM[paper]
    board_w, board_h = board_cfg.squares_x * board_cfg.square_mm, board_cfg.squares_y * board_cfg.square_mm
    if not fits_on(board_cfg, paper):
        raise MacawError(
            f"a {board_cfg.squares_x} x {board_cfg.squares_y} board of {board_cfg.square_mm:g} mm squares is "
            f"{board_w:g} x {board_h:g} mm and does not fit on {paper}; choose a bigger paper or smaller squares"
        )
    per_mm = PRINT_DPI / 25.4
    page = np.full((round(page_h * per_mm), round(page_w * per_mm)), 255, np.uint8)
    image = make_board(board_cfg).generateImage((round(board_w * per_mm), round(board_h * per_mm)), marginSize=0, borderBits=1)
    top, left = round(_PAGE_MARGIN_MM * per_mm), round((page_w - board_w) / 2 * per_mm)
    page[top : top + image.shape[0], left : left + image.shape[1]] = image
    caption = (
        f"ChArUco {board_cfg.squares_x} x {board_cfg.squares_y}, squares {board_cfg.square_mm:g} mm, markers "
        f"{board_cfg.marker_mm:g} mm, {board_cfg.dictionary}. Print at 100%, measure one square and set pose3d.board.square."
    )
    baseline = top + image.shape[0] + round(_CAPTION_MM * 0.6 * per_mm)
    cv2.putText(page, caption, (left, baseline), cv2.FONT_HERSHEY_SIMPLEX, per_mm * 0.12, 0, max(1, round(per_mm * 0.15)), cv2.LINE_AA)
    return page


def write_png(image: np.ndarray, path: Path, dpi: int = PRINT_DPI) -> Path:
    """A PNG that says its dots per inch, so it prints at the size it was drawn for."""
    cv2 = _cv2()
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise MacawError(f"could not encode {path}")
    data = encoded.tobytes()
    per_metre = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", per_metre, per_metre, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    # the header chunk is 8 signature bytes plus 25 of IHDR; pHYs must come before the image data
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data[:33] + chunk + data[33:])
    return path


def find_board(
    path: Path, board, *, name: str, every_s: float, offset_s: float, width_factor: float = 1.0, end_s: float | None = None
) -> CameraClip:
    """The board's corners on frames nearest a common grid every every_s seconds; offset_s takes this
    clip's own clock to the common one, so the same step is the same moment in every camera. width_factor
    stretches non-square pixels square, as prep.video does for the recordings."""
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MacawError(f"cannot open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        square = (int(width * width_factor / 2) * 2, int(height / 2) * 2) if width_factor != 1.0 else (width, height)
        clip = CameraClip(name, square)
        detector = cv2.aruco.CharucoDetector(board)
        index = 0
        while capture.grab():
            common = index / fps + offset_s
            step = round(common / every_s)
            if abs(common - step * every_s) <= 0.5 / fps and step >= 0 and (end_s is None or common <= end_s):
                ok, frame = capture.retrieve()
                if ok:
                    if square != (width, height):
                        frame = cv2.resize(frame, square, interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                    corners, ids, _, _ = detector.detectBoard(gray)
                    if ids is not None and len(ids) >= _MIN_CORNERS:
                        clip.sightings.append(Sighting(step, ids.ravel().astype(int), corners.reshape(-1, 2).astype(float)))
            index += 1
    finally:
        capture.release()
    return clip


def width_factor(path: Path, normalise: bool) -> float:
    """How much wider a clip's pixels are than tall, when prep.video makes the recordings' pixels square."""
    from macaw.media.probe import probe

    if not normalise:
        return 1.0
    streams = probe(path).video_streams()
    aspect = streams[0].pixel_aspect if streams else None
    return float(aspect) if aspect else 1.0


def frame_at(path: Path, at_s: float, width_factor: float = 1.0) -> np.ndarray | None:
    """The clip's frame nearest at_s seconds of its own clock, stretched as find_board stretches it; None outside the clip."""
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MacawError(f"cannot open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        index = round(at_s * fps)
        if index < 0 or (count and index >= count):
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return None
    if width_factor != 1.0:
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (int(width * width_factor / 2) * 2, int(height / 2) * 2), interpolation=cv2.INTER_LINEAR)
    return frame


def corners_on(frame: np.ndarray, board) -> Sighting | None:
    """The board's corners on one frame; None where too few are found to place it."""
    cv2 = _cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < _MIN_CORNERS:
        return None
    return Sighting(0, ids.ravel().astype(int), corners.reshape(-1, 2).astype(float))


def marked_jpeg(frame: np.ndarray, sighting: Sighting | None, width: int = 480) -> bytes:
    """The frame with the board corners found on it marked, shrunk to this width, as a JPEG."""
    cv2 = _cv2()
    frame = frame.copy()
    if sighting is not None:
        for x, y in sighting.corners:
            cv2.circle(frame, (int(round(x)), int(round(y))), max(3, frame.shape[1] // 160), (60, 190, 60), -1)
    height = round(frame.shape[0] * width / frame.shape[1])
    ok, encoded = cv2.imencode(".jpg", cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    return encoded.tobytes() if ok else b""


def clip_seconds(path: Path) -> float:
    cv2 = _cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (capture.get(cv2.CAP_PROP_FPS) or 30.0)
    finally:
        capture.release()


def board_in_frame(camera: Camera, sighting: Sighting, board, units: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The direction the board's face points and its middle, in the calibration's frame and units, from one camera's view."""
    cv2 = _cv2()
    corners_3d = np.asarray(board.getChessboardCorners(), dtype=float)
    ok, rvec, tvec = cv2.solvePnP(corners_3d[sighting.ids], sighting.corners, camera.matrix, camera.distortions)
    if not ok:
        return None
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES

    turn = Rotation.from_rotvec(rvec.ravel())
    # the board is drawn in millimetres
    per_mm = 0.001 / UNITS_TO_METRES[units]
    middle = (turn.apply(corners_3d.mean(axis=0)) + tvec.ravel()) * per_mm
    to_camera = Rotation.from_rotvec(camera.rotation)
    return to_camera.inv().apply(turn.apply([0.0, 0.0, 1.0])), to_camera.inv().apply(middle - camera.translation)


def calibrate(clips: list[CameraClip], board, *, units: str = "mm", fix_k3: bool = True) -> Fit:
    """Each camera's lens, then every camera's place with the first one at the origin."""
    cv2 = _cv2()
    corners_3d = np.asarray(board.getChessboardCorners(), dtype=float)
    notes: list[str] = []
    lenses, lens_error, poses, views = {}, {}, {}, {}
    for clip in clips:
        usable = [sighting for sighting in clip.sightings if not board.checkCharucoCornersCollinear(sighting.ids.reshape(-1, 1))]
        views[clip.name] = len(usable)
        if len(usable) < _MIN_SHARED:
            raise MacawError(f"{clip.name} saw the board well on {len(usable)} frames; film it longer, nearer or more slowly")
        chosen = [usable[index] for index in np.linspace(0, len(usable) - 1, min(len(usable), _MAX_LENS_VIEWS)).round().astype(int)]
        flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
        rms, matrix, distortions, _, _ = cv2.calibrateCamera(
            [corners_3d[sighting.ids].astype(np.float32) for sighting in chosen],
            [sighting.corners.astype(np.float32) for sighting in chosen],
            clip.size, None, None, flags=flags,
        )
        lenses[clip.name] = (matrix, np.resize(distortions.ravel(), 5))
        lens_error[clip.name] = float(rms)
        poses[clip.name] = {}
        for sighting in usable:
            ok, rvec, tvec = cv2.solvePnP(corners_3d[sighting.ids], sighting.corners, matrix, distortions)
            if ok:
                poses[clip.name][sighting.step] = (rvec.ravel(), tvec.ravel())

    names = [clip.name for clip in clips]
    shared = {(a, b): len(set(poses[a]) & set(poses[b])) for index, a in enumerate(names) for b in names[index + 1:]}
    places = _initial_places(names, poses, shared)
    unplaced = [name for name in names if name not in places]
    if unplaced:
        raise MacawError(
            f"{', '.join(unplaced)} never saw the board at the same moment as a placed camera ({_MIN_SHARED} moments "
            "needed); move the board where both cameras see it"
        )
    cameras, error = _refine(clips, lenses, places, poses, corners_3d)
    for (a, b), count in shared.items():
        if 0 < count < 3 * _MIN_SHARED:
            notes.append(f"{a} and {b} saw the board together only {count} times")
    setup = CameraSetup(cameras, units)
    return Fit(setup, lens_error, error, views, shared, notes)


def _initial_places(names, poses, shared) -> dict[str, tuple[Rotation, np.ndarray]]:
    """Room-to-camera transforms, the first camera as the room, each next one through the placed
    camera it shared most moments with."""
    places = {names[0]: (Rotation.identity(), np.zeros(3))}
    while True:
        candidates = [
            (count, a if a in places else b, b if a in places else a)
            for (a, b), count in shared.items()
            if count >= _MIN_SHARED and (a in places) != (b in places)
        ]
        if not candidates:
            return places
        _, known, new = max(candidates)
        steps = sorted(set(poses[known]) & set(poses[new]))
        rotations, shifts = [], []
        for step in steps:
            r_known, t_known = Rotation.from_rotvec(poses[known][step][0]), poses[known][step][1]
            r_new, t_new = Rotation.from_rotvec(poses[new][step][0]), poses[new][step][1]
            # board -> known and board -> new give known -> new
            between = r_new * r_known.inv()
            rotations.append(between)
            shifts.append(t_new - between.apply(t_known))
        between = Rotation.concatenate(rotations).mean()
        shift = np.median(np.asarray(shifts), axis=0)
        room_rotation, room_shift = places[known]
        places[new] = (between * room_rotation, between.apply(room_shift) + shift)


def _refine(clips, lenses, places, poses, corners_3d) -> tuple[list[Camera], dict[str, float]]:
    """Every camera's place and lens and every board pose at once, by how far the board's corners land from
    where they were seen. Lenses are refined too: one camera's own views leave its focal length a few percent off."""
    names = [clip.name for clip in clips]
    steps = sorted({step for name in names for step in poses[name] if sum(step in poses[other] for other in names) >= 2})
    step_index = {step: index for index, step in enumerate(steps)}
    observations = []
    for camera, clip in enumerate(clips):
        for sighting in clip.sightings:
            if sighting.step in step_index and sighting.step in poses[clip.name]:
                for corner, pixel in zip(sighting.ids, sighting.corners):
                    observations.append((camera, step_index[sighting.step], corner, pixel[0], pixel[1]))
    if not observations:
        raise MacawError("no moment saw the board in two cameras at once")
    obs = np.asarray(observations, dtype=float)
    obs_camera, obs_step, obs_corner = obs[:, 0].astype(int), obs[:, 1].astype(int), obs[:, 2].astype(int)
    seen = obs[:, 3:5]

    # a board pose in the room, from the first camera that saw it, through that camera's place
    board = np.zeros((len(steps), 6))
    for step, index in step_index.items():
        name = next(name for name in names if step in poses[name])
        rotation, shift = places[name]
        r_board, t_board = Rotation.from_rotvec(poses[name][step][0]), poses[name][step][1]
        board[index, :3] = (rotation.inv() * r_board).as_rotvec()
        board[index, 3:] = rotation.inv().apply(t_board - shift)
    cams = np.zeros((len(names), 6))
    for index, name in enumerate(names):
        cams[index, :3], cams[index, 3:] = places[name][0].as_rotvec(), places[name][1]

    # per camera fx, fy, cx, cy, k1, k2; the small tangential terms and k3 stay as each camera's own fit left them
    lens = np.stack([[m[0, 0], m[1, 1], m[0, 2], m[1, 2], d[0], d[1]] for m, d in (lenses[name] for name in names)])
    rest = np.stack([lenses[name][1] for name in names])
    n_places, n_lens = 6 * (len(names) - 1), 6 * len(names)

    def unpack(params):
        cam = np.vstack([np.zeros((1, 6)), params[:n_places].reshape(-1, 6)])
        own = params[n_places : n_places + n_lens].reshape(-1, 6)
        matrices = np.zeros((len(names), 3, 3))
        matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 0, 2], matrices[:, 1, 2], matrices[:, 2, 2] = own[:, 0], own[:, 1], own[:, 2], own[:, 3], 1.0
        distortions = rest.copy()
        distortions[:, :2] = own[:, 4:]
        return cam, matrices, distortions, params[n_places + n_lens :].reshape(-1, 6)

    def residuals(params, kept):
        cam, matrices, distortions, pose = unpack(params)
        step, camera, corner = obs_step[kept], obs_camera[kept], obs_corner[kept]
        in_room = Rotation.from_rotvec(pose[step, :3]).apply(corners_3d[corner]) + pose[step, 3:]
        inside = Rotation.from_rotvec(cam[camera, :3]).apply(in_room) + cam[camera, 3:]
        return (through_lens(inside[:, :2] / inside[:, 2:3], matrices[camera], distortions[camera]) - seen[kept]).ravel()

    def solve(start, kept):
        rows = np.arange(kept.sum())
        camera, step = obs_camera[kept], obs_step[kept]
        sparsity = lil_matrix((2 * len(rows), n_places + n_lens + 6 * len(steps)), dtype=int)
        moving = camera > 0
        for offset in range(6):
            for axis in range(2):
                sparsity[2 * rows[moving] + axis, 6 * (camera[moving] - 1) + offset] = 1
                sparsity[2 * rows + axis, n_places + 6 * camera + offset] = 1
                sparsity[2 * rows + axis, n_places + n_lens + 6 * step + offset] = 1
        # a robust loss crawled; plain least squares, once with every corner and once without the stray ones
        return least_squares(residuals, start, args=(kept,), jac_sparsity=sparsity, x_scale="jac", method="trf", max_nfev=_MAX_EVALUATIONS).x

    everything = np.ones(len(obs), dtype=bool)
    solved = solve(np.concatenate([cams[1:].ravel(), lens.ravel(), board.ravel()]), everything)
    misses = np.linalg.norm(residuals(solved, everything).reshape(-1, 2), axis=1)
    kept = misses <= max(_STRAY_TIMES * float(np.median(misses)), _STRAY_MIN_PX)
    if not kept.all():
        solved = solve(solved, kept)
        misses = np.linalg.norm(residuals(solved, everything).reshape(-1, 2), axis=1)

    cam, matrices, distortions, _ = unpack(solved)
    error = {name: float(np.sqrt(np.mean(misses[(obs_camera == index) & kept] ** 2))) for index, name in enumerate(names)}
    cameras = [
        Camera(name, clip.size, matrices[index], distortions[index], cam[index, :3].copy(), cam[index, 3:].copy())
        for index, (name, clip) in enumerate(zip(names, clips))
    ]
    return cameras, error
