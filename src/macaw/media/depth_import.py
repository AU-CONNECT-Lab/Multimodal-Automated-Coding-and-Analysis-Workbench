"""A depth camera's own recording made into a device folder MACAW reads: the colour video, and beside it a depth folder
with one depth image per video frame, aligned with the colour picture by the camera maker's own library."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from macaw.exceptions import BackendNotAvailableError, ExternalToolError, MacawError
from macaw.extractors.video.pose.cameras import Camera
from macaw.extractors.video.pose.depth import DEPTH_FOLDER, LENS_FILE, DepthLens, write_lens
from macaw.media.probe import require_ffmpeg

# the colour video is written close to lossless, since every video step reads it again
_CRF = 16


@dataclass
class Shot:
    t_s: float
    colour: np.ndarray      # height x width x 3, RGB
    depth: np.ndarray       # height x width, aligned with the colour picture


@dataclass
class Source:
    colour: Camera
    unit_m: float
    fps: float
    shots: Iterator[Shot]
    notes: list[str]


def realsense(path: Path) -> Source:
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise BackendNotAvailableError("reading a RealSense .bag needs pyrealsense2: pip install pyrealsense2") from None
    pipe, config = rs.pipeline(), rs.config()
    config.enable_device_from_file(str(path), repeat_playback=False)
    profile = pipe.start(config)
    device = profile.get_device()
    device.as_playback().set_real_time(False)
    streams = {stream.stream_type(): stream.as_video_stream_profile() for stream in profile.get_streams()}
    if rs.stream.color not in streams or rs.stream.depth not in streams:
        pipe.stop()
        raise MacawError(f"{path.name} holds no {'colour' if rs.stream.color not in streams else 'depth'} stream")
    lens = streams[rs.stream.color].get_intrinsics()
    notes = []
    if lens.model != rs.distortion.brown_conrady and any(lens.coeffs):
        notes.append(f"the colour lens is described as {lens.model}, read here as Brown-Conrady")
    matrix = np.array([[lens.fx, 0.0, lens.ppx], [0.0, lens.fy, lens.ppy], [0.0, 0.0, 1.0]])
    colour = Camera("colour", (lens.width, lens.height), matrix, np.asarray(lens.coeffs[:5], float), np.zeros(3), np.zeros(3))
    unit_m = float(device.first_depth_sensor().get_depth_scale())
    rgb = streams[rs.stream.color].format() == rs.format.rgb8

    def shots() -> Iterator[Shot]:
        align = rs.align(rs.stream.color)
        try:
            while True:
                try:
                    frames = pipe.wait_for_frames(5000)
                except RuntimeError:
                    return
                aligned = align.process(frames)
                picture, depth = aligned.get_color_frame(), aligned.get_depth_frame()
                if not picture or not depth:
                    continue
                image = np.asanyarray(picture.get_data())
                yield Shot(picture.get_timestamp() / 1000.0, image if rgb else image[..., ::-1], np.asanyarray(depth.get_data()).copy())
        finally:
            pipe.stop()

    return Source(colour, unit_m, float(streams[rs.stream.color].fps()), shots(), notes)


def kinect(path: Path) -> Source:
    try:
        from pyk4a import CalibrationType, ImageFormat, PyK4APlayback
    except ImportError:
        raise BackendNotAvailableError(
            "reading an Azure Kinect or Orbbec Femto .mkv needs the camera maker's SDK (the Azure Kinect Sensor SDK, or "
            "Orbbec's K4A wrapper) and then pip install pyk4a"
        ) from None
    from macaw.extractors.video.pose.board import _cv2

    cv2 = _cv2()
    playback = PyK4APlayback(str(path))
    playback.open()
    calibration = playback.calibration
    matrix = np.asarray(calibration.get_camera_matrix(CalibrationType.COLOR), float)
    distortions = np.asarray(calibration.get_distortion_coefficients(CalibrationType.COLOR), float)
    form = playback.configuration["color_format"]
    fps = {0: 5.0, 1: 15.0, 2: 30.0}.get(int(playback.configuration["camera_fps"]), 30.0)

    def picture(raw: np.ndarray) -> np.ndarray:
        if form == ImageFormat.COLOR_MJPG:
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        elif form == ImageFormat.COLOR_NV12:
            image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12)
        elif form == ImageFormat.COLOR_YUY2:
            image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
        else:
            image = raw[..., :3]
        return image[..., ::-1]

    def shots() -> Iterator[Shot]:
        try:
            while True:
                try:
                    capture = playback.get_next_capture()
                except EOFError:
                    return
                if capture.color is None or capture.depth is None:
                    continue
                image = picture(capture.color)
                yield Shot(capture.color_timestamp_usec / 1e6, image, np.asarray(capture.transformed_depth, dtype=np.uint16))
        finally:
            playback.close()

    # the picture's size is read off the first frame
    colour = Camera("colour", (0, 0), matrix, distortions, np.zeros(3), np.zeros(3))
    return Source(colour, 0.001, fps, shots(), [])


def import_recording(path: Path, out: Path) -> dict:
    """Write out/<stem>.mp4 and out/depth/ from a .bag or .mkv; what was written, and any notes."""
    path, out = Path(path), Path(out)
    reader = {".bag": realsense, ".mkv": kinect}.get(path.suffix.lower())
    if reader is None:
        raise MacawError(f"{path.name}: give a RealSense .bag or an Azure Kinect or Orbbec Femto .mkv")
    source = reader(path)
    folder = out / DEPTH_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    video = out / f"{path.stem}.mp4"
    written = write_folder(source, video, folder)
    return {"video": video, "depth": folder, **written, "notes": source.notes}


def write_folder(source: Source, video: Path, folder: Path) -> dict:
    """The shots on a grid of the source's frame rate: a missing frame repeats the last picture with an empty depth image."""
    from macaw.extractors.video.pose.board import _cv2

    cv2 = _cv2()
    writer, size, start, last = None, None, None, None
    frames = empty = repeated = 0
    try:
        for shot in source.shots:
            if writer is None:
                height, width = shot.colour.shape[:2]
                size = (width, height)
                writer = _video_writer(video, size, source.fps)
                start = shot.t_s
            if shot.colour.shape[:2] != (size[1], size[0]) or shot.depth.shape != (size[1], size[0]):
                raise MacawError("the colour pictures or the depth images change size within the recording")
            slot = int(round((shot.t_s - start) * source.fps))
            if slot < frames:
                continue
            while frames < slot:
                writer.stdin.write(last.tobytes())
                cv2.imwrite(str(folder / f"{frames:06d}.png"), np.zeros((size[1], size[0]), np.uint16))
                frames, repeated = frames + 1, repeated + 1
            writer.stdin.write(np.ascontiguousarray(shot.colour, dtype=np.uint8).tobytes())
            depth = np.asarray(shot.depth, np.uint16)
            empty += int(not depth.any())
            cv2.imwrite(str(folder / f"{frames:06d}.png"), depth)
            last, frames = shot.colour, frames + 1
    finally:
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise ExternalToolError(f"ffmpeg could not write {video}: {writer.stderr.read().decode(errors='replace').strip()}")
    if writer is None:
        raise MacawError("the recording holds no moment with both a colour picture and a depth image")
    colour = Camera("colour", size, source.colour.matrix, source.colour.distortions, np.zeros(3), np.zeros(3))
    # the maker's library aligned the depth with the colour picture, so both share one lens
    write_lens(DepthLens(colour, colour, source.unit_m), folder / LENS_FILE)
    return {"frames": frames, "repeated": repeated, "empty_depth": empty, "fps": source.fps, "size": size}


def _video_writer(path: Path, size: tuple[int, int], fps: float) -> subprocess.Popen:
    command = [
        require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-crf", str(_CRF),
        "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
