from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from macaw.exceptions import ExternalToolError, UnreadableMediaError

# ffmpeg is a system binary, not a wheel, so it is found at runtime rather than declared
# in pyproject. Same treatment as the other external tools.
FFMPEG_ENV_VAR = "MACAW_FFMPEG"
FFPROBE_ENV_VAR = "MACAW_FFPROBE"

_INSTALL_HINT = (
    "Install it from https://ffmpeg.org/download.html, or point "
    f"${FFMPEG_ENV_VAR} / ${FFPROBE_ENV_VAR} at the binaries."
)


@dataclass
class StreamInfo:
    index: int
    codec_type: str
    codec_name: str
    sample_rate: float | None = None
    channels: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    start_s: float | None = None
    declared_fps: Fraction | None = None    # the rate the container states
    average_fps: Fraction | None = None     # frames over duration
    rotation: float = 0.0                   # degrees players turn the picture
    pixel_aspect: Fraction | None = None    # width of a pixel over its height; None when unstated
    field_order: str | None = None          # progressive, or tt / bb / tb / bt for interlaced


@dataclass
class MediaInfo:
    path: Path
    duration_s: float
    streams: list[StreamInfo]
    variable_frame_rate: bool = False
    creation_time: str | None = None    # as the device wrote it; its clock and time zone are unchecked
    start_s: float = 0.0                # where the container's own timeline starts

    def video_streams(self) -> list[StreamInfo]:
        return [stream for stream in self.streams if stream.codec_type == "video"]

    def audio_streams(self) -> list[StreamInfo]:
        return [stream for stream in self.streams if stream.codec_type == "audio"]


def _require(tool: str, env_var: str) -> str:
    override = os.environ.get(env_var)
    if override:
        if not Path(override).is_file():
            raise ExternalToolError(f"${env_var} points at a missing file: {override}")
        return override
    found = shutil.which(tool)
    if found is None:
        raise ExternalToolError(f"{tool} was not found on PATH. {_INSTALL_HINT}")
    return found


def require_ffmpeg() -> str:
    return _require("ffmpeg", FFMPEG_ENV_VAR)


def require_ffprobe() -> str:
    return _require("ffprobe", FFPROBE_ENV_VAR)


def run_ffprobe(args: list[str]) -> str:
    command = [require_ffprobe(), *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ExternalToolError(f"ffprobe failed on {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout


def _ratio(value: str | None) -> float | None:
    """ffprobe reports frame rates as "30000/1001"."""
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else None
    return float(value)


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fraction(value: str | None, separator: str = "/") -> Fraction | None:
    """"30000/1001", or "32:27" with separator ":"; None for 0/0, N/A and a zero numerator."""
    try:
        numerator, _, denominator = str(value).partition(separator)
        found = Fraction(int(numerator), int(denominator or 1))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return found or None


# the rates cameras record at; a variable-rate file is made constant at the nearest one
STANDARD_RATES = tuple(
    Fraction(rate) for rate in ("10", "12", "15", "20", "24000/1001", "24", "25", "30000/1001", "30", "48", "50",
                                "60000/1001", "60", "100", "120000/1001", "120")
)


def _multiple(declared: Fraction, average: Fraction) -> int | None:
    """k when the declared rate is k times the average one, as with interlaced files that state the field rate."""
    ratio = declared / average
    k = round(ratio)
    return k if k >= 1 and abs(ratio - k) <= Fraction(1, 1000) * k else None


def is_variable_rate(stream: StreamInfo) -> bool:
    if stream.declared_fps is None or stream.average_fps is None:
        return False
    return _multiple(stream.declared_fps, stream.average_fps) is None


def frame_rate(stream: StreamInfo) -> Fraction | None:
    """The constant rate the camera meant; for a variable-rate file the plausible declared rate or the average, snapped to a standard one."""
    declared, average = stream.declared_fps, stream.average_fps
    if declared is None or average is None:
        return declared or average
    k = _multiple(declared, average)
    if k is not None:
        return declared / k
    # some containers declare their time base, e.g. 1000 or 90000
    base = declared if declared <= 2 * average else average
    nearest = min(STANDARD_RATES, key=lambda rate: abs(rate - base))
    return nearest if abs(nearest - base) <= Fraction(15, 1000) * base else base.limit_denominator(1001)


def probe(path: Path) -> MediaInfo:
    path = Path(path)
    if not path.is_file():
        raise UnreadableMediaError(f"No such media file: {path}")

    payload = json.loads(
        run_ffprobe(
            ["-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        )
    )

    streams: list[StreamInfo] = []
    for raw in payload.get("streams", []):
        average = _ratio(raw.get("avg_frame_rate"))
        nominal = _ratio(raw.get("r_frame_rate"))
        video = raw.get("codec_type") == "video"
        rotations = [item["rotation"] for item in raw.get("side_data_list", []) if "rotation" in item]
        streams.append(
            StreamInfo(
                index=int(raw.get("index", 0)),
                codec_type=raw.get("codec_type", "unknown"),
                codec_name=raw.get("codec_name", "unknown"),
                sample_rate=_number(raw.get("sample_rate")),
                channels=raw.get("channels"),
                fps=average or nominal,
                width=raw.get("width"),
                height=raw.get("height"),
                duration_s=_number(raw.get("duration")),
                start_s=_number(raw.get("start_time")),
                declared_fps=_fraction(raw.get("r_frame_rate")) if video else None,
                average_fps=_fraction(raw.get("avg_frame_rate")) if video else None,
                rotation=float(rotations[0]) if rotations else 0.0,
                pixel_aspect=_fraction(raw.get("sample_aspect_ratio"), ":") if video else None,
                field_order=raw.get("field_order"),
            )
        )
    # a container that declares one rate and delivers another has frame times that cannot be counted
    variable = any(is_variable_rate(stream) for stream in streams if stream.codec_type == "video")

    duration = _number(payload.get("format", {}).get("duration"))
    if duration is None:
        known = [stream.duration_s for stream in streams if stream.duration_s is not None]
        duration = max(known) if known else 0.0
    if not duration:
        raise UnreadableMediaError(f"{path}: ffprobe reports no duration, the file may be truncated")

    tagged = [payload.get("format", {}), *payload.get("streams", [])]
    creation = next(
        (item["tags"]["creation_time"] for item in tagged if "creation_time" in (item.get("tags") or {})), None
    )
    return MediaInfo(
        path=path, duration_s=duration, streams=streams, variable_frame_rate=variable, creation_time=creation,
        start_s=_number(payload.get("format", {}).get("start_time")) or 0.0,
    )


def frame_timestamps(path: Path, stream_index: int = 0) -> list[float]:
    """Exact presentation timestamps; currentTime += 1/fps drifts on VFR video."""
    output = run_ffprobe(
        [
            "-v", "error",
            "-select_streams", f"v:{stream_index}",
            "-show_entries", "frame=pts_time",
            "-of", "csv=p=0",
            str(path),
        ]
    )
    return [float(line) for line in output.splitlines() if line.strip() and line.strip() != "N/A"]
