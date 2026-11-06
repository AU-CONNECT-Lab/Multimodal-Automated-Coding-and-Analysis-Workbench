from __future__ import annotations

import re
import subprocess
from fractions import Fraction
from functools import cache
from pathlib import Path

import numpy as np

from macaw.exceptions import ExternalToolError, UnreadableMediaError
from macaw.media.probe import StreamInfo, jumps_exactly, probe, require_ffmpeg

# every call starts with these: no banner, errors only, never prompt to overwrite
_BASE = ["-hide_banner", "-loglevel", "error", "-y"]
# a jump lands on a keyframe at least this far ahead of the wanted frame, so decoding starts before it
SEEK_MARGIN_S = 1.0


def run_ffmpeg(args: list[str], cwd: Path | None = None) -> str:
    """Raises ExternalToolError with ffmpeg's stderr attached on non-zero exit."""
    command = [require_ffmpeg(), *_BASE, *args]
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise ExternalToolError(f"ffmpeg failed: {' '.join(args)}\n{result.stderr.strip()}")
    return result.stderr


def decode_audio(src: Path, *, sample_rate: int = 16000, channels: int = 1) -> np.ndarray:
    """Decode straight into memory as float32. Going through a temp wav would double the
    IO for every alignment pass, and cluster scratch is the scarce resource."""
    command = [
        require_ffmpeg(), *_BASE,
        "-i", str(src),
        "-vn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise ExternalToolError(
            f"ffmpeg could not decode audio from {src}\n{result.stderr.decode(errors='replace').strip()}"
        )
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        raise UnreadableMediaError(f"{src} carries no audio stream")
    return samples.astype(np.float64)


def extract_audio(
    src: Path,
    dest: Path,
    *,
    stream_index: int | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(src)]
    if stream_index is not None:
        args += ["-map", f"0:a:{stream_index}"]
    args += ["-vn", "-ac", str(channels), "-ar", str(sample_rate), str(dest)]
    run_ffmpeg(args)
    return dest


def cut(
    src: Path, dest: Path, *, start_s: float, end_s: float | None = None, recode: bool = False
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i seeks by keyframe, which is fast but lands early; with -c copy that is
    # the only option, so recode when the cut has to be frame accurate
    args = ["-ss", f"{start_s:.6f}", "-i", str(src)]
    if end_s is not None:
        args += ["-to", f"{max(end_s - start_s, 0.0):.6f}"]
    args += [] if recode else ["-c", "copy"]
    args += [str(dest)]
    run_ffmpeg(args)
    return dest


_FPS_COUNTS = re.compile(r"(\d+) frames in, (\d+) frames out; (\d+) frames dropped, (\d+) frames duplicated")
_ENCODERS = {
    "libx264": lambda quality: ["-c:v", "libx264", "-preset", "fast", "-crf", str(quality)],
    "h264_nvenc": lambda quality: ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"],
}


# transfer functions of high dynamic range video, as ffprobe names them
HDR_TRANSFERS = {"arib-std-b67": "HLG", "smpte2084": "PQ"}


@cache
def has_filter(name: str) -> bool:
    result = subprocess.run([require_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, errors="replace")
    return any(line.split()[1:2] == [name] for line in result.stdout.splitlines())


def tone_map_filters(stream: StreamInfo) -> list[str]:
    """High dynamic range to BT.709: a Mobius curve keeps the mid-tones and folds the highlights in under white."""
    # stated in full, because a file whose tags are missing leaves zscale with no path between colour spaces
    primaries = stream.color_primaries if stream.color_primaries not in (None, "unknown") else "bt2020"
    matrix = stream.color_space if stream.color_space not in (None, "unknown") else "bt2020nc"
    source = f"tin={stream.color_transfer}:pin={primaries}:min={matrix}:rin={stream.color_range or 'tv'}"
    return [
        f"zscale={source}:t=linear:npl=100:p={primaries}:m={matrix}", "format=gbrpf32le",
        "zscale=p=bt709", "tonemap=tonemap=mobius:desat=0", "zscale=t=bt709:m=bt709:r=tv", "format=yuv420p",
    ]


def normalise_video(
    src: Path, dest: Path, *, fps: Fraction, start_s: float, quality: int, encoder: str = "libx264",
    tone_map: StreamInfo | None = None,
) -> tuple[int, int] | None:
    """Constant-rate, upright, progressive, square-pixel H.264 with frame 0 at start_s on the container's timeline,
    tone-mapped from tone_map's high dynamic range when given; returns (repeated, dropped)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        "bwdif=mode=send_frame:parity=auto:deint=interlaced",
        *(tone_map_filters(tone_map) if tone_map else []),
        "scale=w='trunc(iw*sar/2)*2':h='trunc(ih/2)*2'", "setsar=1",
        # shifted first so the constant-rate grid starts exactly at start_s rather than at the nearest frame slot
        f"setpts=PTS-({start_s:.6f}/TB)",
        f"fps=fps={fps.numerator}/{fps.denominator}:start_time=0",
    ]
    command = [
        # -copyts: without it a transport stream read for its pictures alone starts at its first picture, not the file's start
        require_ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "verbose", "-y", "-copyts",
        "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", ",".join(filters), "-fps_mode", "passthrough",
        *_ENCODERS[encoder](quality), "-pix_fmt", "yuv420p",
        *(["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"] if tone_map else []),
        "-movflags", "+faststart", str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise ExternalToolError(f"ffmpeg could not re-encode {src}\n{result.stderr.strip()[-2000:]}")
    counts = _FPS_COUNTS.findall(result.stderr)
    if not counts:
        return None
    _, _, dropped, repeated = (int(value) for value in counts[-1])
    return repeated, dropped


def extract_frame(
    src: Path, at_s: float, dest: Path, *, square_pixels: bool = False, long_side: int | None = None
) -> Path:
    """The frame nearest at_s from the first picture; square_pixels shapes it like normalise_video's copies, long_side caps its longer side."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    if not info.video_streams():
        raise UnreadableMediaError(f"{src}: no video stream")
    stream = info.video_streams()[0]
    wanted = (stream.start_s if stream.start_s is not None else info.start_s) + max(at_s, 0.0)
    # half a frame early absorbs rounded timestamps; with -copyts t is the file's own clock
    filters = [f"select=gte(t\\,{wanted - (0.5 / stream.fps if stream.fps else 0.0):.6f})"]
    if square_pixels:
        filters += ["scale=w='trunc(iw*sar/2)*2':h='trunc(ih/2)*2'", "setsar=1"]
    if long_side:
        filters.append(
            f"scale=w='if(gte(iw,ih),min(iw,{long_side}),-2)':h='if(gte(iw,ih),-2,min(ih,{long_side}))'"
        )
    jump = ["-noaccurate_seek", "-ss", f"{max(wanted - info.start_s - SEEK_MARGIN_S, 0.0):.6f}"] if jumps_exactly(info) else []
    run_ffmpeg([
        "-copyts", *jump, "-i", str(src), "-vf", ",".join(filters), "-fps_mode", "passthrough",
        "-frames:v", "1", "-q:v", "2", str(dest),
    ])
    if not dest.is_file():
        raise UnreadableMediaError(f"{src} has no frame at {at_s:.3f} s")
    return dest


def extract_thumbnails(src: Path, dest_dir: Path, *, every_s: float = 10.0) -> list[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-i", str(src), "-vf", f"fps=1/{every_s:g}", str(dest_dir / "thumb_%05d.jpg")]
    )
    return sorted(dest_dir.glob("thumb_*.jpg"))


def make_preview(src: Path, dest: Path, *, height: int = 360) -> Path:
    """A small copy of a recording file that a browser plays, its sound kept, timed like the file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.stem + ".partial.mp4")
    pictures = []
    if probe(src).video_streams():
        pictures = [
            "-map", "0:v:0", "-vf", f"yadif=deint=interlaced,scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        ]
    run_ffmpeg(["-i", str(src), *pictures, "-map", "0:a:0?", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(partial)])
    partial.replace(dest)
    return dest
