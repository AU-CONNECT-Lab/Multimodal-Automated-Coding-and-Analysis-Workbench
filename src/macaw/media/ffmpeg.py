from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from macaw.exceptions import ExternalToolError, UnreadableMediaError
from macaw.media.probe import require_ffmpeg

# every call starts with these: no banner, errors only, never prompt to overwrite
_BASE = ["-hide_banner", "-loglevel", "error", "-y"]


def run_ffmpeg(args: list[str]) -> str:
    """Raises ExternalToolError with ffmpeg's stderr attached on non-zero exit."""
    command = [require_ffmpeg(), *_BASE, *args]
    result = subprocess.run(command, capture_output=True, text=True)
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


def apply_offset(src: Path, dest: Path, *, offset_s: float) -> Path:
    """Shift the whole file onto the recording clock by rewriting its timestamps."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-itsoffset", f"{offset_s:.6f}", "-i", str(src), "-c", "copy", str(dest)]
    )
    return dest


def resample_for_drift(src: Path, dest: Path, *, drift_ppm: float) -> Path:
    """Stretch or squeeze audio so a recorder running fast lines up again. Audio only: the
    video track is passed through, since drift correction of frames needs a recode."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    factor = 1.0 / (1.0 + drift_ppm / 1e6)
    from macaw.media.probe import probe

    info = probe(src)
    audio = info.audio_streams()
    if not audio:
        raise UnreadableMediaError(f"{src} has no audio to resample")
    rate = int(audio[0].sample_rate or 48000)
    run_ffmpeg(
        [
            "-i", str(src),
            "-filter:a", f"asetrate={rate * factor:.6f},aresample={rate}",
            str(dest),
        ]
    )
    return dest


def extract_thumbnails(src: Path, dest_dir: Path, *, every_s: float = 10.0) -> list[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-i", str(src), "-vf", f"fps=1/{every_s:g}", str(dest_dir / "thumb_%05d.jpg")]
    )
    return sorted(dest_dir.glob("thumb_*.jpg"))
