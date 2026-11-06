"""A short clip of a recording's devices on the recording clock, to see and hear whether they line up."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

from macaw.exceptions import MacawError
from macaw.media.align import AlignResult, apply_alignment
from macaw.media.ffmpeg import _ENCODERS, decode_audio, has_filter, run_ffmpeg

CLIP_NAME = "alignment_check.mp4"
MOMENT_S = 10.0
# each moment is the loudest stretch within this share of the time every device records, at its start, middle and end
SEARCH_SHARE = 0.1
LEVEL_RATE = 8000
LEVEL_STEP_S = 0.5
WIDTH = 960
HEAD_HEIGHT = 30
WAVE_HEIGHT = 56
FPS = 25
QUALITY = 28
SAMPLE_RATE = 16000
_FONTS = (
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


def clip_path(layout, recording_id: str) -> Path:
    return layout.renders_dir(recording_id) / "align" / CLIP_NAME


def alignment_clip(recording, results: list[AlignResult], infos: dict, dest: Path, encoder: str = "libx264") -> Path | None:
    """Start, middle and end of the recording: the cameras side by side, a waveform per sound with a moving line, and
    all the sound mixed, so a file out of step sounds doubled. None when there is nothing to line up."""
    if len(recording.media) < 2 or not results:
        return None
    apply_alignment(recording, results)
    cameras = [item for item in recording.media if infos[item.name].video_streams()]
    sounds = [item for item in recording.media if infos[item.name].audio_streams()]
    reference = next((item for item in recording.media if item.name == results[0].reference), None)
    if not sounds or reference is None or reference not in sounds:
        return None
    span = shared_span(recording.media, reference)
    level = loudness(decode_audio(reference.path, sample_rate=LEVEL_RATE), LEVEL_RATE)
    picked = moments(level, LEVEL_STEP_S, span, MOMENT_S)
    labels = has_filter("drawtext")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="macaw_clip_") as folder:
        work = Path(folder)
        parts = []
        for index, (name, start, length) in enumerate(picked):
            part = work / f"part{index}.mp4"
            _render(work, part, f"{name}, {_clock(start)} to {_clock(start + length)}", start, length, cameras, sounds, labels, encoder)
            parts.append(part.name)
        (work / "parts.txt").write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        joined = work / CLIP_NAME
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", "parts.txt", "-c", "copy", joined.name], cwd=work)
        shutil.move(joined, dest)
    return dest


def shared_span(media, reference) -> tuple[float, float]:
    """The stretch of recording time every device records, or the reference's own when they never all overlap."""
    timed = [item for item in media if item.duration_s]
    start, end = max(item.offset_s for item in timed), min(item.offset_s + item.span_s for item in timed)
    if end - start >= 1.0:
        return start, end
    return reference.offset_s, reference.offset_s + (reference.span_s or 0.0)


def loudness(samples: np.ndarray, rate: int) -> np.ndarray:
    """Energy per LEVEL_STEP_S frame, the frame k covering recording time k x LEVEL_STEP_S on the reference."""
    size = int(rate * LEVEL_STEP_S)
    n = samples.size // size
    return np.square(samples[: n * size].reshape(n, size)).mean(axis=1) if n else np.zeros(0)


def moments(level: np.ndarray, step_s: float, span: tuple[float, float], length: float) -> list[tuple[str, float, float]]:
    """(name, start, length) for the loudest stretch near the start, middle and end of span; one when it is short."""
    start, end = span
    total = end - start
    if total <= length:
        return [("whole", start, max(total, 0.0))]
    reach = min(max(SEARCH_SHARE * total, length), total)
    regions = {
        "start": (start, start + reach),
        "middle": (start + (total - reach) / 2, start + (total + reach) / 2),
        "end": (end - reach, end),
    }
    picked: list[tuple[str, float, float]] = []
    for name, (low, high) in regions.items():
        at = _loudest(level, step_s, low, high, length)
        if picked and at < picked[-1][1] + length:
            continue
        picked.append((name, at, length))
    return picked


def _loudest(level: np.ndarray, step_s: float, low: float, high: float, length: float) -> float:
    width = max(int(round(length / step_s)), 1)
    first, last = math.ceil(low / step_s), math.floor((high - length) / step_s)
    if level.size < width or last < first:
        return low
    sums = np.convolve(level, np.ones(width), mode="valid")
    first, last = max(first, 0), min(last, sums.size - 1)
    if last < first:
        return low
    return float(first + int(np.argmax(sums[first : last + 1]))) * step_s


def _render(work: Path, out: Path, title: str, start: float, length: float, cameras, sounds, labels: bool, encoder: str) -> None:
    inputs: list[str] = []
    graph: list[str] = []
    index_of: dict[str, int] = {}
    texts = iter(range(10_000))

    def text(value: str, size: int, x: str = "8", y: str = "6") -> str:
        if not labels:
            return ""
        name = f"label{next(texts)}.txt"
        (work / name).write_text(value, encoding="utf-8")
        return f",drawtext=textfile={name}{_font()}:fontcolor=white:fontsize={size}:x={x}:y={y}:box=1:boxcolor=black@0.5:boxborderw=4"

    def opened(item) -> int | None:
        """The file's input index, opened at the moment; None when it records none of it."""
        own = item.own_time(start)
        if own >= (item.duration_s or 0.0) or own + length / item.clock_scale <= 0:
            return None
        if item.name not in index_of:
            index_of[item.name] = len(index_of)
            inputs.extend(["-ss", f"{max(own, 0.0):.3f}", "-t", f"{length + 1:.3f}", "-i", str(item.path)])
        return index_of[item.name]

    def lead(item) -> float:
        return max(-item.own_time(start), 0.0) * item.clock_scale

    columns = 1 if len(cameras) <= 1 else 2 if len(cameras) <= 4 else 3
    tile_w = WIDTH // columns
    tile_h = int(round(tile_w * 9 / 16 / 2)) * 2
    tiles = []
    for number, item in enumerate(cameras):
        k = opened(item)
        if k is None:
            graph.append(f"color=c=black:s={tile_w}x{tile_h}:d={length:.3f}:r={FPS}{text(item.name + ' (not recording)', 16)}[v{number}]")
        else:
            graph.append(
                f"[{k}:v:0]setpts=PTS-STARTPTS,fps={FPS},scale={tile_w}:{tile_h}:force_original_aspect_ratio=decrease,"
                f"pad={tile_w}:{tile_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p,"
                f"tpad=start_duration={lead(item):.3f}:color=black,tpad=stop_mode=add:stop_duration={length:.3f}:color=black,"
                f"trim=duration={length:.3f},setpts=PTS-STARTPTS{text(item.name, 16)}[v{number}]"
            )
        tiles.append(f"[v{number}]")
    rows = math.ceil(len(tiles) / columns) if tiles else 0
    for blank in range(len(tiles), rows * columns):
        graph.append(f"color=c=black:s={tile_w}x{tile_h}:d={length:.3f}:r={FPS}[v{blank}]")
        tiles.append(f"[v{blank}]")
    if len(tiles) > 1:
        layout = "|".join(f"{(i % columns) * tile_w}_{(i // columns) * tile_h}" for i in range(len(tiles)))
        graph.append(f"{''.join(tiles)}xstack=inputs={len(tiles)}:layout={layout}[grid]")
    elif tiles:
        graph.append(f"{tiles[0]}null[grid]")

    for number, item in enumerate(sounds):
        k = opened(item)
        source = (
            f"anullsrc=r={SAMPLE_RATE}:cl=mono" if k is None
            else f"[{k}:a:0]aresample={SAMPLE_RATE},aformat=channel_layouts=mono,asetpts=PTS-STARTPTS,"
            f"adelay={int(round(lead(item) * 1000))}:all=1,apad"
        )
        # levels evened out, so a quiet room microphone is heard and seen beside a close one
        graph.append(f"{source},atrim=duration={length:.3f},asetpts=PTS-STARTPTS,dynaudnorm=f=200:g=11:m=40,asplit=2[m{number}][w{number}]")
        name = item.name if k is not None else f"{item.name} (not recording)"
        graph.append(
            f"[w{number}]showwavespic=s={WIDTH}x{WAVE_HEIGHT}:colors=0x8fb8ff:scale=sqrt,loop=loop={int(length * FPS)}:size=1,"
            f"setpts=N/{FPS}/TB,format=yuv420p{text(name, 13, y='2')}[s{number}]"
        )
    strips = "".join(f"[s{number}]" for number in range(len(sounds)))
    graph.append(f"{strips}vstack=inputs={len(sounds)}[waves]" if len(sounds) > 1 else f"{strips}null[waves]")
    graph.append(f"color=c=0xff6b6b:s=2x{WAVE_HEIGHT * len(sounds)}:d={length:.3f}:r={FPS}[line]")
    graph.append(f"[waves][line]overlay=x='(W-w)*t/{length:.3f}':y=0:eof_action=pass[played]")
    graph.append(f"color=c=0x1b1f24:s={WIDTH}x{HEAD_HEIGHT}:d={length:.3f}:r={FPS}{text(title, 16)}[head]")
    body = ["[head]", *(["[grid]"] if tiles else []), "[played]"]
    graph.append(f"{''.join(body)}vstack=inputs={len(body)},trim=duration={length:.3f}[video]")
    mixed = "".join(f"[m{number}]" for number in range(len(sounds)))
    graph.append(f"{mixed}amix=inputs={len(sounds)}:normalize=1:duration=first[sound]" if len(sounds) > 1 else f"{mixed}anull[sound]")

    try:
        run_ffmpeg(
            [*inputs, "-filter_complex", ";".join(graph), "-map", "[video]", "-map", "[sound]",
             *_ENCODERS[encoder](QUALITY), "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
             "-t", f"{length:.3f}", out.name],
            cwd=work,
        )
    except MacawError as error:
        raise MacawError(f"the alignment clip could not be made: {error}") from error


def _font() -> str:
    for path in _FONTS:
        if Path(path).is_file():
            return ":fontfile='" + path.replace(":", "\\:") + "'"
    return ""


def _clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{rest:04.1f}"
