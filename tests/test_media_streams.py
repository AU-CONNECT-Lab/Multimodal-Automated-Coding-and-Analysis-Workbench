from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from macaw.exceptions import ExternalToolError
from macaw.extractors.prep import _extract_track
from macaw.media.ffmpeg import decode_audio, run_ffmpeg
from macaw.media.probe import require_ffmpeg, require_ffprobe

try:
    require_ffmpeg()
    require_ffprobe()
except ExternalToolError:
    pytest.skip("needs ffmpeg and ffprobe", allow_module_level=True)

RATE = 16000
BEEP_S = 1.0
FLASH_FRAME = 25
FPS = 25


def beep_at(samples: np.ndarray, rate: int = RATE) -> float:
    return float(np.argmax(np.abs(samples) > 0.3)) / rate


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), "<i2") / 32768.0


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict[str, Path]:
    """One file whose pictures start 0.2 s after its sound and one the other way round; a beep at
    1 s of the sound and a white picture at frame 25."""
    root = tmp_path_factory.mktemp("streams")
    sound, pictures = root / "sound.wav", root / "pictures.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", f"aevalsrc='if(between(t,{BEEP_S},{BEEP_S + 0.05}),sin(2*PI*1000*t),0)':s=48000:d=4", str(sound)])
    run_ffmpeg([
        "-f", "lavfi", "-i", f"color=black:s=320x240:r={FPS}:d=4,drawbox=color=white:t=fill:enable='eq(n,{FLASH_FRAME})'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(pictures),
    ])
    late_pictures, late_sound = root / "late_pictures.mp4", root / "late_sound.mp4"
    run_ffmpeg(["-i", str(sound), "-itsoffset", "0.2", "-i", str(pictures), "-map", "1:v", "-map", "0:a", "-c:v", "copy", "-c:a", "aac", str(late_pictures)])
    run_ffmpeg(["-i", str(pictures), "-itsoffset", "0.2", "-i", str(sound), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(late_sound)])
    return {"late_pictures": late_pictures, "late_sound": late_sound}


class TestPreparedTrack:
    @pytest.mark.parametrize("which", ["late_pictures", "late_sound"])
    @pytest.mark.parametrize("start_s", [0.0, 0.1, 0.5])
    def test_a_cut_lands_on_the_clock_alignment_measured_on(self, sources, tmp_path, which, start_s):
        """Alignment decodes the whole sound; a cut starting at start_s must find the beep start_s earlier."""
        expected = beep_at(decode_audio(sources[which], sample_rate=RATE)) - start_s
        track = _extract_track(sources[which], tmp_path / "track.wav", start_s=start_s, duration_s=2.0, sample_rate=RATE)
        samples = read_wav(track)
        assert len(samples) == 2 * RATE
        assert beep_at(samples) == pytest.approx(expected, abs=1e-3)

    def test_a_negative_start_pads_the_front(self, sources, tmp_path):
        expected = beep_at(decode_audio(sources["late_sound"], sample_rate=RATE)) + 0.5
        track = _extract_track(sources["late_sound"], tmp_path / "track.wav", start_s=-0.5, duration_s=3.0, sample_rate=RATE)
        assert beep_at(read_wav(track)) == pytest.approx(expected, abs=1e-3)
