from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from macaw.media.align import AlignResult
from macaw.media.probe import probe
from macaw.project.recording import MediaFile, Recording
from macaw.qc.align_clip import FPS, LEVEL_STEP_S, alignment_clip, moments, shared_span

SR = 16000


class TestMoments:
    def test_the_loudest_stretch_is_taken_near_start_middle_and_end(self):
        level = np.full(int(600 / LEVEL_STEP_S), 0.1)
        for at in (20.0, 290.0, 580.0):
            level[int(at / LEVEL_STEP_S) : int((at + 5) / LEVEL_STEP_S)] = 5.0
        picked = moments(level, LEVEL_STEP_S, (0.0, 600.0), 10.0)
        assert [name for name, _, _ in picked] == ["start", "middle", "end"]
        for (_, start, length), loud in zip(picked, (20.0, 290.0, 580.0)):
            assert start <= loud and loud + 5 <= start + length

    def test_a_short_recording_is_shown_whole(self):
        assert moments(np.ones(10), LEVEL_STEP_S, (2.0, 8.0), 10.0) == [("whole", 2.0, 6.0)]

    def test_moments_never_overlap(self):
        picked = moments(np.ones(50), LEVEL_STEP_S, (0.0, 25.0), 10.0)
        assert all(later[1] >= earlier[1] + earlier[2] for earlier, later in zip(picked, picked[1:]))

    def test_the_span_is_where_every_device_records(self):
        media = [
            MediaFile(path=Path("a.wav"), kind="audio", duration_s=60.0),
            MediaFile(path=Path("b.wav"), kind="audio", duration_s=50.0, offset_s=5.0),
            MediaFile(path=Path("c.mp4"), kind="video", duration_s=60.0, offset_s=-2.0, clock_scale=1.0001),
        ]
        start, end = shared_span(media, media[0])
        assert start == 5.0 and end == pytest.approx(55.0)

    def test_devices_that_never_meet_fall_back_to_the_reference(self):
        media = [MediaFile(path=Path("a.wav"), kind="audio", duration_s=60.0), MediaFile(path=Path("b.wav"), kind="audio", duration_s=10.0, offset_s=100.0)]
        assert shared_span(media, media[0]) == (0.0, 60.0)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A click at recording time 30 s heard by a mic that started 5 s late, and a flash seen by a silent camera that started 2 s early."""
    from macaw.media.ffmpeg import run_ffmpeg
    from macaw.media.probe import require_ffmpeg

    try:
        require_ffmpeg()
    except Exception:
        pytest.skip("needs ffmpeg")
    folder = tmp_path_factory.mktemp("clip")
    rng = np.random.default_rng(0)

    def sound(name, length_s, click_at):
        signal = 0.002 * rng.standard_normal(int(length_s * SR))
        burst = slice(int(click_at * SR), int((click_at + 0.02) * SR))
        signal[burst] = 0.9 * np.sign(rng.standard_normal(burst.stop - burst.start))
        wavfile.write(folder / name, SR, (signal * 32767).astype("int16"))

    sound("room.wav", 60, 30.0)
    sound("mic.wav", 50, 25.0)
    run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=0x202020:s=320x240:r={FPS}:d=60",
        "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,32,32.039)'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(folder / "camera.mp4"),
    ])
    media = [
        MediaFile(path=folder / "room.wav", kind="audio", device="room"),
        MediaFile(path=folder / "mic.wav", kind="audio", device="mic"),
        MediaFile(path=folder / "camera.mp4", kind="video", device="camera"),
    ]
    infos = {}
    for item in media:
        infos[item.name] = probe(item.path)
        item.duration_s = infos[item.name].duration_s
    results = [
        AlignResult(reference="room", target="room", offset_s=0.0, confidence=1.0),
        AlignResult(reference="room", target="mic", offset_s=5.0, confidence=0.9),
        AlignResult(reference="room", target="camera", offset_s=-2.0, confidence=None, method="manual"),
    ]
    return alignment_clip(Recording(recording_id="r", root=folder, media=media), results, infos, folder / "out" / "check.mp4")


def decoded(path: Path, *args: str) -> bytes:
    from macaw.media.probe import require_ffmpeg

    return subprocess.run([require_ffmpeg(), "-v", "error", "-i", str(path), *args, "-"], capture_output=True, check=True).stdout


class TestClip:
    def test_three_moments_of_ten_seconds(self, clip):
        assert probe(clip).duration_s == pytest.approx(30.0, abs=0.1)

    def test_the_click_is_heard_once_when_the_flash_is_seen(self, clip):
        sound = np.abs(np.frombuffer(decoded(clip, "-ac", "1", "-ar", "8000", "-f", "f32le"), dtype=np.float32))
        loud = np.flatnonzero(sound > 0.5 * sound.max()) / 8000
        # one click, not one from each microphone 5 s apart
        assert loud.max() - loud.min() < 0.1
        frames = np.frombuffer(decoded(clip, "-vf", "scale=96:80,format=gray", "-f", "rawvideo"), dtype=np.uint8).reshape(-1, 80 * 96)
        flash = int(np.argmax(frames.mean(axis=1))) / FPS
        # the middle moment runs from 25 to 35 s, so both land 5 s into it, 15 s into the clip
        assert loud.min() == pytest.approx(15.0, abs=0.06)
        assert flash == pytest.approx(loud.min(), abs=1.5 / FPS)


def test_the_report_plays_the_clip_when_there_is_one(tmp_path):
    from macaw.project.layout import make_layout
    from macaw.qc.align_clip import clip_path
    from macaw.qc.report import _alignment_clip

    layout = make_layout(tmp_path / "out", tmp_path / "work")
    page = layout.qc_report_path("r1")
    assert _alignment_clip(layout, "r1", page) is None
    clip = clip_path(layout, "r1")
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"")
    assert _alignment_clip(layout, "r1", page).endswith("align/alignment_check.mp4")
