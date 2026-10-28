from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macaw.config.schema import VideoSection
from macaw.exceptions import ConfigError, ExternalToolError
from macaw.extractors.prep import VideoPrepExtractor, _extract_track
from macaw.media.ffmpeg import decode_audio, run_ffmpeg
from macaw.media.probe import probe, require_ffmpeg, require_ffprobe
from macaw.project.recording import MediaFile

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
    files = {"late_pictures": late_pictures, "late_sound": late_sound}

    def make(name, source, *video):
        files[name] = root / f"{name}.mp4"
        run_ffmpeg(["-f", "lavfi", "-i", source, *video, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(files[name])])

    # every other frame missing between 1 s and 2 s, with the timestamps kept
    make("variable", "testsrc2=s=320x240:r=30:d=4", "-vf", "select='not(between(t,1,2)*mod(n,2))'", "-fps_mode", "vfr")
    make("anamorphic", "testsrc2=s=720x480:r=25:d=2", "-vf", "setsar=32/27")
    make("interlaced", "testsrc2=s=720x480:r=30000/1001:d=2", "-vf", "tinterlace=interleave_top,fieldorder=tff",
         "-flags", "+ildct+ilme", "-x264opts", "tff=1")
    files["rotated"] = root / "rotated.mp4"
    run_ffmpeg(["-display_rotation", "90", "-i", str(pictures), "-c", "copy", str(files["rotated"])])
    return files


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


def prepare(path: Path, workspace: Path, **settings):
    """(the entry prep.video wrote, its warnings, the copy's path or None)."""
    extractor = VideoPrepExtractor(SimpleNamespace(video=VideoSection(**settings)))
    recording = SimpleNamespace(
        recording_id="r1", media=[MediaFile(path=path, kind="video")], entities=lambda: {"group_id": "r1"},
    )
    workspace.mkdir(parents=True, exist_ok=True)
    result = extractor.run(recording, workspace)
    (entry,) = json.loads(result.artifacts["video.json"].read_text(encoding="utf-8"))
    return entry, result.warnings, (workspace / entry["file"]) if entry["file"] else None


def flash_frame(path: Path) -> int:
    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(path))
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            raise AssertionError("no white frame")
        if image.mean() > 128:
            return index
        index += 1


def checks(warnings) -> dict[str, str]:
    return {warning.check: warning.severity for warning in warnings}


class TestPreparedVideo:
    @pytest.mark.parametrize("which", ["late_pictures", "late_sound"])
    def test_frame_0_sits_at_the_first_sound_so_picture_and_sound_keep_their_distance(self, sources, tmp_path, which):
        """The white picture comes 0.2 s after the beep in one file and 0.2 s before it in the other."""
        entry, _, copy = prepare(sources[which], tmp_path)
        beep = beep_at(decode_audio(sources[which], sample_rate=RATE))
        flash = flash_frame(copy) / entry["fps"]
        expected = 0.2 if which == "late_pictures" else -0.2
        assert flash - beep == pytest.approx(expected, abs=0.5 / FPS)

    def test_late_pictures_repeat_the_first_one_and_say_so(self, sources, tmp_path):
        entry, warnings, _ = prepare(sources["late_pictures"], tmp_path)
        assert entry["frames_repeated"] == 5
        assert checks(warnings)["video_start"] == "info"

    def test_read_as_recorded_the_start_is_left_to_the_clock(self, sources, tmp_path):
        entry, _, copy = prepare(sources["late_pictures"], tmp_path, normalise=False)
        assert copy is None
        assert entry["frames_start_s"] == pytest.approx(0.2)
        assert entry["fps"] == FPS

    def test_a_variable_rate_becomes_constant(self, sources, tmp_path):
        entry, warnings, copy = prepare(sources["variable"], tmp_path)
        stream = probe(copy).video_streams()[0]
        assert entry["variable_rate"] and entry["fps"] == 30
        assert stream.declared_fps == stream.average_fps == 30
        assert entry["frames_repeated"] == 15
        assert checks(warnings)["video_variable_rate"] == "info"

    def test_a_variable_rate_read_as_recorded_is_a_warning(self, sources, tmp_path):
        _, warnings, _ = prepare(sources["variable"], tmp_path, normalise=False)
        assert checks(warnings)["video_variable_rate"] == "warn"

    def test_a_turned_picture_is_made_upright(self, sources, tmp_path):
        entry, warnings, copy = prepare(sources["rotated"], tmp_path)
        stream = probe(copy).video_streams()[0]
        assert (stream.width, stream.height, stream.rotation) == (240, 320, 0.0)
        assert checks(warnings)["video_rotated"] == "info"

    def test_wide_pixels_are_stretched_square(self, sources, tmp_path):
        entry, warnings, copy = prepare(sources["anamorphic"], tmp_path)
        stream = probe(copy).video_streams()[0]
        assert (stream.width, stream.height, stream.pixel_aspect) == (852, 480, 1)
        assert checks(warnings)["video_pixel_shape"] == "info"

    def test_interlaced_pictures_are_deinterlaced(self, sources, tmp_path):
        _, warnings, copy = prepare(sources["interlaced"], tmp_path)
        assert probe(copy).video_streams()[0].field_order == "progressive"
        assert checks(warnings)["video_interlaced"] == "info"

    def test_one_rate_for_all_keeps_every_other_frame(self, sources, tmp_path):
        entry, warnings, copy = prepare(sources["late_pictures"], tmp_path, fps=12.5)
        assert entry["fps"] == 12.5
        assert probe(copy).video_streams()[0].declared_fps == 12.5
        assert checks(warnings)["video_rate_changed"] == "info"

    def test_a_rate_that_does_not_divide_the_file_is_refused(self, sources, tmp_path):
        with pytest.raises(ConfigError, match="25, 12.5"):
            prepare(sources["late_pictures"], tmp_path, fps=30)
