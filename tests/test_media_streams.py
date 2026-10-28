from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import VideoSection
from macaw.exceptions import ConfigError, ExternalToolError
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, _extract_track
from macaw.media.ffmpeg import decode_audio, has_filter, run_ffmpeg
from macaw.media.probe import probe, require_ffmpeg, require_ffprobe
from macaw.project.recording import MediaFile, build_recording

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
    files["sdr"] = root / "sdr.mp4"
    run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(files["sdr"])])
    if has_filter("zscale"):
        files["hlg"] = root / "hlg.mp4"
        run_ffmpeg([
            "-i", str(files["sdr"]), "-vf", "zscale=tin=bt709:pin=bt709:min=bt709:rin=tv:t=arib-std-b67:p=bt2020:m=bt2020nc,format=yuv420p10le",
            "-c:v", "libx264", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", str(files["hlg"]),
        ])
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

    @pytest.mark.parametrize("which, sounding", [("late_sound", True), ("variable", False)])
    def test_a_lone_camera_gives_its_sound_as_the_shared_track(self, sources, tmp_path, which, sounding):
        raw = tmp_path / "raw" / "003"
        raw.mkdir(parents=True)
        shutil.copy(sources[which], raw / "003_room.mp4")
        project = tmp_path / "project.yaml"
        scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
        config = resolve_config(project)
        (tmp_path / "work").mkdir()
        MediaPrepExtractor(config).run(build_recording("003", config), tmp_path / "work")
        shared = json.loads((tmp_path / "work" / "prep.json").read_text(encoding="utf-8"))["shared_tracks"]
        if not sounding:
            assert shared == {}
            return
        expected = beep_at(decode_audio(sources[which], sample_rate=RATE))
        assert beep_at(read_wav(tmp_path / "work" / shared["003_room.mp4"])) == pytest.approx(expected, abs=1e-3)


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


def saturation(path: Path) -> float:
    cv2 = pytest.importorskip("cv2")
    ok, image = cv2.VideoCapture(str(path)).read()
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].mean())


needs_hlg = pytest.mark.skipif(not has_filter("zscale"), reason="making a high dynamic range clip needs zscale")


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

    @needs_hlg
    def test_high_dynamic_range_is_tone_mapped_to_standard_range(self, sources, tmp_path):
        entry, warnings, copy = prepare(sources["hlg"], tmp_path)
        stream = probe(copy).video_streams()[0]
        assert entry["transfer"] == "arib-std-b67" and entry["tone_mapped"]
        assert (stream.color_transfer, stream.color_primaries) == ("bt709", "bt709")
        assert checks(warnings)["video_hdr"] == "info"
        # read without tone mapping the colours come out faded
        assert saturation(sources["hlg"]) < 0.75 * saturation(sources["sdr"]) < saturation(copy)

    @needs_hlg
    def test_high_dynamic_range_without_zscale_is_a_warning(self, sources, tmp_path, monkeypatch):
        from macaw.extractors import prep as prep_module

        monkeypatch.setattr(prep_module, "has_filter", lambda name: False)
        entry, warnings, _ = prepare(sources["hlg"], tmp_path)
        assert not entry["tone_mapped"] and checks(warnings)["video_hdr"] == "warn"

    @needs_hlg
    def test_high_dynamic_range_read_as_recorded_is_a_warning(self, sources, tmp_path):
        _, warnings, _ = prepare(sources["hlg"], tmp_path, normalise=False)
        assert checks(warnings)["video_hdr"] == "warn"

    def test_standard_range_is_left_alone(self, sources, tmp_path):
        entry, warnings, _ = prepare(sources["sdr"], tmp_path)
        assert not entry["tone_mapped"] and "video_hdr" not in checks(warnings)
