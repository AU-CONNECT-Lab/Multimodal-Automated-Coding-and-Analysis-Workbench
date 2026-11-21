"""The Review page's player: a small copy of one camera's file with sound, and how its clock reads on the recording's."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from macaw.media.ffmpeg import make_preview
from macaw.media.probe import picture_lead, probe
from macaw.project.layout import layout_for
from macaw.project.recording import MediaFile, Recording, build_recording


@dataclass
class Playing:
    camera: str
    sound: str                      # the device heard: the camera itself or another file
    offset_s: float                 # recording time of the copy's time 0, the file's first picture
    clock_scale: float
    aligned: bool                   # the offsets come from prep.media rather than being taken as 0
    window: tuple[float, float] | None  # the analysed part on the recording clock, once prep.media has run
    duration_s: float | None        # of the camera's file, on its own clock

    def own_time(self, t: float) -> float:
        return (t - self.offset_s) / self.clock_scale


def timed_recording(config, rid: str) -> tuple[Recording, tuple[float, float] | None]:
    """The recording with its files' offsets once prep.media has run, and the analysed part."""
    from macaw.extractors.prep import apply_prep_timing

    recording = build_recording(rid, config)
    try:
        prep = apply_prep_timing(config, recording)
    except Exception:   # before prep.media each file is taken to start at 0
        return recording, None
    return recording, (prep.t_origin_s, prep.t_origin_s + prep.duration_s)


def first_picture_at(config, rid: str, item: MediaFile, timed: bool) -> float:
    """Recording time of a file's first picture, where its copy's time 0 lies; timed once prep.media's offsets are applied."""
    from macaw.extractors.prep import load_video_prep

    video = load_video_prep(config, rid).get(item.name) if timed else None
    if video is not None:
        # the offset then places frame 0 of the file the video steps read
        return item.offset_s + (video["first_picture_s"] - video["frames_start_s"]) * item.clock_scale
    return item.offset_s + picture_lead(probe(item.path)) * item.clock_scale if item.kind == "video" else item.offset_s


def heard_with(recording: Recording, camera: MediaFile) -> MediaFile:
    """The camera's own sound where its file has any, else the file that hears the most people."""
    if probe(camera.path).audio_streams():
        return camera
    others = [item for item in recording.media if item.kind == "audio"]
    if not others:
        return camera
    return max(others, key=lambda item: (len(item.hears), -others.index(item)))


def playing(config, rid: str, camera: str) -> Playing:
    recording, window = timed_recording(config, rid)
    item = recording.media_named(camera)
    sound = heard_with(recording, item)
    duration = item.duration_s if item.duration_s is not None else probe(item.path).duration_s
    start = first_picture_at(config, rid, item, window is not None)
    return Playing(camera, sound.name, start, item.clock_scale, window is not None, window, duration)


def played(config, rid: str) -> Path:
    """The file the recording's copy is made of, the one its clock is set by; config is the recording's own settings."""
    return build_recording(rid, config).reference_media().path


def _played_in(config, rid: str, recording: Recording) -> MediaFile:
    path = played(config, rid)
    return next(item for item in recording.media if item.path == path)


def recording_clock(config, rid: str) -> tuple[float, float]:
    """Where the recording copy's own seconds fall on the recording's clock: its time 0 and clock scale."""
    recording, window = timed_recording(config, rid)
    item = _played_in(config, rid, recording)
    return first_picture_at(config, rid, item, window is not None), item.clock_scale


def preview_path(config, rid: str, camera: str) -> Path:
    return layout_for(config, create=False).work_dir / "previews" / rid / f"{camera}.mp4"


def _stamp(path: Path) -> Path:
    return path.with_suffix(".json")


def _made_for(config, rid: str, camera: str | None) -> dict:
    recording, window = timed_recording(config, rid)
    item = _played_in(config, rid, recording) if camera is None else recording.media_named(camera)
    sound = item if camera is None else heard_with(recording, item)
    start = first_picture_at(config, rid, item, window is not None)
    shift = 0.0 if sound is item else round((sound.offset_s - start) / item.clock_scale, 3)
    return {"source": str(item.path), "sound": str(sound.path), "sound_at_s": shift, "time_0": "first picture"}


def _current(config, rid: str, camera: str | None, path: Path) -> Path | None:
    stamp = _stamp(path)
    if not path.is_file() or not stamp.is_file():
        return None
    return path if json.loads(stamp.read_text(encoding="utf-8")) == _made_for(config, rid, camera) else None


def _make(config, rid: str, camera: str | None, path: Path) -> Path:
    made = _made_for(config, rid, camera)
    own = made["sound"] == made["source"]
    make_preview(Path(made["source"]), path, sound=None if own else Path(made["sound"]), sound_at_s=made["sound_at_s"])
    _stamp(path).write_text(json.dumps(made), encoding="utf-8")
    return path


def current_preview(config, rid: str, camera: str) -> Path | None:
    """The copy, when it was made from the files and offsets there are now."""
    return _current(config, rid, camera, preview_path(config, rid, camera))


def make_camera_preview(config, rid: str, camera: str) -> Path:
    return _make(config, rid, camera, preview_path(config, rid, camera))


def current_recording_preview(config, rid: str) -> Path | None:
    """The copy of the file the recording's clock is set by, which the Segments page and the 3D view play."""
    return _current(config, rid, None, layout_for(config, create=False).preview_path(rid))


def make_recording_preview(config, rid: str) -> Path:
    return _make(config, rid, None, layout_for(config, create=False).preview_path(rid))


LOUDNESS_STEP_S = 0.1
_SAMPLE_RATE = 4000


def loudness(path: Path) -> dict:
    """The copy's loudness every step on a 0 to 1 scale over 50 dB below its loud part; worked out once per copy."""
    from macaw.media.ffmpeg import decode_audio

    cache = path.with_suffix(".loudness.npy")
    if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
        values = np.load(cache)
    else:
        samples = decode_audio(path, sample_rate=_SAMPLE_RATE)
        width = int(_SAMPLE_RATE * LOUDNESS_STEP_S)
        bins = samples[: len(samples) // width * width].reshape(-1, width)
        level = 10 * np.log10(np.mean(bins**2, axis=1) + 1e-12)
        top = np.percentile(level, 99.5) if len(level) else 0.0
        values = np.clip((level - top + 50) / 50, 0, 1).astype(np.float32)
        np.save(cache, values)
    return {"step": LOUDNESS_STEP_S, "values": [round(float(v), 2) for v in values]}
