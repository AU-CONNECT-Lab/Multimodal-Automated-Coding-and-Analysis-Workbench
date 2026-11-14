"""The Review page's player: a small copy of one camera's file with sound, and how its clock reads on the recording's."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from macaw.media.ffmpeg import make_preview
from macaw.media.probe import probe
from macaw.project.layout import layout_for
from macaw.project.recording import MediaFile, Recording, build_recording


@dataclass
class Playing:
    camera: str
    sound: str                      # the device heard: the camera itself or another file
    offset_s: float                 # recording time of the copy's time 0
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
    return Playing(camera, sound.name, item.offset_s, item.clock_scale, window is not None, window, duration)


def preview_path(config, rid: str, camera: str) -> Path:
    return layout_for(config, create=False).work_dir / "previews" / rid / f"{camera}.mp4"


def _stamp(path: Path) -> Path:
    return path.with_suffix(".json")


def _made_for(config, rid: str, camera: str) -> dict:
    recording, _ = timed_recording(config, rid)
    item = recording.media_named(camera)
    sound = heard_with(recording, item)
    shift = 0.0 if sound is item else round((sound.offset_s - item.offset_s) / item.clock_scale, 3)
    return {"source": str(item.path), "sound": str(sound.path), "sound_at_s": shift}


def current_preview(config, rid: str, camera: str) -> Path | None:
    """The copy, when it was made from the files and offsets there are now."""
    path = preview_path(config, rid, camera)
    stamp = _stamp(path)
    if not path.is_file() or not stamp.is_file():
        return None
    return path if json.loads(stamp.read_text(encoding="utf-8")) == _made_for(config, rid, camera) else None


def make_camera_preview(config, rid: str, camera: str) -> Path:
    made = _made_for(config, rid, camera)
    path = preview_path(config, rid, camera)
    own = made["sound"] == made["source"]
    make_preview(Path(made["source"]), path, sound=None if own else Path(made["sound"]), sound_at_s=made["sound_at_s"])
    _stamp(path).write_text(json.dumps(made), encoding="utf-8")
    return path
