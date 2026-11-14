from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Literal

from macaw.config.schema import MacawConfig, SetupSection
from macaw.exceptions import ConfigError, RecordingNotFoundError, UnreadableMediaError
from macaw.project.setup import device_people

MediaKind = Literal["audio", "video", "depth"]

ENTITY_NAMES = ("group_id", "session_id", "task", "run")

AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".mts", ".m4v"})

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


@dataclass
class MediaFile:
    path: Path
    kind: MediaKind
    stream_index: int | None = None
    subject_id: str | None = None      # set when this is a per-person mic or camera
    # the folder it was found in; names it everywhere, since files in two folders may share a name
    device: str = ""
    # whose voices its sound carries, and whom a camera is read for per kind of measure, as the setup says
    hears: tuple[str, ...] = ()
    uses: dict[str, tuple[str, ...]] = field(default_factory=dict)
    sample_rate: float | None = None
    fps: float | None = None
    duration_s: float | None = None
    offset_s: float = 0.0              # seconds to add to this file's clock to reach recording time
    clock_scale: float = 1.0           # recording seconds per second of this file's clock, once drift is corrected
    drift_ppm: float | None = None
    offset_confidence: float | None = None
    prepared_video: Path | None = None  # prep.video's re-encoded copy
    depth: bool = False                 # a camera that also records depth

    @property
    def name(self) -> str:
        return self.device or self.path.name

    @property
    def frames_path(self) -> Path:
        """The file video steps decode frames from."""
        return self.prepared_video or self.path

    def own_time(self, t: float) -> float:
        """Where recording time t falls on this file's own clock."""
        return (t - self.offset_s) / self.clock_scale

    @property
    def span_s(self) -> float | None:
        """How long the file lasts in recording time."""
        return None if self.duration_s is None else self.duration_s * self.clock_scale


@dataclass
class Subject:
    subject_id: str
    role: str | None = None            # free-form label, never a binary A/B assumption
    age_months: int | None = None
    sex: str | None = None
    audio_track: str | None = None
    roi: tuple[int, int, int, int] | None = None


@dataclass
class Recording:
    """One stretch of media sharing a single clock. The parallel unit of the whole package:
    one array index, one output directory, one alignment."""

    recording_id: str
    root: Path
    # BIDS entities parsed out of the path; all optional, a flat dataset leaves them None
    group_id: str | None = None        # the people recorded together, BIDS sub-
    session_id: str | None = None      # visit or wave, BIDS ses-
    task: str | None = None
    run: str | None = None
    subjects: list[Subject] = field(default_factory=list)
    media: list[MediaFile] = field(default_factory=list)
    t_task_start: float | None = None
    t_task_end: float | None = None
    t_keep_last: float | None = None
    metadata: dict = field(default_factory=dict)

    def entities(self) -> dict[str, str | None]:
        """The identity columns every long-table row carries."""
        return {name: getattr(self, name) for name in ENTITY_NAMES}

    def subject(self, subject_id: str) -> Subject:
        for candidate in self.subjects:
            if candidate.subject_id == subject_id:
                return candidate
        known = ", ".join(s.subject_id for s in self.subjects) or "none"
        raise ValueError(f"{self.recording_id} has no subject '{subject_id}'. Known: {known}")

    def audio_for(self, subject_id: str) -> MediaFile | None:
        """The device whose sound carries this person's voice alone."""
        for item in self.media:
            if item.hears == (subject_id,):
                return item
        return None

    def shared_sound(self) -> list[MediaFile]:
        """Devices whose sound carries several people's voices."""
        return sorted((item for item in self.media if len(item.hears) > 1), key=lambda item: item.name)

    def cameras(self) -> list[MediaFile]:
        """The cameras read for anyone's body, face or gaze, in the setup's order."""
        return [item for item in self.media if item.kind == "video" and any(item.uses.values())]

    def views(self, use: str) -> dict[str, MediaFile]:
        """person -> the camera read for that measure of theirs: with several, their own camera (one read for them
        alone), else the first in the setup."""
        found: dict[str, MediaFile] = {}
        for own in (True, False):
            for item in self.cameras():
                if own and len(set().union(*item.uses.values())) != 1:
                    continue
                for person in item.uses.get(use, ()):
                    found.setdefault(person, item)
        return found

    def pair_view(self, a: str, b: str) -> MediaFile | None:
        """The first camera read for both people's bodies, where a distance between them means something."""
        return next((item for item in self.cameras() if {a, b} <= set(item.uses.get("body", ()))), None)

    def media_named(self, name: str) -> MediaFile:
        for item in self.media:
            if item.name == name:
                return item
        known = ", ".join(item.name for item in self.media) or "none"
        raise ValueError(f"{self.recording_id} has no device {name}. Known: {known}")

    def reference_media(self, among: list[MediaFile] | None = None) -> MediaFile:
        """The clock everything else is aligned to, picked from among when given."""
        pool = self.media if among is None else among
        for kind in ("video", "audio"):
            candidates = [item for item in pool if item.kind == kind]
            if candidates:
                # longest first, then by path so the choice never depends on scan order
                return sorted(candidates, key=lambda m: (-(m.duration_s or 0.0), str(m.path)))[0]
        raise UnreadableMediaError(f"{self.recording_id} has no audio or video to align to")

    def pairs(self) -> list[tuple[str, str]]:
        """Ordered subject pairs; pairwise features are stored per pair, dyad is just N=2."""
        return list(combinations([subject.subject_id for subject in self.subjects], 2))

    def duration_s(self) -> float:
        known = [item.duration_s for item in self.media if item.duration_s is not None]
        if not known:
            raise UnreadableMediaError(
                f"{self.recording_id}: media have not been probed yet, no duration known"
            )
        return max(known)


def _entity_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    cursor = 0
    for match in _PLACEHOLDER.finditer(pattern):
        name = match.group(1)
        if name not in ENTITY_NAMES:
            raise ConfigError(
                f"recording_pattern uses unknown entity '{name}'. "
                f"Available: {', '.join(ENTITY_NAMES)}"
            )
        parts.append(re.escape(pattern[cursor : match.start()]))
        parts.append(f"(?P<{name}>[^/\\\\]+?)")
        cursor = match.end()
    parts.append(re.escape(pattern[cursor:]))
    return re.compile(f"^{''.join(parts)}$")


def parse_entities(relative_path: Path, pattern: str) -> dict[str, str | None]:
    """Pull group_id / session_id / task / run out of a path using the project's pattern.
    Entities the pattern does not mention come back as None."""
    text = Path(relative_path).as_posix()
    match = _entity_regex(pattern).match(text)
    if match is None:
        raise RecordingNotFoundError(f"'{text}' does not match recording_pattern '{pattern}'")
    found = match.groupdict()
    return {name: found.get(name) for name in ENTITY_NAMES}


def make_recording_id(entities: dict[str, str | None], pattern: str) -> str:
    """Fill the entities back into the pattern and flatten it into one path segment, so the
    id and the directory the data sits in always look the same."""
    text = pattern
    for name in ENTITY_NAMES:
        placeholder = "{" + name + "}"
        if placeholder in text:
            value = entities.get(name)
            if value is None:
                raise RecordingNotFoundError(f"recording_pattern needs {name}, which is unset")
            text = text.replace(placeholder, value)
    return text.replace("/", "_")


def detect_recordings(raw_dir: Path, pattern: str, prefix: str = "") -> list[str]:
    """Walk raw_dir to the depth the pattern describes and return the ids of the directories
    that match, sorted. The prefix filters which top-level directories are considered."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise RecordingNotFoundError(f"raw_dir does not exist: {raw_dir}")

    depth = pattern.count("/") + 1
    regex = _entity_regex(pattern)

    found: list[str] = []
    for candidate in raw_dir.glob("/".join(["*"] * depth)):
        if not candidate.is_dir():
            continue
        relative = candidate.relative_to(raw_dir)
        if prefix and not relative.parts[0].startswith(prefix):
            continue
        match = regex.match(relative.as_posix())
        if match is None:
            continue
        entities = {name: match.groupdict().get(name) for name in ENTITY_NAMES}
        found.append(make_recording_id(entities, pattern))
    return sorted(found)


def parse_recordings_input(value: str) -> list[str]:
    """A comma-separated list, or a path to a file holding one id per line."""
    if not value:
        return []
    # os.path.isfile, not Path.is_file: a comma-joined list of ids can exceed NAME_MAX,
    # where Path.stat() raises ENAMETOOLONG instead of answering False
    if os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def recording_override_path(recording_id: str, config: MacawConfig) -> Path | None:
    candidate = Path(config.project.recording_config_dir) / f"{recording_id}.yaml"
    return candidate if candidate.is_file() else None


def media_kind(path: Path) -> MediaKind | None:
    suffix = path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def playable_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and not path.name.startswith(".") and media_kind(path) is not None
    )


def camera_clips(folder: Path) -> dict[str, Path]:
    """camera -> its one clip, from a folder holding one folder per camera, as a recording does."""
    loose = [path.name for path in folder.iterdir() if path.is_file() and media_kind(path) == "video"] if folder.is_dir() else []
    if loose:
        raise RecordingNotFoundError(f"{folder}: put each clip in a folder named after its camera ({', '.join(sorted(loose))})")
    clips: dict[str, Path] = {}
    for camera in sorted(path for path in folder.iterdir() if path.is_dir()) if folder.is_dir() else []:
        videos = [path for path in playable_files(camera) if media_kind(path) == "video"]
        if len(videos) > 1:
            raise RecordingNotFoundError(f"{camera}: holds {len(videos)} videos; keep the one clip this camera filmed")
        if videos:
            clips[camera.name] = videos[0]
    return clips


def scan_devices(root: Path, setup: SetupSection) -> tuple[list[MediaFile], list[str]]:
    """The file in each device folder the setup names, and notes on what was missing or left unread.
    Nothing is read from file names: the folder says which device a file is, the setup whom it records."""
    media: list[MediaFile] = []
    notes: list[str] = []
    for device in setup.devices():
        folder = root / device
        if not folder.is_dir():
            notes.append(f"no {device} folder, so this recording has no {device}")
            continue
        files = playable_files(folder)
        if not files:
            notes.append(f"the {device} folder holds no audio or video file")
            continue
        if len(files) > 1:
            listed = ", ".join(path.relative_to(folder).as_posix() for path in files)
            raise RecordingNotFoundError(
                f"{root.name}: the {device} folder holds {len(files)} files ({listed}); keep the one {device} recorded"
            )
        path = files[0]
        # a microphone's file is read for its sound only, whatever else the file holds
        kind = media_kind(path) if device in setup.cameras else "audio"
        if kind != "video" and device in setup.cameras:
            raise RecordingNotFoundError(f"{root.name}: {device} is a camera, but its folder holds {path.name}, which is not a video")
        people, hears = device_people(setup, device)
        camera = setup.cameras.get(device)
        uses = {use: tuple(getattr(camera, use)) for use in ("body", "face", "gaze")} if camera is not None else {}
        media.append(
            MediaFile(
                path=path, kind=kind, device=device, hears=tuple(hears), uses=uses,
                subject_id=people[0] if len(people) == 1 else None, depth=bool(camera is not None and camera.depth),
            )
        )
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.name in setup.devices():
            continue
        if path.is_dir() and playable_files(path):
            notes.append(f"the {path.name} folder is not a device of setup {setup.name}, so it was not read")
        elif path.is_file() and media_kind(path) is not None:
            notes.append(f"{path.name} lies outside the device folders, so it was not read")
    return media, notes


def _readings(recording_id: str, pattern: str) -> list[dict[str, str]]:
    """Every way the id reads as the pattern with its folders joined by _; a value may hold _ itself."""
    pieces = _PLACEHOLDER.split(pattern.replace("/", "_"))
    unknown = sorted(set(pieces[1::2]) - set(ENTITY_NAMES))
    if unknown:
        raise ConfigError(f"recording_pattern uses unknown entity '{unknown[0]}'. Available: {', '.join(ENTITY_NAMES)}")
    found: list[dict[str, str]] = []

    def walk(position: int, index: int, taken: dict[str, str]) -> None:
        if not recording_id.startswith(pieces[index], position):
            return
        position += len(pieces[index])
        if index == len(pieces) - 1:
            if position == len(recording_id):
                found.append(taken)
            return
        name = pieces[index + 1]
        for end in range(position + 1, len(recording_id) + 1):
            value = recording_id[position:end]
            if taken.get(name, value) == value:
                walk(end, index + 2, {**taken, name: value})

    walk(0, 0, {})
    return found


def _folder(pattern: str, entities: dict[str, str | None]) -> str:
    relative = pattern
    for name in ENTITY_NAMES:
        if entities.get(name) is not None:
            relative = relative.replace("{" + name + "}", entities[name])
    return relative


def recording_entities(recording_id: str, config: MacawConfig) -> dict[str, str | None]:
    """The entities a recording id stands for; where _ inside a value makes it read more than one way, the folder under
    raw_dir that exists says which."""
    pattern = config.project.recording_pattern
    readings = _readings(recording_id, pattern)
    if len(readings) > 1:
        readings = [found for found in readings if (Path(config.project.raw_dir) / _folder(pattern, found)).is_dir()] or readings
    if not readings:
        raise RecordingNotFoundError(f"'{recording_id}' does not look like a recording_pattern '{pattern}' id")
    if len(readings) > 1:
        folders = ", ".join(_folder(pattern, found) for found in readings)
        raise RecordingNotFoundError(f"'{recording_id}' reads as more than one recording folder ({folders}) and none is under raw_dir")
    return {name: readings[0].get(name) for name in ENTITY_NAMES}


def build_recording(recording_id: str, config: MacawConfig) -> Recording:
    """Resolve one recording on demand from directory convention plus its optional override file."""
    pattern = config.project.recording_pattern
    entities = recording_entities(recording_id, config)

    root = Path(config.project.raw_dir) / _folder(pattern, entities)
    if not root.is_dir():
        raise RecordingNotFoundError(f"{recording_id}: no directory at {root}")

    subjects = [
        Subject(subject_id=role, role=role, roi=config.motion.rois.get(role)) for role in config.project.roles
    ]
    media, notes = scan_devices(root, config.setup)

    return Recording(
        recording_id=recording_id,
        root=root,
        subjects=subjects,
        media=media,
        t_task_start=config.segment.task_start_s,
        t_task_end=config.segment.task_end_s,
        t_keep_last=config.segment.keep_last_s,
        metadata={"setup": config.setup.name, "setup_notes": notes},
        **entities,
    )
