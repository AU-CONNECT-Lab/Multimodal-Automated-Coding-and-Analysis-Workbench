"""Everything downstream depends on this: probe, trim to the task window, split audio tracks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import pandas as pd

from macaw.exceptions import ConfigError, UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Gate, Phase, QCWarning
from macaw.media.align import (
    UNALIGNED,
    apply_alignment,
    describe_agreement,
    describe_drift,
    drift_flagged,
    low_confidence,
    needs_person,
    read_offset_table,
)
from macaw.media.ffmpeg import normalise_video, run_ffmpeg
from macaw.media.probe import frame_rate, is_variable_rate, probe
from macaw.project.identity import identity_basis, read_confirmation
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import MediaFile, Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# every prepared track is written at this rate so downstream extractors never resample
PREP_SAMPLE_RATE = 16000
PREP_INFO_NAME = "prep.json"
VIDEO_INFO_NAME = "video.json"
SHARED_TRACK_PREFIX = "shared_"


@dataclass
class PrepInfo:
    """What prep.media leaves behind for every other audio extractor to build on."""

    recording_id: str
    t_origin_s: float                       # recording-clock time of sample 0 in every track
    duration_s: float
    sample_rate: int
    tracks: dict[str, str] = field(default_factory=dict)      # subject_id -> file name
    shared: list[str] = field(default_factory=list)
    shared_tracks: dict[str, str] = field(default_factory=dict)  # shared audio file, or the camera standing in -> prepared file name
    media: list[dict] = field(default_factory=list)
    root: Path | None = None                # where the files above actually sit

    def track_path(self, subject_id: str) -> Path:
        if subject_id not in self.tracks:
            known = ", ".join(sorted(self.tracks)) or "none"
            raise UnreadableMediaError(
                f"{self.recording_id}: no prepared track for '{subject_id}'. Have: {known}"
            )
        return Path(self.root or ".") / self.tracks[subject_id]

    def shared_track_path(self, media_name: str) -> Path:
        return Path(self.root or ".") / self.shared_tracks[media_name]

    def to_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "t_origin_s": self.t_origin_s,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "tracks": self.tracks,
            "shared": self.shared,
            "shared_tracks": self.shared_tracks,
            "media": self.media,
        }


def load_prep(config, recording_id: str) -> PrepInfo:
    """Read what prep.media wrote for this recording. Every audio extractor starts here."""
    root = layout_for(config, create=False).artifacts_dir(
        recording_id, MediaPrepExtractor.name
    )
    path = root / PREP_INFO_NAME
    if not path.is_file():
        raise UnreadableMediaError(
            f"{recording_id}: {path} is missing, run the '{MediaPrepExtractor.name}' extractor first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PrepInfo(root=root, **payload)


def load_video_prep(config, recording_id: str) -> dict[str, dict]:
    """prep.video's record of each video by file name, with the path of its copy; empty before it has run."""
    root = layout_for(config, create=False).artifacts_dir(recording_id, VideoPrepExtractor.name)
    path = root / VIDEO_INFO_NAME
    if not path.is_file():
        return {}
    return {
        entry["name"]: {**entry, "path": root / entry["file"] if entry["file"] else None}
        for entry in json.loads(path.read_text(encoding="utf-8"))
    }


def apply_prep_timing(config, recording: Recording) -> PrepInfo:
    """Give each media file the frame rate and offset prep.media found for it, so a later
    process reads the same clock without probing again. Video starts here the way audio starts
    at the prepared tracks."""
    prep = load_prep(config, recording.recording_id)
    entries = {entry["name"]: entry for entry in prep.media}
    videos = load_video_prep(config, recording.recording_id)
    for item in recording.media:
        entry = entries.get(item.path.name)
        if entry is None:
            continue
        item.duration_s = entry["duration_s"]
        item.sample_rate = entry["sample_rate"]
        item.fps = entry["fps"]
        item.offset_s = entry["offset_s"]
        item.drift_ppm = entry["drift_ppm"]
        item.offset_confidence = entry["offset_confidence"]
        video = videos.get(item.path.name)
        if video is not None:
            item.fps = video["fps"]
            # from here on offset_s places frame 0 of the file the video steps read
            item.offset_s += video["frames_start_s"]
            item.prepared_video = video["path"]
    return prep


def _extract_track(
    src: Path, dest: Path, *, start_s: float, duration_s: float, sample_rate: int
) -> Path:
    """Cut one media file down to the task window on the recording clock.

    A negative start means the source begins after the window opens, so the front is padded
    with silence: sample 0 of every prepared track has to be the same instant."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # counted in samples from the first one decoded, the clock alignment measured on; seeking
    # with -ss lands elsewhere when a file's sound and pictures start at different times
    filters = [f"aresample={sample_rate}"]
    if start_s >= 0:
        filters += [f"atrim=start_sample={int(round(start_s * sample_rate))}", "asetpts=PTS-STARTPTS"]
    else:
        filters.append(f"adelay={int(round(-start_s * 1000))}:all=1")
    filters.append("apad")              # a source shorter than the window still fills it
    args = [
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-af", ",".join(filters),
        "-t", f"{duration_s:.6f}",
        "-c:a", "pcm_s16le",
        str(dest),
    ]
    run_ffmpeg(args)
    return dest


def _window(recording: Recording) -> tuple[float, float]:
    """The task window on the recording clock, defaulting to everything that was recorded."""
    start = recording.t_task_start if recording.t_task_start is not None else 0.0
    if recording.t_task_end is not None:
        end = recording.t_task_end
    else:
        ends = [
            (item.duration_s or 0.0) + item.offset_s
            for item in recording.media
            if item.duration_s is not None
        ]
        if not ends:
            raise UnreadableMediaError(f"{recording.recording_id}: no media duration to work from")
        end = max(ends)
    if end <= start:
        raise UnreadableMediaError(
            f"{recording.recording_id}: task window is empty ({start:.3f}s to {end:.3f}s)"
        )
    return start, end


_SPECS = {
    "media_duration_s": ("Length of the prepared task window.", "s"),
    "n_media": ("Playable files found for this recording.", "count"),
    "n_audio_tracks": ("Per-subject audio tracks prepared.", "count"),
    "track_offset_s": ("Seconds added to this track's clock to reach recording time.", "s"),
    "track_offset_confidence": ("Alignment peak sharpness, 0-1.", None),
    "track_drift_ppm": ("Clock drift of this track against the reference.", "ppm"),
}


@register_extractor
class AlignReviewGate(Gate):
    """Offsets a person has to look at before anything is cut by them: flagged, missing, or a file without audio."""

    name = "align.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    config_sections = ("align",)

    def inputs_for(self, recording: Recording) -> list[Path]:
        path = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
        return [path] if path.is_file() else []

    def pending(self, recording: Recording) -> str | None:
        cfg = self.config.align
        if cfg.review == "skip" or cfg.method == "none" or len(recording.media) < 2:
            return None
        rid = recording.recording_id
        layout = layout_for(self.config, create=False)
        in_sync = "or set align.method: none if the files were recorded in sync"
        table = self.inputs_for(recording)
        if not table:
            if cfg.method == "manual":
                return f"no offsets yet: set each file's with `macaw set-offset`, {in_sync}"
            return f"no offset table yet: run `macaw align -r {rid}`, {in_sync}"

        by_name = {result.target: result for result in read_offset_table(table[0])}
        missing = [item.path.name for item in recording.media if item.path.name not in by_name]
        if missing:
            return f"{', '.join(missing)} came after aligning: run `macaw align -r {rid}` again"
        flagged = [
            f"{item.path.name} ({reason})"
            for item in recording.media
            if (reason := needs_person(by_name[item.path.name], cfg)) is not None
        ]
        if not flagged:
            return None
        return (
            f"{len(flagged)} file(s) to look at: {'; '.join(flagged)}. The figures are in "
            f"{layout.renders_dir(rid) / 'align'}; keep them with `macaw accept-offset -r {rid}` "
            "or change one with `macaw set-offset`"
        )


@register_extractor
class IdentityConfirmGate(Gate):
    """Which file belongs to whom, confirmed once per project; everything per person rests on it."""

    name = "identity.confirm"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    config_sections = ("identity",)

    def inputs_for(self, recording: Recording) -> list[Path]:
        path = layout_for(self.config, create=False).identity_path
        return [path] if path.is_file() else []

    def pending(self, recording: Recording) -> str | None:
        if self.config.identity.confirm == "skip":
            return None
        confirmation = read_confirmation(layout_for(self.config, create=False).identity_path)
        if confirmation is None:
            return "nobody has confirmed which file belongs to whom: run `macaw identify` once for the project"
        if confirmation.get("basis") != identity_basis(self.config):
            return (
                "the confirmation was for another media_pattern or list of people: run `macaw identify` again"
            )
        return None

    def notes(self, recording: Recording) -> list[QCWarning]:
        if self.config.identity.confirm != "skip":
            return []
        return [
            QCWarning(
                check="identity_unconfirmed",
                recording_id=recording.recording_id,
                subject_id=None,
                severity="warn",
                message="identity.confirm is skip: nobody confirmed which file belongs to whom",
            )
        ]


@register_extractor
class MediaPrepExtractor(Extractor):
    name = "prep.media"
    version = "0.4.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.PREP
    config_sections = ("segment", "align")
    requires = (AlignReviewGate.name,)
    resource_profile = "data_manage"

    def check_available(self) -> None:
        from macaw.media.probe import require_ffmpeg, require_ffprobe

        require_ffmpeg()
        require_ffprobe()

    def inputs_for(self, recording: Recording) -> list[Path]:
        paths = [item.path for item in recording.media]
        offsets = self._offset_table(recording)
        if offsets is not None:
            paths.append(offsets)
        return paths

    def _offset_table(self, recording: Recording) -> Path | None:
        path = layout_for(self.config, create=False).offset_table_path(
            recording.recording_id
        )
        return path if path.is_file() else None

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Reads the offset table produced by `macaw align`; it does not estimate offsets."""
        warnings: list[QCWarning] = []
        self._probe_media(recording)
        self._apply_offsets(recording, warnings)

        start, end = _window(recording)
        duration = end - start

        artifacts: dict[str, Path] = {}
        shared_audio = sorted(
            (item for item in recording.media if item.kind == "audio" and item.subject_id is None),
            key=lambda item: item.path.name,
        )
        camera = None if shared_audio else _camera_sound(recording)
        if camera is not None:
            shared_audio = [camera]
            warnings.append(
                QCWarning(
                    check="prep_camera_sound",
                    recording_id=recording.recording_id,
                    subject_id=None,
                    severity="info",
                    message=f"No shared audio file; the sound of {camera.path.name} is the shared microphone.",
                )
            )
        shared_tracks: dict[str, str] = {}
        for item in shared_audio:
            name = f"{SHARED_TRACK_PREFIX}{item.path.stem}.wav"
            _extract_track(
                item.path, workspace / name, start_s=start - item.offset_s, duration_s=duration,
                sample_rate=PREP_SAMPLE_RATE,
            )
            artifacts[name] = workspace / name
            shared_tracks[item.path.name] = name

        tracks: dict[str, str] = {}
        for subject in recording.subjects:
            track = recording.audio_for(subject.subject_id)
            if track is None:
                # with a shared microphone their voice is in the room track, found by diarization
                warnings.append(
                    QCWarning(
                        check="prep_missing_track",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="info" if shared_tracks else "warn",
                        message=(
                            f"No audio file is attributed to '{subject.subject_id}'; their voice can come "
                            f"only from the shared track(s) {', '.join(shared_tracks)}."
                            if shared_tracks else
                            f"No audio file is attributed to '{subject.subject_id}'. "
                            "Per-speaker audio features cannot be computed for them."
                        ),
                    )
                )
                continue
            name = f"{subject.subject_id}.wav"
            _extract_track(
                track.path,
                workspace / name,
                # recording time start maps to start - offset_s on this file's own clock
                start_s=start - track.offset_s,
                duration_s=duration,
                sample_rate=PREP_SAMPLE_RATE,
            )
            artifacts[name] = workspace / name
            tracks[subject.subject_id] = name

        shared = [item.path.name for item in recording.media if item.subject_id is None]
        info = PrepInfo(
            recording_id=recording.recording_id,
            t_origin_s=start,
            duration_s=duration,
            sample_rate=PREP_SAMPLE_RATE,
            tracks=tracks,
            shared=shared,
            shared_tracks=shared_tracks,
            media=[_media_entry(item) for item in recording.media],
        )
        info_path = workspace / PREP_INFO_NAME
        info_path.write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
        artifacts[PREP_INFO_NAME] = info_path

        return ExtractorResult(
            rows=self._rows(recording, start, end, len(tracks)),
            specs=self._specs(),
            artifacts=artifacts,
            warnings=warnings,
        )

    def _probe_media(self, recording: Recording) -> None:
        for item in recording.media:
            info = probe(item.path)
            item.duration_s = info.duration_s
            audio = info.audio_streams()
            video = info.video_streams()
            if audio:
                item.sample_rate = audio[0].sample_rate
            if video:
                item.fps = video[0].fps

    def _apply_offsets(self, recording: Recording, warnings: list[QCWarning]) -> None:
        cfg = self.config.align
        table = self._offset_table(recording)
        if table is None:
            if cfg.method != "none" and len(recording.media) > 1:
                warnings.append(
                    QCWarning(
                        check="alignment_missing",
                        recording_id=recording.recording_id,
                        subject_id=None,
                        severity="warn",
                        message=(
                            f"{len(recording.media)} media files but no offset table. "
                            "Run `macaw align` first, or every track is assumed to start together."
                        ),
                    )
                )
            return

        results = read_offset_table(table)
        apply_alignment(recording, results, recut=False)

        def warn(check: str, item, message: str, **details) -> None:
            warnings.append(
                QCWarning(
                    check=check,
                    recording_id=recording.recording_id,
                    subject_id=item.subject_id,
                    severity="warn",
                    message=message,
                    details=details,
                )
            )

        by_name = {result.target: result for result in results}
        for item in recording.media:
            result = by_name.get(item.path.name)
            if result is None:
                warn("alignment_missing", item, f"{item.path.name} is not in the offset table; run `macaw align` again")
                continue
            if result.method == UNALIGNED:
                warn(
                    "alignment_missing", item,
                    f"{item.path.name} was not aligned ({result.note}) and is taken to start with "
                    f"{result.reference}; check it and set it with `macaw set-offset`",
                    metadata_offset_s=result.metadata_offset_s,
                )
            elif low_confidence(result, cfg):
                warn(
                    "alignment_confidence", item,
                    f"{item.path.name} aligned at confidence {result.confidence:.3f}, below {cfg.min_confidence:.3f}, "
                    f"and {describe_agreement(result)}",
                    offset_s=result.offset_s, agreement=result.agreement,
                )
            if drift_flagged(result):
                warn(
                    "clock_drift", item,
                    f"{item.path.name} has {describe_drift(result)}; drift is not corrected, so the offset is "
                    "off by that much at the far end",
                )

    def _rows(self, recording: Recording, start: float, end: float, n_tracks: int):
        entities = recording.entities()
        common = dict(
            recording_id=recording.recording_id,
            source=self.name,
            scale=Scale.RECORDING,
            entities=entities,
        )

        frames = [
            build_long_rows(
                values=pd.DataFrame(
                    {
                        "t_start": [start],
                        "t_end": [end],
                        "media_duration_s": [end - start],
                        "n_media": [float(len(recording.media))],
                        "n_audio_tracks": [float(n_tracks)],
                    }
                ),
                **common,
            )
        ]
        for subject in recording.subjects:
            track = recording.audio_for(subject.subject_id)
            if track is None:
                continue
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {
                            "t_start": [start],
                            "t_end": [end],
                            "track_offset_s": [track.offset_s],
                            "track_offset_confidence": [track.offset_confidence],
                            "track_drift_ppm": [track.drift_ppm],
                        }
                    ),
                    subject_id=subject.subject_id,
                    **common,
                )
            )
        return concat_long_tables(frames)

    def _specs(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(
                name=name,
                description=description,
                unit=unit,
                scale=Scale.RECORDING,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                validation="none",
            )
            for name, (description, unit) in _SPECS.items()
        ]


_VIDEO_SPECS = {
    "n_videos": ("Video files prepared.", "count"),
    "n_variable_rate_videos": ("Videos recorded at a variable frame rate.", "count"),
    "video_frames_repeated": ("Frames repeated to give the videos a constant rate and start them with their sound.", "count"),
    "video_frames_dropped": ("Frames left out for the same reasons, or to reach video.fps.", "count"),
}
_INTERLACED = {"tt", "bb", "tb", "bt"}


@register_extractor
class VideoPrepExtractor(Extractor):
    """Each video re-encoded once, so every video step reads the same frames on the same clock."""

    name = "prep.video"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.PREP
    config_sections = ("video",)
    resource_profile = "light_short"
    cached = False                      # whole videos; only a changed setting remakes them

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return False                    # the steps that read video pull it in

    def check_available(self) -> None:
        from macaw.media.probe import require_ffmpeg, require_ffprobe

        require_ffmpeg()
        require_ffprobe()

    def inputs_for(self, recording: Recording) -> list[Path]:
        return sorted(item.path for item in recording.media if item.kind == "video")

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        entries, warnings = [], []
        for item in sorted((item for item in recording.media if item.kind == "video"), key=lambda item: item.path.name):
            entry = self._prepare(item, workspace)
            entries.append(entry)
            warnings += self._notes(recording, item, entry)
        info_path = workspace / VIDEO_INFO_NAME
        info_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        artifacts = {VIDEO_INFO_NAME: info_path, **{entry["file"]: workspace / entry["file"] for entry in entries if entry["file"]}}
        return ExtractorResult(rows=self._rows(recording, entries), specs=self._specs(), artifacts=artifacts, warnings=warnings)

    def _prepare(self, item, workspace: Path) -> dict:
        cfg = self.config.video
        info = probe(item.path)
        if not info.video_streams():
            raise UnreadableMediaError(f"{item.path}: no video stream")
        stream = info.video_streams()[0]
        rate = frame_rate(stream)
        if rate is None:
            raise UnreadableMediaError(f"{item.path}: no frame rate to count frames by")
        # a file's clock is its sound's, the one alignment measures on; a silent file's is its pictures'
        pictures = stream.start_s if stream.start_s is not None else info.start_s
        sound = info.audio_streams()[0].start_s if info.audio_streams() else None
        clock = sound if sound is not None else pictures
        target = rate if cfg.fps is None else self._target(item, rate)
        entry = {
            "name": item.path.name,
            "file": None,
            "fps": float(target) if cfg.normalise else stream.fps,
            "frames_start_s": 0.0 if cfg.normalise else pictures - clock,
            "first_picture_s": pictures - clock,
            "source_fps": float(rate),
            "average_fps": float(stream.average_fps) if stream.average_fps else None,
            "declared_fps": float(stream.declared_fps) if stream.declared_fps else None,
            "variable_rate": is_variable_rate(stream),
            "rotation": stream.rotation,
            "pixel_aspect": str(stream.pixel_aspect) if stream.pixel_aspect else None,
            "field_order": stream.field_order,
            "codec": stream.codec_name,
            "size": [stream.width, stream.height],
            "prepared_size": None,
            "frames_repeated": None,
            "frames_dropped": None,
        }
        if cfg.normalise:
            name = f"{item.path.name}.mp4"
            counts = normalise_video(
                item.path, workspace / name, fps=target, start_s=clock - info.start_s, quality=cfg.quality, encoder=cfg.encoder,
            )
            prepared = probe(workspace / name).video_streams()[0]
            entry.update(file=name, prepared_size=[prepared.width, prepared.height])
            if counts is not None:
                entry.update(frames_repeated=counts[0], frames_dropped=counts[1])
        return entry

    def _target(self, item, rate: Fraction) -> Fraction:
        wanted = Fraction(self.config.video.fps).limit_denominator(1001)
        step = rate / wanted
        every = round(step)
        if every < 1 or abs(step - every) > Fraction(1, 1000) * every:
            choices = ", ".join(f"{float(rate / k):g}" for k in range(1, 5))
            raise ConfigError(
                f"{item.path.name} runs at {float(rate):g} fps, which video.fps {float(wanted):g} does not divide by a whole "
                f"number, so frames would be kept unevenly; rates this file can take: {choices}"
            )
        return rate / every

    def _notes(self, recording: Recording, item, entry: dict) -> list[QCWarning]:
        fixed = entry["file"] is not None
        notes: list[QCWarning] = []

        def note(check: str, message: str, severity: str | None = None) -> None:
            notes.append(
                QCWarning(
                    check=check, recording_id=recording.recording_id, subject_id=item.subject_id,
                    severity=severity or ("info" if fixed else "warn"), message=f"{item.path.name}: {message}",
                )
            )

        if entry["variable_rate"]:
            rates = f"variable frame rate (on average {entry['average_fps']:.3g} fps, stated {entry['declared_fps']:g})"
            note("video_variable_rate", f"{rates}; made constant at {entry['fps']:g} fps" if fixed else (
                f"{rates}; frame times are counted at the average rate and drift from the sound where the rate changed. "
                "video.normalise makes it constant"
            ))
        if entry["rotation"] % 360:
            turn = abs(entry["rotation"])
            note("video_rotated", f"turned {turn:g} degrees upright, as the file asks players to" if fixed else (
                f"asks players to turn it {turn:g} degrees, which some tools do and some do not, so regions and positions "
                "may not match the picture. video.normalise turns it upright"
            ))
        if entry["field_order"] in _INTERLACED:
            note("video_interlaced", f"interlaced ({entry['field_order']}), deinterlaced" if fixed else (
                f"interlaced ({entry['field_order']}); the comb between its two fields disturbs face and body detection "
                "and frame differences. video.normalise deinterlaces it"
            ))
        if entry["pixel_aspect"] not in (None, "1"):
            shape = f"pixels {entry['pixel_aspect']} as wide as they are tall"
            note("video_pixel_shape", (
                f"{shape}, stretched to square ({'x'.join(map(str, entry['size']))} to {'x'.join(map(str, entry['prepared_size']))})"
            ) if fixed else f"{shape}: distances across and down the picture are on different scales. video.normalise makes them square")
        lead = entry["first_picture_s"]
        if abs(lead) >= 0.5 / entry["source_fps"]:
            if not fixed:
                note("video_start", f"its pictures start {lead:+.3f} s from its sound; frame times allow for it", "info")
            elif lead > 0:
                note("video_start", f"its pictures start {lead:.3f} s after its sound; the first is repeated so frame 0 is the sound's first instant")
            else:
                note("video_start", f"its pictures start {-lead:.3f} s before its sound; those frames were left out so frame 0 is the sound's first instant")
        if fixed and not math.isclose(entry["fps"], entry["source_fps"]):
            note("video_rate_changed", f"one frame in {round(entry['source_fps'] / entry['fps'])} kept, {entry['source_fps']:g} to {entry['fps']:g} fps (video.fps)")
        return notes

    def _rows(self, recording: Recording, entries: list[dict]) -> pd.DataFrame:
        def total(key: str) -> float:
            values = [entry[key] for entry in entries]
            return float(sum(values)) if values and None not in values else float("nan")

        values = pd.DataFrame(
            {
                "t_start": [0.0],
                "n_videos": [float(len(entries))],
                "n_variable_rate_videos": [float(sum(entry["variable_rate"] for entry in entries))],
                "video_frames_repeated": [total("frames_repeated")],
                "video_frames_dropped": [total("frames_dropped")],
            }
        )
        return build_long_rows(
            values=values, recording_id=recording.recording_id, source=self.name, scale=Scale.RECORDING,
            entities=recording.entities(),
        )

    def _specs(self) -> list[FeatureSpec]:
        cfg = self.config.video
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                extractor=self.name, extractor_version=self.version, validation="none",
                params={"normalise": cfg.normalise, "fps": cfg.fps, "quality": cfg.quality, "encoder": cfg.encoder},
            )
            for name, (description, unit) in _VIDEO_SPECS.items()
        ]


def _camera_sound(recording: Recording) -> MediaFile | None:
    """The shared camera whose sound stands in for a room microphone when someone has no mic of their own."""
    if all(recording.audio_for(subject.subject_id) is not None for subject in recording.subjects):
        return None
    sounding = [
        item for item in recording.media
        if item.kind == "video" and item.subject_id is None and item.sample_rate is not None
    ]
    return recording.reference_media(among=sounding) if sounding else None


def _media_entry(item) -> dict:
    return {
        "path": str(item.path),
        "name": item.path.name,
        "kind": item.kind,
        "subject_id": item.subject_id,
        "duration_s": item.duration_s,
        "sample_rate": item.sample_rate,
        "fps": item.fps,
        "offset_s": item.offset_s,
        "drift_ppm": item.drift_ppm,
        "offset_confidence": item.offset_confidence,
    }
