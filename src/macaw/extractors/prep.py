"""Everything downstream depends on this: probe, trim to the task window, split audio tracks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from macaw.exceptions import UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.media.align import apply_alignment, read_offset_table
from macaw.media.ffmpeg import run_ffmpeg
from macaw.media.probe import probe
from macaw.project.layout import make_layout
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

# every prepared track is written at this rate so downstream extractors never resample
PREP_SAMPLE_RATE = 16000
PREP_INFO_NAME = "prep.json"


@dataclass
class PrepInfo:
    """What prep.media leaves behind for every other audio extractor to build on."""

    recording_id: str
    t_origin_s: float                       # recording-clock time of sample 0 in every track
    duration_s: float
    sample_rate: int
    tracks: dict[str, str] = field(default_factory=dict)      # subject_id -> file name
    shared: list[str] = field(default_factory=list)
    media: list[dict] = field(default_factory=list)
    root: Path | None = None                # where the files above actually sit

    def track_path(self, subject_id: str) -> Path:
        if subject_id not in self.tracks:
            known = ", ".join(sorted(self.tracks)) or "none"
            raise UnreadableMediaError(
                f"{self.recording_id}: no prepared track for '{subject_id}'. Have: {known}"
            )
        return Path(self.root or ".") / self.tracks[subject_id]

    def to_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "t_origin_s": self.t_origin_s,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "tracks": self.tracks,
            "shared": self.shared,
            "media": self.media,
        }


def load_prep(config, recording_id: str) -> PrepInfo:
    """Read what prep.media wrote for this recording. Every audio extractor starts here."""
    root = make_layout(config.project.output_dir, create=False).artifacts_dir(
        recording_id, MediaPrepExtractor.name
    )
    path = root / PREP_INFO_NAME
    if not path.is_file():
        raise UnreadableMediaError(
            f"{recording_id}: {path} is missing, run the '{MediaPrepExtractor.name}' extractor first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PrepInfo(root=root, **payload)


def _extract_track(
    src: Path, dest: Path, *, start_s: float, duration_s: float, sample_rate: int
) -> Path:
    """Cut one media file down to the task window on the recording clock.

    A negative start means the source begins after the window opens, so the front is padded
    with silence: sample 0 of every prepared track has to be the same instant."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = []
    filters: list[str] = []
    if start_s >= 0:
        args += ["-ss", f"{start_s:.6f}"]
    else:
        filters.append(f"adelay={int(round(-start_s * 1000))}:all=1")
    filters.append("apad")              # a source shorter than the window still fills it
    args += [
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
class MediaPrepExtractor(Extractor):
    name = "prep.media"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.PREP
    config_sections = ("segment", "align")
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
        path = make_layout(self.config.project.output_dir, create=False).offset_table_path(
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
        tracks: dict[str, str] = {}
        for subject in recording.subjects:
            track = recording.audio_for(subject.subject_id)
            if track is None:
                warnings.append(
                    QCWarning(
                        check="prep_missing_track",
                        recording_id=recording.recording_id,
                        subject_id=subject.subject_id,
                        severity="warn",
                        message=(
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

        for item in recording.media:
            if item.offset_confidence is not None and item.offset_confidence < cfg.min_confidence:
                warnings.append(
                    QCWarning(
                        check="alignment_confidence",
                        recording_id=recording.recording_id,
                        subject_id=item.subject_id,
                        severity="warn",
                        message=(
                            f"{item.path.name} aligned at confidence "
                            f"{item.offset_confidence:.3f}, below {cfg.min_confidence:.3f}"
                        ),
                        details={"offset_s": item.offset_s},
                    )
                )
            if item.drift_ppm is not None and abs(item.drift_ppm) > cfg.max_drift_ppm:
                warnings.append(
                    QCWarning(
                        check="clock_drift",
                        recording_id=recording.recording_id,
                        subject_id=item.subject_id,
                        severity="warn",
                        message=(
                            f"{item.path.name} drifts {item.drift_ppm:.1f} ppm, "
                            f"above {cfg.max_drift_ppm:.1f}"
                        ),
                    )
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


class AlignReviewGate(Extractor):
    """A human confirms low-confidence offsets. Present so the DAG knows the edge exists."""

    name = "align.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.INTERACTIVE
    config_sections = ("align",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Fails with a clear message when run non-interactively and confirmation is missing."""
        raise NotImplementedError


class IdentityConfirmGate(Extractor):
    """Three-source vote is automatic; accepting it is not."""

    name = "identity.confirm"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.INTERACTIVE
    config_sections = ("text",)
    requires = ("text.asr", "prosody.frame")

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError
