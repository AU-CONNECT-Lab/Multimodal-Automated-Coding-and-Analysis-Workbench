"""Config schema. extra="forbid" everywhere; a typo'd key must fail loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectSection(_Section):
    name: str
    raw_dir: Path
    output_dir: Path
    # bookkeeping that can be deleted without losing a result: logs, job records, cache.
    # null means it lives under output_dir; on a cluster point it at scratch
    work_dir: Path | None = None
    scratch_dir: Path | None = None
    # how a path under raw_dir maps to BIDS entities; "{group_id}" is the flat case
    recording_pattern: str = "{group_id}"
    # which media file belongs to which person; a capture that is not a known subject id
    # means the file is shared, e.g. a room camera
    media_pattern: str = "*_{subject_id}.*"
    recording_prefix: str = ""
    recording_config_dir: Path = Path("recordings")
    excluded_recordings: list[str] = Field(default_factory=list)
    n_subjects: int = 2
    roles: list[str] = Field(default_factory=lambda: ["parent", "child"])

    @model_validator(mode="after")
    def _one_role_per_person(self) -> "ProjectSection":
        if len(self.roles) != self.n_subjects:
            raise ValueError(f"roles has {len(self.roles)} entries but n_subjects is {self.n_subjects}; give one role per person")
        return self


class AlignSection(_Section):
    method: Literal["gcc_phat", "manual", "none"] = "gcc_phat"
    search_window_s: float = 600.0
    min_confidence: float = 0.6
    min_agreement: float = 0.5
    drift_check: bool = True
    drift_segment_s: float = 60.0
    max_drift_ppm: float = 200.0
    apply: Literal["offset_table", "recut"] = "offset_table"
    review: Literal["always", "flagged", "skip"] = "flagged"


class IdentitySection(_Section):
    confirm: Literal["always", "skip"] = "always"


class VideoSection(_Section):
    normalise: bool = True
    fps: float | None = Field(default=None, gt=0)
    quality: int = Field(default=17, ge=0, le=51)
    encoder: Literal["libx264", "h264_nvenc"] = "libx264"

    @model_validator(mode="after")
    def _fps_needs_normalise(self) -> "VideoSection":
        if self.fps is not None and not self.normalise:
            raise ValueError("video.fps changes the frame rate while re-encoding, so it needs video.normalise: true")
        return self


class SegmentSection(_Section):
    task_start_s: float | None = None
    task_end_s: float | None = None


class ProsodySection(_Section):
    backend: str = "parselmouth"
    fallback_chain: list[str] = Field(default_factory=lambda: ["parselmouth", "librosa", "builtin"])
    frame_step_s: float = 0.01
    f0_min_hz: float = 75.0
    f0_max_hz: float = 500.0
    f0_range_by_role: dict[str, tuple[float, float]] = Field(default_factory=dict)
    standard_sets: list[Literal["eGeMAPSv02", "ComParE_2016"]] = Field(
        default_factory=lambda: ["eGeMAPSv02"]
    )
    min_voiced_fraction: float = 0.1


class TextSection(_Section):
    asr_backend: str = "whisperx"
    asr_model: str = "large-v3"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "float16"
    diarization: str = "per_track"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    extra_voices: int = Field(default=0, ge=0)
    voice_review: Literal["always", "flagged", "skip"] = "always"
    crosstalk_margin_db: float = 6.0
    word_timestamps: bool = True
    embedding_model: str | None = None
    syntax_model: str = "en_core_web_sm"
    first_use_min_speakers: int = Field(default=2, ge=2)
    first_use_exclude: Literal["none", "english_stopwords"] | Path = "english_stopwords"
    first_use_words: Path | None = None


class PoseSection(_Section):
    backend: str | None = None
    model: str | None = None
    device: str = "auto"
    min_confidence: float = 0.3
    input_size: int = 1280
    review: Literal["always", "flagged", "skip"] = "flagged"
    track_repair: bool = True
    track_repair_max_gap_s: float = 1.0
    identity_anchors: dict[str, float] = Field(default_factory=dict)
    read_shared: bool = False


class KinematicsSection(_Section):
    smoothing_s: float = 0.3


class FaceSection(_Section):
    backend: str = "pyfeat"
    model: str | None = None
    device: str = "auto"
    batch_size: int = 16
    action_units: bool = True
    landmarks: bool = True
    blink_min_eye_px: float = 5.0
    gaze: bool = False
    mutual_gaze_angle_deg: float = 15.0
    check_model: str | None = None
    check_backend: str | None = None


class MotionSection(_Section):
    rois: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict)
    threshold: int = 10
    smoothing_s: float = 0.0


class TurnsSection(_Section):
    max_gap_s: float = 2.0
    interruption_overlap_s: float = 0.5
    smooth_transfer_gap_s: float = 0.2
    silence_min_s: float = 1.0
    window_s: float | None = None
    window_hop_s: float | None = None
    permutation_entropy_order: int = 3


class LLMSection(_Section):
    provider: str = "ollama"
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout_s: int = 300
    n_samples: int = Field(default=5, ge=1)
    seed: int = 0
    allow_abstain: bool = True
    review_threshold: float = 0.7
    batch_size: int = Field(default=20, ge=1)
    context_units: int = Field(default=5, ge=0)
    # lines that bring a local server up inside a cluster job, plus how long to wait for it
    launch: list[str] = Field(default_factory=list)
    ready_timeout_s: int = 300

    def is_local(self) -> bool:
        from macaw.coding.llm.backends import BACKEND_GROUP
        from macaw.extractors import get_backend

        return get_backend(BACKEND_GROUP, self.provider).is_local


class CodingSection(_Section):
    layers: list[Literal["A", "B", "C"]] = Field(default_factory=lambda: ["A"])
    codebook: Path | None = None
    backchannel_lexicon: list[str] = Field(default_factory=list)
    backchannel_max_s: float = 1.0
    min_turn_internal_pause_s: float = 0.2
    llm: LLMSection = Field(default_factory=LLMSection)


class QCSection(_Section):
    checks: list[str] = Field(
        default_factory=lambda: [
            "transcript_match",
            "mouth_match",
            "track_quality",
            "octave_error",
            "motion_energy",
            "au_agreement",
        ]
    )
    fail_fast: bool = False
    report_downsample_px: int = 2000
    review_max_candidates: int = 40
    review_baseline: int = 25
    review_seed: int = 0


class ExecutionSection(_Section):
    backend: Literal["local", "slurm"] = "local"
    n_workers: int = 4
    cache: bool = True
    cache_dir: Path | None = None
    check_outputs: bool = True
    output_rules: Path | None = None
    job_name_prefix: str = "macaw"


class PipelineSection(_Section):
    # extractor or group names; empty means every installed extractor
    run: list[str] = Field(default_factory=list)
    # the user's own grouping; the package imposes no division into modalities
    groups: dict[str, list[str]] = Field(default_factory=dict)


class GUISection(_Section):
    mode: Literal["window", "browser", "server"] = "window"
    host: str = "127.0.0.1"
    port: int = 8050


class MacawConfig(_Section):
    project: ProjectSection
    align: AlignSection = Field(default_factory=AlignSection)
    identity: IdentitySection = Field(default_factory=IdentitySection)
    segment: SegmentSection = Field(default_factory=SegmentSection)
    video: VideoSection = Field(default_factory=VideoSection)
    prosody: ProsodySection = Field(default_factory=ProsodySection)
    text: TextSection = Field(default_factory=TextSection)
    pose: PoseSection = Field(default_factory=PoseSection)
    kinematics: KinematicsSection = Field(default_factory=KinematicsSection)
    face: FaceSection = Field(default_factory=FaceSection)
    motion: MotionSection = Field(default_factory=MotionSection)
    turns: TurnsSection = Field(default_factory=TurnsSection)
    coding: CodingSection = Field(default_factory=CodingSection)
    qc: QCSection = Field(default_factory=QCSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    gui: GUISection = Field(default_factory=GUISection)
    pipeline: PipelineSection = Field(default_factory=PipelineSection)
    # parameters for extractors installed from other packages, keyed by extractor name
    plugins: dict[str, dict[str, Any]] = Field(default_factory=dict)
    primary: list[str] = Field(default_factory=list)


# Which sections a recordings/<id>.yaml file may touch. Everything else is a hard error:
# a recording must never be able to redirect output_dir or change the scheduler.
RECORDING_OVERRIDABLE: tuple[str, ...] = (
    "segment",
    "prosody",
    "text",
    "pose",
    "kinematics",
    "face",
    "motion",
    "turns",
    "coding",
    "plugins",
)

# Which config keys hold a path whose *contents* belong in the hash. Hashing the path
# instead would let a changed codebook reuse a stale cache entry.
# Keys that take a name or a file or folder; a relative one found next to project.yaml is read from there.
NAME_OR_PATH_FIELDS: tuple[str, ...] = (
    "text.asr_model",
    "text.diarization_model",
    "text.embedding_model",
    "text.syntax_model",
    "pose.model",
    "face.model",
)

CONTENT_HASHED_PATHS: tuple[str, ...] = (
    "coding.codebook",
    "execution.output_rules",
    *NAME_OR_PATH_FIELDS,
)
