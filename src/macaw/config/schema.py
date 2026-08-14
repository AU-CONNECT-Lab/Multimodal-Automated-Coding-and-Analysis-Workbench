"""Config schema. extra="forbid" everywhere; a typo'd key must fail loudly."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectSection(_Section):
    name: str
    raw_dir: Path
    output_dir: Path
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


class AlignSection(_Section):
    method: Literal["gcc_phat", "manual", "none"] = "gcc_phat"
    search_window_s: float = 30.0
    min_confidence: float = 0.6
    drift_check: bool = True
    drift_segment_s: float = 60.0
    max_drift_ppm: float = 200.0
    apply: Literal["offset_table", "recut"] = "offset_table"


class SegmentSection(_Section):
    task_start_s: float | None = None
    task_end_s: float | None = None
    researcher_exit_marker: str | None = None


class ProsodySection(_Section):
    backend: Literal["parselmouth", "librosa", "builtin"] = "parselmouth"
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
    asr_backend: Literal["whisperx", "faster_whisper"] = "whisperx"
    asr_model: str = "large-v3"
    language: str | None = None
    device: str = "auto"
    compute_type: str = "float16"
    diarization: Literal["pyannote", "per_track", "none"] = "per_track"
    crosstalk_margin_db: float = 6.0
    word_timestamps: bool = True
    embedding_model: str | None = None
    kinship_anchors: dict[str, list[str]] = Field(default_factory=dict)


class PoseSection(_Section):
    backend: Literal["rtmpose", "mediapipe", "yolo", "dwpose"] = "rtmpose"
    model: str | None = None
    device: str = "auto"
    min_confidence: float = 0.3
    track_repair: bool = True
    track_repair_max_gap_s: float = 1.0
    identity_anchors: dict[str, float] = Field(default_factory=dict)


class FaceSection(_Section):
    backend: Literal["pyfeat", "openface", "mediapipe"] = "pyfeat"
    action_units: bool = True
    landmarks: bool = True
    gaze: bool = False
    mutual_gaze_angle_deg: float = 15.0
    cross_backend_check: str | None = None


class MotionSection(_Section):
    enabled: bool = True
    rois: dict[str, tuple[int, int, int, int]] = Field(default_factory=dict)
    threshold: int = 10
    smoothing_s: float = 0.0


class TurnsSection(_Section):
    max_gap_s: float = 2.0
    lookback: int = 3
    interruption_overlap_s: float = 0.5
    smooth_transfer_gap_s: float = 0.2
    silence_min_s: float = 1.0
    window_s: float | None = None
    window_hop_s: float | None = None
    permutation_entropy_order: int = 3


class LLMSection(_Section):
    provider: Literal["ollama", "vllm", "llamacpp", "anthropic", "openai"] = "ollama"
    model: str = "llama3.1:8b"
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    allow_abstain: bool = True
    review_threshold: float = 0.7
    batch_size: int = 20
    # lines that bring a local server up inside a cluster job, plus how long to wait for it
    launch: list[str] = Field(default_factory=list)
    ready_timeout_s: int = 300

    def is_local(self) -> bool:
        raise NotImplementedError


class CodingSection(_Section):
    layers: list[Literal["A", "B", "C"]] = Field(default_factory=lambda: ["A"])
    codebook: Path | None = None
    backchannel_lexicon: list[str] = Field(default_factory=list)
    backchannel_max_s: float = 1.0
    llm: LLMSection = Field(default_factory=LLMSection)


class QCSection(_Section):
    checks: list[str] = Field(
        default_factory=lambda: [
            "alignment_confidence",
            "clock_drift",
            "transcript_match",
            "track_quality",
            "octave_error",
        ]
    )
    fail_fast: bool = False
    report_downsample_px: int = 2000


class ExecutionSection(_Section):
    backend: Literal["local", "slurm"] = "local"
    n_workers: int = 4
    cache: bool = True
    cache_dir: Path | None = None
    check_outputs: bool = True
    output_rules: Path | None = None
    job_name_prefix: str = "macaw"


class GUISection(_Section):
    mode: Literal["window", "browser", "server"] = "window"
    host: str = "127.0.0.1"
    port: int = 8050


class MacawConfig(_Section):
    project: ProjectSection
    align: AlignSection = Field(default_factory=AlignSection)
    segment: SegmentSection = Field(default_factory=SegmentSection)
    prosody: ProsodySection = Field(default_factory=ProsodySection)
    text: TextSection = Field(default_factory=TextSection)
    pose: PoseSection = Field(default_factory=PoseSection)
    face: FaceSection = Field(default_factory=FaceSection)
    motion: MotionSection = Field(default_factory=MotionSection)
    turns: TurnsSection = Field(default_factory=TurnsSection)
    coding: CodingSection = Field(default_factory=CodingSection)
    qc: QCSection = Field(default_factory=QCSection)
    execution: ExecutionSection = Field(default_factory=ExecutionSection)
    gui: GUISection = Field(default_factory=GUISection)
    primary: list[str] = Field(default_factory=list)


# Which sections a recordings/<id>.yaml file may touch. Everything else is a hard error:
# a recording must never be able to redirect output_dir or change the scheduler.
RECORDING_OVERRIDABLE: tuple[str, ...] = (
    "segment",
    "prosody",
    "text",
    "pose",
    "face",
    "motion",
    "turns",
    "coding",
)

# Which config keys hold a path whose *contents* belong in the hash. Hashing the path
# instead would let a changed codebook reuse a stale cache entry.
CONTENT_HASHED_PATHS: tuple[str, ...] = (
    "coding.codebook",
    "pose.model",
    "execution.output_rules",
)
