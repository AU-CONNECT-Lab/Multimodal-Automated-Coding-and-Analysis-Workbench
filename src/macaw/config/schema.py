"""Config schema. extra="forbid" everywhere; a typo'd key must fail loudly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    recording_prefix: str = ""
    recording_config_dir: Path = Path("recordings")
    # one file per setup, saying what each device folder inside a recording folder holds
    setups_dir: Path = Path("setups")
    scripts_dir: Path = Path("scripts")
    # clips and files filmed or made before recording: an object of known length, a board per camera setup
    calibration_dir: Path = Path("calibration")
    # named parts of recordings that the summaries are worked out again over; no file, no segments
    segments: Path = Path("segments.csv")
    excluded_recordings: list[str] = Field(default_factory=list)
    n_subjects: int = 2
    roles: list[str] = Field(default_factory=lambda: ["parent", "child"])

    @model_validator(mode="after")
    def _one_role_per_person(self) -> "ProjectSection":
        if len(self.roles) != self.n_subjects:
            raise ValueError(f"roles has {len(self.roles)} entries but n_subjects is {self.n_subjects}; give one role per person")
        # the names a voice on a shared microphone gets when it is no participant's
        taken = sorted({"none", "other"} & set(self.roles))
        if taken:
            raise ValueError(f"{', '.join(taken)} cannot be a role: it names a voice that is not one of the people")
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


class VoiceHintSection(_Section):
    # a pattern searched in each line a voice on a shared microphone said, ignoring case
    says: str
    role: str

    @field_validator("says")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(f"{value!r} is not a pattern Python can read: {error}") from None
        return value


class IdentitySection(_Section):
    voice_hints: list[VoiceHintSection] = Field(default_factory=list)
    # roles from the lowest voice to the highest
    pitch_order: list[str] = Field(default_factory=list)


DEVICE_NAME = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"


class CameraDevice(_Section):
    # the people this camera is read for, per kind of measure
    body: list[str] = Field(default_factory=list)
    face: list[str] = Field(default_factory=list)
    gaze: list[str] = Field(default_factory=list)
    # whose voices the camera's own sound carries; empty leaves its sound to alignment only
    hears: list[str] = Field(default_factory=list)

    def people(self) -> list[str]:
        return sorted(set(self.body) | set(self.face) | set(self.gaze))


class MicrophoneDevice(_Section):
    hears: list[str] = Field(min_length=1)


class SetupSection(_Section):
    """What each device folder in a recording holds; read from setups/<name>.yaml."""

    name: str = Field(default="main", pattern=DEVICE_NAME)
    cameras: dict[str, CameraDevice] = Field(default_factory=dict)
    microphones: dict[str, MicrophoneDevice] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _devices(self) -> "SetupSection":
        names = list(self.cameras) + list(self.microphones)
        bad = sorted(name for name in names if not re.fullmatch(DEVICE_NAME, name))
        if bad:
            raise ValueError(f"{', '.join(bad)}: a device's name is its folder's name, letters, digits, - and _ only")
        both = sorted(set(self.cameras) & set(self.microphones))
        if both:
            raise ValueError(f"{', '.join(both)} is both a camera and a microphone; give each device its own name")
        unused = sorted(name for name, camera in self.cameras.items() if not camera.people() and not camera.hears)
        if unused:
            raise ValueError(f"camera {', '.join(unused)} is read for nobody: say whom it films or hears, or remove it")
        alone: dict[str, list[str]] = {}
        for name, device in self.devices().items():
            if len(set(device.hears)) == 1:
                alone.setdefault(device.hears[0], []).append(name)
        twice = {person: names for person, names in alone.items() if len(names) > 1}
        if twice:
            person, names = sorted(twice.items())[0]
            raise ValueError(f"{' and '.join(sorted(names))} both carry {person}'s voice alone; say it of one of them")
        shared = sorted(name for name, device in self.devices().items() if len(set(device.hears)) > 1)
        if len(shared) > 1:
            raise ValueError(f"{', '.join(shared)} all carry several people's voices; voices are separated on one, so say it of one of them")
        # a person's face and gaze are read from one picture; several cameras on a body serve three dimensions
        for use in ("face", "gaze"):
            claimed: dict[str, list[str]] = {}
            for name, camera in self.cameras.items():
                for person in set(getattr(camera, use)):
                    claimed.setdefault(person, []).append(name)
            twice = {person: names for person, names in claimed.items() if len(names) > 1}
            if twice:
                person, names = sorted(twice.items())[0]
                raise ValueError(f"{' and '.join(sorted(names))} are both read for {person}'s {use}; say it of one of them")
        return self

    def devices(self) -> dict[str, CameraDevice | MicrophoneDevice]:
        return {**self.cameras, **self.microphones}

    def named_people(self) -> set[str]:
        named: set[str] = set()
        for camera in self.cameras.values():
            named |= set(camera.people()) | set(camera.hears)
        for microphone in self.microphones.values():
            named |= set(microphone.hears)
        return named


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
    # on a camera nobody clicked, suggest who is who: most in view for one person; head motion, own mics or qc.look for several
    suggest_names: bool = False


class MeasureSection(_Section):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    person: str
    joints: list[str] = Field(default_factory=list)
    # "self" measures within the person
    to_person: str
    to_joints: list[str] = Field(default_factory=list)
    # a built-in name, or file.function from project.scripts_dir
    method: str = "distance"
    per_body_size: bool = False
    # with 3D, a script method runs on the points in the room too; built-in methods always do
    in_3d: bool = False


class GestureSection(_Section):
    tier: str | None = None
    annotator: str | None = None
    # in the keypoints' length unit per second, so each project sets them for its own videos
    still_speed: float | None = Field(default=None, gt=0)
    peak_prominence: float | None = Field(default=None, gt=0)
    # metres per second, for pose.gestures3d; without them gestures are measured from the picture only
    still_speed_3d: float | None = Field(default=None, gt=0)
    peak_prominence_3d: float | None = Field(default=None, gt=0)
    holds: Literal["each_hand", "both_hands"] = "each_hand"

    @model_validator(mode="after")
    def _thresholds_with_a_tier(self) -> "GestureSection":
        if self.tier and (self.still_speed is None or self.peak_prominence is None):
            raise ValueError(
                "gestures.tier is set, so still_speed and peak_prominence are needed too, in pixels per second "
                "for these videos; the wrist speeds pose.kinematics writes help to choose them"
            )
        return self


class KinematicsSection(_Section):
    smoothing_s: float = 0.3
    measures: list[MeasureSection] = Field(default_factory=list)
    gestures: GestureSection = Field(default_factory=GestureSection)

    @model_validator(mode="after")
    def _distinct_names(self) -> "KinematicsSection":
        names = [measure.name for measure in self.measures]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"two measures are called {', '.join(repeated)}; give each its own name")
        return self


class ScaleSection(_Section):
    # written by hand, these win over any measurement; a role's own number wins over the one for everyone
    cm_per_px: float | None = Field(default=None, gt=0)
    cm_per_px_by_role: dict[str, float] = Field(default_factory=dict)
    # a clip in project.calibration_dir measured with macaw set-scale --calibration, for recordings not measured on their own video
    calibration: str | None = None
    # share by which a person's apparent size may change over a recording before the report flags it
    max_size_change: float = Field(default=0.1, gt=0)
    # length of the stretches compared; long ones miss people walking about, short ones flag a moment bent over
    size_window_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _positive_by_role(self) -> "ScaleSection":
        bad = sorted(role for role, value in self.cm_per_px_by_role.items() if value <= 0)
        if bad:
            raise ValueError(f"scale.cm_per_px_by_role must be above 0, not for {', '.join(bad)}")
        return self


class BoardSection(_Section):
    squares_x: int = Field(default=7, ge=3)
    squares_y: int = Field(default=5, ge=3)
    # as printed and measured with a ruler; it sets the scale of everything in 3D
    square_mm: float = Field(default=75.0, gt=0)
    marker_mm: float = Field(default=56.0, gt=0)
    dictionary: Literal["4X4_50", "4X4_100", "5X5_50", "5X5_100", "6X6_50", "6X6_250"] = "5X5_100"

    @model_validator(mode="after")
    def _markers_inside_squares(self) -> "BoardSection":
        if self.marker_mm >= self.square_mm:
            raise ValueError("pose3d.board.marker_mm must be smaller than square_mm")
        return self


class Pose3DSection(_Section):
    # a folder of board clips, one per camera, or a calibration .toml, in project.calibration_dir
    calibration: str | None = None
    # what an imported calibration's translations are in, when the file does not say
    units: Literal["mm", "cm", "m"] = "mm"
    board: BoardSection = Field(default_factory=BoardSection)
    # a camera whose view of a point lands further than this from where the others place it is left out of that point
    max_reprojection_px: float = Field(default=15.0, gt=0)
    # board clips are read at this interval
    board_every_s: float = Field(default=0.2, gt=0)


class FaceSection(_Section):
    backend: str = "pyfeat"
    model: str | None = None
    device: str = "auto"
    batch_size: int = 16
    action_units: bool = True
    landmarks: bool = True
    blink_min_eye_px: float = 5.0
    gaze: bool = False
    check_model: str | None = None
    check_backend: str | None = None


class GazeSection(_Section):
    backend: str = "gazelle"
    model: str | None = None
    device: str = "auto"
    batch_size: int = Field(default=16, ge=1)
    targets: list[str] = Field(default_factory=list)
    object_backend: str = "grounding_dino"
    object_model: str | None = None
    object_score: float = Field(default=0.35, gt=0, lt=1)
    seats: dict[str, tuple[float, float, float]] = Field(default_factory=dict)
    tolerance_deg: float | None = Field(default=None, gt=0, lt=180)
    min_look_s: float = Field(default=0.2, ge=0)

    @model_validator(mode="after")
    def _names(self) -> "GazeSection":
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("gaze.targets names a thing twice")
        clash = sorted(set(self.targets) & {"elsewhere", "off_screen", "unsure", "camera"})
        if clash:
            raise ValueError(f"gaze.targets cannot use {', '.join(clash)}, which MACAW already uses")
        if self.seats and "camera" not in self.seats:
            raise ValueError("gaze.seats needs the camera's place too, as camera: [x, depth, eye height]")
        return self


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


ShownToModel = Literal["timing", "pitch", "loudness", "rate", "frames"]


class LLMSection(_Section):
    provider: str = "ollama"
    model: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 8192
    timeout_s: int = 300
    n_samples: int = Field(default=5, ge=1)
    seed: int = 0
    allow_abstain: bool = True
    review_threshold: float = 0.7
    review_random: int = Field(default=5, ge=0)
    batch_size: int = Field(default=20, ge=1)
    context_units: int = Field(default=5, ge=0)
    coded_roles: list[str] = Field(default_factory=list)
    show: list[ShownToModel] = Field(default_factory=lambda: ["timing"])
    notes: str = ""
    examples: Path | None = None
    reasons: bool = False
    think: bool | None = None

    def is_local(self) -> bool:
        from macaw.coding.llm.backends import BACKEND_GROUP
        from macaw.extractors import get_backend

        return get_backend(BACKEND_GROUP, self.provider).is_local


class JudgeVersion(_Section):
    """What another coding by the language model changes; anything left out is as in coding.codebook and coding.llm."""

    codebook: Path | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    n_samples: int | None = Field(default=None, ge=1)
    seed: int | None = None
    allow_abstain: bool | None = None
    review_threshold: float | None = None
    batch_size: int | None = Field(default=None, ge=1)
    context_units: int | None = Field(default=None, ge=0)
    coded_roles: list[str] | None = None
    show: list[ShownToModel] | None = None
    notes: str | None = None
    examples: Path | None = None
    reasons: bool | None = None
    think: bool | None = None


class ImportedTier(_Section):
    person: str | None = None
    tier: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class CodingSection(_Section):
    layers: list[Literal["A", "B", "C"]] = Field(default_factory=lambda: ["A"])
    codebook: Path | None = None
    backchannel_lexicon: list[str] = Field(default_factory=list)
    backchannel_max_s: float = 1.0
    min_turn_internal_pause_s: float = 0.2
    llm: LLMSection = Field(default_factory=LLMSection)
    judges: dict[str, JudgeVersion] = Field(default_factory=dict)
    imports: dict[str, ImportedTier] = Field(default_factory=dict)

    @field_validator("judges")
    @classmethod
    def _judge_names(cls, value: dict[str, JudgeVersion]) -> dict[str, JudgeVersion]:
        # the name becomes a step name, a file name and a coder in agreement
        bad = [name for name in value if not re.fullmatch(r"[A-Za-z0-9_]+", name)]
        if bad:
            raise ValueError(f"{', '.join(bad)}: a version's name takes letters, digits and _ only")
        return value

    def judge_settings(self, version: str | None = None) -> tuple[Path | None, LLMSection]:
        """The codebook and model settings of one coding by the language model; None is coding.llm's own."""
        if version is None:
            return self.codebook, self.llm
        changes = self.judges[version].model_dump(exclude_none=True)
        codebook = changes.pop("codebook", self.codebook)
        return codebook, self.llm.model_copy(update=changes)


class LookSection(_Section):
    provider: str = "ollama"
    model: str | None = None
    base_url: str | None = None
    timeout_s: int = 300
    # role -> what that person looks like; only the people described have their names checked
    people: dict[str, str] = Field(default_factory=dict)
    random_check: int = Field(default=5, ge=0)

    def llm(self) -> LLMSection:
        """The same server settings as coding.llm takes, answering as plainly as it can."""
        return LLMSection(provider=self.provider, model=self.model, base_url=self.base_url, timeout_s=self.timeout_s, temperature=0.0, max_tokens=512)


class QCSection(_Section):
    checks: list[str] = Field(
        default_factory=lambda: [
            "transcript_match",
            "mouth_match",
            "track_quality",
            "octave_error",
            "motion_energy",
            "au_agreement",
            "scale",
            "gaze_labels",
        ]
    )
    fail_fast: bool = False
    # per role or subject id; a median above it means the whole track may be an octave high
    f0_median_max_by_role: dict[str, float] = Field(default_factory=dict)
    report_downsample_px: int = 2000
    # long-table features the report's overview draws; empty draws movement, distance and kinematics.measures
    overview_features: list[str] = Field(default_factory=list)
    review_max_candidates: int = 40
    review_baseline: int = 25
    review_seed: int = 0
    look: LookSection = Field(default_factory=LookSection)


class ExecutionSection(_Section):
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
    setup: SetupSection = Field(default_factory=SetupSection)
    align: AlignSection = Field(default_factory=AlignSection)
    identity: IdentitySection = Field(default_factory=IdentitySection)
    segment: SegmentSection = Field(default_factory=SegmentSection)
    video: VideoSection = Field(default_factory=VideoSection)
    prosody: ProsodySection = Field(default_factory=ProsodySection)
    text: TextSection = Field(default_factory=TextSection)
    pose: PoseSection = Field(default_factory=PoseSection)
    kinematics: KinematicsSection = Field(default_factory=KinematicsSection)
    scale: ScaleSection = Field(default_factory=ScaleSection)
    pose3d: Pose3DSection = Field(default_factory=Pose3DSection)
    face: FaceSection = Field(default_factory=FaceSection)
    gaze: GazeSection = Field(default_factory=GazeSection)
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

    @model_validator(mode="after")
    def _identity_roles(self) -> "MacawConfig":
        unknown = sorted(self.setup.named_people() - set(self.project.roles))
        if unknown:
            raise ValueError(
                f"setup {self.setup.name} names {', '.join(unknown)}, which project.roles does not have ({', '.join(self.project.roles)})"
            )
        named = [hint.role for hint in self.identity.voice_hints] + list(self.identity.pitch_order)
        unknown = sorted(set(named) - set(self.project.roles))
        if unknown:
            raise ValueError(
                f"identity names {', '.join(unknown)}, which project.roles does not have ({', '.join(self.project.roles)})"
            )
        unknown = sorted(set(self.scale.cm_per_px_by_role) - set(self.project.roles))
        if unknown:
            raise ValueError(
                f"scale.cm_per_px_by_role names {', '.join(unknown)}, which project.roles does not have ({', '.join(self.project.roles)})"
            )
        unknown = sorted(set(self.gaze.seats) - set(self.project.roles) - {"camera"})
        if unknown:
            raise ValueError(f"gaze.seats names {', '.join(unknown)}, which project.roles does not have ({', '.join(self.project.roles)})")
        clash = sorted(set(self.gaze.targets) & set(self.project.roles))
        if clash:
            raise ValueError(f"gaze.targets names {', '.join(clash)}, which are roles; people are looked for anyway")
        return self


# Which sections a recordings/<id>.yaml file may touch. Everything else is a hard error:
# a recording must never be able to redirect output_dir or change the scheduler.
RECORDING_OVERRIDABLE: tuple[str, ...] = (
    "setup",
    "segment",
    "prosody",
    "text",
    "pose",
    "kinematics",
    "scale",
    "pose3d",
    "face",
    "gaze",
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
    "coding.judges.*.codebook",
    "coding.llm.examples",
    "coding.judges.*.examples",
    "execution.output_rules",
    *NAME_OR_PATH_FIELDS,
)
