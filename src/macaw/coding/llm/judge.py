"""Layer B: the codebook applied to each unit of the transcript by a language model.

This is the half that has no single truth, so its output keeps confidence and abstentions and
goes through the review queue. It reads the whole coding section, codebook included, which is
what makes a codebook edit recompute this pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.coding.codebook import Codebook, read_codebook, validate_codebook
from macaw.coding.examples import read_examples
from macaw.coding.llm.backends import LLMBackend, resolve_llm_backend
from macaw.coding.llm.parsing import parse_annotation_response
from macaw.coding.llm.prompts import ABSTAIN, FRAME_AT, FRAMES, VOICE, build_annotation_prompt
from macaw.config.schema import LLMSection, MacawConfig
from macaw.exceptions import BackendNotAvailableError, CodebookError, ConfigError, LLMBackendError, UnreadableMediaError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import ASRExtractor, load_transcript
from macaw.extractors.audio.prosody.frame import FrameProsodyExtractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, apply_prep_timing
from macaw.media.ffmpeg import extract_frame
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.segments import clip_intervals
from macaw.store import _io
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import ANNOTATION_SCHEMA, Scale, coerce_frame, empty_frame
from macaw.turns.segmentation import TurnStructureExtractor, load_turns

LAYER = "B"
# each version under coding.judges is its own step with its own layer, so none overwrites another
VERSION_STEP = "coding.judge."
VERSION_LAYER = "B."
SAMPLES_NAME = "samples.parquet"
_FRAME_FEATURES = {"pitch": "f0_semitone", "loudness": "intensity_db"}
# px; enough for faces at a table in a full frame without the picture costing thousands of tokens more
FRAME_LONG_SIDE = 960

_SUBJECT_SPECS = {
    "n_units_judged": ("Units of this speaker the model coded.", "count"),
    "share_abstained": ("Share of those units where the model's majority answer was that it could not tell.", "proportion"),
    "share_needs_review": ("Share of those units sent to the review queue: abstained, tied, or below coding.llm.review_threshold.", "proportion"),
}


@register_extractor
class JudgementExtractor(Extractor):
    name = "coding.judge"
    version = "0.4.0"
    scale = Scale.EVENT
    group = "coding"
    # turns: the turn table it codes when the unit is a turn. Where the model answers (base_url,
    # timeout_s) is the machine's business, so moving to a cluster does not code everything again
    config_sections = (
        "coding.codebook", "turns", *(f"coding.llm.{key}" for key in (
            "provider", "model", "temperature", "max_tokens", "n_samples", "seed", "allow_abstain",
            "review_threshold", "batch_size", "context_units", "coded_roles", "show", "notes", "examples",
            "reasons", "think",
        )),
    )
    requires = (ASRExtractor.name, TurnStructureExtractor.name)
    judge: str | None = None  # the version under coding.judges this step codes; None is coding.llm's own

    @classmethod
    def settings(cls, config: MacawConfig) -> tuple[Path | None, LLMSection]:
        if cls.judge is not None and cls.judge not in config.coding.judges:
            raise ConfigError(f"coding.judges has no version {cls.judge}")
        return config.coding.judge_settings(cls.judge)

    @classmethod
    def _present(cls, config: MacawConfig) -> bool:
        return cls.judge is None or cls.judge in config.coding.judges

    @classmethod
    def wanted_by_default(cls, config: MacawConfig) -> bool:
        return "B" in config.coding.layers and cls._present(config)

    @classmethod
    def dependencies(cls, config: MacawConfig) -> tuple[str, ...]:
        shown = set(cls.settings(config)[1].show) if cls._present(config) else set()
        voiced = (FrameProsodyExtractor.name,) if {"pitch", "loudness"} & shown else ()
        return cls.requires + voiced + ((MediaPrepExtractor.name, VideoPrepExtractor.name) if FRAMES in shown else ())

    @classmethod
    def resources(cls, config: MacawConfig) -> str:
        # a model served inside the job needs the GPU; a remote one only the network
        return "gpu_standard" if cls.settings(config)[1].is_local() else cls.resource_profile

    def check_available(self) -> None:
        self._codebook()
        _, cfg = self.settings(self.config)
        unknown = sorted(set(cfg.coded_roles) - set(self.config.project.roles))
        if unknown:
            raise ConfigError(
                f"coded_roles of {self.name} names {', '.join(unknown)}, which project.roles does not have "
                f"({', '.join(self.config.project.roles)})"
            )
        resolve_llm_backend(cfg).check_available()

    def inputs_for(self, recording: Recording) -> list[Path]:
        inputs = TurnStructureExtractor(self.config).inputs_for(recording)
        shown = set(self.settings(self.config)[1].show)
        voice = prosody_rows_file(self.config, recording.recording_id)
        if {"pitch", "loudness"} & shown and voice.is_file():
            inputs.append(voice)
        if FRAMES in shown:
            # the offset table moves every frame on the recording clock
            offsets = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
            inputs += sorted(item.path for item in recording.media if item.kind == "video")
            inputs += [offsets] if offsets.is_file() else []
        return inputs

    def extra_key(self, recording: Recording) -> str:
        """Shown frames come from each speaker's body camera, so the setup's choice of it decides them."""
        if FRAMES not in set(self.settings(self.config)[1].show):
            return ""
        return ",".join(f"{person}={item.name}" for person, item in sorted(recording.views("body").items()))

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = resolve_llm_backend(self.settings(self.config)[1])
        return backend.name, backend.model_version()

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        _, cfg = self.settings(self.config)
        codebook = self._codebook()
        backend = resolve_llm_backend(cfg)
        lines = coding_units(self.config, recording, codebook.unit, cfg.coded_roles)
        if set(VOICE) & set(cfg.show):
            lines = voice_figures(self.config, recording.recording_id, lines, cfg.show)
        units = lines[lines["unit_id"].notna()]
        examples = self._examples(codebook)
        # an example from the conversation being coded would hand the model the answer
        own = examples["recording_id"] == recording.recording_id
        frames = unit_frames(self.config, recording, units, workspace / "frames") if FRAMES in cfg.show else None
        samples = sample_labels(lines, codebook, backend, cfg, examples=examples[~own], frames=frames)
        judged = aggregate_samples(units, samples, cfg)

        path = workspace / SAMPLES_NAME
        samples.merge(units[["unit_id", "subject_id", "t_start", "t_end"]], on="unit_id").to_parquet(path, index=False)
        annotator = self.judge or f"llm:{backend.name}/{cfg.model}"
        return ExtractorResult(
            rows=self._rows(recording, judged),
            specs=self._specs(codebook, examples),
            annotations=self._annotations(recording, judged, codebook.name, annotator),
            artifacts={SAMPLES_NAME: path},
            warnings=self._warnings(
                recording, judged, backend, samples.attrs, int(own.sum()),
                sum(not pictures for pictures in frames.values()) if frames is not None else 0,
            ),
        )

    def _examples(self, codebook: Codebook) -> pd.DataFrame:
        _, cfg = self.settings(self.config)
        if cfg.examples is None:
            return pd.DataFrame(columns=["code", "speaker", "text", "before", "why", "recording_id"])
        return read_examples(cfg.examples, codebook)

    def _codebook(self) -> Codebook:
        path, cfg = self.settings(self.config)
        if path is None:
            raise BackendNotAvailableError(f"{self.name} needs a codebook: set coding.codebook to its YAML file.")
        codebook = read_codebook(path)
        problems = validate_codebook(codebook)
        if problems:
            raise CodebookError(f"{path}: " + "; ".join(problems))
        if not codebook.mutually_exclusive:
            raise CodebookError(f"{path}: only mutually exclusive codebooks can be coded so far (one code per unit).")
        if codebook.unit == "window":
            raise CodebookError(f"{path}: unit 'window' cannot be coded yet; use utterance or turn.")
        if cfg.examples is not None:
            read_examples(cfg.examples, codebook)
        return codebook

    def _annotations(self, recording: Recording, judged: pd.DataFrame, tier: str, annotator: str) -> pd.DataFrame:
        entities = recording.entities()
        frames = [
            build_annotations(
                recording_id=recording.recording_id, tier=tier, layer=self.layer(), annotator=annotator,
                intervals=group[["t_start", "t_end", "label", "confidence", "abstained", "needs_review"]].reset_index(drop=True),
                entities=entities, subject_id=str(subject_id),
            )
            for subject_id, group in judged.groupby("subject_id", sort=True)
        ]
        if not frames:
            return empty_frame(ANNOTATION_SCHEMA)
        return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")

    @classmethod
    def layer(cls) -> str:
        return LAYER if cls.judge is None else f"{VERSION_LAYER}{cls.judge}"

    @classmethod
    def feature(cls, name: str) -> str:
        return name if cls.judge is None else f"{name}_{cls.judge}"

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        # the layer's own file: people's review decisions share its layer but are kept apart
        own = layout_for(self.config, create=False).annotations_path(recording.recording_id) / f"{self.layer()}.parquet"
        if not own.is_file():
            return None
        return self._rows(recording, clip_intervals(_io.read_file(own, ANNOTATION_SCHEMA, "annotations"), start_s, end_s))

    def _rows(self, recording: Recording, judged: pd.DataFrame) -> pd.DataFrame:
        span = (float(judged["t_start"].min()), float(judged["t_end"].max())) if not judged.empty else (np.nan, np.nan)
        coded = set(self.settings(self.config)[1].coded_roles)
        frames = []
        for subject in recording.subjects:
            if coded and subject.subject_id not in coded:
                continue
            mine = judged[judged["subject_id"] == subject.subject_id]
            n = len(mine)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame({
                        "t_start": [span[0]],
                        "t_end": [span[1]],
                        self.feature("n_units_judged"): [float(n)],
                        self.feature("share_abstained"): [float(mine["abstained"].mean()) if n else np.nan],
                        self.feature("share_needs_review"): [float(mine["needs_review"].mean()) if n else np.nan],
                    }),
                    scale=Scale.RECORDING, subject_id=subject.subject_id,
                    recording_id=recording.recording_id, source=self.name, entities=recording.entities(),
                )
            )
        return concat_long_tables(frames)

    def _warnings(
        self, recording: Recording, judged: pd.DataFrame, backend: LLMBackend, sampled: dict, own_examples: int = 0,
        unseen: int = 0,
    ) -> list[QCWarning]:
        _, cfg = self.settings(self.config)
        warnings = []

        def note(check: str, severity: str, message: str) -> None:
            warnings.append(QCWarning(check=check, recording_id=recording.recording_id, subject_id=None, severity=severity, message=message))

        notice = cloud_notice(backend, cfg)
        if notice:
            note("llm_cloud", "info", notice)
        if own_examples:
            note(
                "llm_own_examples", "info",
                f"{own_examples} of the examples come from this recording and were left out of its prompt; the others were shown.",
            )
        if unseen:
            note(
                "llm_no_frames", "info",
                f"{unseen} of {len(judged)} units fall where the video does not reach and were coded from the words alone.",
            )
        if sampled.get("cut_off"):
            note(
                "llm_cut_off", "warn",
                f"{sampled['cut_off']} of {sampled['requests']} answers were cut off at max_tokens ({cfg.max_tokens}); the units "
                "they did not reach count as ones the model could not tell. Raise coding.llm.max_tokens.",
            )
        if cfg.n_samples == 1:
            note("llm_single_sample", "info", "coding.llm.n_samples is 1, so no confidence was measured; only abstentions go to the review queue.")
        if not backend.takes_seed and cfg.n_samples > 1:
            note("llm_no_seed", "info", f"{backend.name} takes no seed, so a rerun can give different labels.")
        if not judged.empty and judged["abstained"].mean() > 0.5:
            note(
                "llm_mostly_abstained", "warn",
                f"The model could not tell for {judged['abstained'].mean():.0%} of the units. "
                "Check that the codebook fits this conversation and that the model follows the answer format.",
            )
        review = int(judged["needs_review"].sum()) if not judged.empty else 0
        # a version is coded for comparison, so its unsure units do not join the review queue
        if review and self.judge is None:
            note("llm_review", "info", f"{review} of {len(judged)} units wait for a person: macaw review --export <file>.csv.")
        return warnings

    def _specs(self, codebook: Codebook, examples: pd.DataFrame) -> list[FeatureSpec]:
        _, cfg = self.settings(self.config)
        params = {
            "codebook": codebook.name, "codebook_unit": codebook.unit, "names_only": codebook.names_only(),
            "provider": cfg.provider, "model": cfg.model, "n_samples": cfg.n_samples,
            "temperature": cfg.temperature, "seed": cfg.seed, "review_threshold": cfg.review_threshold,
            "batch_size": units_per_request(cfg), "context_units": cfg.context_units, "coded_roles": list(cfg.coded_roles),
            "show": list(cfg.show), "notes": cfg.notes.strip(), "n_examples": len(examples), "reasons": cfg.reasons,
            "think": cfg.think, "judge": self.judge,
        }
        return [
            FeatureSpec(
                name=self.feature(feature), description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                extractor=self.name, extractor_version=self.version, params=params, aggregation="none", validation="none",
            )
            for feature, (description, unit) in _SUBJECT_SPECS.items()
        ]


def _letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"S{index + 1}"


def coding_units(config: MacawConfig, recording: Recording, unit: str, coded_roles: list[str] | tuple = ()) -> pd.DataFrame:
    """Every line in time order, other voices' too: subject_id, speaker letter, times, the pause or overlap
    before, and the words. Lines to code (coded_roles, or everyone's when empty) have a unit_id, the rest none."""
    loader = load_transcript if unit == "utterance" else load_turns
    source = loader(config, recording.recording_id, others=True)
    frame = source[["subject_id", "t_start", "t_end", "text"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["text"] = frame["text"].astype("string").fillna("").str.strip()
    frame = frame[frame["text"] != ""].sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)

    # letters, not subject ids, go to the model: nothing guarantees the ids are pseudonymous
    letters = {subject.subject_id: _letter(index) for index, subject in enumerate(recording.subjects)}
    frame["speaker"] = [letters.get(subject, subject) for subject in frame["subject_id"]]
    coded = frame["subject_id"].isin(list(coded_roles) or list(letters)).to_numpy()
    frame["unit_id"] = pd.array([pd.NA] * len(frame), dtype="string")
    frame.loc[coded, "unit_id"] = [f"u{index}" for index in range(int(coded.sum()))]
    reached = frame["t_end"].cummax().shift()
    frame["gap_before"] = frame["t_start"] - reached
    return frame[["unit_id", "subject_id", "speaker", "t_start", "t_end", "gap_before", "text"]]


def prosody_rows_file(config: MacawConfig, recording_id: str) -> Path:
    return layout_for(config, create=False).long_table_path(recording_id) / f"{FrameProsodyExtractor.name}.parquet"


def voice_figures(config: MacawConfig, recording_id: str, lines: pd.DataFrame, show) -> pd.DataFrame:
    """Each line's pitch (semitones), loudness (dB) and speech rate (percent) against the median of that speaker's lines."""
    out = lines.copy()
    wanted = [item for item in VOICE if item in show]
    if "rate" in wanted:
        seconds = (out["t_end"] - out["t_start"]).to_numpy(dtype=float)
        words = out["text"].astype(str).str.split().str.len().to_numpy(dtype=float)
        out["rate"] = np.where(seconds > 0, words / np.where(seconds > 0, seconds, 1.0), np.nan)
    measured = [item for item in wanted if item in _FRAME_FEATURES]
    if measured:
        path = prosody_rows_file(config, recording_id)
        rows = (
            read_long_table(path, recording_id=recording_id, features=[_FRAME_FEATURES[item] for item in measured])
            if path.is_file() else pd.DataFrame(columns=["subject_id", "t_start", "feature", "value"])
        )
        for item in measured:
            out[item] = _line_medians(out, rows[rows["feature"] == _FRAME_FEATURES[item]])
    for item in wanted:
        usual = out.groupby("subject_id")[item].transform("median")
        out[item] = (out[item] / usual - 1.0) * 100.0 if item == "rate" else out[item] - usual
    return out


def _line_medians(lines: pd.DataFrame, frames: pd.DataFrame) -> np.ndarray:
    values = np.full(len(lines), np.nan)
    for subject_id, own in frames.groupby("subject_id", sort=False):
        own = own.sort_values("t_start")
        times, measured = own["t_start"].to_numpy(dtype=float), own["value"].to_numpy(dtype=float)
        for position in np.flatnonzero((lines["subject_id"] == str(subject_id)).to_numpy()):
            first, last = np.searchsorted(times, [lines["t_start"].iloc[position], lines["t_end"].iloc[position]])
            inside = measured[first:last]
            inside = inside[np.isfinite(inside)]
            if inside.size:
                values[position] = float(np.median(inside))
    return values


_VERSIONS: dict[str, type[JudgementExtractor]] = {}


def register_judge_versions(config: MacawConfig) -> None:
    """Each version under coding.judges becomes the step coding.judge.<name>; a class per name, made once."""
    for name in config.coding.judges:
        if name not in _VERSIONS:
            _VERSIONS[name] = register_extractor(
                type(
                    f"JudgementVersion_{name}",
                    (JudgementExtractor,),
                    {
                        "name": f"{VERSION_STEP}{name}",
                        "judge": name,
                        "config_sections": (*JudgementExtractor.config_sections, f"coding.judges.{name}"),
                        "__module__": __name__,
                    },
                )
            )


def version_steps(config: MacawConfig) -> list[str]:
    return [f"{VERSION_STEP}{name}" for name in config.coding.judges]


def sample_annotations(config: MacawConfig, recording_ids: list[str]) -> pd.DataFrame:
    """Every answer of every coding by the language model as a coder of its own, <coder>#<sample>, so that
    agreement can set the model against itself."""
    from macaw.extractors import get_extractor
    from macaw.store import _io

    layout = layout_for(config, create=False)
    frames = []
    for rid in recording_ids:
        for step in (JudgementExtractor.name, *version_steps(config)):
            path = layout.artifacts_dir(rid, step) / SAMPLES_NAME
            # the layer's own file: people's review decisions share its layer but are kept apart
            own = layout.annotations_path(rid) / f"{get_extractor(step).layer()}.parquet"
            if not own.is_file() or not path.is_file():
                continue
            mine = _io.read_file(own, ANNOTATION_SCHEMA, "annotations")
            if mine.empty:
                continue
            samples = pd.read_parquet(path)
            keys = _unit_keys(mine)
            for sample, answers in samples.groupby("sample", sort=True):
                said = dict(zip(_unit_keys(answers), answers["label"]))
                labels = [said.get(key) for key in keys]
                frames.append(mine.assign(
                    annotator=f"{mine['annotator'].iloc[0]}#{int(sample) + 1}",
                    label=pd.array([pd.NA if label in (None, ABSTAIN) else label for label in labels], dtype="string"),
                    abstained=[label == ABSTAIN for label in labels], confidence=np.nan, needs_review=False,
                ))
    if not frames:
        return empty_frame(ANNOTATION_SCHEMA)
    return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")


def _unit_keys(frame: pd.DataFrame) -> list[tuple]:
    return list(zip(frame["subject_id"].astype(str), frame["t_start"].astype(float).round(6), frame["t_end"].astype(float).round(6)))


def units_per_request(cfg: LLMSection) -> int:
    # a model reading pictures gets one unit's frames at a time
    return 1 if FRAMES in cfg.show else cfg.batch_size


def cloud_notice(backend: LLMBackend, cfg: LLMSection) -> str | None:
    if not backend.reporting:
        return None
    return backend.reporting + (" Frames of the video are sent there too." if FRAMES in cfg.show else "")


def unit_frames(config: MacawConfig, recording: Recording, units: pd.DataFrame, directory: Path) -> dict[str, list[bytes]]:
    """Each unit's frames from the camera the setup reads its speaker's body from, else from the recording's main
    video. A moment the video does not reach, as when the camera stopped before the microphones, gives no frame."""
    apply_prep_timing(config, recording)
    videos = [item for item in recording.media if item.kind == "video"]
    if not videos:
        raise UnreadableMediaError(f"{recording.recording_id}: coding.llm.show has frames, but the recording has no video")
    own = recording.views("body")
    main = recording.reference_media(videos)
    lengths = {}
    frames = {}
    for row in units.itertuples(index=False):
        item = own.get(row.subject_id, main)
        if item.frames_path not in lengths:
            lengths[item.frames_path] = video_length(item.frames_path)
        pictures = []
        for index, share in enumerate(FRAME_AT):
            at = item.own_time(row.t_start + share * (row.t_end - row.t_start))
            if 0 <= at < lengths[item.frames_path]:
                dest = extract_frame(item.frames_path, at, directory / f"{row.unit_id}_{index}.jpg", long_side=FRAME_LONG_SIDE)
                pictures.append(dest.read_bytes())
        frames[row.unit_id] = pictures
    return frames


def video_length(path: Path) -> float:
    """Seconds up to the start of the last frame, the last moment a frame can be taken at."""
    from macaw.media.probe import probe

    info = probe(path)
    stream = info.video_streams()[0]
    duration = stream.duration_s or info.duration_s
    return duration - 1.0 / stream.fps if stream.fps else duration


def sample_labels(
    lines: pd.DataFrame, codebook: Codebook, backend: LLMBackend, cfg: LLMSection, *,
    examples: pd.DataFrame | None = None, frames: dict[str, list[bytes]] | None = None,
) -> pd.DataFrame:
    """One row per unit and sample: unit_id, sample, label (a code or abstain), reason. A batch is
    units_per_request units with the uncoded lines between them in place."""
    coded = np.flatnonzero(lines["unit_id"].notna().to_numpy())
    size = units_per_request(cfg)
    answers = []
    cut_off = 0
    for sample in range(cfg.n_samples):
        seed = cfg.seed + sample if backend.takes_seed else None
        for start in range(0, len(coded), size):
            first, last = coded[start], coded[min(start + size, len(coded)) - 1]
            batch = lines.iloc[first : last + 1]
            context = lines.iloc[max(0, first - cfg.context_units) : first]
            system, user = build_annotation_prompt(
                codebook, batch, context, allow_abstain=cfg.allow_abstain, show=cfg.show, notes=cfg.notes,
                examples=examples, reasons=cfg.reasons,
            )
            ids = batch["unit_id"].dropna().tolist()
            images = [picture for unit_id in ids for picture in frames[unit_id]] if frames else ()
            response = backend.complete(system, user, seed=seed, images=images)
            parsed = parse_annotation_response(response.text, codebook, ids, allow_abstain=cfg.allow_abstain)
            # an answer that never came is not the model saying it cannot tell
            if response.cut_off and (parsed["label"] == ABSTAIN).all():
                raise LLMBackendError(
                    f"{backend.name} {cfg.model}: the answer was cut off at max_tokens ({cfg.max_tokens}) before any code "
                    "came back. A model that reasons before it answers spends tokens on that first; raise coding.llm.max_tokens."
                )
            cut_off += response.cut_off
            answers.append(parsed.assign(sample=sample))
    if not answers:
        out = pd.DataFrame({column: pd.Series(dtype=int if column == "sample" else str) for column in ("unit_id", "sample", "label", "reason")})
    else:
        out = pd.concat(answers, ignore_index=True)[["unit_id", "sample", "label", "reason"]]
    out.attrs.update(cut_off=cut_off, requests=len(answers))
    return out


def aggregate_samples(units: pd.DataFrame, samples: pd.DataFrame, cfg: LLMSection) -> pd.DataFrame:
    """The majority answer per unit with its share as confidence. A tie has no label and goes to review."""
    labels, confidence, abstained, review = [], [], [], []
    answers = samples.groupby("unit_id")["label"].apply(list).to_dict()
    for unit_id in units["unit_id"]:
        counts = pd.Series(answers.get(unit_id, [ABSTAIN] * cfg.n_samples)).value_counts()
        top = counts[counts == counts.iloc[0]]
        share = float(counts.iloc[0]) / cfg.n_samples
        tied = len(top) > 1
        gave_up = not tied and top.index[0] == ABSTAIN
        labels.append(pd.NA if tied or gave_up else str(top.index[0]))
        abstained.append(gave_up)
        if cfg.n_samples == 1:
            confidence.append(np.nan)
            review.append(gave_up)
        else:
            confidence.append(share)
            review.append(tied or gave_up or share < cfg.review_threshold)
    return units.assign(label=pd.array(labels, dtype="string"), confidence=confidence, abstained=abstained, needs_review=review)
