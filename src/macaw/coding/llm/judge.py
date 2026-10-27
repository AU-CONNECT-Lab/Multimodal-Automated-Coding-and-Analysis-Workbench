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
from macaw.coding.llm.backends import LLMBackend, resolve_llm_backend
from macaw.coding.llm.parsing import parse_annotation_response
from macaw.coding.llm.prompts import ABSTAIN, build_annotation_prompt
from macaw.config.schema import LLMSection, MacawConfig
from macaw.exceptions import BackendNotAvailableError, CodebookError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import ASRExtractor, load_transcript
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.annotations import build_annotations
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import ANNOTATION_SCHEMA, Scale, coerce_frame, empty_frame
from macaw.turns.segmentation import TurnStructureExtractor, load_turns

LAYER = "B"
SAMPLES_NAME = "samples.parquet"

_SUBJECT_SPECS = {
    "n_units_judged": ("Units of this speaker the model coded.", "count"),
    "share_abstained": ("Share of those units where the model's majority answer was that it could not tell.", "proportion"),
    "share_needs_review": ("Share of those units sent to the review queue: abstained, tied, or below coding.llm.review_threshold.", "proportion"),
}


@register_extractor
class JudgementExtractor(Extractor):
    name = "coding.judge"
    version = "0.1.0"
    scale = Scale.EVENT
    group = "coding"
    config_sections = ("coding", "turns")
    requires = (ASRExtractor.name, TurnStructureExtractor.name)

    @classmethod
    def wanted_by_default(cls, config: MacawConfig) -> bool:
        return "B" in config.coding.layers

    def check_available(self) -> None:
        self._codebook()
        resolve_llm_backend(self.config.coding.llm).check_available()

    def inputs_for(self, recording: Recording) -> list[Path]:
        return TurnStructureExtractor(self.config).inputs_for(recording)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = resolve_llm_backend(self.config.coding.llm)
        return backend.name, backend.model_version()

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.coding.llm
        codebook = self._codebook()
        backend = resolve_llm_backend(cfg)
        units = coding_units(self.config, recording, codebook.unit)
        samples = sample_labels(units, codebook, backend, cfg)
        judged = aggregate_samples(units, samples, cfg)

        path = workspace / SAMPLES_NAME
        samples.merge(units[["unit_id", "subject_id", "t_start", "t_end"]], on="unit_id").to_parquet(path, index=False)
        annotator = f"llm:{backend.name}/{cfg.model}"
        return ExtractorResult(
            rows=self._rows(recording, judged),
            specs=self._specs(codebook),
            annotations=self._annotations(recording, judged, codebook.name, annotator),
            artifacts={SAMPLES_NAME: path},
            warnings=self._warnings(recording, judged, backend),
        )

    def _codebook(self) -> Codebook:
        path = self.config.coding.codebook
        if path is None:
            raise BackendNotAvailableError("coding.judge needs a codebook: set coding.codebook to its YAML file.")
        codebook = read_codebook(path)
        problems = validate_codebook(codebook)
        if problems:
            raise CodebookError(f"{path}: " + "; ".join(problems))
        if not codebook.mutually_exclusive:
            raise CodebookError(f"{path}: only mutually exclusive codebooks can be coded so far (one code per unit).")
        if codebook.unit == "window":
            raise CodebookError(f"{path}: unit 'window' cannot be coded yet; use utterance or turn.")
        return codebook

    def _annotations(self, recording: Recording, judged: pd.DataFrame, tier: str, annotator: str) -> pd.DataFrame:
        entities = recording.entities()
        frames = [
            build_annotations(
                recording_id=recording.recording_id, tier=tier, layer=LAYER, annotator=annotator,
                intervals=group[["t_start", "t_end", "label", "confidence", "abstained", "needs_review"]].reset_index(drop=True),
                entities=entities, subject_id=str(subject_id),
            )
            for subject_id, group in judged.groupby("subject_id", sort=True)
        ]
        if not frames:
            return empty_frame(ANNOTATION_SCHEMA)
        return coerce_frame(pd.concat(frames, ignore_index=True), ANNOTATION_SCHEMA, "annotations")

    def _rows(self, recording: Recording, judged: pd.DataFrame) -> pd.DataFrame:
        span = (float(judged["t_start"].min()), float(judged["t_end"].max())) if not judged.empty else (np.nan, np.nan)
        frames = []
        for subject in recording.subjects:
            mine = judged[judged["subject_id"] == subject.subject_id]
            n = len(mine)
            frames.append(
                build_long_rows(
                    values=pd.DataFrame({
                        "t_start": [span[0]],
                        "t_end": [span[1]],
                        "n_units_judged": [float(n)],
                        "share_abstained": [float(mine["abstained"].mean()) if n else np.nan],
                        "share_needs_review": [float(mine["needs_review"].mean()) if n else np.nan],
                    }),
                    scale=Scale.RECORDING, subject_id=subject.subject_id,
                    recording_id=recording.recording_id, source=self.name, entities=recording.entities(),
                )
            )
        return concat_long_tables(frames)

    def _warnings(self, recording: Recording, judged: pd.DataFrame, backend: LLMBackend) -> list[QCWarning]:
        cfg = self.config.coding.llm
        warnings = []

        def note(check: str, severity: str, message: str) -> None:
            warnings.append(QCWarning(check=check, recording_id=recording.recording_id, subject_id=None, severity=severity, message=message))

        if backend.reporting:
            note("llm_cloud", "info", backend.reporting)
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
        if review:
            note("llm_review", "info", f"{review} of {len(judged)} units wait for a person: macaw review --export <file>.csv.")
        return warnings

    def _specs(self, codebook: Codebook) -> list[FeatureSpec]:
        cfg = self.config.coding.llm
        params = {
            "codebook": codebook.name, "codebook_unit": codebook.unit, "provider": cfg.provider, "model": cfg.model, "n_samples": cfg.n_samples,
            "temperature": cfg.temperature, "seed": cfg.seed, "review_threshold": cfg.review_threshold,
            "batch_size": cfg.batch_size, "context_units": cfg.context_units,
        }
        return [
            FeatureSpec(
                name=feature, description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                extractor=self.name, extractor_version=self.version, params=params, aggregation="none", validation="none",
            )
            for feature, (description, unit) in _SUBJECT_SPECS.items()
        ]


def _letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"S{index + 1}"


def coding_units(config: MacawConfig, recording: Recording, unit: str) -> pd.DataFrame:
    """unit_id, subject_id, speaker letter, times, the pause or overlap before, and the words, in time order."""
    if unit == "utterance":
        source = load_transcript(config, recording.recording_id)
    else:
        source = load_turns(config, recording.recording_id)
    frame = source[["subject_id", "t_start", "t_end", "text"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["text"] = frame["text"].astype("string").fillna("").str.strip()
    frame = frame[frame["text"] != ""].sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)

    # letters, not subject ids, go to the model: nothing guarantees the ids are pseudonymous
    letters = {subject.subject_id: _letter(index) for index, subject in enumerate(recording.subjects)}
    frame["speaker"] = [letters.get(subject, subject) for subject in frame["subject_id"]]
    frame["unit_id"] = [f"u{index}" for index in range(len(frame))]
    reached = frame["t_end"].cummax().shift()
    frame["gap_before"] = frame["t_start"] - reached
    return frame[["unit_id", "subject_id", "speaker", "t_start", "t_end", "gap_before", "text"]]


def sample_labels(units: pd.DataFrame, codebook: Codebook, backend: LLMBackend, cfg: LLMSection) -> pd.DataFrame:
    """One row per unit and sample: unit_id, sample, label (a code or abstain)."""
    answers = []
    for sample in range(cfg.n_samples):
        seed = cfg.seed + sample if backend.takes_seed else None
        for start in range(0, len(units), cfg.batch_size):
            batch = units.iloc[start : start + cfg.batch_size]
            context = units.iloc[max(0, start - cfg.context_units) : start]
            system, user = build_annotation_prompt(codebook, batch, context, allow_abstain=cfg.allow_abstain)
            response = backend.complete(system, user, seed=seed)
            parsed = parse_annotation_response(
                response.text, codebook, batch["unit_id"].tolist(), allow_abstain=cfg.allow_abstain
            )
            answers.append(parsed.assign(sample=sample))
    if not answers:
        return pd.DataFrame({"unit_id": pd.Series(dtype=str), "label": pd.Series(dtype=str), "sample": pd.Series(dtype=int)})
    return pd.concat(answers, ignore_index=True)[["unit_id", "sample", "label"]]


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
