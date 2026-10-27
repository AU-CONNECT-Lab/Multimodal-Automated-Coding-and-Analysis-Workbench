"""Syntax per speaker from a dependency parse of each utterance. Readability scores are left out:
they rest on sentence boundaries, which in speech are wherever the recogniser cut."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, load_transcript
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

POS_TAGS = ("NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "INTJ", "NUM")
CONTENT_TAGS = frozenset({"NOUN", "PROPN", "VERB", "ADJ", "ADV"})
# a clause hung off another one: complements, adverbial and relative clauses, clausal subjects
SUBORDINATE_RELATIONS = frozenset({"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "csubjpass"})
_SKIPPED = frozenset({"PUNCT", "SPACE", "SYM", "X"})

_SPECS = {
    "n_utterances_parsed": ("Utterances with at least one word, each parsed on its own.", "count"),
    "dependency_distance_mean": (
        "Mean dependency distance: per utterance, the mean number of words between a word and its head, then the mean "
        "over utterances; longer means harder to process.",
        "word",
    ),
    "adjacent_dependency_share": ("Share of dependencies between neighbouring words.", None),
    "subordinate_clauses_per_utterance": (
        "Clausal complements, adverbial, relative and clausal-subject clauses per utterance.", "count",
    ),
    "lexical_density": ("Nouns, proper nouns, verbs, adjectives and adverbs over all words.", None),
    **{f"pos_share_{tag.lower()}": (f"Share of words tagged {tag} (Universal Dependencies).", None) for tag in POS_TAGS},
}

_INSTALL = "text.syntax needs spaCy and a parser for the language: pip install spacy, then python -m spacy download {model}"


@register_extractor
class SyntaxExtractor(Extractor):
    name = "text.syntax"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = ("text.syntax_model", "segment")
    requires = (ASRExtractor.name,)

    def check_available(self) -> None:
        model = self.config.text.syntax_model
        try:
            import spacy
        except ImportError as error:
            raise BackendNotAvailableError(_INSTALL.format(model=model)) from error
        if not (spacy.util.is_package(model) or Path(model).is_dir()):
            raise BackendNotAvailableError(_INSTALL.format(model=model))

    def backend_info(self) -> tuple[str | None, str | None]:
        import spacy

        return ("spacy", spacy.__version__)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [layout_for(self.config, create=False).artifacts_dir(recording.recording_id, ASRExtractor.name) / TRANSCRIPT_NAME]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        import spacy

        nlp = spacy.load(self.config.text.syntax_model, exclude=["ner"])
        transcript = load_transcript(self.config, recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)
        texts = transcript["text"].fillna("").astype(str).tolist()
        parsed = [utterance_syntax(doc) for doc in nlp.pipe(texts)]
        transcript = transcript.reset_index(drop=True).assign(parsed=parsed)
        frames = []
        for subject_id, spoken in transcript.groupby("subject_id", observed=True):
            values = {"t_start": [prep.t_origin_s], "t_end": [prep.t_origin_s + prep.duration_s]}
            values.update({name: [value] for name, value in speaker_syntax([item for item in spoken["parsed"] if item]).items()})
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(values), scale=Scale.RECORDING, recording_id=recording.recording_id,
                    source=self.name, entities=recording.entities(), subject_id=str(subject_id),
                )
            )
        language = self.config.text.language
        warnings = [
            QCWarning(
                check="syntax_parser", recording_id=recording.recording_id, subject_id=None, severity="info",
                message="the parser is trained on written adult text; read the syntax of children's or disfluent speech with care",
            )
        ]
        if language and nlp.lang != language:
            warnings.append(
                QCWarning(
                    check="syntax_parser", recording_id=recording.recording_id, subject_id=None, severity="warn",
                    message=f"text.language is {language} but the parser {self.config.text.syntax_model} reads {nlp.lang}",
                )
            )
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(nlp), warnings=warnings)

    def _specs(self, nlp) -> list[FeatureSpec]:
        model = f"{nlp.meta.get('lang', '')}_{nlp.meta.get('name', '')}"
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=Scale.RECORDING, group=self.group,
                extractor=self.name, extractor_version=self.version,
                params={"syntax_model": model, "syntax_model_version": nlp.meta.get("version")},
                validation="synthetic",
            )
            for name, (description, unit) in _SPECS.items()
        ]


def utterance_syntax(doc) -> dict | None:
    """Counts for one parsed utterance, None when it has no words."""
    words = [token for token in doc if token.pos_ not in _SKIPPED]
    if not words:
        return None
    distances = [abs(token.i - token.head.i) for token in words if token.head.i != token.i and token.dep_ != "ROOT"]
    return {
        "n_words": len(words),
        "distance_mean": float(np.mean(distances)) if distances else np.nan,
        "n_dependencies": len(distances),
        "n_adjacent": sum(1 for distance in distances if distance == 1),
        "n_subordinate": sum(1 for token in words if token.dep_ in SUBORDINATE_RELATIONS),
        "pos": pd.Series([token.pos_ for token in words]).value_counts().to_dict(),
    }


def speaker_syntax(parsed: list[dict]) -> dict[str, float]:
    if not parsed:
        return {name: (0.0 if name == "n_utterances_parsed" else np.nan) for name in _SPECS}
    n_words = sum(item["n_words"] for item in parsed)
    n_dependencies = sum(item["n_dependencies"] for item in parsed)
    tags: dict[str, int] = {}
    for item in parsed:
        for tag, count in item["pos"].items():
            tags[tag] = tags.get(tag, 0) + count
    return {
        "n_utterances_parsed": float(len(parsed)),
        "dependency_distance_mean": float(np.nanmean([item["distance_mean"] for item in parsed]))
        if any(np.isfinite(item["distance_mean"]) for item in parsed) else np.nan,
        "adjacent_dependency_share": sum(item["n_adjacent"] for item in parsed) / n_dependencies if n_dependencies else np.nan,
        "subordinate_clauses_per_utterance": sum(item["n_subordinate"] for item in parsed) / len(parsed),
        "lexical_density": sum(tags.get(tag, 0) for tag in CONTENT_TAGS) / n_words,
        **{f"pos_share_{tag.lower()}": tags.get(tag, 0) / n_words for tag in POS_TAGS},
    }
