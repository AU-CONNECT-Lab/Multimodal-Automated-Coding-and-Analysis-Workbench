from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, InputNotFoundError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, load_transcript, resolve_device
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.extractors.prep import load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

EMBEDDINGS_NAME = "embeddings.parquet"

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")

_SPECS = {
    "mlu_words": ("Mean length of utterance in words; an utterance is one recogniser segment.", "word"),
    "n_tokens": ("Words this speaker produced, after lower-casing and dropping punctuation.", "count"),
    "n_types": ("Distinct words this speaker used.", "count"),
    "type_token_ratio": ("Distinct words over all words; it falls as a speaker talks more, so compare only equal amounts of talk.", None),
    "unigram_entropy_bits": ("Shannon entropy of this speaker's word frequencies; it also grows with the amount of talk.", "bit"),
    "first_use_share": (
        "Of the counted words at least text.first_use_min_speakers speakers used, the share this speaker said first; "
        "who brings in the shared vocabulary.",
        None,
    ),
}

STOPWORDS = "english_stopwords"

_INSTALL = (
    "text.embedding_model is set, which needs sentence-transformers. Install PyTorch for your "
    "CUDA version first, then: pip install sentence-transformers"
)


@register_extractor
class TextFeatureExtractor(Extractor):
    name = "text.features"
    version = "0.3.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = (
        "text.embedding_model", "text.device", "text.first_use_min_speakers", "text.first_use_exclude",
        "text.first_use_words", "segment",
    )
    requires = (ASRExtractor.name,)

    def check_available(self) -> None:
        for path in self._word_lists():
            if not path.is_file():
                raise InputNotFoundError(f"word list {path} in text.first_use_exclude or text.first_use_words does not exist")
        if self.config.text.embedding_model is None:
            return
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(_INSTALL) from error

    def backend_info(self) -> tuple[str | None, str | None]:
        if self.config.text.embedding_model is None:
            return (None, None)
        import sentence_transformers

        return ("sentence-transformers", sentence_transformers.__version__)

    def prefetch(self) -> None:
        if self.config.text.embedding_model is not None:
            sentence_embeddings([], self.config.text.embedding_model, device=self.config.text.device)

    def inputs_for(self, recording: Recording) -> list[Path]:
        root = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, ASRExtractor.name)
        return [root / TRANSCRIPT_NAME, *self._word_lists()]

    def _word_lists(self) -> list[Path]:
        text = self.config.text
        return [path for path in (text.first_use_exclude, text.first_use_words) if isinstance(path, Path)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        transcript = load_transcript(self.config, recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)
        transcript = transcript.assign(tokens=transcript["text"].fillna("").astype(str).map(tokenize))
        text = self.config.text
        first = first_use_share(
            transcript,
            min_speakers=text.first_use_min_speakers,
            exclude=excluded_words(text.first_use_exclude),
            only=None if text.first_use_words is None else read_word_list(text.first_use_words),
        )
        frames = []
        for subject_id, spoken in transcript.groupby("subject_id", observed=True):
            tokens = [token for tokens in spoken["tokens"] for token in tokens]
            values = pd.DataFrame(
                {
                    "t_start": [prep.t_origin_s],
                    "t_end": [prep.t_origin_s + prep.duration_s],
                    "mlu_words": [mlu(spoken)],
                    "n_tokens": [float(len(tokens))],
                    "n_types": [float(len(set(tokens)))],
                    "type_token_ratio": [type_token_ratio(tokens)],
                    "unigram_entropy_bits": [unigram_entropy(tokens)],
                    "first_use_share": [first.get(str(subject_id), np.nan)],
                }
            )
            frames.append(
                build_long_rows(
                    values=values, scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=str(subject_id),
                )
            )
        artifacts = {}
        model = self.config.text.embedding_model
        if model is not None:
            embedded = transcript.loc[:, ["subject_id", "utterance_index", "t_start", "t_end"]].reset_index(drop=True)
            vectors = sentence_embeddings(transcript["text"].fillna("").astype(str).tolist(), model, device=self.config.text.device)
            path = workspace / EMBEDDINGS_NAME
            pd.concat([embedded, vectors], axis=1).to_parquet(path, index=False)
            artifacts[EMBEDDINGS_NAME] = path
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(), artifacts=artifacts)

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
                params={
                    "tokenizer": "english_words",
                    "embedding_model": self.config.text.embedding_model,
                    "first_use_min_speakers": self.config.text.first_use_min_speakers,
                    "first_use_exclude": _list_name(self.config.text.first_use_exclude),
                    "first_use_words": _list_name(self.config.text.first_use_words),
                },
                validation="synthetic",
            )
            for name, (description, unit) in _SPECS.items()
        ]


def tokenize(text: str) -> list[str]:
    """English words, lower-cased, apostrophes kept inside a word; the one place another language plugs in."""
    return _WORD.findall(text.lower())


def mlu(utterances: pd.DataFrame) -> float:
    lengths = [len(tokens) for tokens in utterances["tokens"]]
    return float(np.mean(lengths)) if lengths else np.nan


def type_token_ratio(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else np.nan


def unigram_entropy(tokens: list[str]) -> float:
    if not tokens:
        return np.nan
    counts = np.array(list(pd.Series(tokens).value_counts()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def read_word_list(path: Path) -> frozenset[str]:
    return word_list(path.read_text(encoding="utf-8"))


def word_list(text: str) -> frozenset[str]:
    """One word per line, # starts a comment; each line is tokenized like the transcript."""
    return frozenset(token for line in text.splitlines() for token in tokenize(line.split("#", 1)[0]))


def excluded_words(setting: str | Path) -> frozenset[str]:
    if setting == "none":
        return frozenset()
    if setting == STOPWORDS:
        return word_list(files("macaw.extractors.audio.content").joinpath("english_stopwords.txt").read_text(encoding="utf-8"))
    return read_word_list(Path(setting))


def _list_name(setting: str | Path | None) -> str | None:
    return setting.name if isinstance(setting, Path) else setting


def first_use_share(
    transcript: pd.DataFrame,
    *,
    min_speakers: int = 2,
    exclude: frozenset[str] = frozenset(),
    only: frozenset[str] | None = None,
) -> dict[str, float]:
    """Utterances with a tokens column; a word counts once min_speakers speakers have used it."""
    first: dict[str, str] = {}
    users: dict[str, set[str]] = {}
    for subject_id, tokens in transcript.sort_values(["t_start", "t_end"])[["subject_id", "tokens"]].itertuples(index=False):
        for token in tokens:
            if token in exclude or (only is not None and token not in only):
                continue
            first.setdefault(token, str(subject_id))
            users.setdefault(token, set()).add(str(subject_id))
    shared = [token for token, speakers in users.items() if len(speakers) >= min_speakers]
    speakers = {str(subject) for subject in transcript["subject_id"].unique()}
    if not shared:
        return {subject: np.nan for subject in speakers}
    introduced = pd.Series([first[token] for token in shared]).value_counts()
    return {subject: float(introduced.get(subject, 0)) / len(shared) for subject in speakers}


def sentence_embeddings(texts: list[str], model_name: str, *, device: str = "auto") -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=resolve_device(device))
    width = model.get_sentence_embedding_dimension()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False) if texts else np.empty((0, width))
    return pd.DataFrame(np.asarray(vectors, dtype=np.float32), columns=[f"e{index}" for index in range(width)])
