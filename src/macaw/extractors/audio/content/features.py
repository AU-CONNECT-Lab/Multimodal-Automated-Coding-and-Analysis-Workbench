from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, InputNotFoundError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import (
    TRANSCRIPT_NAME, ASRExtractor, load_transcript, resolve_device, transcript_between, transcript_saved,
)
from macaw.extractors.audio.content.diarization import OTHER
from macaw.extractors.base import Extractor, ExtractorResult
from macaw.extractors.prep import load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

EMBEDDINGS_NAME = "embeddings.parquet"
TURNS_STEP = "turns.structure"

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
    "questions_per_minute": (
        "Sentences this speaker ended with a question mark, per minute of the analysed part; it rests on the transcript's "
        "punctuation.",
        "1/min",
    ),
}
# a sentence ends where . ! or ? is followed by a space; 3.5 does not end one
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+")
_CLOSING = re.compile(r"[.!?\"')\]\s]*$")

# a row per change of speaker and its mean per pair, subject_id taking the floor from subject_id_b
_OVERLAP_SPECS = {
    "lexical_overlap": (
        "Share of the distinct words in subject_id's turn that subject_id_b used in the turn just before it, the words in "
        "text.exclude_words left out; missing for a turn with no counted word.",
        None, Scale.EVENT,
    ),
    "mean_lexical_overlap": ("Mean lexical_overlap over the turns subject_id took straight after subject_id_b's.", None, Scale.RECORDING),
}

# made only with text.embedding_model; a pair of speakers per row, subject_id following subject_id_b
_EMBEDDING_SPECS = {
    "semantic_following": (
        "Mean cosine similarity of this speaker's utterances to the utterance just before each, where subject_id_b said it: "
        "how closely they follow on from that person. Its level depends on the embedding model.",
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
    version = "0.6.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = (
        "text.embedding_model", "text.device", "text.first_use_min_speakers", "text.exclude_words",
        "text.first_use_words", "text.word_lists", "segment",
    )
    requires = (ASRExtractor.name, TURNS_STEP)

    def check_available(self) -> None:
        for path in self._word_lists():
            if not path.is_file():
                raise InputNotFoundError(f"word list {path} in text.exclude_words or text.first_use_words does not exist")
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
        from macaw.turns.segmentation import TURNS_NAME

        layout = layout_for(self.config, create=False)
        root = layout.artifacts_dir(recording.recording_id, ASRExtractor.name)
        return [root / TRANSCRIPT_NAME, layout.artifacts_dir(recording.recording_id, TURNS_STEP) / TURNS_NAME, *self._word_lists()]

    def _word_lists(self) -> list[Path]:
        text = self.config.text
        return [path for path in (text.exclude_words, text.first_use_words, *text.word_lists.values()) if isinstance(path, Path)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        transcript = load_transcript(self.config, recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)
        window = (prep.t_origin_s, prep.t_origin_s + prep.duration_s)
        from macaw.turns.segmentation import load_turns

        turns = load_turns(self.config, recording.recording_id, others=True)
        frames = [self._summary(recording, transcript, window), self._overlap(recording, turns, window)]
        artifacts = {}
        model = self.config.text.embedding_model
        if model is not None:
            embedded = transcript.loc[:, ["subject_id", "utterance_index", "t_start", "t_end"]].reset_index(drop=True)
            vectors = sentence_embeddings(transcript["text"].fillna("").astype(str).tolist(), model, device=self.config.text.device)
            path = workspace / EMBEDDINGS_NAME
            pd.concat([embedded, vectors], axis=1).to_parquet(path, index=False)
            artifacts[EMBEDDINGS_NAME] = path
            every = load_transcript(self.config, recording.recording_id, others=True)
            frames.append(self._following(recording, window, every, _by_utterance(embedded, vectors)))
        return ExtractorResult(rows=concat_long_tables(frames), specs=self._specs(), artifacts=artifacts)

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        if not transcript_saved(self.config, recording.recording_id):
            return None
        from macaw.turns.segmentation import turns_between

        every, turns = turns_between(self.config, recording.recording_id, start_s, end_s)
        overlap = self._overlap(recording, turns, (start_s, end_s))
        frames = [self._summary(recording, every[every["subject_id"] != OTHER], (start_s, end_s)), overlap[overlap["scale"] == Scale.RECORDING.value]]
        path = layout_for(self.config, create=False).artifacts_dir(recording.recording_id, self.name) / EMBEDDINGS_NAME
        if self.config.text.embedding_model is not None and path.is_file():
            saved = pd.read_parquet(path)
            vectors = _by_utterance(saved[["subject_id", "utterance_index"]], saved.drop(columns=["subject_id", "utterance_index", "t_start", "t_end"]))
            whole = {(str(who), int(index)): (start, end) for who, index, start, end in saved[["subject_id", "utterance_index", "t_start", "t_end"]].itertuples(index=False)}
            # a line cut at an edge is embedded again from the words left inside it
            cut = [row for row in every.itertuples() if (str(row.subject_id), int(row.utterance_index)) in whole
                   and whole[(str(row.subject_id), int(row.utterance_index))] != (row.t_start, row.t_end)]
            if cut:
                again = sentence_embeddings([str(row.text) for row in cut], self.config.text.embedding_model, device=self.config.text.device)
                vectors.update({(str(row.subject_id), int(row.utterance_index)): again.iloc[i].to_numpy(dtype=float) for i, row in enumerate(cut)})
            frames.append(self._following(recording, (start_s, end_s), every, vectors))
        return concat_long_tables(frames)

    def _summary(self, recording: Recording, transcript: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
        transcript = transcript.assign(tokens=transcript["text"].fillna("").astype(str).map(tokenize))
        text = self.config.text
        first = first_use_share(
            transcript,
            min_speakers=text.first_use_min_speakers,
            exclude=excluded_words(text.exclude_words),
            only=None if text.first_use_words is None else read_word_list(text.first_use_words),
        )
        lists = {name: read_phrases(path) for name, path in text.word_lists.items()}
        minutes = (window[1] - window[0]) / 60.0
        frames = []
        for subject_id, spoken in transcript.groupby("subject_id", observed=True):
            tokens = [token for tokens in spoken["tokens"] for token in tokens]
            said = [sentence for line in spoken["text"].fillna("").astype(str) for sentence in sentences(line)]
            values = pd.DataFrame(
                {
                    "t_start": [window[0]],
                    "t_end": [window[1]],
                    "mlu_words": [mlu(spoken)],
                    "n_tokens": [float(len(tokens))],
                    "n_types": [float(len(set(tokens)))],
                    "type_token_ratio": [type_token_ratio(tokens)],
                    "unigram_entropy_bits": [unigram_entropy(tokens)],
                    "first_use_share": [first.get(str(subject_id), np.nan)],
                    "questions_per_minute": [_per_minute(sum(map(is_question, said)), minutes)],
                    **{
                        f"{name}_per_minute": [_per_minute(sum(any_phrase(tokenize(sentence), phrases) for sentence in said), minutes)]
                        for name, phrases in lists.items()
                    },
                }
            )
            frames.append(
                build_long_rows(
                    values=values, scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name,
                    entities=recording.entities(), subject_id=str(subject_id),
                )
            )
        return concat_long_tables(frames)

    def _overlap(self, recording: Recording, turns: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
        found = lexical_overlap(turns, excluded_words(self.config.text.exclude_words))
        found = found[(found["taker"] != OTHER) & (found["giver"] != OTHER)]
        common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities())
        frames = []
        for (taker, giver), mine in found.groupby(["taker", "giver"], sort=True):
            frames.append(
                build_long_rows(
                    values=mine[["t_start", "t_end"]].assign(lexical_overlap=mine["overlap"]).reset_index(drop=True),
                    scale=Scale.EVENT, subject_id=taker, subject_id_b=giver, **common,
                )
            )
            frames.append(
                build_long_rows(
                    values=pd.DataFrame({"t_start": [window[0]], "t_end": [window[1]], "mean_lexical_overlap": [float(mine["overlap"].mean())]}),
                    scale=Scale.RECORDING, subject_id=taker, subject_id_b=giver, **common,
                )
            )
        return concat_long_tables(frames)

    def _following(
        self, recording: Recording, window: tuple[float, float], every: pd.DataFrame, vectors: dict[tuple[str, int], np.ndarray],
    ) -> pd.DataFrame:
        # a line in a voice that is no participant's breaks the chain rather than being skipped over
        every = every.sort_values(["t_start", "t_end"], kind="stable")
        if every.empty or not vectors:
            return concat_long_tables([])
        missing = np.full(len(next(iter(vectors.values()))), np.nan)
        keys = [(str(who), int(index)) for who, index in zip(every["subject_id"], every["utterance_index"])]
        ordered = np.array([vectors.get(key, missing) for key in keys])
        span = {"t_start": [window[0]], "t_end": [window[1]]}
        return concat_long_tables([
            build_long_rows(
                values=pd.DataFrame({**span, "semantic_following": [value]}), scale=Scale.RECORDING,
                recording_id=recording.recording_id, source=self.name, entities=recording.entities(), subject_id=follower,
                subject_id_b=followed,
            )
            for (follower, followed), value in sorted(semantic_following([who for who, _ in keys], ordered).items())
        ])

    def _specs(self) -> list[FeatureSpec]:
        listed = {
            f"{name}_per_minute": (
                f"Sentences of this speaker holding a word or phrase of the word list {path.name} (text.word_lists {name}), per "
                "minute of the analysed part.",
                "1/min", Scale.RECORDING,
            )
            for name, path in self.config.text.word_lists.items()
        }
        described = {
            **{name: (description, unit, Scale.RECORDING) for name, (description, unit) in _SPECS.items()},
            **listed,
            **_OVERLAP_SPECS,
            **{name: (description, unit, Scale.RECORDING) for name, (description, unit) in _EMBEDDING_SPECS.items() if self.config.text.embedding_model},
        }
        return [
            FeatureSpec(
                name=name,
                description=description,
                unit=unit,
                scale=scale,
                group=self.group,
                extractor=self.name,
                extractor_version=self.version,
                params={
                    "tokenizer": "english_words",
                    "embedding_model": self.config.text.embedding_model,
                    "first_use_min_speakers": self.config.text.first_use_min_speakers,
                    "exclude_words": _list_name(self.config.text.exclude_words),
                    "first_use_words": _list_name(self.config.text.first_use_words),
                    "word_lists": {name: path.name for name, path in self.config.text.word_lists.items()},
                },
                validation="synthetic",
            )
            for name, (description, unit, scale) in described.items()
        ]


def _by_utterance(keys: pd.DataFrame, vectors: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {
        (str(who), int(index)): vectors.iloc[row].to_numpy(dtype=float)
        for row, (who, index) in enumerate(zip(keys["subject_id"], keys["utterance_index"]))
    }


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


def sentences(text: str) -> list[str]:
    """A line split after each . ! or ? that a space follows; a line without one is a sentence of its own."""
    return [part for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def is_question(sentence: str) -> bool:
    return "?" in _CLOSING.search(sentence).group(0)


def read_phrases(path: Path) -> list[tuple[str, ...]]:
    """One word or phrase per line, # starts a comment; each is tokenized like the transcript."""
    phrases = (tuple(tokenize(line.split("#", 1)[0])) for line in Path(path).read_text(encoding="utf-8").splitlines())
    return sorted({phrase for phrase in phrases if phrase})


def any_phrase(tokens: list[str], phrases: list[tuple[str, ...]]) -> bool:
    """Whether the tokens hold one of the phrases, its words in a row."""
    words = set(tokens)
    for phrase in phrases:
        if len(phrase) == 1:
            if phrase[0] in words:
                return True
        elif any(tuple(tokens[start : start + len(phrase)]) == phrase for start in range(len(tokens) - len(phrase) + 1)):
            return True
    return False


def _per_minute(count: int, minutes: float) -> float:
    return float(count) / minutes if minutes > 0 else np.nan


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


def lexical_overlap(turns: pd.DataFrame, exclude: frozenset[str]) -> pd.DataFrame:
    """A row per turn taken straight after someone else's: taker, giver, the taker's t_start and t_end, and overlap,
    the share of the taker's distinct counted words found in the giver's turn."""
    columns = ["taker", "giver", "t_start", "t_end", "overlap"]
    turns = turns.sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    words = [set(tokenize(str(text))) - exclude for text in turns["text"].fillna("")]
    speakers = turns["subject_id"].astype(str).tolist()
    rows = [
        (speakers[i], speakers[i - 1], float(turns.at[i, "t_start"]), float(turns.at[i, "t_end"]),
         len(words[i] & words[i - 1]) / len(words[i]) if words[i] else np.nan)
        for i in range(1, len(turns))
        if speakers[i] != speakers[i - 1]
    ]
    return pd.DataFrame(rows, columns=columns)


def semantic_following(speakers: list[str], vectors: np.ndarray) -> dict[tuple[str, str], float]:
    """Per (follower, followed), the mean cosine similarity of a line to the line just before it by someone else; lines in time order, NaN for one not measured."""
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    found: dict[tuple[str, str], list[float]] = {}
    for index in range(1, len(speakers)):
        if speakers[index] == speakers[index - 1] or not (np.isfinite(unit[index]).all() and np.isfinite(unit[index - 1]).all()):
            continue
        found.setdefault((speakers[index], speakers[index - 1]), []).append(float(unit[index] @ unit[index - 1]))
    return {pair: float(np.mean(values)) for pair, values in found.items()}


def sentence_embeddings(texts: list[str], model_name: str, *, device: str = "auto") -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=resolve_device(device))
    width = model.get_sentence_embedding_dimension()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False) if texts else np.empty((0, width))
    return pd.DataFrame(np.asarray(vectors, dtype=np.float32), columns=[f"e{index}" for index in range(width)])
