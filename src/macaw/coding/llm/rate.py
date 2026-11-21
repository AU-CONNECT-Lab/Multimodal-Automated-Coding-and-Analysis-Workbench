"""Whole recordings, or named segments of them, rated on scales by a language model: one scale per request, asked
several times, the median answer kept as the score."""

from __future__ import annotations

import json
import re
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from macaw.coding.llm.backends import (
    KeptAnswers, LLMBackend, concurrency_params, request_key, resolve_llm_backend, run_in_order, seed_caveat,
)
from macaw.coding.llm.evidence import EVIDENCE_NAME, evidence_table, evidence_warnings
from macaw.coding.llm.judge import prosody_rows_file, voice_figures
from macaw.coding.llm.notes import NOTE_KEYS, given_note, notes_by_key, notes_params
from macaw.coding.llm.parsing import parse_rating
from macaw.coding.llm.prompts import VOICE, build_rating_prompt
from macaw.coding.scales import RatingScale, ScaleExample, ScaleSet, read_examples, read_scales, validate_scales
from macaw.config.hashing import canonical
from macaw.config.schema import RUN_ONLY_KEYS, LLMSection, MacawConfig, RatingVersion
from macaw.exceptions import CodebookError, ConfigError, InputNotFoundError, LLMBackendError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import (
    CORRECTED_CHECK, TRANSCRIPT_NAME, WORDS_NAME, ASRExtractor, cut_transcript, read_transcript, read_words,
)
from macaw.extractors.audio.content.diarization import OTHER, SHARED_MODE
from macaw.extractors.audio.prosody.frame import FrameProsodyExtractor
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, load_prep
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.project.segments import segments_of
from macaw.store.longtable import build_long_rows
from macaw.store.schema import Scale
from macaw.turns.talk import clip_talk, talk_inputs, talk_spans, voice_gaps, voices_time_talk

RATE_STEP = "coding.rate."
RATINGS_NAME = "ratings.parquet"
ANSWERS_NAME = "answers.parquet"
PROMPTS_NAME = "prompts.parquet"
# the segment column of a rating of the whole analysed window
WHOLE = ""
CUT_OFF = "cut off"
_EDGE_S = 1e-3
# an answer that lists its evidence first runs longer
_EVIDENCE_ROOM = 1024
# words in Latin letters; any other character but a space counts alone, as in scripts written without spaces
_DIGIT, _WORD, _SYMBOL = re.compile(r"\d"), re.compile(r"[A-Za-zÀ-ɏ]+"), re.compile(r"[^\sA-Za-zÀ-ɏ\d]")

RATING_COLUMNS = [
    "recording_id", "segment", "t_start", "t_end", "scale", "person", "rater", "score", "score_sd", "score_min", "score_max",
    "n_answers", "n_asked", "not_rated",
]
ANSWER_COLUMNS = [
    "recording_id", "segment", "scale", "rater", "sample", "score", "problem", "reason", "answer", "prompt_tokens",
    "completion_tokens", "reasoning_tokens", "seconds", "n_evidence", "evidence_problem",
]
KEPT_NAME = "answers.jsonl"
_COUNTS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")
PROMPT_COLUMNS = ["recording_id", "segment", "scale", "rater", "system", "user"]
_MISSING = (
    "Empty for a recording where this rating asked the model nothing: its transcript is recognised and no person corrected "
    "this one, nothing was said in the parts it rates, or none of the segments it names lies in the analysed part. 0 means "
    "every answer gave a whole score in the scale's range."
)


class Prompt(NamedTuple):
    """One scale asked about one part: the text sent, and the numbered lines it shows."""

    segment: str
    start: float
    end: float
    scale: RatingScale
    system: str
    user: str
    lines: pd.DataFrame


class RatingExtractor(Extractor):
    """Never registered itself: each version under coding.ratings is a step of its own."""

    name = "coding.rate"
    version = "0.2.0"
    scale = Scale.RECORDING
    group = "coding"
    requires = (ASRExtractor.name, MediaPrepExtractor.name)
    rating: str = ""

    @classmethod
    def settings(cls, config: MacawConfig) -> tuple[RatingVersion, LLMSection]:
        if cls.rating not in config.coding.ratings:
            raise ConfigError(f"coding.ratings has no version {cls.rating}")
        return config.coding.rating_settings(cls.rating)

    @classmethod
    def wanted_by_default(cls, config: MacawConfig) -> bool:
        return cls.rating in config.coding.ratings

    @classmethod
    def dependencies(cls, config: MacawConfig) -> tuple[str, ...]:
        shown = set(cls.settings(config)[1].show) if cls.rating in config.coding.ratings else set()
        return cls.requires + ((FrameProsodyExtractor.name,) if {"pitch", "loudness"} & shown else ())

    @classmethod
    def resources(cls, config: MacawConfig) -> str:
        return "gpu_standard" if cls.settings(config)[1].is_local() else cls.resource_profile

    def check_available(self) -> None:
        own, cfg = self.settings(self.config)
        scales = self.scale_set()
        self.examples()
        if own.excerpts is not None:
            read_excerpts(own.excerpts, scales)
        resolve_llm_backend(cfg).check_available()

    def examples(self) -> dict[str, list[ScaleExample]]:
        """The examples people rated, by scale; none without an examples file."""
        own, _ = self.settings(self.config)
        if own.examples is None:
            return {}
        return read_examples(own.examples, self.scale_set(), [*self.config.project.roles, OTHER])

    def marked(self, recording_id: str) -> dict[str, list[tuple[float, float]]] | None:
        """Each scale's stretches marked for this recording in the excerpts table; None without one."""
        own, _ = self.settings(self.config)
        if own.excerpts is None:
            return None
        rows = read_excerpts(own.excerpts, self.scale_set())
        rows = rows[rows["recording"] == recording_id].sort_values(["scale", "start_s", "end_s"])
        return {scale: list(zip(group["start_s"], group["end_s"])) for scale, group in rows.groupby("scale", sort=True)}

    def scale_set(self) -> ScaleSet:
        own, _ = self.settings(self.config)
        scales = read_scales(own.scales)
        problems = validate_scales(scales, list(self.config.project.roles))
        if problems:
            raise CodebookError(f"{own.scales}: " + "; ".join(problems))
        return scales

    def inputs_for(self, recording: Recording) -> list[Path]:
        own, cfg = self.settings(self.config)
        rid = recording.recording_id
        layout = layout_for(self.config, create=False)
        inputs = [*(transcript_files(self.config, rid, own.transcript) or ()), layout.artifacts_dir(rid, MediaPrepExtractor.name) / PREP_INFO_NAME]
        if own.transcript == "recognised" and layout.corrected_transcript_path(rid).is_file():
            inputs.append(layout.corrected_transcript_path(rid))
        voice = prosody_rows_file(self.config, rid)
        if {"pitch", "loudness"} & set(cfg.show) and voice.is_file():
            inputs.append(voice)
        return inputs + talk_inputs(self.config, rid, by_voices=self.timed_by_voices(rid))

    def timed_by_voices(self, recording_id: str) -> bool:
        """Whether the pause or overlap before each line comes from the voices: shown, on a shared microphone, for the
        recogniser's lines."""
        own, cfg = self.settings(self.config)
        return "timing" in cfg.show and self.config.text.diarization == SHARED_MODE and (
            own.transcript == "recognised" or voices_time_talk(self.config, recording_id)
        )

    @classmethod
    def notes_settings(cls, config: MacawConfig) -> tuple[str, LLMSection]:
        """Where its notes_by comes from, and the model settings that hold it."""
        return notes_by_key(config, "ratings", cls.rating), cls.settings(config)[1]

    def prompt_note(self, recording: Recording) -> str:
        where, cfg = self.notes_settings(self.config)
        return given_note(cfg, recording.entities(), where)

    def extra_key(self, recording: Recording) -> str:
        """The note this recording's prompts hold, and where its segments lie."""
        own, _ = self.settings(self.config)
        named = {segment.name: segment for segment in segments_of(self.config, recording.recording_id)}
        segments = [[name, named[name].start_s, named[name].end_s] for name in own.segments if name in named]
        key = {"note": self.prompt_note(recording), "segments": segments}
        if self.timed_by_voices(recording.recording_id):
            # the pause shown before each line is cut from talk timed by the voices
            key["min_voice_s"] = self.config.turns.min_voice_s
        marked = self.marked(recording.recording_id)
        # only this recording's stretches, so marking another recording rates only that one again
        if marked is not None:
            key["excerpts"] = {scale: [list(span) for span in spans] for scale, spans in marked.items()}
        return canonical(key)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = resolve_llm_backend(self.settings(self.config)[1])
        return backend.name, backend.model_version()

    def backends(self) -> list[Backend]:
        return [resolve_llm_backend(self.settings(self.config)[1])]

    def summarise(self, recording: Recording, start_s: float, end_s: float) -> pd.DataFrame | None:
        # a segment is rated only when a version names it, since each asks the model again
        return pd.DataFrame()

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        own, cfg = self.settings(self.config)
        scales = self.scale_set()
        backend = resolve_llm_backend(cfg)
        rid = recording.recording_id
        warnings: list[QCWarning] = []

        def note(check: str, severity: str, message: str) -> None:
            warnings.append(QCWarning(check=check, recording_id=rid, subject_id=None, severity=severity, message=message))

        # (segment, start, end, scales, why) for what was not asked, so each shows as a row without a score
        skipped: list[tuple] = []
        every = [scale.scale for scale in scales.scales]
        if backend.reporting:
            note("rating_cloud", "info", rating_notice(self.name, backend))
        files = transcript_files(self.config, rid, own.transcript)
        parts = [] if files is None else self._parts(recording, note, skipped, every)
        if files is None:
            note("rating_skipped", "info", "no person corrected this transcript, so the recogniser's is the one in use, which the other ratings read")
            skipped.append((WHOLE, np.nan, np.nan, every, "no person corrected the transcript"))
        elif not files[0].is_file():
            raise InputNotFoundError(
                f"{rid}: the recogniser's transcript was not kept when the correction came in. Run macaw transcript reset -r {rid}, "
                f"run text.asr again, then import the corrected file again"
            )
        prompts = []
        told = self.prompt_note(recording)
        examples = self.examples()
        marked = self.marked(rid)
        if parts:
            transcript = read_transcript(files[0])
            lines = rating_lines(transcript)
            words = read_words(files[1]) if files[1].is_file() else pd.DataFrame()
            talk = talk_spans(self.config, rid, transcript, words, by_voices=True)[0] if self.timed_by_voices(rid) else None
            if set(VOICE) & set(cfg.show):
                lines = voice_figures(self.config, rid, lines, cfg.show)
            people = [subject.subject_id for subject in recording.subjects]
            for segment, start, end in parts:
                shown = part_lines(lines, words, start, end, talk)
                if shown.empty:
                    note("rating_skipped", "info", f"nothing was said in {segment or 'the analysed window'}, so it was not rated")
                    skipped.append((segment, start, end, every, "nothing was said"))
                    continue
                unmarked = []
                for scale in scales.scales:
                    asked = shown if marked is None else excerpt_lines(shown, marked.get(scale.scale, []), start)
                    if asked.empty:
                        unmarked.append(scale.scale)
                        continue
                    system, user = build_rating_prompt(
                        scale, asked, people=people, minutes=(end - start) / 60.0, show=cfg.show, notes=told, reasons=cfg.reasons,
                        evidence=own.evidence, examples=examples.get(scale.scale, ()), excerpts=marked is not None,
                    )
                    prompts.append(Prompt(segment, start, end, scale, system, user, asked))
                if unmarked:
                    skipped.append((segment, start, end, unmarked, "the excerpts table marks no stretch with a line in it"))
                    note(
                        "rating_skipped", "info",
                        f"the excerpts table marks no stretch with a line in it for {', '.join(unmarked)} in this recording, so "
                        f"{'that scale was' if len(unmarked) == 1 else 'those scales were'} not rated",
                    )
        kept = layout_for(self.config, create=False).unfinished_path(rid, self.name) / KEPT_NAME
        answers = ask(prompts, backend, cfg, kept, room=_EVIDENCE_ROOM if own.evidence else 0)
        answers.insert(0, "recording_id", rid)
        answers.insert(3, "rater", self.rating)
        ratings = rating_table(answers, prompts, self.rating, recording, skipped, scales)
        warnings += answer_warnings(answers, ratings, cfg, backend, rid)

        paths = {name: workspace / name for name in (RATINGS_NAME, ANSWERS_NAME, PROMPTS_NAME)}
        if own.evidence:
            evidence, answers = evidence_table(answers, prompts)
            warnings += evidence_warnings(evidence, answers, rid)
            paths[EVIDENCE_NAME] = workspace / EVIDENCE_NAME
            evidence.to_parquet(paths[EVIDENCE_NAME], index=False)
        else:
            answers = answers.assign(n_evidence=pd.array([pd.NA] * len(answers), dtype="Int64"), evidence_problem=pd.NA)
        ratings.to_parquet(paths[RATINGS_NAME], index=False)
        answers[ANSWER_COLUMNS].to_parquet(paths[ANSWERS_NAME], index=False)
        prompt_table(prompts, self.rating, rid).to_parquet(paths[PROMPTS_NAME], index=False)
        return ExtractorResult(
            rows=self._rows(recording, answers, parts), specs=self._specs(scales, recording), artifacts=paths, warnings=warnings,
        )

    def _parts(self, recording: Recording, note, skipped: list, every: list[str]) -> list[tuple[str, float, float]]:
        """(segment, start, end) to rate: the analysed window, or each named segment inside it."""
        own, _ = self.settings(self.config)
        prep = load_prep(self.config, recording.recording_id)
        low, high = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        if not own.segments:
            return [(WHOLE, low, high)]
        named = {segment.name: segment for segment in segments_of(self.config, recording.recording_id)}
        parts = []
        for name in own.segments:
            segment = named.get(name)
            if segment is None:
                note("rating_skipped", "info", f"this recording has no segment {name}, so it was not rated there")
                skipped.append((name, np.nan, np.nan, every, "the recording has no such segment"))
            elif segment.start_s < low - _EDGE_S or segment.end_s > high + _EDGE_S:
                skipped.append((name, segment.start_s, segment.end_s, every, "the segment reaches outside the analysed part"))
                note(
                    "rating_skipped", "warn",
                    f"segment {name} ({segment.start_s:g} to {segment.end_s:g} s) reaches outside the analysed part of the recording "
                    f"({low:g} to {high:g} s), so it was not rated",
                )
            else:
                parts.append((name, segment.start_s, segment.end_s))
        return parts

    def _rows(self, recording: Recording, answers: pd.DataFrame, parts) -> pd.DataFrame:
        if answers.empty:
            return pd.DataFrame()
        unread = float((answers["problem"] != "").mean())
        span = (min(part[1] for part in parts), max(part[2] for part in parts))
        return build_long_rows(
            values=pd.DataFrame({"t_start": [span[0]], "t_end": [span[1]], f"share_unread_answers_{self.rating}": [unread]}),
            scale=Scale.RECORDING, recording_id=recording.recording_id, source=self.name, entities=recording.entities(),
        )

    def _specs(self, scales: ScaleSet, recording: Recording) -> list[FeatureSpec]:
        own, cfg = self.settings(self.config)
        params = {
            "rating": self.rating, "scales": scales.name, "n_scales": len(scales.scales), "names_only": scales.names_only(),
            "provider": cfg.provider, "model": cfg.model, "n_samples": cfg.n_samples, "temperature": cfg.temperature,
            "seed": cfg.seed, "show": list(cfg.show), "notes": cfg.notes.strip(), **notes_params(cfg, recording),
            "reasons": cfg.reasons, "think": cfg.think, "transcript": own.transcript, "segments": list(own.segments), "evidence": own.evidence,
            **_example_params(own, self.examples()), **({"excerpts": Path(own.excerpts).name} if own.excerpts is not None else {}),
            **concurrency_params(cfg),
        }
        return [
            FeatureSpec(
                name=f"share_unread_answers_{self.rating}",
                description="Share of the model's answers in this rating that gave no whole score in the scale's range; those scales' scores are the median of the rest.",
                unit="proportion", scale=Scale.RECORDING, group=self.group, extractor=self.name, extractor_version=self.version,
                params=params, aggregation="none", missing=_MISSING, validation="none",
            )
        ]


def transcript_files(config, recording_id: str, source: str) -> tuple[Path, Path] | None:
    """(transcript, words) a rating reads; None for the recogniser's where nobody corrected it, as it is then the one in use."""
    layout = layout_for(config, create=False)
    if source == "in_use":
        folder = layout.artifacts_dir(recording_id, ASRExtractor.name)
        return folder / TRANSCRIPT_NAME, folder / WORDS_NAME
    if not layout.corrected_transcript_path(recording_id).is_file():
        return None
    return layout.recognised_transcript_path(recording_id), layout.recognised_words_path(recording_id)


def keep_recognised(config, recording_id: str) -> bool:
    """Keeps text.asr's transcript beside the correction about to replace it, when it is still the recogniser's."""
    layout = layout_for(config, create=False)
    folder = layout.artifacts_dir(recording_id, ASRExtractor.name)
    if layout.corrected_transcript_path(recording_id).is_file() or not (folder / TRANSCRIPT_NAME).is_file():
        return False
    said = layout.warnings_path(recording_id, ASRExtractor.name)
    # a run that read a correction since reset left no recogniser's transcript behind
    if said.is_file() and any(item.get("check") == CORRECTED_CHECK for item in json.loads(said.read_text(encoding="utf-8"))):
        return False
    target = layout.recognised_transcript_path(recording_id)
    target.write_bytes((folder / TRANSCRIPT_NAME).read_bytes())
    if (folder / WORDS_NAME).is_file():
        layout.recognised_words_path(recording_id).write_bytes((folder / WORDS_NAME).read_bytes())
    return True


def rating_lines(transcript: pd.DataFrame) -> pd.DataFrame:
    """Everyone's utterances in time order, outside voices too, each named by its role."""
    frame = transcript.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["text"] = frame["text"].astype("string").fillna("").str.strip()
    frame = frame[frame["text"] != ""].sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    frame["speaker"] = frame["subject_id"]
    return frame


def part_lines(lines: pd.DataFrame, words: pd.DataFrame, start: float, end: float, talk: pd.DataFrame | None = None) -> pd.DataFrame:
    """The lines within start to end, cut at its edges, timed from start, with the pause or overlap before each: by the
    lines themselves, or with talk by the voices they were said in."""
    columns = ["n_words"] if "n_words" not in lines.columns else []
    source = lines.assign(n_words=lines["text"].str.split().str.len()) if columns else lines
    cut = cut_transcript(source, words, start, end)
    cut = cut[cut["text"].astype(str).str.strip() != ""].drop(columns=columns).reset_index(drop=True)
    cut["line"] = np.arange(1, len(cut) + 1)
    gaps = voice_gaps(cut, clip_talk(talk, start, end)) if talk is not None else cut["t_start"] - cut["t_end"].cummax().shift()
    cut["t_start"] = cut["t_start"] - start
    cut["t_end"] = cut["t_end"] - start
    cut["gap_before"] = gaps
    return cut


def excerpt_lines(shown: pd.DataFrame, spans: list[tuple[float, float]], start: float) -> pd.DataFrame:
    """The lines of shown, timed from start, that overlap one of spans on the recording clock, whole and numbered again;
    opens marks the first line of each run of lines that followed one another."""
    keep = np.zeros(len(shown), dtype=bool)
    for low, high in spans:
        keep |= (shown["t_end"].to_numpy(dtype=float) >= low - start) & (shown["t_start"].to_numpy(dtype=float) <= high - start)
    position = np.flatnonzero(keep)
    kept = shown.iloc[position].reset_index(drop=True)
    return kept.assign(opens=np.r_[True, np.diff(position) > 1][: len(kept)], line=np.arange(1, len(kept) + 1))


EXCERPT_COLUMNS = ("recording", "scale", "start_s", "end_s")


def read_excerpts(path: Path, scales: ScaleSet) -> pd.DataFrame:
    """The excerpts table, checked line by line: each row a stretch of a recording, on its clock, marked for a scale."""
    path = Path(path)
    if not path.is_file():
        raise InputNotFoundError(f"Excerpts table not found: {path}")
    table = pd.read_csv(path, dtype={"recording": str, "scale": str})
    missing = [column for column in EXCERPT_COLUMNS if column not in table.columns]
    if missing:
        raise ConfigError(f"{path} has no column {', '.join(missing)}; it needs {', '.join(EXCERPT_COLUMNS)}")
    times = table[["start_s", "end_s"]].apply(pd.to_numeric, errors="coerce")
    for line, (row, low, high) in enumerate(zip(table.itertuples(index=False), times["start_s"], times["end_s"]), 2):
        if row.scale not in scales.names():
            raise ConfigError(f"{path} line {line}: {row.scale} is not a scale of {scales.name}")
        if not (np.isfinite(low) and np.isfinite(high) and low <= high):
            raise ConfigError(f"{path} line {line}: start_s and end_s must be seconds, the start not after the end")
    return table.assign(start_s=times["start_s"].astype(float), end_s=times["end_s"].astype(float)).loc[:, list(EXCERPT_COLUMNS)]


def _example_params(own: RatingVersion, examples: dict[str, list[ScaleExample]]) -> dict:
    if own.examples is None:
        return {}
    shown = [example for listed in examples.values() for example in listed]
    return {"examples": Path(own.examples).name, "n_examples": len(shown), "n_examples_with_reasons": sum(bool(example.why) for example in shown)}


def _answer(backend: LLMBackend, system: str, user: str, seed: int | None) -> dict:
    started = time.monotonic()
    response = backend.complete(system, user, seed=seed)
    return {
        "text": response.text, "cut_off": response.cut_off, **{name: getattr(response, name, None) for name in _COUNTS},
        "seconds": round(time.monotonic() - started, 2),
    }


def ask(prompts: list, backend: LLMBackend, cfg: LLMSection, kept_at: Path | None = None, *, room: int = 0) -> pd.DataFrame:
    """One row per prompt and sample: segment, scale, sample, score, problem, reason, answer, the tokens and seconds it
    took, and whether it was kept from an earlier try. A server whose context is shorter than the longest prompt and its
    answer, room more tokens of it, stops the step after the first answer. With kept_at each answer is written there as it
    comes, and one already there is not asked for again."""
    longest = max((estimated_tokens(prompt[4] + prompt[5]) for prompt in prompts), default=0) + room
    asked = [(prompt[0], prompt[3], sample, prompt[4], prompt[5]) for prompt in prompts for sample in range(cfg.n_samples)]
    seeds = [cfg.seed + sample if backend.takes_seed else None for _, _, sample, _, _ in asked]
    kept = KeptAnswers(kept_at)
    keys = [
        request_key(backend, cfg, system, user, sample, seed) if kept_at is not None else None
        for (_, _, sample, system, user), seed in zip(asked, seeds)
    ]
    entries = [kept.get(key) for key in keys]
    rest = [index for index, entry in enumerate(entries) if entry is None]
    held: list[tuple[str | None, dict]] = []

    def answer(index: int, first: bool) -> dict:
        entry = _answer(backend, asked[index][3], asked[index][4], seeds[index])
        if first:
            held.append((keys[index], entry))
        else:
            kept.keep(keys[index], entry)
        return entry

    def checked() -> None:
        check_context(backend, longest, cfg)
        # only now: an answer to a prompt the server cut short is not one to keep
        kept.keep(*held[0])

    fresh = run_in_order([partial(answer, index, position == 0) for position, index in enumerate(rest)], cfg.concurrency, after_first=checked)
    for index, entry in zip(rest, fresh):
        entries[index] = {**entry, "fresh": True}
    rows = []
    for (segment, scale, sample, *_), entry in zip(asked, entries):
        score, reason, problem = parse_rating(entry["text"], scale.low, scale.high)
        if score is None and entry["cut_off"]:
            problem = CUT_OFF
        rows.append((
            segment, scale.scale, sample, float(score) if score is not None else np.nan, problem, reason, entry["text"],
            *(entry.get(name) for name in _COUNTS), entry.get("seconds"), not entry.get("fresh", False),
        ))
    answers = pd.DataFrame(rows, columns=["segment", "scale", "sample", "score", "problem", "reason", "answer", *_COUNTS, "seconds", "kept"])
    if len(answers) and (answers["problem"] == CUT_OFF).all():
        raise LLMBackendError(
            f"{backend.name} {cfg.model}: every answer was cut off at max_tokens ({cfg.max_tokens}) before a score came back. A model "
            "that reasons before it answers spends tokens on that first; raise coding.llm.max_tokens."
        )
    return answers.astype({"sample": int, "score": float, "seconds": float, "kept": bool, **{name: "Int64" for name in _COUNTS}})


def estimated_tokens(text: str) -> int:
    """Tokens a prompt takes, on the high side: many tokenizers give each digit a token of its own."""
    return len(_DIGIT.findall(text)) + int(1.3 * len(_WORD.findall(text))) + len(_SYMBOL.findall(text))


def check_context(backend: LLMBackend, tokens: int, cfg: LLMSection) -> None:
    limit = backend.context_length()
    if limit is None:
        return
    # a model that reasons first needs room for that too
    answer = 512 if cfg.think is False else min(cfg.max_tokens, 4096)
    needed = tokens + answer
    if needed > limit:
        raise LLMBackendError(
            f"{backend.name} {cfg.model} reads {limit} tokens at most, and the longest prompt with its answer needs about {needed}; "
            "the server would cut the start of the prompt without saying so. Give it a longer context, for Ollama by starting it "
            f"with OLLAMA_CONTEXT_LENGTH={_room(needed)}."
        )


def _room(needed: int) -> int:
    size = 4096
    while size < needed:
        size *= 2
    return size


def prompt_table(prompts: list, rater: str, recording_id: str) -> pd.DataFrame:
    """The text sent for each segment and scale, once: every sample of a scale is asked with the same."""
    rows = [(recording_id, prompt[0], prompt[3].scale, rater, prompt[4], prompt[5]) for prompt in prompts]
    return pd.DataFrame(rows, columns=PROMPT_COLUMNS)


def rating_table(
    answers: pd.DataFrame, prompts: list, rater: str, recording: Recording, skipped: list = (), scales=None,
) -> pd.DataFrame:
    """One row per segment and scale: the median of the answers that gave a score, their spread and how many there were;
    and for each segment and scale left out, a row without a score saying why."""
    rows = []
    for segment, start, end, scale, *_ in (prompt[:4] for prompt in prompts):
        mine = answers[(answers["segment"] == segment) & (answers["scale"] == scale.scale)]
        scores = mine["score"].dropna().to_numpy(dtype=float)
        rows.append({
            "recording_id": recording.recording_id, "segment": segment, "t_start": start, "t_end": end, "scale": scale.scale,
            "person": scale.person, "rater": rater,
            "score": float(np.median(scores)) if scores.size else np.nan,
            "score_sd": float(np.std(scores, ddof=1)) if scores.size > 1 else np.nan,
            "score_min": float(scores.min()) if scores.size else np.nan,
            "score_max": float(scores.max()) if scores.size else np.nan,
            "n_answers": int(scores.size), "n_asked": int(len(mine)), "not_rated": "",
        })
    person = {scale.scale: scale.person for scale in scales.scales} if scales is not None else {}
    rows += [
        {"recording_id": recording.recording_id, "segment": segment, "t_start": start, "t_end": end, "scale": name,
         "person": person.get(name), "rater": rater, "score": np.nan, "score_sd": np.nan, "score_min": np.nan, "score_max": np.nan,
         "n_answers": 0, "n_asked": 0, "not_rated": why}
        for segment, start, end, names, why in skipped for name in names
    ]
    table = pd.DataFrame(rows, columns=RATING_COLUMNS)
    for entity, value in recording.entities().items():
        table[entity] = value
    return table


def answer_warnings(answers: pd.DataFrame, ratings: pd.DataFrame, cfg: LLMSection, backend: LLMBackend, recording_id: str) -> list[QCWarning]:
    found = []

    def note(check: str, severity: str, message: str) -> None:
        found.append(QCWarning(check=check, recording_id=recording_id, subject_id=None, severity=severity, message=message))

    bad = answers[answers["problem"] != ""]
    if len(bad):
        counts = ", ".join(f"{problem} {count}" for problem, count in bad["problem"].value_counts().sort_index().items())
        note(
            "rating_unread", "warn",
            f"{len(bad)} of {len(answers)} answers gave no whole score in the scale's range ({counts}); those were left out and each "
            "score is the median of the rest.",
        )
    if (bad["problem"] == CUT_OFF).any():
        note("rating_cut_off", "warn", f"Answers were cut off at max_tokens ({cfg.max_tokens}) before a score; raise coding.llm.max_tokens.")
    empty = ratings[ratings["n_answers"] == 0]
    if len(empty):
        named = ", ".join(f"{row.scale}{f' in {row.segment}' if row.segment else ''}" for row in empty.itertuples(index=False))
        note("rating_no_score", "warn", f"No answer gave a score for {named}.")
    if len(answers) and cfg.n_samples == 1:
        note("rating_single_sample", "info", "coding.llm.n_samples is 1, so how much the model's answers vary was not measured.")
    if len(answers) and cfg.n_samples > 1 and not backend.takes_seed:
        note("rating_no_seed", "info", f"{backend.name} takes no seed, so a rerun can give other scores.")
    if len(answers) and seed_caveat(backend, cfg):
        note("rating_at_once", "info", seed_caveat(backend, cfg))
    if "kept" in answers and answers["kept"].any():
        note(
            "rating_resumed", "info",
            f"{int(answers['kept'].sum())} of {len(answers)} answers were kept from a run of this rating that stopped before its end, "
            "and were not asked for again.",
        )
    return found


def rating_notice(step: str, backend: LLMBackend) -> str:
    return (
        f"{step}: whole transcripts, with each line's role and times, are sent to {backend.name}'s servers. Whether that is "
        "allowed is for your ethics approval to say."
    )


_VERSIONS: dict[str, type[RatingExtractor]] = {}


def register_rating_versions(config: MacawConfig) -> None:
    """Each version under coding.ratings becomes the step coding.rate.<name>; a class per name, made once. The note is
    in extra_key, as each recording's own, so a changed note for one task rates only that task again."""
    sections = tuple(f"coding.llm.{key}" for key in ("provider", "model", "temperature", "max_tokens", "n_samples", "seed", "show", "reasons", "think"))
    for name in config.coding.ratings:
        if name not in _VERSIONS:
            own = tuple(f"coding.ratings.{name}.{key}" for key in RatingVersion.model_fields if key not in (*NOTE_KEYS, *RUN_ONLY_KEYS))
            _VERSIONS[name] = register_extractor(
                type(
                    f"RatingVersion_{name}",
                    (RatingExtractor,),
                    {"name": f"{RATE_STEP}{name}", "rating": name, "config_sections": (*sections, *own), "__module__": __name__},
                )
            )


def rating_steps(config: MacawConfig) -> list[str]:
    return [f"{RATE_STEP}{name}" for name in config.coding.ratings]


def scales_asked(config: MacawConfig, name: str) -> int | None:
    """How many scales a version asks about, None when its file does not read."""
    try:
        return len(read_scales(config.coding.ratings[name].scales).scales)
    except (CodebookError, InputNotFoundError, OSError):
        return None


def examples_notice(config: MacawConfig, step: str) -> str | None:
    """How many tokens a version's examples add to a request at most, so a server's context can be checked before the run."""
    from macaw.coding.llm.prompts import example_lines
    from macaw.extractors import get_extractor

    rating = get_extractor(step)(config)
    try:
        examples = rating.examples()
    except (CodebookError, InputNotFoundError, OSError):
        return None
    if not examples:
        return None
    sizes = {scale: estimated_tokens(" ".join(example_lines(listed))) for scale, listed in examples.items()}
    largest = max(sizes, key=sizes.get)
    return (
        f"{step} shows {sum(len(listed) for listed in examples.values())} examples people rated; they add up to about "
        f"{sizes[largest]} tokens to a request, for {largest}"
    )
