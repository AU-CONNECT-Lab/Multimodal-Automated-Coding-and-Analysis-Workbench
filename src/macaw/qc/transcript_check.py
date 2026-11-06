"""A language model reads the transcript and marks lines that look misheard, made up or cut off, for a person to
check against the recording. The transcript itself is never changed here."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd

from macaw.coding.llm.backends import LLMBackend
from macaw.coding.llm.parsing import answer_object
from macaw.extractors.audio.content.diarization import OTHER

CONTEXT_LINES = 5
# each kind of mark and how it reads; a kind the model makes up counts as the first
KINDS = {"misheard": "possibly misheard", "cut": "cut off"}
COLUMNS = ["subject_id", "t_start", "t_end", "text", "kind", "why"]


def _letter(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else f"S{index + 1}"


def transcript_lines(transcript: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    """Every line in time order with its speaker's letter and an id; letters, not subject ids, go to a model."""
    letters = {subject: _letter(index) for index, subject in enumerate(subjects)}
    lines = transcript[["subject_id", "t_start", "t_end", "text"]].copy()
    lines["subject_id"] = lines["subject_id"].astype(str)
    lines["text"] = lines["text"].astype("string").fillna("").str.strip()
    lines = lines[lines["text"] != ""].sort_values(["t_start", "t_end"], kind="stable").reset_index(drop=True)
    lines["speaker"] = [letters.get(subject, OTHER if subject == OTHER else "?") for subject in lines["subject_id"]]
    return lines.assign(id=[f"l{index + 1}" for index in range(len(lines))])


def fingerprint(lines: pd.DataFrame) -> str:
    said = lines[["subject_id", "t_start", "t_end", "text"]].astype(str).values.tolist()
    return hashlib.sha1(json.dumps(said).encode("utf-8")).hexdigest()[:16]


def check_prompt(batch: pd.DataFrame, context: pd.DataFrame) -> tuple[str, str]:
    system = [
        "You check an automatic transcript of a recorded conversation for lines a speech recogniser may have got wrong. "
        "Speakers are named by letters.",
        *([f"The speaker {OTHER} is someone besides the people recorded, such as a researcher in the room."]
          if (pd.concat([context, batch])["speaker"] == OTHER).any() else []),
        "A recogniser can mishear words, giving a line that makes little sense where it stands, or write words nobody said, "
        "such as a phrase repeated over and over, a sign-off like 'thanks for watching', or many words in a very short time. "
        "Mark only such lines. Do not mark a line for being casual, ungrammatical, unfinished or a young child's speech. "
        "Give each line you mark a kind: misheard when its words make little sense where they stand or nobody said them; "
        "cut when it is only a word or two that seem cut off from what the speaker said before or after.",
        'Answer with one JSON object and nothing else: {"lines": [{"id": "<line id>", "kind": "misheard" or "cut", '
        '"why": "<a few words>"}]}, listing only the lines you mark; an empty list when there are none.',
    ]
    user = []
    if not context.empty:
        user += ["Earlier in the conversation (context only, do not check):", *(_line(row) for row in context.itertuples(index=False)), ""]
    user += ["Lines to check:", *(f"id {row.id} | {_line(row)}" for row in batch.itertuples(index=False))]
    return "\n".join(system), "\n".join(user)


def _line(row) -> str:
    seconds = float(row.t_end) - float(row.t_start)
    return f"speaker {row.speaker} | {seconds:.1f} s, {len(str(row.text).split())} words: {row.text}"


def check_lines(backend: LLMBackend, lines: pd.DataFrame, *, batch_size: int, seed: int | None = None) -> pd.DataFrame:
    """The lines the model marked, each with its kind and why; an id it made up is ignored."""
    marked: dict[str, tuple[str, str]] = {}
    for start in range(0, len(lines), batch_size):
        batch = lines.iloc[start : start + batch_size]
        context = lines.iloc[max(0, start - CONTEXT_LINES) : start]
        answer = answer_object(backend.complete(*check_prompt(batch, context), seed=seed).text)
        ids = set(batch["id"])
        for item in answer.get("lines") or []:
            if isinstance(item, dict) and str(item.get("id", "")).strip() in ids:
                kind = str(item.get("kind") or "").strip().lower()
                marked[str(item["id"]).strip()] = (kind if kind in KINDS else next(iter(KINDS)), str(item.get("why") or "").strip())
    found = lines[lines["id"].isin(marked)]
    return found.assign(
        kind=[marked[line_id][0] for line_id in found["id"]], why=[marked[line_id][1] for line_id in found["id"]],
    )[COLUMNS]


def check_transcript(
    backend: LLMBackend, transcript: pd.DataFrame, subjects: list[str], path, *, provider: str, batch_size: int, seed: int | None,
) -> dict:
    lines = transcript_lines(transcript, subjects)
    marked = check_lines(backend, lines, batch_size=batch_size, seed=seed)
    found = {
        "model": backend.model_version(), "provider": provider, "asked_at": datetime.now().isoformat(timespec="seconds"),
        "transcript": fingerprint(lines), "n_lines": len(lines), "lines": marked.to_dict("records"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
    return found


def read_check(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def still_there(found: dict, transcript: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The marked lines the transcript still has as they were, and how many it no longer has, after a correction."""
    marked = pd.DataFrame(found.get("lines") or [], columns=COLUMNS)
    if marked.empty:
        return marked, 0
    now = {(str(row.subject_id), round(float(row.t_start), 3), str(row.text).strip()) for row in transcript.itertuples(index=False)}
    keep = [(str(row.subject_id), round(float(row.t_start), 3), str(row.text).strip()) in now for row in marked.itertuples(index=False)]
    return marked[keep].reset_index(drop=True), int(len(keep) - sum(keep))


def describe(found: dict, transcript: pd.DataFrame | None = None) -> str:
    lines = found.get("lines") or []
    text = f"{len(lines)} of {found.get('n_lines', 0)} line(s) marked by {found.get('model')}"
    if lines:
        counts = {kind: sum(1 for line in lines if line.get("kind") == kind) for kind in KINDS}
        text += ": " + ", ".join(f"{count} {KINDS[kind]}" for kind, count in counts.items() if count)
    if transcript is not None:
        _, gone = still_there(found, transcript)
        text += f"; {gone} of them changed since" if gone else ""
    return text
