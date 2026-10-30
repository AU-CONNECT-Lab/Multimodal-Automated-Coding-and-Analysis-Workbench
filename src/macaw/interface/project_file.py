"""project.yaml as the settings page edits it: values as the file writes them, edits checked
before they are saved, and the file's comments and layout kept."""

from __future__ import annotations

import io
import os
import re
import tempfile
import types
import typing
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from macaw.config.hashing import hash_extractor_config
from macaw.config.loader import DEFAULTS_PATH, PROJECT_TEMPLATE, deep_merge, load_defaults, load_yaml, resolve_config
from macaw.config.schema import MacawConfig
from macaw.execution.dag import build_dag
from macaw.extractors import get_extractor, list_backends, list_extractors
from macaw.extractors.base import Gate

# the settings most projects change; everything else is edited as text below the form
FORM: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Project", "Where the recordings are, how their files are named, and the part of each recording to analyse.", (
        "project.name", "project.raw_dir", "project.output_dir", "project.work_dir", "project.recording_pattern",
        "project.media_pattern", "project.n_subjects", "project.roles", "segment.task_start_s", "segment.task_end_s",
    )),
    ("Steps", "What a run without a choice of steps runs, and how many recordings run at once.", (
        "pipeline.run", "execution.n_workers",
    )),
    ("Tools", "Which program does each kind of measurement, and with which model.", (
        "prosody.backend", "text.asr_backend", "text.asr_model", "text.language", "text.device", "text.diarization",
        "text.extra_voices", "pose.backend", "pose.model", "face.backend", "face.model",
    )),
    ("Checks by a person", "Where the software waits for a person before going on: always, only when it is unsure, or never.", (
        "identity.confirm", "align.review", "pose.review", "text.voice_review",
    )),
    ("Coding", "Coding layers, and the language model that codes speech acts in layer B.", (
        "coding.layers", "coding.codebook", "coding.llm.provider", "coding.llm.model", "coding.llm.base_url",
        "coding.llm.n_samples", "coding.llm.review_threshold", "coding.llm.coded_roles",
    )),
)

LABELS = {
    "project.name": "Project name",
    "project.raw_dir": "Recordings folder",
    "project.output_dir": "Results folder",
    "project.work_dir": "Working folder",
    "project.recording_pattern": "Recording folder pattern",
    "project.media_pattern": "Media file pattern",
    "project.n_subjects": "People per recording",
    "project.roles": "Roles",
    "segment.task_start_s": "Task starts at (s)",
    "segment.task_end_s": "Task ends at (s)",
    "pipeline.run": "Steps to run",
    "execution.n_workers": "Recordings at once",
    "prosody.backend": "Voice measures",
    "text.asr_backend": "Transcription",
    "text.asr_model": "Transcription model",
    "text.language": "Language",
    "text.device": "Transcription device",
    "text.diarization": "Microphones",
    "text.extra_voices": "Other voices on a shared microphone",
    "pose.backend": "Body pose",
    "pose.model": "Body pose model",
    "face.backend": "Face",
    "face.model": "Face model",
    "identity.confirm": "Which file is whose",
    "align.review": "Offsets between files",
    "pose.review": "Who is on camera",
    "text.voice_review": "Voices on a shared microphone",
    "coding.layers": "Coding layers",
    "coding.codebook": "Codebook",
    "coding.llm.provider": "Language model server",
    "coding.llm.model": "Language model",
    "coding.llm.base_url": "Server address",
    "coding.llm.n_samples": "Answers per unit",
    "coding.llm.review_threshold": "Review below this agreement",
    "coding.llm.coded_roles": "People to code",
}

# keys that name a registered backend, so the choices are whatever is installed, plugins included
BACKEND_GROUPS = {
    "prosody.backend": "prosody",
    "text.asr_backend": "asr",
    "text.diarization": "diarization",
    "pose.backend": "pose",
    "face.backend": "face",
    "coding.llm.provider": "llm",
}
TEXT_DEVICES = ("auto", "cuda", "cpu")

_KEY_LINE = re.compile(r"^(?P<indent> *)(?P<key>[A-Za-z_][\w.]*):(?P<rest>.*)$")
_COMMENT_LINE = re.compile(r"^ *# ?(?P<text>.*)$")


def _comments(text: str) -> dict[str, str]:
    """Each key's inline comment, or the comment lines right above it."""
    found: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    above: list[str] = []
    for line in text.splitlines():
        comment = _COMMENT_LINE.match(line)
        if comment:
            above.append(comment["text"].strip())
            continue
        match = _KEY_LINE.match(line)
        if not match:
            above = []
            continue
        indent = len(match["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([key for _, key in stack] + [match["key"]])
        stack.append((indent, match["key"]))
        inline = re.search(r"\s#\s?(.*)$", match["rest"])
        hint = inline.group(1).strip() if inline else " ".join(above)
        if hint:
            found[dotted] = hint
        above = []
    return found


def hints() -> dict[str, str]:
    return {**_comments(PROJECT_TEMPLATE), **_comments(DEFAULTS_PATH.read_text(encoding="utf-8"))}


def _annotation(dotted: str):
    model: type[BaseModel] = MacawConfig
    annotation = None
    for part in dotted.split("."):
        annotation = model.model_fields[part].annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            model = annotation
    return annotation


def _strip_none(annotation) -> tuple[Any, bool]:
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType) and type(None) in args:
        rest = [arg for arg in args if arg is not type(None)]
        return (rest[0] if len(rest) == 1 else typing.Union[tuple(rest)]), True
    return annotation, False


def control(dotted: str) -> dict[str, Any]:
    """How the form asks for a key: its kind (choice, multi, number, text, list), its options, and whether it may be empty."""
    annotation, optional = _strip_none(_annotation(dotted))
    if dotted in BACKEND_GROUPS:
        return {"kind": "choice", "options": list_backends(BACKEND_GROUPS[dotted]), "optional": optional}
    if dotted == "text.device":
        return {"kind": "choice", "options": list(TEXT_DEVICES), "optional": optional}
    if dotted == "pipeline.run":
        return {"kind": "multi", "options": [name for name in list_extractors() if not issubclass(get_extractor(name), Gate)]}
    if typing.get_origin(annotation) is Literal:
        return {"kind": "choice", "options": list(typing.get_args(annotation)), "optional": optional}
    if typing.get_origin(annotation) is list:
        (item,) = typing.get_args(annotation)
        if typing.get_origin(item) is Literal:
            return {"kind": "multi", "options": list(typing.get_args(item))}
        return {"kind": "list"}
    if annotation in (int, float):
        return {"kind": "number", "integer": annotation is int, "optional": optional}
    return {"kind": "text", "optional": optional}


def _get(data: dict, dotted: str) -> Any:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def file_values(path: Path) -> dict[str, Any]:
    """The form's keys as the file writes them, so a relative folder stays relative; defaults fill the rest."""
    written, _ = load_yaml(path)
    merged = deep_merge(load_defaults(), written)
    return {dotted: _get(merged, dotted) for _, _, keys in FORM for dotted in keys}


def to_form(dotted: str, value: Any) -> Any:
    if control(dotted)["kind"] == "list":
        return ", ".join(str(item) for item in value or [])
    if control(dotted)["kind"] == "text":
        return "" if value is None else str(value)
    return value


def from_form(dotted: str, value: Any) -> Any:
    kind = control(dotted)
    if kind["kind"] == "list":
        return [item.strip() for item in str(value or "").split(",") if item.strip()]
    if kind["kind"] == "multi":
        return list(value or [])
    if kind["kind"] in ("text", "choice") and value in ("", None):
        return None if kind.get("optional") else ("" if kind["kind"] == "text" else None)
    if kind["kind"] == "number" and value is not None and kind["integer"]:
        return int(value)
    return value


def changes(path: Path, form: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Keys whose form value differs from the file, as (in the file, in the form)."""
    now = file_values(path)
    found = {}
    for dotted, value in form.items():
        new = from_form(dotted, value)
        if new != now.get(dotted) and not (new in ("", None, []) and now.get(dotted) in ("", None, [])):
            found[dotted] = (now.get(dotted), new)
    return found


def with_values(text: str, values: dict[str, Any]) -> str:
    """The file's text with these keys set; a key it lacks is added under its section."""
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    data = yaml.load(text) or CommentedMap()
    for dotted, value in values.items():
        parts = dotted.split(".")
        cursor = data
        for depth, part in enumerate(parts[:-1]):
            if not isinstance(cursor.get(part), dict):
                cursor[part] = CommentedMap()
                if depth == 0 and len(data) > 1:
                    data.yaml_set_comment_before_after_key(part, before="\n")
            cursor = cursor[part]
        if isinstance(value, list):
            value = CommentedSeq(value)
            value.fa.set_flow_style()
        cursor[parts[-1]] = value
    out = io.StringIO()
    yaml.dump(data, out)
    return out.getvalue()


def check(path: Path, text: str) -> MacawConfig:
    """The config this text would give, read beside the real file so relative paths mean the same; raises on any error."""
    handle, temporary = tempfile.mkstemp(suffix=".yaml", prefix=".macaw-check-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        return resolve_config(Path(temporary))
    finally:
        Path(temporary).unlink(missing_ok=True)


def runs_again(before: MacawConfig, after: MacawConfig) -> list[str]:
    """Steps whose results a change makes out of date: those that hash a changed setting, and everything after them."""
    dag = build_dag(after, list_extractors())
    order = dag.topological_order()
    hit = {name for name in order if hash_extractor_config(before, name) != hash_extractor_config(after, name)}
    reached = set(hit).union(*(dag.descendants(name) for name in hit)) if hit else set()
    return [name for name in order if name in reached and not issubclass(get_extractor(name), Gate)]


def save(path: Path, text: str) -> None:
    """Writes the text with the line endings the file already has."""
    newline = "\r\n" if path.is_file() and b"\r\n" in path.read_bytes() else "\n"
    path.write_text(text, encoding="utf-8", newline=newline)
