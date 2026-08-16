"""Layered merge: packaged defaults, then project.yaml, then recordings/<id>.yaml, then --set."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Iterator, Sequence

import yaml
from pydantic import BaseModel, ValidationError
from yaml.constructor import SafeConstructor

from macaw.config.schema import RECORDING_OVERRIDABLE, MacawConfig
from macaw.exceptions import (
    ConfigError,
    ConfigValidationError,
    InputNotFoundError,
    UnknownConfigKeyError,
)

DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"
RESOLVED_FILENAME = "resolved_config.yaml"

# where the output goes is a machine fact, like the hpc.yaml path and the ffmpeg binary,
# so it can be answered from the environment instead of by editing a config that is in git
OUTPUT_DIR_ENV_VAR = "MACAW_OUTPUT_DIR"
WORK_DIR_ENV_VAR = "MACAW_WORK_DIR"

PROJECT_TEMPLATE = """\
# MACAW project configuration. Only the keys you change need to appear here;
# everything else comes from the packaged defaults. Run `macaw config show` to see
# the full resolved result, and `macaw config diff` to see what you actually changed.

project:
  name: $name              # short project name, used in report titles
  raw_dir: $raw_dir        # directory holding the recordings
  output_dir: $output_dir  # tables, artifacts, reports, manifest, archived config
  # Logs, job records and cache. Empty means they sit under output_dir; on a cluster point
  # this at scratch. Deleting it costs recomputation and no result.
  work_dir: $work_dir
  # "{group_id}" is one directory per recording. Longitudinal datasets want
  # "sub-{group_id}/ses-{session_id}"; add "/task-{task}_run-{run}" for several per visit.
  recording_pattern: "{group_id}"
  recording_prefix: ""     # e.g. "dyad-" to only pick up dyad-001, dyad-002, ...
  n_subjects: $n_subjects
  roles: $roles

# Cluster resources are configured separately; see hpc.yaml.
execution:
  backend: local
  n_workers: 4
"""


@dataclass(frozen=True)
class KeySource:
    """Where a value came from, so an error can point at a file and a line."""

    layer: str                  # defaults | project | recording | cli
    path: Path | None = None
    line: int | None = None


def load_defaults() -> dict[str, Any]:
    values, _ = load_yaml(DEFAULTS_PATH, layer="defaults")
    return values


def load_yaml(path: Path, *, layer: str = "project") -> tuple[dict[str, Any], dict[str, KeySource]]:
    """Values plus a dotted-key to source map built from the YAML node line numbers."""
    if not path.is_file():
        raise InputNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        # compose() keeps the node tree, and with it the line numbers safe_load() throws away
        root = yaml.compose(handle)

    sources: dict[str, KeySource] = {}
    if root is None:
        return {}, sources

    values = _build_node(root, "", sources, layer, path, SafeConstructor())
    if not isinstance(values, dict):
        raise ConfigValidationError(f"{path}: the top level of a config file must be a mapping")
    return values, sources


def _build_node(
    node: yaml.Node,
    prefix: str,
    sources: dict[str, KeySource],
    layer: str,
    path: Path,
    constructor: SafeConstructor,
) -> Any:
    if isinstance(node, yaml.MappingNode):
        out: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = str(constructor.construct_object(key_node, deep=True))
            dotted = f"{prefix}.{key}" if prefix else key
            sources[dotted] = KeySource(layer, path, key_node.start_mark.line + 1)
            out[key] = _build_node(value_node, dotted, sources, layer, path, constructor)
        return out
    if isinstance(node, yaml.SequenceNode):
        return [
            _build_node(item, f"{prefix}[{i}]", sources, layer, path, constructor)
            for i, item in enumerate(node.value)
        ]
    return constructor.construct_object(node, deep=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Dicts merge key by key; lists replace wholesale. Write an explicit null to clear."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def parse_cli_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Each value is parsed as YAML, so 2.5 is a float and [a,b] is a list."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got: {pair}")
        dotted, raw = pair.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise ConfigError(f"--set is missing a key: {pair}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"--set {dotted}: value is not valid YAML: {raw}") from exc

        cursor = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"--set {dotted}: '{part}' was already given a scalar value")
        cursor[parts[-1]] = value
    return out


def check_recording_overridable(data: dict[str, Any], path: Path) -> None:
    """Raises when a recording file touches a section outside RECORDING_OVERRIDABLE."""
    illegal = [key for key in data if key not in RECORDING_OVERRIDABLE]
    if illegal:
        allowed = ", ".join(RECORDING_OVERRIDABLE)
        raise ConfigError(
            f"{path}: a recording file may not set {', '.join(sorted(illegal))}. "
            f"Sections a recording may override: {allowed}."
        )


def suggest_key(unknown: str, candidates: Sequence[str]) -> str | None:
    matches = difflib.get_close_matches(unknown, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def describe_unknown_key(key: str, sources: dict[str, KeySource], candidates: Sequence[str]) -> str:
    """Builds the 'project.yaml:14: unknown key ... did you mean ...' message."""
    source = _lookup_source(key, sources)
    where = "config"
    if source is not None and source.path is not None:
        where = f"{source.path}:{source.line}" if source.line else str(source.path)

    message = f"{where}: unknown config key '{key}'"
    suggestion = suggest_key(key.rsplit(".", 1)[-1], candidates)
    if suggestion:
        message += f". Did you mean '{suggestion}'?"
    elif candidates:
        message += f". Valid keys here: {', '.join(sorted(candidates))}"
    return message


def _lookup_source(dotted: str, sources: dict[str, KeySource]) -> KeySource | None:
    parts = dotted.split(".")
    while parts:
        found = sources.get(".".join(parts))
        if found is not None:
            return found
        parts.pop()
    return None


def _dotted_keys(data: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        yield dotted
        if isinstance(value, dict):
            yield from _dotted_keys(value, dotted)


def _directory_override(explicit: Path | str | None, env_var: str) -> Path | None:
    """A shell flag or an exported variable is read relative to the shell, unlike a path
    written in project.yaml, which follows the file so the project stays portable."""
    value = explicit if explicit is not None else os.environ.get(env_var)
    return Path(value).expanduser().resolve() if value else None


def resolve_config(
    project_yaml: Path,
    *,
    recording_id: str | None = None,
    cli_overrides: Sequence[str] = (),
    output_dir: Path | str | None = None,
    work_dir: Path | str | None = None,
) -> MacawConfig:
    """Without recording_id this is the project baseline; with one it is what that recording runs."""
    merged = load_defaults()
    sources: dict[str, KeySource] = {
        key: KeySource("defaults", DEFAULTS_PATH) for key in _dotted_keys(merged)
    }

    project_data, project_sources = load_yaml(project_yaml, layer="project")
    merged = deep_merge(merged, project_data)
    sources.update(project_sources)

    if recording_id is not None:
        recording_yaml = recording_override_path(project_yaml, recording_id)
        if recording_yaml is not None:
            recording_data, recording_sources = load_yaml(recording_yaml, layer="recording")
            check_recording_overridable(recording_data, recording_yaml)
            merged = deep_merge(merged, recording_data)
            sources.update(recording_sources)

    overrides = parse_cli_overrides(cli_overrides)
    if overrides:
        merged = deep_merge(merged, overrides)
        sources.update({key: KeySource("cli") for key in _dotted_keys(overrides)})

    config = _validate(merged, sources)
    # relative paths follow project.yaml, not the shell's cwd, so a project directory
    # keeps working after being copied to another machine
    _absolutise(config, project_yaml.resolve().parent)

    for field, explicit, env_var in (
        ("output_dir", output_dir, OUTPUT_DIR_ENV_VAR),
        ("work_dir", work_dir, WORK_DIR_ENV_VAR),
    ):
        override = _directory_override(explicit, env_var)
        if override is not None:
            setattr(config.project, field, override)
    return config


def _validate(merged: dict[str, Any], sources: dict[str, KeySource]) -> MacawConfig:
    try:
        return MacawConfig.model_validate(merged)
    except ValidationError as exc:
        raise _translate(exc, sources) from exc


def _translate(exc: ValidationError, sources: dict[str, KeySource]) -> ConfigError:
    messages: list[str] = []
    unknown = False
    for error in exc.errors():
        dotted = ".".join(str(part) for part in error["loc"])
        if error["type"] == "extra_forbidden":
            unknown = True
            parent = dotted.rsplit(".", 1)[0] if "." in dotted else ""
            messages.append(describe_unknown_key(dotted, sources, _candidates_for(parent)))
        else:
            source = _lookup_source(dotted, sources)
            where = ""
            if source is not None and source.path is not None:
                where = f"{source.path}:{source.line}: " if source.line else f"{source.path}: "
            messages.append(f"{where}{dotted}: {error['msg']}")

    text = "\n".join(messages)
    return UnknownConfigKeyError(text) if unknown else ConfigValidationError(text)


def _candidates_for(dotted_parent: str) -> list[str]:
    model: type[BaseModel] | None = MacawConfig
    for part in filter(None, dotted_parent.split(".")):
        field = model.model_fields.get(part) if model else None
        annotation = field.annotation if field else None
        model = (
            annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel)
            else None
        )
        if model is None:
            return []
    return list(model.model_fields) if model else []


def _absolutise(model: BaseModel, base: Path) -> None:
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, Path) and not value.is_absolute():
            setattr(model, name, (base / value).resolve())
        elif isinstance(value, BaseModel):
            _absolutise(value, base)


def load_config(path: Path) -> MacawConfig:
    """Load an already-resolved config from an output directory, without merging anything."""
    values, sources = load_yaml(path, layer="project")
    return _validate(values, sources)


def recording_override_path(project_yaml: Path, recording_id: str) -> Path | None:
    project_yaml = project_yaml.resolve()
    with open(project_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    configured = (data.get("project") or {}).get("recording_config_dir", "recordings")

    directory = Path(configured)
    if not directory.is_absolute():
        directory = project_yaml.parent / directory
    candidate = directory / f"{recording_id}.yaml"
    return candidate if candidate.is_file() else None


def archive_resolved(config: MacawConfig, output_dir: Path) -> Path:
    """Reproducibility depends entirely on this file existing."""
    from macaw.project.layout import make_layout

    return _dump_config(config, make_layout(output_dir).resolved_config_path)


def archive_recording_resolved(config: MacawConfig, output_dir: Path, recording_id: str) -> Path:
    """The per-recording copy is the one that explains a given recording's numbers."""
    from macaw.project.layout import make_layout

    layout = make_layout(output_dir)
    return _dump_config(config, layout.recording_dir(recording_id) / RESOLVED_FILENAME)


def _dump_config(config: MacawConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return path


def _yaml_scalar(value: Any) -> str:
    """Single quotes, so a Windows path's backslashes stay literal."""
    text = "" if value is None else str(value)
    return f"'{text.replace(chr(39), chr(39) * 2)}'" if text else ""


def scaffold_project_config(
    dest: Path,
    *,
    force: bool = False,
    name: str | None = None,
    raw_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    work_dir: Path | str | None = None,
    n_subjects: int = 2,
) -> Path:
    if dest.exists() and not force:
        raise ConfigError(f"{dest} already exists. Pass --force to overwrite it.")
    roles = (
        ["parent", "child"]
        if n_subjects == 2
        else [f"subject_{position}" for position in range(1, n_subjects + 1)]
    )
    text = Template(PROJECT_TEMPLATE).substitute(
        name=_yaml_scalar(name),
        raw_dir=_yaml_scalar(raw_dir),
        output_dir=_yaml_scalar(output_dir),
        work_dir=_yaml_scalar(work_dir),
        n_subjects=n_subjects,
        roles="[" + ", ".join(roles) + "]",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return dest
