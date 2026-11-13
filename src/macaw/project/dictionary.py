"""Data dictionaries written beside an exported table: one entry per column, as a CSV and as a JSON sidecar keyed by
column name."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from macaw import __version__
from macaw.project.manifest import FeatureSpec
from macaw.project.scale import in_unit, metre_power, pixel_power, suffix
from macaw.store.schema import SEGMENT_TABLE_SCHEMA, Scale

COLUMNS = [
    "column", "meaning", "unit", "who", "rows", "aggregation", "missing", "step", "backend", "settings",
    "comparable_across_people", "comparable_across_backends", "validation", "converted_from",
]
# the BIDS sidecar keys; the rest keep their CSV names
_JSON_KEYS = {"meaning": "Description", "unit": "Units", "levels": "Levels"}
PACKAGE = f"pymacaw {__version__}"
NOT_STATED = "The step that made it does not say."
NEVER = "Never empty."
NO_SPEC = "No step described this measure."
_WHO = ("person", "pair", "recording")
_ROWS = (Scale.FRAME.value, Scale.EVENT.value, Scale.WINDOW.value, Scale.RECORDING.value, "segment")
LEVELS = {
    "scale": {
        Scale.FRAME.value: "one video or audio frame",
        Scale.EVENT.value: "one event, such as a turn or a gesture",
        Scale.WINDOW.value: "one window of fixed length",
        Scale.RECORDING.value: "the whole analysed part of the recording, or one segment of it",
    },
}


def _entity(name: str, what: str) -> tuple[str, str | None, str]:
    return f"{what}, read from the recording's name.", None, f"Empty when the recording folder pattern has no {{{name}}}."


FIXED: dict[str, tuple[str, str | None, str]] = {
    "recording_id": ("The recording the row comes from.", None, NEVER),
    "group_id": _entity("group_id", "The people recorded together, such as a family (BIDS sub-)"),
    "session_id": _entity("session_id", "The visit or wave (BIDS ses-)"),
    "task": _entity("task", "The task"),
    "run": _entity("run", "The run of the task"),
    "segment": (
        "A segment named in the project's segments table; every measure in the row was worked out again inside it.", None,
        "Empty for the whole analysed part of the recording and for its frames, events and windows.",
    ),
    "subject_id": ("The person the row belongs to, or the first person of a pair.", None, "Empty for a row of the recording as a whole."),
    "subject_id_b": (
        "The second person of a pair, for measures between two people.", None, "Empty for a row of one person or of the recording as a whole.",
    ),
    "t_start": ("Start of the span the row covers, on the recording's clock.", "s", NEVER),
    "t_end": ("End of the span the row covers, on the recording's clock.", "s", NEVER),
    "source": ("The step that wrote the row.", None, NEVER),
    "scale": ("What span the row covers.", None, NEVER),
    "feature": ("The measure; each one has an entry of its own in this dictionary.", None, NEVER),
    "value": ("The measure's value, in the unit its own entry gives.", None, "See the measure's own entry."),
    "execution_id": ("The run of macaw that wrote the row.", None, NEVER),
    "config_hash": ("A fingerprint of the settings the row was made under; rows with the same one share their settings.", None, NEVER),
}


def dictionary_paths(out: Path) -> tuple[Path, Path]:
    return out.with_name(f"{out.stem}_dictionary.csv"), out.with_name(f"{out.stem}_dictionary.json")


def fixed_entry(column: str) -> dict:
    meaning, unit, missing = FIXED[column]
    return {"column": column, "meaning": meaning, "unit": unit, "missing": missing, "step": PACKAGE, "levels": LEVELS.get(column)}


def feature_dictionary(table: pd.DataFrame, columns: list[str], specs: dict[str, dict[str, FeatureSpec]], units: str) -> list[dict]:
    """One entry per column of the export, in its order. table is the export in its long layout; specs holds each
    recording's own feature specs, so a step run under a recording's own settings shows each variant."""
    found = _found(table)
    entries = []
    for column in columns:
        if column in FIXED:
            entries.append(fixed_entry(column))
            continue
        recordings = found.at[column, "recordings"] if column in found.index else []
        entry = _feature_entry(column, recordings, specs, units)
        if column in found.index:
            entry["who"] = ", ".join(kind for kind in _WHO if found.at[column, kind])
            entry["rows"] = ", ".join(kind for kind in _ROWS if found.at[column, kind])
        entries.append(entry)
    return entries


def long_columns(table: pd.DataFrame) -> list[str]:
    """The long layout's own columns, then each measure it holds, as the entries come in its dictionary."""
    return [*SEGMENT_TABLE_SCHEMA, *sorted(table["feature"].dropna().astype(str).unique())]


def _found(table: pd.DataFrame) -> pd.DataFrame:
    """Per feature: whose rows hold it, which spans, and which recordings."""
    if table.empty:
        return pd.DataFrame(columns=[*_WHO, *_ROWS, "recordings"])
    first, second = table["subject_id"].notna().to_numpy(), table["subject_id_b"].notna().to_numpy()
    scale = table["scale"].astype(str).to_numpy()
    named = table["segment"].notna().to_numpy()
    flags = pd.DataFrame({
        "feature": table["feature"].astype(str).to_numpy(),
        "person": first & ~second, "pair": second, "recording": ~first & ~second,
        **{kind: (scale == kind) & ~named for kind in _ROWS[:4]}, "segment": named,
    })
    out = flags.groupby("feature").any()
    names = table["feature"].astype(str).to_numpy()
    recordings = pd.Series(table["recording_id"].astype(str).to_numpy()).groupby(names).unique()
    out["recordings"] = recordings.map(sorted).reindex(out.index)
    return out


def _feature_entry(column: str, recordings: list[str], specs: dict[str, dict[str, FeatureSpec]], units: str) -> dict:
    by_recording, converted = _specs_of(column, recordings, specs, units)
    if not by_recording:
        return {"column": column, "meaning": NO_SPEC, "missing": NOT_STATED}
    fields = {rid: _fields(spec) for rid, spec in by_recording.items()}
    entry = {"column": column}
    for key in ("meaning", "unit", "aggregation", "missing", "step", "backend", "comparable_across_people", "comparable_across_backends", "validation"):
        entry[key] = _agreed({rid: values[key] for rid, values in fields.items()})
    entry["settings"] = varying_settings({rid: spec.params for rid, spec in by_recording.items()})
    if converted is not None:
        base, power_of_pixels = converted
        entry["unit"] = _agreed({rid: in_unit(spec.unit, units) for rid, spec in by_recording.items()})
        if power_of_pixels:
            entry["converted_from"] = f"{base}, in pixels, times the {units} per pixel of that recording and person listed in scale.csv"
            entry["missing"] = (
                f"{entry['missing']} Also empty where the recording has no usable scale for that person, or for a pair whose two "
                "people have different scales; scale.csv lists the scales used."
            )
        else:
            entry["converted_from"] = f"{base}, in metres"
    return entry


def _specs_of(column, recordings, specs, units) -> tuple[dict[str, FeatureSpec], tuple[str, bool] | None]:
    """The spec of the column in each recording that holds it, and for a column in the project's unit, the measure it
    was converted from and whether that one is in pixels."""
    own = {rid: specs[rid][column] for rid in recordings if column in specs.get(rid, {})}
    if own or not column.endswith(suffix(units)):
        return own, None
    base = column[: -len(suffix(units))]
    found = {rid: specs[rid][base] for rid in recordings if base in specs.get(rid, {})}
    found = {rid: spec for rid, spec in found.items() if pixel_power(spec.unit) or metre_power(spec.unit)}
    if not found:
        return {}, None
    return found, (base, any(pixel_power(spec.unit) for spec in found.values()))


def _fields(spec: FeatureSpec) -> dict:
    backend = " ".join(part for part in (spec.backend, spec.backend_version) if part) or None
    return {
        "meaning": spec.description, "unit": spec.unit, "aggregation": spec.aggregation, "missing": spec.missing or NOT_STATED,
        "step": f"{spec.extractor} {spec.extractor_version} ({PACKAGE})", "backend": backend,
        "comparable_across_people": spec.comparable_across_subjects, "comparable_across_backends": spec.comparable_across_backends,
        "validation": spec.validation,
    }


def _agreed(values: dict[str, object]):
    """The one value every recording gives, or each value with the recordings that gave it."""
    groups = _grouped(values)
    if len(groups) == 1:
        return next(iter(groups.values()))[0]
    return "; ".join(f"{_said(value)} (for {', '.join(rids)})" for value, rids in _pairs(groups))


def varying_settings(params: dict[str, dict]):
    """The settings every recording ran under, or each variant with its recordings."""
    groups = _grouped(params)
    if len(groups) == 1:
        return next(iter(groups.values()))[0] or None
    return [{"recordings": rids, "settings": value} for value, rids in _pairs(groups)]


def _grouped(values: dict[str, object]) -> dict[str, tuple[object, list[str]]]:
    groups: dict[str, tuple[object, list[str]]] = {}
    for rid, value in values.items():
        key = json.dumps(value, sort_keys=True, default=str)
        groups.setdefault(key, (value, []))[1].append(rid)
    return groups


def _pairs(groups: dict) -> list[tuple[object, list[str]]]:
    return [(value, sorted(rids)) for value, rids in groups.values()]


def _said(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "none" if value is None else str(value)


def flat_settings(settings) -> str:
    """key=value; ... with nested keys dotted, and each variant after the recordings that ran it."""
    if not settings:
        return ""
    if isinstance(settings, list):
        return " | ".join(f"{', '.join(variant['recordings'])}: {flat_settings(variant['settings']) or 'none'}" for variant in settings)
    return "; ".join(f"{key}={value}" for key, value in _flatten(settings))


def _flatten(params: dict, prefix: str = "") -> list[tuple[str, str]]:
    pairs = []
    for key in sorted(params):
        value, name = params[key], f"{prefix}{key}"
        if isinstance(value, dict) and value:
            pairs += _flatten(value, f"{name}.")
        else:
            pairs.append((name, _value(value)))
    return pairs


def _value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return _said(value).lower() if isinstance(value, bool) or value is None else str(value)


def _with_levels(meaning: str | None, levels: dict | None) -> str | None:
    if not levels:
        return meaning
    return f"{meaning} " + "; ".join(f"{level}: {said}" for level, said in levels.items()) + "."


def write_dictionary(entries: list[dict], out: Path, columns: list[str] = COLUMNS) -> tuple[Path, Path]:
    """<stem>_dictionary.csv and <stem>_dictionary.json beside the export at out."""
    csv_path, json_path = dictionary_paths(out)
    rows = [
        {
            **{key: entry.get(key) for key in columns},
            "meaning": _with_levels(entry.get("meaning"), entry.get("levels")),
            "settings": flat_settings(entry.get("settings")),
            **{key: _said(entry[key]) for key in ("comparable_across_people", "comparable_across_backends") if entry.get(key) is not None},
        }
        for entry in entries
    ]
    pd.DataFrame(rows).reindex(columns=columns).to_csv(csv_path, index=False, encoding="utf-8")
    sidecar = {
        entry["column"]: {_JSON_KEYS.get(key, key): value for key, value in entry.items() if key != "column" and value not in (None, "", [], {})}
        for entry in entries
    }
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return csv_path, json_path
