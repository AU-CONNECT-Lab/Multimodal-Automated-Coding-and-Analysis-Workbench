"""Canonical table schemas. Changing these later means touching every extractor."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from macaw.exceptions import SchemaViolationError


class Scale(str, Enum):
    FRAME = "frame"
    EVENT = "event"
    WINDOW = "window"
    RECORDING = "recording"


# The signal + feature layer. One row per (entity, time span, feature).
# subject_id is null for recording-level rows; subject_id_b is non-null only for
# pairwise quantities (interpersonal distance, gap/overlap, mutual gaze).
#
# group_id / session_id / task / run are the BIDS entities recording_id was built from.
# They are carried on every row, dictionary encoded and so nearly free, so that an
# exported table goes straight into a longitudinal model without joining an index.
LONG_TABLE_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "group_id": "string",     # the people recorded together, BIDS sub-
    "session_id": "string",   # visit or wave, BIDS ses-
    "task": "string",
    "run": "string",
    "subject_id": "string",
    "subject_id_b": "string",
    "source": "string",       # extractor name, e.g. "prosody.frame"
    "scale": "string",        # one of Scale
    "t_start": "float64",     # seconds, recording clock
    "t_end": "float64",
    "feature": "string",
    "value": "float64",
    "execution_id": "string",
    "config_hash": "string",
}

LONG_TABLE_KEY = ("recording_id", "subject_id", "subject_id_b", "source", "t_start", "feature")

# Everything that identifies a row apart from the time span it covers and the value it
# carries. Aggregation groups by these, so provenance survives instead of being averaged
# away, and each feature stays its own series.
LONG_TABLE_IDENTITY = tuple(
    name for name in LONG_TABLE_SCHEMA if name not in {"scale", "t_start", "t_end", "value"}
)

# The four entities parsed out of the path. Null whenever the pattern omits them.
ENTITY_COLUMNS = ("group_id", "session_id", "task", "run")


# Keypoints live outside the long table: 30 fps x 17 joints x 4 values x N people
# is ~10^6 rows per 5 min recording, which the long table handles badly.
# Only derived kinematics (velocity, distance, openness) go into the long table.
KEYPOINT_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "source": "string",       # pose backend, e.g. "pose.rtmpose"
    "frame": "int64",
    "t": "float64",
    "joint": "string",
    "x": "float64",
    "y": "float64",
    "z": "float64",           # NaN when input is 2D
    "confidence": "float64",
    "track_id": "int64",      # raw backend track before repair
}


# The annotation layer. Discrete labelled intervals, losslessly mappable to .eaf.
ANNOTATION_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "group_id": "string",
    "session_id": "string",
    "task": "string",
    "run": "string",
    "subject_id": "string",
    "tier": "string",
    "t_start": "float64",
    "t_end": "float64",
    "label": "string",
    "annotator": "string",     # human user id, or model id for automatic coding
    "layer": "string",         # A | B | C
    "confidence": "float64",   # NaN for human annotations
    "abstained": "bool",
    "needs_review": "bool",
    "execution_id": "string",
    "config_hash": "string",
}


# bool maps to the nullable dtype: a missing flag has to survive a parquet round trip
# instead of crashing the cast.
_PANDAS_DTYPES: dict[str, str] = {
    "string": "string[pyarrow]",
    "float64": "float64",
    "int64": "int64",
    "bool": "boolean",
}


def pandas_dtype(spec: str) -> str:
    return _PANDAS_DTYPES[spec]


def validate_frame(df, schema: dict[str, str], name: str) -> None:
    """Raise SchemaViolationError if df does not carry exactly these columns and dtypes."""
    missing = [column for column in schema if column not in df.columns]
    unexpected = [column for column in df.columns if column not in schema]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(unexpected)}")
        raise SchemaViolationError(f"{name}: {'; '.join(problems)}")

    wrong = [
        f"{column} is {df[column].dtype}, expected {_PANDAS_DTYPES[spec]}"
        for column, spec in schema.items()
        if df[column].dtype != pd.api.types.pandas_dtype(_PANDAS_DTYPES[spec])
    ]
    if wrong:
        raise SchemaViolationError(f"{name}: {'; '.join(wrong)}")


def coerce_frame(df, schema: dict[str, str], name: str):
    """Reorder to schema order and cast every column. Missing or extra columns are still
    an error; only dtypes are fixed up."""
    missing = [column for column in schema if column not in df.columns]
    unexpected = [column for column in df.columns if column not in schema]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected {', '.join(unexpected)}")
        raise SchemaViolationError(f"{name}: {'; '.join(problems)}")

    out = df.loc[:, list(schema)].copy()
    for column, spec in schema.items():
        target = _PANDAS_DTYPES[spec]
        if out[column].dtype != pd.api.types.pandas_dtype(target):
            out[column] = out[column].astype(target)
    return out.reset_index(drop=True)


def empty_frame(schema: dict[str, str]):
    """Typed empty DataFrame, so extractors that find nothing still return valid output."""
    return pd.DataFrame(
        {column: pd.Series([], dtype=_PANDAS_DTYPES[spec]) for column, spec in schema.items()}
    )
