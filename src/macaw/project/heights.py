"""People's heights, from the project's heights table, by which one camera's 3D is set to each person's true size."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from macaw.exceptions import ConfigError

COLUMNS = ("recording", "person", "height")


def heights_path(config) -> Path:
    return Path(config.project.heights)


def read_heights(config) -> dict[tuple[str, str], float]:
    """(recording, person) -> standing height in project units; no table means none."""
    path = heights_path(config)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(COLUMNS)}")
    frame = frame.apply(lambda column: column.str.strip())
    found: dict[tuple[str, str], float] = {}
    for index, row in frame[(frame != "").any(axis=1)].iterrows():
        line = index + 2   # the header is line 1
        recording, person = row["recording"], row["person"]
        if person not in config.project.roles:
            raise ConfigError(f"{path} line {line}: {person!r} is not one of project.roles ({', '.join(config.project.roles)})")
        try:
            height = float(row["height"])
        except ValueError:
            height = float("nan")
        if not math.isfinite(height) or height <= 0:
            raise ConfigError(f"{path} line {line}: height '{row['height']}' is not a length in {config.project.units}")
        if (recording, person) in found:
            raise ConfigError(f"{path} line {line}: {person} in {recording} is given twice")
        found[(recording, person)] = height
    return found
