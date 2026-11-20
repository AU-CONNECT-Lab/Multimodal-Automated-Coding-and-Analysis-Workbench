"""The project's signals table: which file of other signals belongs to which recording, person and stage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macaw.exceptions import ConfigError

COLUMNS = ("recording", "person", "stage", "file")
DEFAULT_STAGE = "raw"
# what an empty person stands for: a file of the whole recording, which no role can be called
EVERYONE = "all"
_STAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class SignalFile:
    recording: str
    person: str | None
    stage: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.person or EVERYONE}_{self.stage}"


def signals_path(config) -> Path:
    return Path(config.project.signals)


def read_signals(config) -> dict[str, list[SignalFile]]:
    """recording -> its files in table order; no table means none. A relative file is taken from the table's folder."""
    path = signals_path(config)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(COLUMNS)}")
    frame = frame.apply(lambda column: column.str.strip())
    found: dict[str, list[SignalFile]] = {}
    seen: set[tuple[str, str | None, str]] = set()
    for index, row in frame[(frame != "").any(axis=1)].iterrows():
        line = index + 2   # the header is line 1
        recording, person, stage, name = row["recording"], row["person"] or None, row["stage"] or DEFAULT_STAGE, row["file"]
        if not recording:
            raise ConfigError(f"{path} line {line}: no recording")
        if person is not None and person not in config.project.roles:
            raise ConfigError(
                f"{path} line {line}: {person!r} is not one of project.roles ({', '.join(config.project.roles)}); "
                "leave it empty for a file of the whole recording"
            )
        if not _STAGE.fullmatch(stage):
            raise ConfigError(f"{path} line {line}: stage {stage!r} should be letters, digits, - and _, such as raw or preproc")
        if not name:
            raise ConfigError(f"{path} line {line}: no file")
        if (recording, person, stage) in seen:
            raise ConfigError(f"{path} line {line}: {stage} of {person or 'the whole recording'} in {recording} is given twice")
        seen.add((recording, person, stage))
        file = Path(name)
        found.setdefault(recording, []).append(
            SignalFile(recording, person, stage, file if file.is_absolute() else (path.parent / file).resolve())
        )
    return found


def signal_files(config, recording_id: str) -> list[SignalFile]:
    return read_signals(config).get(recording_id, [])
