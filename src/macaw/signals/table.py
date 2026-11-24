"""The project's signals table: which file of other signals, or stream of an XDF file, belongs to which recording, person and stage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from macaw.exceptions import ConfigError
from macaw import people_tables

COLUMNS = ("recording", "person", "stage", "file")
# the stream of an XDF file a row means, by its name or source_id; the column may be left out
STREAM = "stream"
DEFAULT_STAGE = "raw"
# what an empty person stands for: a file of the whole recording, which no role can be called
EVERYONE = "all"
_STAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_UNSAFE = re.compile(r"[^A-Za-z0-9-]+")


@dataclass(frozen=True)
class SignalFile:
    recording: str
    person: str | None
    stage: str
    path: Path
    stream: str | None = None

    @property
    def key(self) -> str:
        base = f"{self.person or EVERYONE}_{self.stage}"
        return f"{base}_{_UNSAFE.sub('-', self.stream).strip('-') or 'stream'}" if self.stream else base

    @property
    def clock(self) -> str:
        """What the file's own times count on: a person's SNIRF files share their device's, an XDF file's streams the file's."""
        suffix = self.path.suffix.lower()
        if suffix == ".snirf":
            return f"snirf {self.person or EVERYONE}"
        if suffix in (".xdf", ".xdfz"):
            return f"xdf {self.path}"
        return f"own {self.key}"


def signals_path(config) -> Path:
    return Path(config.project.signals)


def read_signals(config) -> dict[str, list[SignalFile]]:
    """recording -> its files in table order; no table means none. A relative file is taken from the table's folder."""
    path = signals_path(config)
    if not path.is_file():
        return {}
    frame = people_tables.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise ConfigError(f"{path} has no {', '.join(missing)} column; it needs {', '.join(COLUMNS)}")
    frame = frame.apply(lambda column: column.str.strip())
    found: dict[str, list[SignalFile]] = {}
    seen: set[tuple[str, str | None, str, str | None]] = set()
    for index, row in frame[(frame != "").any(axis=1)].iterrows():
        line = index + 2   # the header is line 1
        recording, person, stage, name = row["recording"], row["person"] or None, row["stage"] or DEFAULT_STAGE, row["file"]
        stream = row.get(STREAM) or None
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
        if (recording, person, stage, stream) in seen:
            said = f"the stream {stream} at {stage}" if stream else stage
            raise ConfigError(f"{path} line {line}: {said} of {person or 'the whole recording'} in {recording} is given twice")
        seen.add((recording, person, stage, stream))
        file = Path(name)
        if stream and file.suffix.lower() not in (".xdf", ".xdfz"):
            raise ConfigError(f"{path} line {line}: a stream is named, but {file.name} is not an XDF file; leave stream empty")
        found.setdefault(recording, []).append(
            SignalFile(recording, person, stage, file if file.is_absolute() else (path.parent / file).resolve(), stream)
        )
    return found


def signal_files(config, recording_id: str) -> list[SignalFile]:
    return read_signals(config).get(recording_id, [])


def stream_mates(files: list[SignalFile], chosen: list[SignalFile]) -> list[SignalFile]:
    """The chosen files and the other listed streams of any XDF file among them, which share its clock, in table order."""
    shared = {item.clock for item in chosen if item.clock.startswith("xdf ")}
    return [item for item in files if item in chosen or item.clock in shared]
