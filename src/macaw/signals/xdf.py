"""XDF files Lab Streaming Layer's recorder wrote: a stream of numbers read as a SNIRF file is, with the file's events."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, UnreadableSignalError
from macaw.signals.snirf import MARKER_COLUMNS, Snirf

SUFFIXES = (".xdf", ".xdfz")
# a stream of numbers whose type says it holds codes, as an experiment program's trigger stream does
EVENT_TYPES = ("marker", "trigger", "event", "stim")


def is_xdf(path: Path) -> bool:
    return path.suffix.lower() in SUFFIXES


def load_xdf(path: Path) -> tuple[list[dict], dict]:
    """(streams, file header) as pyxdf reads them, every stream on the recording computer's clock."""
    try:
        import pyxdf
    except ImportError:
        raise BackendNotAvailableError("signals.read reads XDF files with pyxdf: pip install pymacaw[signals]") from None
    try:
        return pyxdf.load_xdf(str(path), verbose=False)
    except Exception as error:   # pyxdf raises plain exceptions of several kinds on a broken file
        raise UnreadableSignalError(f"{path.name}: cannot read the XDF file: {error}") from None


def _first(info: dict, key: str, default: str = "") -> str:
    value = info.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return default if value is None else str(value).strip()


def holds_events(info: dict) -> bool:
    kind = _first(info, "type").lower()
    return _first(info, "channel_format") == "string" or any(word in kind for word in EVENT_TYPES)


def describe_stream(info: dict) -> str:
    extra = [f"{key} {_first(info, key)}" for key in ("type", "source_id", "hostname") if _first(info, key)]
    return f"{_first(info, 'name')!r}" + (f" ({', '.join(extra)})" if extra else "")


def data_streams(streams: list[dict]) -> list[dict]:
    return [stream for stream in streams if not holds_events(stream["info"])]


def pick_stream(streams: list[dict], name: str, wanted: str | None) -> dict:
    """The stream of numbers named wanted, by its name or else its source_id; with none named, the file's only one."""
    candidates = data_streams(streams)
    listed = "; ".join(describe_stream(stream["info"]) for stream in candidates) or "none"
    if wanted is None:
        filled = [stream for stream in candidates if len(stream["time_stamps"])]
        if len(filled) == 1:
            return filled[0]
        if not filled:
            raise UnreadableSignalError(f"{name} holds no stream of numbers with samples ({listed})")
        raise UnreadableSignalError(
            f"{name} holds {len(candidates)} streams of numbers: {listed}. Name the one this row means in the signals "
            "table's stream column, by its name, or its source_id where two share a name"
        )
    for key in ("name", "source_id"):
        found = [stream for stream in candidates if _first(stream["info"], key) == wanted]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise UnreadableSignalError(
                f"{name}: {len(found)} streams are called {wanted!r}: "
                f"{'; '.join(describe_stream(stream['info']) for stream in found)}. Write the source_id of the one meant instead"
            )
    raise UnreadableSignalError(f"{name} has no stream of numbers called {wanted!r}; its streams of numbers: {listed}")


def channel_labels(info: dict) -> tuple[list[str], list[str]]:
    """(labels, kinds) of a stream's channels from its description, numbered where the description gives none."""
    count = int(float(_first(info, "channel_count", "0")))
    labels, kinds = [], []
    desc = info.get("desc")
    desc = desc[0] if isinstance(desc, list) and desc else desc
    try:
        channels = desc["channels"][0]["channel"] if desc else []
    except (KeyError, IndexError, TypeError):
        channels = []
    for channel in channels if len(channels) == count else []:
        labels.append(_first(channel, "label"))
        kinds.append(_first(channel, "type"))
    if not labels:
        labels, kinds = [""] * count, [""] * count
    stream_kind = _first(info, "type").lower() or "value"
    out, seen = [], set()
    for index, label in enumerate(labels, start=1):
        label = label or f"ch{index}"
        if label in seen:
            label = f"{label} {index}"
        seen.add(label)
        out.append(label)
    return out, [kind.lower() or stream_kind for kind in kinds]


def event_name(sample) -> str | None:
    values = [str(value).strip() for value in np.atleast_1d(sample) if str(value).strip()]
    if not values:
        return None
    named = []
    for value in values:
        try:
            number = float(value)
            named.append(str(int(number)) if number.is_integer() else value)
        except ValueError:
            named.append(value)
    return "/".join(named)


def file_events(streams: list[dict]) -> pd.DataFrame:
    """Every event stream's events, on the file's clock."""
    rows = []
    for stream in streams:
        if not holds_events(stream["info"]):
            continue
        for at, sample in zip(stream["time_stamps"], stream["time_series"]):
            name = event_name(sample)
            if name is not None:
                rows.append((name, float(at), 0.0))
    found = pd.DataFrame(rows, columns=list(MARKER_COLUMNS))
    return found.sort_values("onset", kind="stable", ignore_index=True)


def _started(header: dict, streams: list[dict], first: float) -> str | None:
    """The wall-clock time of the stream's first sample, taking the file's written time as its earliest sample's."""
    written = _first(header.get("info", {}), "datetime")
    stamps = [float(stream["time_stamps"][0]) for stream in streams if len(stream["time_stamps"])]
    if not written or not stamps:
        return None
    try:
        start = datetime.strptime(written, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
    except ValueError:
        try:
            start = datetime.fromisoformat(written).replace(tzinfo=None)
        except ValueError:
            return None
    return (start + timedelta(seconds=first - min(stamps))).isoformat()


def read_xdf(path: Path, stream: str | None, loaded: tuple[list[dict], dict] | None = None) -> Snirf:
    """The named stream of numbers, its times on the file's clock in s, and the file's events."""
    streams, header = loaded if loaded is not None else load_xdf(path)
    chosen = pick_stream(streams, path.name, stream)
    times = np.asarray(chosen["time_stamps"], dtype=float)
    if not times.size:
        raise UnreadableSignalError(f"{path.name}: the stream {describe_stream(chosen['info'])} holds no samples")
    data = np.asarray(chosen["time_series"], dtype=float).reshape(times.size, -1)
    labels, kinds = channel_labels(chosen["info"])
    if len(labels) != data.shape[1]:
        labels, kinds = [f"ch{index}" for index in range(1, data.shape[1] + 1)], [kinds[0] if kinds else "value"] * data.shape[1]
    return Snirf(times=times, data=data, channels=labels, kinds=kinds, markers=file_events(streams),
                 started=_started(header, streams, float(times[0])))
