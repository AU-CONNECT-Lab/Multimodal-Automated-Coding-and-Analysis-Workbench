"""SNIRF files read with h5py: the data and its channels, the stimulus markers, and the auxiliary channels such as an IMU."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import UnreadableSignalError

# seconds per unit of the file's TimeUnit tag; any other value is taken as seconds, which most writers mean
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6}
# dataType of the specification for continuous-wave amplitude
CW_AMPLITUDE = 1
HAEMOGLOBIN = ("hbo", "hbr", "hbt")
MARKER_COLUMNS = ("name", "onset", "duration")


@dataclass
class AuxSeries:
    times: np.ndarray
    values: np.ndarray
    unit: str | None = None


@dataclass
class Snirf:
    times: np.ndarray                  # s on the file's own clock
    data: np.ndarray                   # one column per channel
    channels: list[str]                # named as MNE names them: "S1_D1 760", "S1_D1 hbo"
    kinds: list[str]                   # raw | od | hbo | hbr | hbt, or the file's own label
    markers: pd.DataFrame              # name, onset, duration in s on the file's own clock
    aux: dict[str, AuxSeries] = field(default_factory=dict)
    started: str | None = None         # MeasurementDate and MeasurementTime as written, joined by T
    time_unit: str = "s"

    @property
    def rate_hz(self) -> float | None:
        steps = np.diff(self.times)
        return float(1.0 / np.median(steps)) if steps.size and np.median(steps) > 0 else None


def _text(node) -> str:
    value = node[()]
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    return value.decode("utf-8", "replace").strip() if isinstance(value, bytes) else str(value).strip()


def _numbered(group, stem: str) -> list[str]:
    """The children named stem, stem1, stem2 ... in numeric order."""
    pattern = re.compile(rf"{stem}(\d*)")
    found = [(int(match.group(1) or 0), name) for name in group if (match := pattern.fullmatch(name))]
    return [name for _, name in sorted(found)]


def _time_axis(raw, n: int, scale: float) -> np.ndarray:
    """A time vector, or the specification's [start, spacing] for evenly sampled data."""
    raw = np.asarray(raw, dtype=float).ravel()
    if raw.size == 2 and n != 2:
        return (raw[0] + raw[1] * np.arange(n)) * scale
    if raw.size != n:
        raise ValueError(f"{raw.size} time points for {n} samples")
    return raw * scale


def _field(group, name: str, index: int | None = None):
    if name not in group:
        return None
    value = np.asarray(group[name][()]).ravel()
    if index is not None:
        value = value[index : index + 1]
    if not value.size:
        return None
    first = value[0]
    return first.decode("utf-8", "replace").strip() if isinstance(first, bytes) else first


def _measurements(block) -> list[dict]:
    keys = ("sourceIndex", "detectorIndex", "wavelengthIndex", "dataType", "dataTypeLabel")
    if "measurementLists" in block:   # the columnar form of SNIRF 1.1
        lists = block["measurementLists"]
        count = np.asarray(lists["sourceIndex"][()]).size
        return [{key: _field(lists, key, index) for key in keys} for index in range(count)]
    return [{key: _field(block[name], key) for key in keys} for name in _numbered(block, "measurementList")]


def _channel(measurement: dict, wavelengths: np.ndarray) -> tuple[str, str]:
    source, detector = int(measurement["sourceIndex"] or 0), int(measurement["detectorIndex"] or 0)
    label = str(measurement["dataTypeLabel"] or "").strip().lower()
    if label in HAEMOGLOBIN:
        return f"S{source}_D{detector} {label}", label
    index = int(measurement["wavelengthIndex"] or 0)
    wavelength = f"{wavelengths[index - 1]:g}" if 0 < index <= wavelengths.size else str(index)
    if int(measurement["dataType"] or 0) == CW_AMPLITUDE:
        kind = "raw"
    else:
        kind = "od" if label in ("dod", "od") else (label or "other")
    return f"S{source}_D{detector} {wavelength}", kind


def _unique(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for name in names:
        counts[name] = counts.get(name, 0) + 1
        out.append(name if counts[name] == 1 else f"{name} #{counts[name]}")
    return out


def _markers(nirs, scale: float) -> pd.DataFrame:
    rows = []
    for name in _numbered(nirs, "stim"):
        stim = nirs[name]
        label = _text(stim["name"]) if "name" in stim else name
        table = np.asarray(stim["data"][()], dtype=float) if "data" in stim else np.empty((0, 3))
        if table.ndim == 1:   # a single marker is often stored flat
            table = table.reshape(1, -1) if table.size >= 2 else np.empty((0, 3))
        rows.extend((label, float(onset) * scale, float(duration) * scale) for onset, duration in table[:, :2])
    return pd.DataFrame(rows, columns=list(MARKER_COLUMNS)).sort_values("onset", kind="stable").reset_index(drop=True)


def _aux(nirs, scale: float) -> dict[str, AuxSeries]:
    found: dict[str, AuxSeries] = {}
    for name in _numbered(nirs, "aux"):
        group = nirs[name]
        if "dataTimeSeries" not in group or "time" not in group:
            continue
        label = _text(group["name"]) if "name" in group else name
        values = np.asarray(group["dataTimeSeries"][()], dtype=float)
        raw_time = np.asarray(group["time"][()], dtype=float).ravel()
        if values.ndim == 2 and values.shape[0] != raw_time.size and values.shape[1] == raw_time.size:
            values = values.T
        values = values.reshape(values.shape[0], -1)
        times = _time_axis(raw_time, values.shape[0], scale)
        offset = _field(group, "timeOffset")
        if offset is not None:
            times = times + float(offset) * scale
        unit = _text(group["dataUnit"]) if "dataUnit" in group else None
        columns = [label] if values.shape[1] == 1 else [f"{label}_{k + 1}" for k in range(values.shape[1])]
        for column, series in zip(columns, values.T):
            key = column if column not in found else f"{column} #{len(found) + 1}"
            found[key] = AuxSeries(times, series, unit or None)
    return found


def read_snirf(path: Path) -> Snirf:
    import h5py

    path = Path(path)
    try:
        handle = h5py.File(path, "r")
    except OSError as error:
        raise UnreadableSignalError(f"{path} is not a SNIRF file: {error}") from None
    try:
        with handle:
            blocks = _numbered(handle, "nirs")
            if not blocks:
                raise UnreadableSignalError(f"{path} has no nirs group, so it is not a SNIRF file")
            nirs = handle[blocks[0]]
            tags = nirs["metaDataTags"] if "metaDataTags" in nirs else {}
            unit = _text(tags["TimeUnit"]) if "TimeUnit" in tags else "s"
            scale = TIME_UNITS.get(unit, 1.0)
            started = None
            if "MeasurementDate" in tags and "MeasurementTime" in tags:
                started = f"{_text(tags['MeasurementDate'])}T{_text(tags['MeasurementTime'])}"
            datas = _numbered(nirs, "data")
            if not datas:
                raise UnreadableSignalError(f"{path} has no data block")
            block = nirs[datas[0]]
            values = np.asarray(block["dataTimeSeries"][()], dtype=float)
            values = values[:, None] if values.ndim == 1 else values
            measurements = _measurements(block)
            if measurements and values.shape[1] != len(measurements) and values.shape[0] == len(measurements):
                values = values.T
            if measurements and values.shape[1] != len(measurements):
                raise UnreadableSignalError(f"{path}: {values.shape[1]} data columns but {len(measurements)} channels described")
            times = _time_axis(block["time"][()], values.shape[0], scale)
            probe = nirs["probe"] if "probe" in nirs else {}
            wavelengths = np.asarray(probe["wavelengths"][()], dtype=float).ravel() if "wavelengths" in probe else np.array([])
            named = [_channel(m, wavelengths) for m in measurements] or [(f"channel {k + 1}", "other") for k in range(values.shape[1])]
            return Snirf(
                times=times,
                data=values,
                channels=_unique([name for name, _ in named]),
                kinds=[kind for _, kind in named],
                markers=_markers(nirs, scale),
                aux=_aux(nirs, scale),
                started=started,
                time_unit=unit,
            )
    except (KeyError, ValueError, TypeError) as error:
        raise UnreadableSignalError(f"{path} could not be read as SNIRF: {error}") from None
