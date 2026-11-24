"""The coherence between two people's signals over time, from the wavelet maps nirspipe-hyper saves with --wtc-save-maps."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import UnreadableSignalError
from macaw.signals.snirf import MARKER_COLUMNS, Snirf

# what nirspipe puts between the parts of a map's key
SEPARATOR = "\x1f"
MEAN = "mean"


@dataclass
class Coherence:
    series: Snirf                                 # time on the members' shared clock, a column per pair and their mean
    band_hz: tuple[float, float]
    members: dict[str, float]                     # BIDS subject -> s into its own recording where the shared clock starts
    sources: list[str] = field(default_factory=list)
    masked_cone: bool = True


def is_coherence(path: Path) -> bool:
    return path.name.lower().endswith("_stat-wtc_relmat.npz")


def band_series(coherence: np.ndarray, cone: np.ndarray, freqs: np.ndarray, band: tuple[float, float], mask_cone: bool) -> np.ndarray:
    """Per time point, the mean squared coherence over the band's frequencies; with mask_cone, only those at or above
    1 / cone of influence, the cells free of edge effects; NaN where none is left."""
    inside = (freqs >= band[0]) & (freqs <= band[1])
    cells = np.asarray(coherence, dtype=float)[inside]
    keep = np.isfinite(cells)
    if mask_cone:
        with np.errstate(divide="ignore"):
            keep &= freqs[inside][:, None] >= 1.0 / np.asarray(cone, dtype=float)[None, :]
    total, count = np.where(keep, cells, 0.0).sum(axis=0), keep.sum(axis=0)
    out = np.full(cells.shape[1], np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out


def read_coherence(path: Path) -> Coherence:
    """Each pair's band-mean coherence over time and their mean, with the band and each member's start from the
    sidecar nirspipe writes beside the maps."""
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise UnreadableSignalError(f"{path.name}: no {sidecar.name} beside it, which says the band and where each member starts")
    params = json.loads(sidecar.read_text(encoding="utf-8")).get("parameters", {})
    if not params.get("aligned") or not params.get("align_offset_s"):
        raise UnreadableSignalError(f"{sidecar.name} does not say where each member starts on the shared clock (align_offset_s)")
    band = (float(params["band_fmin"]), float(params["band_fmax"]))
    mask_cone = bool(params.get("mask_coi", True))
    try:
        npz = np.load(path)
    except (OSError, ValueError) as error:
        raise UnreadableSignalError(f"{path.name}: cannot read the maps: {error}") from None
    with npz:
        freqs, times = np.asarray(npz["freqs"], dtype=float), np.asarray(npz["times"], dtype=float)
        pairs = [name.split(SEPARATOR)[1:] for name in npz.files if name.startswith("wtc" + SEPARATOR)]
        several = len({(first, second) for first, second, *_ in pairs}) > 1
        columns = {}
        for first, second, label, other in pairs:
            key = SEPARATOR.join((first, second, label, other))
            series = band_series(npz["wtc" + SEPARATOR + key], npz["coi" + SEPARATOR + key], freqs, band, mask_cone)
            excluded = SEPARATOR.join(("excluded", first, second))
            if excluded in npz.files:
                series[np.asarray(npz[excluded], dtype=bool)[: series.size]] = np.nan
            name = f"{label}-{other}" if other else label
            columns[f"{first}-{second} {name}" if several else name] = series
    if not columns:
        raise UnreadableSignalError(f"{path.name} holds no coherence map")
    data = np.column_stack(list(columns.values()))
    known = np.isfinite(data)
    mean = np.full(times.size, np.nan)
    np.divide(np.where(known, data, 0.0).sum(axis=1), known.sum(axis=1), out=mean, where=known.any(axis=1))
    found = Snirf(times=times, data=np.column_stack([mean, data]), channels=[MEAN, *columns], kinds=["coherence"] * (len(columns) + 1),
                  markers=pd.DataFrame(columns=list(MARKER_COLUMNS)))
    return Coherence(series=found, band_hz=band, members={str(k): float(v) for k, v in params["align_offset_s"].items()},
                     sources=list(json.loads(sidecar.read_text(encoding="utf-8")).get("Sources", [])), masked_cone=mask_cone)
