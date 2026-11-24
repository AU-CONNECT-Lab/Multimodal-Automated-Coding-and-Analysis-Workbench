"""The signals table's files read once into the project, each on its own clock."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, InputNotFoundError, UnreadableSignalError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, Phase, QCWarning
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.signals.snirf import Snirf
from macaw.signals.table import SignalFile, read_signals, signal_files, signals_path
from macaw.store.schema import Scale

# a step this far from the median spacing, as a share of it, means the samples are not evenly spaced
UNEVEN = 0.01
SNIRF, TABLE, COHERENCE, XDF = "snirf", "table", "coherence", "xdf"


def samples_name(item: SignalFile) -> str:
    return f"{item.key}.parquet"


def aux_name(item: SignalFile) -> str:
    return f"{item.key}_aux.parquet"


def info_name(item: SignalFile) -> str:
    return f"{item.key}.json"


def read_dir(config, recording_id: str) -> Path:
    return layout_for(config, create=False).artifacts_dir(recording_id, SignalReadExtractor.name)


def load_samples(config, recording_id: str, item: SignalFile) -> pd.DataFrame:
    """time on the file's own clock, then one column per channel."""
    return pd.read_parquet(read_dir(config, recording_id) / samples_name(item))


def load_aux(config, recording_id: str, item: SignalFile) -> pd.DataFrame:
    """aux, time on the file's own clock, value: one row per sample of each auxiliary channel."""
    return pd.read_parquet(read_dir(config, recording_id) / aux_name(item))


def load_info(config, recording_id: str, item: SignalFile) -> dict:
    return json.loads((read_dir(config, recording_id) / info_name(item)).read_text(encoding="utf-8"))


@register_extractor
class SignalReadExtractor(Extractor):
    name = "signals.read"
    version = "0.3.0"
    scale = Scale.RECORDING
    group = "signals"
    phase = Phase.PREP

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(read_signals(config))

    def check_available(self) -> None:
        from macaw.signals.xdf import is_xdf

        listed = [item for items in read_signals(self.config).values() for item in items]
        for wanted, module, kind in ((lambda path: path.suffix.lower() == ".snirf", "h5py", "SNIRF"), (is_xdf, "pyxdf", "XDF")):
            if not any(wanted(item.path) for item in listed):
                continue
            try:
                __import__(module)
            except ImportError:
                raise BackendNotAvailableError(f"signals.read reads {kind} files with {module}: pip install pymacaw[signals]") from None

    def inputs_for(self, recording: Recording) -> list[Path]:
        return sorted({item.path for item in signal_files(self.config, recording.recording_id) if item.path.is_file()})

    def extra_key(self, recording: Recording) -> str:
        return json.dumps([[item.person, item.stage, item.path.name, item.stream]
                           for item in signal_files(self.config, recording.recording_id)])

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        from macaw.signals.snirf import read_snirf

        from macaw.signals.coherence import is_coherence, read_coherence
        from macaw.signals.tables import is_table, read_table
        from macaw.signals.xdf import is_xdf, load_xdf, read_xdf

        rid = recording.recording_id
        artifacts: dict[str, Path] = {}
        warnings: list[QCWarning] = []
        loaded: dict[Path, tuple] = {}   # an XDF file with several streams listed is read once
        for item in signal_files(self.config, rid):
            if not item.path.is_file():
                raise InputNotFoundError(f"{rid}: {item.path}, listed in {signals_path(self.config)}, does not exist")
            info, skipped = {}, []
            if item.path.suffix.lower() == ".snirf":
                found = read_snirf(item.path)
            elif is_xdf(item.path):
                if item.path not in loaded:
                    loaded[item.path] = load_xdf(item.path)
                found = read_xdf(item.path, item.stream, loaded[item.path])
                info = {"format": XDF, "stream": item.stream}
            elif is_table(item.path):
                found, skipped = read_table(item.path)
                info = {"format": TABLE}
            elif is_coherence(item.path):
                coherence = read_coherence(item.path)
                found = coherence.series
                info = {"format": COHERENCE, "coherence": {
                    "band_hz": list(coherence.band_hz), "members": coherence.members, "sources": coherence.sources,
                    "masked_cone": coherence.masked_cone}}
            else:
                raise UnreadableSignalError(
                    f"{rid}: {item.path.name} is none of what is read: a .snirf file, an .xdf file, a table (.csv or .tsv with a "
                    "time column), or the coherence maps nirspipe-hyper saves (*_stat-wtc_relmat.npz)"
                )
            samples = pd.DataFrame(found.data, columns=found.channels)
            samples.insert(0, "time", found.times)
            samples.to_parquet(workspace / samples_name(item), index=False)
            _aux_table(found).to_parquet(workspace / aux_name(item), index=False)
            (workspace / info_name(item)).write_text(json.dumps({**_info(item, found), **info}, indent=1), encoding="utf-8")
            artifacts.update({name: workspace / name for name in (samples_name(item), aux_name(item), info_name(item))})
            warnings.extend(self._checks(rid, item, found))
            if skipped:
                warnings.append(QCWarning(check=self.name, recording_id=rid, subject_id=item.person, severity="info",
                                          message=f"{item.path.name}: {', '.join(skipped)} hold no numbers and were left out"))
        return ExtractorResult(rows=pd.DataFrame(), artifacts=artifacts, warnings=warnings)

    def _checks(self, recording_id: str, item: SignalFile, found: Snirf) -> list[QCWarning]:
        out = []
        common = dict(check=self.name, recording_id=recording_id, subject_id=item.person)
        steps = np.diff(found.times)
        if steps.size and np.median(steps) > 0 and np.max(np.abs(steps - np.median(steps))) > UNEVEN * np.median(steps):
            out.append(QCWarning(**common, severity="warn", message=(
                f"{item.path.name}: the samples are not evenly spaced (from {steps.min():.4g} to {steps.max():.4g} s apart); "
                "each is kept at its own time"
            )))
        if not found.data.size or not np.isfinite(found.data).any():
            out.append(QCWarning(**common, severity="warn", message=f"{item.path.name}: the data holds no numbers"))
        return out


def _aux_table(found: Snirf) -> pd.DataFrame:
    pieces = [pd.DataFrame({"aux": name, "time": series.times, "value": series.values}) for name, series in found.aux.items()]
    if not pieces:
        return pd.DataFrame({"aux": pd.Series(dtype=str), "time": pd.Series(dtype=float), "value": pd.Series(dtype=float)})
    return pd.concat(pieces, ignore_index=True)


def _info(item: SignalFile, found: Snirf) -> dict:
    return {
        "file": str(item.path),
        "person": item.person,
        "stage": item.stage,
        "format": SNIRF,
        "started": found.started,
        "time_unit": found.time_unit,
        "rate_hz": found.rate_hz,
        "start_s": float(found.times[0]) if found.times.size else None,
        "end_s": float(found.times[-1]) if found.times.size else None,
        "channels": [{"name": name, "kind": kind} for name, kind in zip(found.channels, found.kinds)],
        "markers": [
            {"name": str(row.name), "onset": float(row.onset), "duration": float(row.duration)}
            for row in found.markers.itertuples(index=False)
        ],
        "aux": [{"name": name, "unit": series.unit, "samples": int(series.values.size)} for name, series in found.aux.items()],
    }
