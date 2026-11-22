"""Signal files with the behaviour measured on the video put on their own samples, and nirspipe's events and bad spans."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import ConfigError, MacawError
from macaw.media.align import UNALIGNED
from macaw.signals.align import needs_person, signal_clock, signal_offsets
from macaw.signals.table import SignalFile

# what an --events-path template and a --bad-spans table can name, in file name order
BIDS_KEYS = ("sub", "ses", "task", "acq", "run")
BAD_SPAN_COLUMNS = {"sub": "participant_id", "ses": "session", "task": "task", "acq": "acq", "run": "run"}
EVENT_COLUMNS = ("onset", "duration", "trial_type")


def bids_entities(path: Path) -> dict[str, str]:
    """sub, task and whichever of ses, acq and run a BIDS file name carries."""
    found = {key: match.group(1) for key in BIDS_KEYS if (match := re.search(rf"(?:^|_){key}-([A-Za-z0-9]+)(?=_|$)", path.stem))}
    missing = [key for key in ("sub", "task") if key not in found]
    if missing:
        raise ConfigError(
            f"{path.name} has no {' and no '.join(f'{key}-' for key in missing)} in its name; nirspipe needs BIDS names "
            "such as sub-01_task-play_nirs.snirf to know whose run it is"
        )
    return found


def events_name(entities: dict[str, str]) -> str:
    return "_".join(f"{key}-{entities[key]}" for key in BIDS_KEYS if key in entities) + "_events.tsv"


def events_template(entities: dict[str, str]) -> str:
    return "_".join(f"{key}-{{{key}}}" for key in BIDS_KEYS if key in entities) + "_events.tsv"


def ready_clock(config, recording_id: str, item: SignalFile) -> tuple[float, float]:
    """(offset_s, clock_scale) of a file whose offset may be used; says what is missing otherwise."""
    result = signal_offsets(config, recording_id).get(item.key)
    if result is None:
        raise MacawError(f"{recording_id}: {item.key} has no offset yet; run signals.align")
    if result.method == UNALIGNED:
        raise MacawError(f"{recording_id}: {item.key} is not aligned ({result.note}); set its offset on the Signals page")
    waiting = needs_person(result, config.signals) if config.signals.review != "skip" else None
    if waiting:
        raise MacawError(f"{recording_id}: {item.key}'s offset waits for a person ({waiting}); keep or set it on the Signals page")
    return signal_clock(result)


def recording_times(own: np.ndarray, clock: tuple[float, float]) -> np.ndarray:
    offset, scale = clock
    return offset + np.asarray(own, dtype=float) * scale


def own_times(recorded: np.ndarray, clock: tuple[float, float]) -> np.ndarray:
    offset, scale = clock
    return (np.asarray(recorded, dtype=float) - offset) / scale


def measure_column(row) -> str:
    who = row.subject_id if pd.notna(row.subject_id) else None
    if who is not None and pd.notna(row.subject_id_b):
        who = f"{who}-{row.subject_id_b}"
    return f"{who}.{row.feature}" if who else str(row.feature)


def measures_on_samples(rows: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Each frame measure averaged over the frames nearer each sample time than its neighbours', a column per person and
    measure; a sample with no frame is empty. Averaging over the sample's span keeps faster frames from aliasing."""
    times = np.asarray(times, dtype=float)
    if rows.empty or times.size == 0:
        return pd.DataFrame(index=range(times.size))
    middles = (times[1:] + times[:-1]) / 2
    first = times[0] - (middles[0] - times[0] if middles.size else 0.5)
    last = times[-1] + (times[-1] - middles[-1] if middles.size else 0.5)
    edges = np.concatenate([[first], middles, [last]])
    at = rows["t_start"].to_numpy(dtype=float)
    index = np.searchsorted(edges, at, side="right") - 1
    keep = (index >= 0) & (index < times.size) & np.isfinite(rows["value"].to_numpy(dtype=float))
    found = rows[keep].assign(_sample=index[keep])
    if found.empty:
        return pd.DataFrame(index=range(times.size))
    found["_column"] = [measure_column(row) for row in found[["subject_id", "subject_id_b", "feature"]].itertuples(index=False)]
    wide = found.groupby(["_sample", "_column"], observed=True)["value"].mean().unstack("_column")
    wide = wide.reindex(range(times.size))
    wide.columns.name = None
    return wide[sorted(wide.columns)]


def signal_table(samples: pd.DataFrame, clock: tuple[float, float], measures: pd.DataFrame) -> pd.DataFrame:
    """The file's samples on its own clock, their time on the recording's clock, then the measures."""
    out = samples.reset_index(drop=True).copy()
    out.insert(1, "recording_time", recording_times(out["time"].to_numpy(), clock))
    return pd.concat([out, measures.reset_index(drop=True)], axis=1)


def event_spans(config, recording_id: str, annotations: pd.DataFrame, marks, segments) -> tuple[pd.DataFrame, list[str]]:
    """(start_s, end_s, trial_type) on the recording's clock for each name in signals.events, and the names this
    recording has nothing of. A tier's code is its trial_type, after its person's role when it belongs to one."""
    spans, absent = [], []
    for name in config.signals.events:
        tier, _, coder = name.partition("/")
        coded = annotations[annotations["tier"].astype(str) == tier] if not annotations.empty else annotations
        if coder:
            coded = coded[coded["annotator"].astype(str) == coder]
        elif not coded.empty and coded["annotator"].nunique() > 1:
            coders = ", ".join(sorted(coded["annotator"].astype(str).unique()))
            raise ConfigError(f"{recording_id}: {tier} was coded by {coders}; name one in signals.events, as {tier}/<coder>")
        named = [(mark.start_s, mark.end_s, mark.name) for mark in marks if mark.name == name]
        named += [(segment.start_s, segment.end_s, segment.name) for segment in segments if segment.name == name]
        if not coded.empty and named:
            raise ConfigError(f"{recording_id}: {name} names both a coding tier and a mark or segment; rename one of them")
        for row in coded.itertuples(index=False):
            label = "" if pd.isna(row.label) else str(row.label).strip()
            abstained = getattr(row, "abstained", None)
            if not label or (pd.notna(abstained) and bool(abstained)):
                continue
            who = None if pd.isna(row.subject_id) else str(row.subject_id)
            spans.append((float(row.t_start), float(row.t_end), f"{who}.{label}" if who else label))
        spans += named
        if coded.empty and not named:
            absent.append(name)
    return pd.DataFrame(spans, columns=["start_s", "end_s", "trial_type"]), absent


def on_file(spans: pd.DataFrame, clock: tuple[float, float], first_s: float, last_s: float) -> tuple[pd.DataFrame, int]:
    """Spans as onset and duration on a file's own clock, cut to the file's samples; and how many fell outside it."""
    if spans.empty:
        return pd.DataFrame(columns=list(EVENT_COLUMNS)), 0
    start = np.maximum(own_times(spans["start_s"], clock), first_s)
    end = np.minimum(own_times(spans["end_s"], clock), last_s)
    inside = end > start
    out = pd.DataFrame({"onset": start[inside], "duration": (end - start)[inside], "trial_type": spans["trial_type"].to_numpy()[inside]})
    return out.sort_values(["onset", "trial_type"], kind="stable").reset_index(drop=True), int((~inside).sum())


def bad_span_rows(spans: pd.DataFrame, entities: dict[str, str]) -> pd.DataFrame:
    labels = {column: (f"sub-{entities[key]}" if key == "sub" and key in entities else entities.get(key, "")) for key, column in BAD_SPAN_COLUMNS.items()}
    return pd.DataFrame([{**labels, "onset": row.onset, "duration": row.duration} for row in spans.itertuples(index=False)],
                        columns=[*BAD_SPAN_COLUMNS.values(), "onset", "duration"])


def write_tsv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, float_format="%.4f", encoding="utf-8")
    return path
