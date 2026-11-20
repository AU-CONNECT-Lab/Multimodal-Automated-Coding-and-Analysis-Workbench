"""Where each recording stands: every step's state, and each check by a person, waiting or decided by whom."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.dag import build_dag
from macaw.execution.job_db import JobDB
from macaw.execution.runner import resolve_recordings, task_id_for, task_states
from macaw.extractors import get_extractor, list_extractors
from macaw.extractors.base import Gate
from macaw.project.dictionary import fixed_entry, write_dictionary
from macaw.project.layout import layout_for
from macaw.project.tables import DICTIONARY_COLUMNS

# the coded units waiting for a person count as a check of their own, after the judge
SPEECH_ACTS = "speech acts"


def steps() -> list[str]:
    return [name for name in list_extractors() if not issubclass(get_extractor(name), Gate)]

def failed_last_time(config_path: Path) -> set[str]:
    database = JobDB(layout_for(resolve_config(config_path), create=False).job_db_path)
    known = database.executions()
    return {record.task_id for record in database.failed(known[-1])} if known else set()

def task_grid(config_path: Path, recordings: list[str]) -> tuple[list[str], list[str], dict[tuple[str, str], str], dict[str, list[str]]]:
    """Every step in run order, the ones this project runs, the state of each (recording, step), and what each step needs."""
    config = resolve_config(config_path)
    everything = build_dag(config, steps())
    project = set(build_dag(config).topological_order())
    states = task_states(config, everything, recordings)
    failed = failed_last_time(config_path)
    order = [name for name in everything.topological_order() if not issubclass(get_extractor(name), Gate)]
    grid = {}
    for name in order:
        for rid in recordings:
            state = states[task_id_for(rid, name)]
            grid[rid, name] = "failed" if task_id_for(rid, name) in failed and state != "current" else state
    needs = {name: sorted(everything.dependencies(name)) for name in order}
    return order, [name for name in order if name in project], grid, needs

def people_waiting(config) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Each recording's gates waiting for a person, and the coded units waiting for one across the project."""
    from macaw.coding.review_queue import export_review
    from macaw.project.recording import build_recording
    from macaw.qc.report import _waiting

    recordings = resolve_recordings(config, None)
    gates = {}
    for rid in recordings:
        try:
            found = _waiting(config, build_recording(rid, config))
        except MacawError:
            continue
        if found:
            gates[rid] = found
    try:
        units = len(export_review(config, recordings))
    except Exception:   # no codebook or nothing coded: no units are waiting
        units = 0
    return gates, units

def check_states(config_path: Path, recordings: list[str], grid: dict, gates: dict) -> tuple[list[str], dict[tuple[str, str], str], dict[str, list[str]]]:
    """The project's steps with its checks by a person among them, per recording whether each check waits, who
    decided it, or that it has not been reached, and what each column needs."""
    from macaw.coding.review_queue import export_review
    from macaw.execution.dag import build_dag
    from macaw.extractors import get_extractor
    from macaw.extractors.base import Gate
    from macaw.project.layout import layout_for
    from macaw.project.recording import build_recording

    config = resolve_config(config_path)
    dag = build_dag(config)
    columns = list(dag.topological_order())
    names = [name for name in columns if issubclass(get_extractor(name), Gate)]
    judged = "coding.judge" in columns
    if judged:
        columns.insert(columns.index("coding.judge") + 1, SPEECH_ACTS)
    states: dict[tuple[str, str], str] = {}
    try:
        queue = export_review(config, recordings) if judged else None
    except Exception:   # no codebook or nothing coded: no units wait
        queue = None
    waiting_units = set(queue["recording_id"].astype(str)) if queue is not None and not queue.empty else set()
    layout = layout_for(config, create=False)
    for rid in recordings:
        held = {gate for gate, _ in gates.get(rid, [])}
        try:
            own = resolve_config(config_path, recording_id=rid)
            recording = build_recording(rid, own)
        except MacawError:
            recording = None
        for name in names:
            reached = all(grid.get((rid, need)) == "current" for need in dag.dependencies(name) if (rid, need) in grid)
            if name in held:
                states[rid, name] = "waiting"
            elif not reached or recording is None:
                states[rid, name] = "new"
            else:
                states[rid, name] = get_extractor(name)(own).decided_by(recording) or "none"
        if judged:
            states[rid, SPEECH_ACTS] = (
                "waiting" if rid in waiting_units else "new" if grid.get((rid, "coding.judge")) != "current"
                else "person" if layout.reviewed_codes_path(rid).is_file() else "software"
            )
    shown = [name for name in columns if name in names or name == SPEECH_ACTS or (recordings and (recordings[0], name) in grid)]
    needs = {name: sorted(set(dag.dependencies(name)) & set(shown)) for name in shown if name != SPEECH_ACTS}
    if judged:
        needs[SPEECH_ACTS] = ["coding.judge"]
    return shown, states, needs


def write_status(config_path: Path, path: Path) -> Path:
    """The Overview's grid as a table: a row per recording, a column per step and per check by a person."""
    config = resolve_config(config_path)
    recordings = resolve_recordings(config, None)
    _, steps_run, grid, _ = task_grid(config_path, recordings)
    gates, _ = people_waiting(config)
    columns, checks, _ = check_states(config_path, recordings, grid, gates)
    rows = [
        {"recording_id": rid, **{name: checks.get((rid, name), grid.get((rid, name))) for name in columns}}
        for rid in recordings
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["recording_id", *columns]).to_csv(path, index=False, encoding="utf-8")
    checked = {name for _, name in checks}
    write_dictionary([fixed_entry("recording_id"), *(status_entry(name, name in checked) for name in columns)], path, DICTIONARY_COLUMNS)
    return path


def status_entry(name: str, check: bool) -> dict:
    if check:
        meaning = (
            f"Where the check by a person {name} stands for the recording: waiting (for a person), person (decided by a person), "
            "software (left to the software, as its settings allow), none (nothing to check) or new (not reached yet)."
        )
    else:
        meaning = (
            f"Where the step {name} stands for the recording: current (up to date), stale (to run again, its inputs or settings "
            "changed), new (not run yet), held (waiting for a check by a person before it) or failed (failed in the last run)."
        )
    return {"column": name, "meaning": meaning, "unit": None, "missing": "Never empty.", "step": name}
