"""The project at a glance: how much there is, what waits for whom, and where each recording stands."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

from dash import Input, Output, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.pages import results, review, run, setup
from macaw.interface.step_labels import label
from macaw.project.setup import setup_path

LABEL = "Overview"
PATH = "/overview"
# steps listed under Next up before the rest are counted as "and N more"
NEXT_STEPS = 6
REVIEW_PATH = f"/{review.LABEL.lower()}"
# the grid's key, in the order a project moves through
LEGEND = ("current", "stale", "new", "held", "failed")
# a check by a person has a round square of its own beside the steps
CHECK_TEXT = {"waiting": "waits for a person", "person": "checked by a person", "software": "left to the software", "none": "nothing to check", "new": "not reached yet"}
CHECK_LEGEND = ("person", "software", "waiting", "none")
SPEECH_ACTS = "speech acts"
CHECK_LABELS = {SPEECH_ACTS: "Check: speech acts"}


@dataclass
class Standing:
    recordings: list[str]
    steps: list[str]
    grid: dict[tuple[str, str], str]
    waiting: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    units_waiting: int = 0
    task_s: float = 0.0
    n_prepared: int = 0
    setup_saved: bool = True
    # the steps and the checks by a person in run order, and each check's state per recording
    columns: list[str] = field(default_factory=list)
    checks: dict[tuple[str, str], str] = field(default_factory=dict)


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


def standing(config_path: Path) -> Standing:
    from macaw.extractors.prep import load_prep

    config = resolve_config(config_path)
    recordings = resolve_recordings(config, None)
    _, steps, grid, _ = run.task_grid(config_path, recordings)
    gates, units = people_waiting(config)
    task_s, prepared = 0.0, 0
    for rid in recordings:
        try:
            task_s += load_prep(config, rid).duration_s
        except MacawError:
            continue
        prepared += 1
    columns, checks = check_states(config_path, recordings, grid, gates)
    return Standing(recordings, steps, grid, gates, units, task_s, prepared, setup_path(config).is_file(), columns, checks)


def check_states(config_path: Path, recordings: list[str], grid: dict, gates: dict) -> tuple[list[str], dict[tuple[str, str], str]]:
    """The project's steps with its checks by a person among them, and per recording whether each check waits, who
    decided it, or that it has not been reached."""
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
    return [name for name in columns if name in names or name == SPEECH_ACTS or (recordings and (recordings[0], name) in grid)], states


def column_label(name: str) -> str:
    return CHECK_LABELS.get(name) or label(name)


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{seconds / 60:.0f} min"


def _tile(label: str, value: str, sub: str | None = None):
    return html.Div([html.Div(label, className="label"), html.Div(value, className="value"),
                     *([html.Div(sub, className="sub")] if sub else [])], className="tile")


def tiles(config, found: Standing):
    n = len(found.recordings)
    tasks = [found.grid[rid, step] for rid in found.recordings for step in found.steps]
    current = sum(state == "current" for state in tasks)
    if not found.n_prepared:
        time = _tile("Task time", "–", "known once prep.media has run")
    else:
        time = _tile("Task time", _duration(found.task_s),
                     f"over {n} recordings" if found.n_prepared == n else f"over the {found.n_prepared} of {n} recordings prepared")
    return [
        _tile("Recordings", str(n)),
        _tile("People in each", str(config.project.n_subjects), ", ".join(config.project.roles)),
        time,
        _tile("Up to date", f"{round(100 * current / len(tasks))}%" if tasks else "–", f"{current} of {len(tasks)} tasks" if tasks else None),
    ]


def _names(recordings: list[str], limit: int = 4) -> str:
    return ", ".join(recordings[:limit]) + (f" and {len(recordings) - limit} more" if len(recordings) > limit else "")


def _item(kind: str, text: str, detail: str | None, href: str, where: str):
    return html.Li(dcc.Link(
        [html.Span(className=f"dot {kind}"),
         html.Span([text, *([html.Span(detail, className="detail")] if detail else [])], className="what"),
         html.Span(where, className="where")],
        href=href,
    ))


def next_up(found: Standing) -> list:
    """What failed, what waits for a person, then what is left to run, each linked to the page that deals with it."""
    items = []
    if not found.setup_saved:
        items.append(_item("bad", "Describe the setup", "the cameras and microphones; nothing runs before it is saved", setup.PATH, setup.LABEL))
    for step in found.steps:
        failed = [rid for rid in found.recordings if found.grid[rid, step] == "failed"]
        if failed:
            items.append(_item("bad", f"{label(step)} failed", f"{step}, for {_names(failed)}", run.PATH, "Run"))
    if found.waiting:
        n_checks = sum(len(gates) for gates in found.waiting.values())
        detail = "; ".join(f"{rid}: {', '.join(label(name) for name, _ in gates)}" for rid, gates in list(found.waiting.items())[:4])
        more = len(found.waiting) - 4
        items.append(_item("warn", f"{n_checks} check{'s wait' if n_checks != 1 else ' waits'} for a person",
                           detail + (f"; and {more} more recordings" if more > 0 else ""), REVIEW_PATH, "Review"))
    if found.units_waiting:
        items.append(_item("warn", f"{found.units_waiting} coded unit{'s wait' if found.units_waiting != 1 else ' waits'} for review",
                           None, REVIEW_PATH, "Review"))
    tasks = [found.grid[rid, step] for rid in found.recordings for step in found.steps]
    if tasks and all(state == "new" for state in tasks):
        items.append(_item("", "Nothing has run yet", f"{len(found.steps)} steps for {len(found.recordings)} recordings", run.PATH, "Run"))
        return items
    left = []
    for step in found.steps:
        counts = Counter(found.grid[rid, step] for rid in found.recordings)
        parts = [f"{counts[state]} {run.STATE_TEXT[state]}" for state in ("stale", "new") if counts[state]]
        if parts:
            left.append(_item("", label(step), f"{step}: " + ", ".join(parts), run.PATH, "Run"))
    items += left[:NEXT_STEPS]
    if len(left) > NEXT_STEPS:
        items.append(_item("", f"and {len(left) - NEXT_STEPS} more steps to run", None, run.PATH, "Run"))
    return items


def cell_link(rid: str, step: str, state: str) -> tuple[str, str]:
    """Where a square leads, and that page's name: what it produced, the checks it waits on, or the Run page."""
    if state == "current":
        return f"{results.PATH}?{urlencode({'recording': rid, 'step': step})}", results.LABEL
    if state == "held":
        return f"{REVIEW_PATH}?{urlencode({'recording': rid})}", review.LABEL
    return f"{run.PATH}?{urlencode({'recording': rid, 'step': step})}", run.LABEL


def _cell(rid: str, step: str, state: str):
    href, page = cell_link(rid, step, state)
    return html.Td(dcc.Link(html.Span(className=f"cell {state}"), href=href, title=f"{rid}, {label(step)} ({step}): {run.STATE_TEXT[state]}. Opens {page}."))


def _check_cell(rid: str, name: str, state: str):
    href = f"{REVIEW_PATH}?{urlencode({'recording': rid})}"
    return html.Td(dcc.Link(html.Span(className=f"cell check {state}"), href=href, title=f"{rid}, {column_label(name)}: {CHECK_TEXT[state]}. Opens Review."))


def grid(found: Standing):
    columns = found.columns or found.steps
    head = html.Tr([html.Th(), *[
        html.Th(html.Span(column_label(name), title=name), className="step-name check" if (found.recordings and (found.recordings[0], name) in found.checks) else "step-name")
        for name in columns
    ]])
    rows = [
        html.Tr([html.Th(rid, className="recording"), *[
            _check_cell(rid, name, found.checks[rid, name]) if (rid, name) in found.checks else _cell(rid, name, found.grid[rid, name])
            for name in columns
        ]])
        for rid in found.recordings
    ]
    legend = [html.Span([html.Span(className=f"cell {state}"), run.STATE_TEXT[state]]) for state in LEGEND]
    if found.checks:
        legend += [html.Span([html.Span(className=f"cell check {state}"), CHECK_TEXT[state]]) for state in CHECK_LEGEND]
    return [html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="standing"), className="standing-wrap"),
            html.Div(legend, className="legend")]


def latest_run(config_path: Path) -> str:
    execution, counts = cli_bridge.poll_status(config_path)
    if execution is None:
        return "Nothing has run yet."
    # execution ids start with when they began, 20261031-214220-...
    stamp = re.match(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", execution)
    when = f"{stamp[1]}-{stamp[2]}-{stamp[3]} {stamp[4]}:{stamp[5]}" if stamp else execution
    return f"Latest run {when}: " + ", ".join(f"{count} {name}" for name, count in counts.items() if count)


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head("Overview", html.Span(id="overview-summary")),
            html.Div(id="overview-message"),
            dcc.Loading(
                [
                    html.Div(id="overview-tiles", className="tiles"),
                    ui.card("Next up", html.Ul(id="overview-next", className="next"), flush=True),
                    ui.card("Where each recording stands", html.Div(id="overview-grid"),
                            desc="Point at a square for its state, click it to go there. Round ones are checks by a person.",
                            flush=True),
                ],
                delay_show=300,
            ),
        ]
    )


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("overview-summary", "children"), Output("overview-message", "children"), Output("overview-tiles", "children"),
        Output("overview-next", "children"), Output("overview-grid", "children"),
        Input("url", "pathname"),
    )
    def fill(pathname):
        if (pathname or "/").rstrip("/") not in ("", PATH):
            return (no_update,) * 5
        config = resolve_config(config_path)
        try:
            found = standing(config_path)
        except Exception as error:   # the page still opens; the Run page and the run itself say what is wrong
            return latest_run(config_path), ui.note(f"Could not work out where the recordings stand: {error}", "bad"), [], [], []
        items = next_up(found) or [html.Li(html.P("Nothing to do: every step is up to date and nothing waits for a person.", className="none"))]
        return latest_run(config_path), None, tiles(config, found), items, grid(found)
