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
from macaw.interface.step_labels import label, short
from macaw.project.setup import setup_path

LABEL = "Overview"
FLOW = "overview-flow"
FLOW_DATA = "overview-flow-data"
PATH = "/overview"
REVIEW_PATH = f"/{review.LABEL.lower()}"
# the grid's key, in the order a project moves through
LEGEND = ("current", "stale", "new", "held", "failed")
# a check by a person has a round square of its own beside the steps
CHECK_TEXT = {"waiting": "waits for a person", "person": "checked by a person", "software": "left to the software", "none": "nothing to check", "new": "not reached yet"}
CHECK_LEGEND = ("person", "software", "waiting", "none")
SPEECH_ACTS = "speech acts"
CHECK_LABELS = {SPEECH_ACTS: "Check: speech acts"}
CHECK_SHORT = {SPEECH_ACTS: "Acts checked"}


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
    needs: dict[str, list[str]] = field(default_factory=dict)


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
    columns, checks, needs = check_states(config_path, recordings, grid, gates)
    return Standing(recordings, steps, grid, gates, units, task_s, prepared, setup_path(config).is_file(), columns, checks, needs)


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


def column_label(name: str) -> str:
    return CHECK_LABELS.get(name) or label(name)


def column_short(name: str) -> str:
    return CHECK_SHORT.get(name) or short(name)


def _duration(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{seconds / 60:.0f} min"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _tile(label: str, value: str, sub: str | None = None, href: str | None = None, where: str | None = None,
          kind: str = "", meter: float | None = None):
    head = html.Div([label, *([html.Span(f"{where} ›")] if where else [])], className="label")
    body = [head, html.Div(value, className="value"), *([html.Div(sub, className="sub")] if sub else []),
            *([html.Div(html.I(style={"width": f"{100 * meter:.1f}%"}), className="meter")] if meter is not None else [])]
    if href:
        return dcc.Link(body, href=href, className=f"tile {kind}".strip())
    return html.Div(body, className=f"tile {kind}".strip())


def tiles(config, found: Standing):
    n = len(found.recordings)
    tasks = [found.grid[rid, step] for rid in found.recordings for step in found.steps]
    current = sum(state == "current" for state in tasks)
    people = ", ".join(config.project.roles)
    if not found.n_prepared:
        time = "task time known once prep.media has run"
    elif found.n_prepared == n:
        time = f"{_duration(found.task_s)} of task time"
    else:
        time = f"{_duration(found.task_s)} over the {found.n_prepared} prepared"
    failed = [(rid, step) for rid in found.recordings for step in found.steps if found.grid[rid, step] == "failed"]
    checks = sum(len(gates) for gates in found.waiting.values())
    waits = checks + found.units_waiting
    return [
        _tile("Recordings", str(n), f"{people} · {time}" if people else time),
        _tile("Up to date", f"{round(100 * current / len(tasks))}%" if tasks else "–", f"{current} of {len(tasks)} tasks" if tasks else None,
              run.PATH, "Run", meter=current / len(tasks) if tasks else None),
        _tile("Failed last time", str(len(failed)),
              f"{_plural(len({step for _, step in failed}), 'step')}, {_plural(len({rid for rid, _ in failed}), 'recording')}" if failed else "nothing failed",
              run.PATH, "Run", "bad" if failed else ""),
        _tile("Waits for you", str(waits), f"{_plural(checks, 'check')}, {_plural(found.units_waiting, 'coded unit')}",
              REVIEW_PATH, "Review", "warn" if waits else ""),
    ]


def setup_message(found: Standing):
    if found.setup_saved:
        return None
    return ui.note(["Describe the setup first: nothing runs before it is saved. ", dcc.Link("Open Setup", href=setup.PATH)], "warn")


# columns are grouped by what they work on, the group named by the step's first part
LANES = (("Prepare", ("prep", "align", "segments")), ("Speech", ("text", "turns")), ("Voice", ("prosody",)),
         ("Body", ("pose", "motion")), ("Face", ("face",)), ("Gaze", ("gaze",)), ("Coding", ("coding",)))
OTHER_LANE = "Other"
# a step's state in the workflow, by what is most pressing among its recordings
FLOW_ORDER = ("failed", "held", "waiting", "stale", "new", "software", "none", "person", "current")
DONE = ("current", "person", "software", "none")


def lane_of(name: str) -> str:
    if name == SPEECH_ACTS:
        return "Coding"
    first = name.split(".")[0]
    return next((lane for lane, prefixes in LANES if first in prefixes), OTHER_LANE)


def _is_check(found: Standing, name: str) -> bool:
    return bool(found.recordings) and (found.recordings[0], name) in found.checks


def _state(found: Standing, rid: str, name: str) -> str:
    return found.checks[rid, name] if (rid, name) in found.checks else found.grid[rid, name]


def depths(columns: list[str], needs: dict[str, list[str]]) -> dict[str, int]:
    level: dict[str, int] = {}

    def depth(name: str) -> int:
        if name not in level:
            level[name] = 1 + max((depth(need) for need in needs.get(name, []) if need in columns), default=-1)
        return level[name]

    for name in columns:
        depth(name)
    return level


def lanes(found: Standing) -> list[tuple[str, list[str]]]:
    """The columns in their lanes, each lane in run order."""
    columns = found.columns or found.steps
    level = depths(columns, found.needs)
    order = [lane for lane, _ in LANES] + [OTHER_LANE]
    grouped = {lane: sorted((name for name in columns if lane_of(name) == lane), key=lambda name: (level[name], columns.index(name)))
               for lane in order}
    return [(lane, names) for lane, names in grouped.items() if names]


def direct_needs(columns: list[str], needs: dict[str, list[str]]) -> dict[str, list[str]]:
    """What each column needs, leaving out what it needs only through another of its needs."""
    above: dict[str, set[str]] = {}

    def ancestors(name: str) -> set[str]:
        if name not in above:
            above[name] = set()
            for need in needs.get(name, []):
                if need in columns:
                    above[name] |= {need} | ancestors(need)
        return above[name]

    own = {name: [need for need in needs.get(name, []) if need in columns] for name in columns}
    return {name: [need for need in mine if not any(other != need and need in ancestors(other) for other in mine)] for name, mine in own.items()}


def workflow_data(found: Standing) -> dict:
    """What the workflow drawing needs: lanes, each step's place and state, and the direct needs between steps."""
    columns = found.columns or found.steps
    level = depths(columns, found.needs)
    grouped = lanes(found)
    nodes = []
    for lane, names in grouped:
        for name in names:
            is_check = _is_check(found, name)
            counts = Counter(_state(found, rid, name) for rid in found.recordings)
            worst = next((state for state in FLOW_ORDER if counts.get(state)), "new")
            stuck = counts["failed"] or counts["waiting"] + counts["held"]
            nodes.append({
                "id": name, "label": column_label(name), "lane": lane, "depth": level[name], "check": is_check, "state": worst,
                "issue": [stuck, "failed" if counts["failed"] else "waiting"] if stuck else None,
                "href": REVIEW_PATH if is_check else f"{run.PATH}?{urlencode({'step': name})}",
                "tip": f"{column_label(name)} ({name})\n" + "\n".join(
                    f"{count} {(CHECK_TEXT if is_check else run.STATE_TEXT)[state]}" for state, count in counts.most_common()),
            })
    edges = [[need, name] for name, needs in direct_needs(columns, found.needs).items() for need in needs]
    return {"lanes": [lane for lane, _ in grouped], "nodes": nodes, "edges": edges}


def cell_link(rid: str, step: str, state: str) -> tuple[str, str]:
    """Where a square leads, and that page's name: what it produced, the checks it waits on, or the Run page."""
    if state == "current":
        return f"{results.PATH}?{urlencode({'recording': rid, 'step': step})}", results.LABEL
    if state == "held":
        return f"{REVIEW_PATH}?{urlencode({'recording': rid})}", review.LABEL
    return f"{run.PATH}?{urlencode({'recording': rid, 'step': step})}", run.LABEL


def _cell(found: Standing, rid: str, name: str):
    state = _state(found, rid, name)
    if _is_check(found, name):
        href, page, text, kind = f"{REVIEW_PATH}?{urlencode({'recording': rid})}", review.LABEL, CHECK_TEXT[state], f"cell check {state}"
    else:
        (href, page), text, kind = cell_link(rid, name, state), run.STATE_TEXT[state], f"cell {state}"
    return html.Td(dcc.Link(html.Span(className=kind), href=href), className="c",
                   **{"data-col": name, "data-rec": rid, "data-label": column_label(name), "data-state": text, "data-page": page})


def row_summary(found: Standing, rid: str) -> str:
    states = Counter(_state(found, rid, name) for name in (found.columns or found.steps))
    waiting, left = states["waiting"] + states["held"], states["new"] + states["stale"]
    parts = [f"{states['failed']} failed" if states["failed"] else "", f"{waiting} wait" if waiting else "", f"{left} to run" if left else ""]
    return ", ".join(part for part in parts if part) or "done"


INSPECT_HINT = [html.H3("Nothing pointed at"), html.Ul([html.Li("Point at a square: its recording, step and state show here."),
                                                        html.Li("Click it to go to the page it names.")])]


def _key(found: Standing, columns: list[str]) -> list:
    def entry(kind: str, text: str, state: str, checks: bool) -> list:
        count = sum(_state(found, rid, name) == state for rid in found.recordings for name in columns if _is_check(found, name) == checks)
        return [html.Span(className=kind), html.Span(text), html.Span(str(count), className="count")]

    rows = [html.Div("Steps", className="head")]
    for state in LEGEND:
        rows += entry(f"cell {state}", run.STATE_TEXT[state], state, False)
    if found.checks:
        rows.append(html.Div("Checks by a person", className="head"))
        for state in CHECK_LEGEND:
            rows += entry(f"cell check {state}", CHECK_TEXT[state], state, True)
    return rows


def grid(found: Standing):
    grouped = lanes(found)
    columns = [name for _, names in grouped for name in names]
    bands, heads = [html.Th(className="recording")], [html.Th(className="recording")]
    for k, (lane, names) in enumerate(grouped):
        if k:
            bands.append(html.Th(className="gap"))
            heads.append(html.Th(className="gap"))
        bands.append(html.Th(lane, colSpan=len(names), className="band"))
        heads += [html.Th(html.Span(column_short(name), title=f"{column_label(name)} ({name})"),
                          className="step-name check" if _is_check(found, name) else "step-name", **{"data-col": name}) for name in names]
    bands.append(html.Th())
    heads.append(html.Th())

    def row(rid: str):
        cells = [html.Th(rid, className="recording")]
        for k, (_, names) in enumerate(grouped):
            if k:
                cells.append(html.Td(className="gap"))
            cells += [_cell(found, rid, name) for name in names]
        summary = row_summary(found, rid)
        cells.append(html.Td(summary, className="summary" + (" bad" if "failed" in summary else " warn" if "wait" in summary else "")))
        return html.Tr(cells)

    done = [rid for rid in found.recordings if all(_state(found, rid, name) in DONE for name in columns)]
    open_ = [rid for rid in found.recordings if rid not in done]
    if done and open_:
        width = len(columns) + len(grouped) + 1
        body = [html.Tbody([row(rid) for rid in open_]),
                html.Tbody(html.Tr(html.Th(f"▸ {_plural(len(done), 'more recording')}, everything done", colSpan=width), className="fold-row")),
                html.Tbody([row(rid) for rid in done], className="done closed")]
    else:
        body = [html.Tbody([row(rid) for rid in found.recordings])]
    table = html.Table([html.Thead(html.Tr(bands)), *body, html.Tfoot(html.Tr(heads))], className="standing")
    n_checks = sum(_is_check(found, name) for name in columns)
    totals = [html.B(str(len(open_))), f" recording{'' if len(open_) == 1 else 's'} with something to do", html.Br(),
              html.B(str(len(done))), " with everything done", html.Br(),
              html.B(str(len(columns) - n_checks)), " steps, ", html.B(str(n_checks)), " checks"]
    side = html.Aside([html.Div(_key(found, columns), className="key"), html.Div(totals, className="totals"),
                       html.Div(INSPECT_HINT, className="inspect")], className="standing-side")
    return html.Div([html.Div(table, className="standing-wrap"), side], className="standing-card")


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
            dcc.Store(id=FLOW_DATA),
            dcc.Loading(html.Div(id="overview-tiles", className="tiles"), delay_show=300),
            ui.card("Workflow", html.Div(id=FLOW, className="flow-canvas"),
                    desc="Each step where it comes in the run. Point at one to see what it needs; click it to go there."),
            dcc.Loading(ui.card("Where each recording stands", html.Div(id="overview-grid"),
                                desc="Recordings with something to do come first.", flush=True), delay_show=300),
        ]
    )


def register(app, config_path: Path) -> None:
    # the workflow is drawn in the browser by assets/overview.js, which also opens a step's page on a click
    app.clientside_callback(
        """function (data) {
            if (window.macawOverview) { window.macawOverview.draw(data); } else { window.macawOverviewPending = data; }
            return window.dash_clientside.no_update;
        }""",
        Output(FLOW, "title"), Input(FLOW_DATA, "data"),
    )

    @app.callback(
        Output("overview-summary", "children"), Output("overview-message", "children"), Output("overview-tiles", "children"),
        Output("overview-grid", "children"), Output(FLOW_DATA, "data"),
        Input("url", "pathname"),
    )
    def fill(pathname):
        if (pathname or "/").rstrip("/") not in ("", PATH):
            return (no_update,) * 5
        config = resolve_config(config_path)
        try:
            found = standing(config_path)
        except Exception as error:   # the page still opens; the Run page and the run itself say what is wrong
            return latest_run(config_path), ui.note(f"Could not work out where the recordings stand: {error}", "bad"), [], [], None
        return latest_run(config_path), setup_message(found), tiles(config, found), grid(found), workflow_data(found)
