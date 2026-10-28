"""Choose recordings and steps, see which are up to date, start `macaw run`, and watch it."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from dash import Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.execution.dag import build_dag, resolve_selection
from macaw.execution.runner import resolve_recordings, task_id_for, task_states
from macaw.extractors import get_extractor, list_extractors
from macaw.extractors.base import Gate
from macaw.interface import cli_bridge, ui
from macaw.project.layout import layout_for

LABEL = "Run"
PATH = "/run"
# one run at a time per project, as the run lock enforces anyway
_CURRENT: dict[str, cli_bridge.LaunchedCommand] = {}
STATE_TEXT = {"failed": "failed last time", "held": "waiting for a person", "stale": "to run again", "new": "not run yet", "current": "up to date"}
STATE_KIND = {"failed": "bad", "held": "warn", "stale": "warn", "new": "info", "current": "ok"}


def steps() -> list[str]:
    return [name for name in list_extractors() if not issubclass(get_extractor(name), Gate)]


def default_steps(config) -> list[str]:
    """What `macaw run` runs with no -e: pipeline.run, or every step that joins a run by default."""
    requested = config.pipeline.run or [name for name in steps() if get_extractor(name).wanted_by_default(config)]
    return resolve_selection(config, requested)


def run_command(config_path: Path, recordings: list[str], chosen: list[str], every_recording: list[str], default: list[str] = ()) -> list[str]:
    narrowed = recordings if set(recordings) != set(every_recording) else None
    named = chosen if set(chosen) != set(default) else None
    return cli_bridge.build_command(["run"], config_path=config_path, recordings=narrowed, extractors=named)


def step_states(config_path: Path, recordings: list[str]) -> tuple[list[str], list[str], dict[str, Counter], dict[str, list[str]]]:
    """Every step in run order, the ones this project runs, each step's task states, and what each needs."""
    config = resolve_config(config_path)
    everything = build_dag(config, steps())
    project = set(build_dag(config).topological_order())
    states = task_states(config, everything, recordings)
    failed = cli_bridge.failed_last_time(config_path)
    order = [name for name in everything.topological_order() if not issubclass(get_extractor(name), Gate)]
    counts = {
        name: Counter(
            "failed" if task_id_for(rid, name) in failed and states[task_id_for(rid, name)] != "current" else states[task_id_for(rid, name)]
            for rid in recordings
        )
        for name in order
    }
    needs = {name: sorted(everything.dependencies(name)) for name in order}
    return order, [name for name in order if name in project], counts, needs


def _step_label(name: str, counts: Counter, needs: list[str], n_recordings: int):
    present = [state for state in STATE_TEXT if counts.get(state)]
    pills = [
        ui.pill(STATE_TEXT[state] if counts[state] == n_recordings else f"{counts[state]} {STATE_TEXT[state]}", STATE_KIND[state])
        for state in present
    ]
    worst = present[0] if present else "new"
    return html.Div(
        [
            html.Span(className=f"dot {STATE_KIND[worst] if worst != 'new' else ''}"),
            html.Span([name, *pills], className="name"),
            *([html.Span("needs " + ", ".join(needs), className="meta")] if needs else []),
        ],
        className="step",
    )


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head(
                "Run",
                html.Span(id="run-summary"),
                [
                    ui.button("Preview", "run-preview", icon_name="preview"),
                    ui.button("Stop", "run-stop", icon_name="stop"),
                    ui.button("Run", "run-start", icon_name="play", primary=True),
                ],
            ),
            html.Div(id="run-message"),
            ui.card("Recordings", dcc.Checklist([], None, id="run-recordings", className="chips", inline=True), desc="Which recordings to run."),
            ui.card(
                "Steps",
                dcc.Checklist([], None, id="run-steps", className="steps"),
                html.Details(
                    [html.Summary(id="run-more-summary"), dcc.Checklist([], None, id="run-more", className="steps")],
                    className="more",
                ),
                html.Div(html.Div(id="run-command"), className="card-body"),
                desc="In the order they run. What a chosen step needs runs too, and what is up to date is not run again.",
                flush=True,
            ),
            ui.card("Progress", html.Div(id="run-progress"), dcc.Interval(id="run-poll", interval=2000), dcc.Store(id="run-state")),
        ]
    )


def progress(config_path: Path):
    current = _CURRENT.get(str(config_path))
    execution, counts = cli_bridge.poll_status(config_path)
    parts = []
    if current is None:
        parts.append(html.P("Nothing started from this window yet.", className="none"))
    else:
        code = cli_bridge.exit_code(current)
        state = ui.pill("running", "warn") if code is None else ui.pill("finished" if code == 0 else f"stopped, exit {code}", "ok" if code == 0 else "bad")
        parts += [html.P([state, f" started {current.started_at}"]), ui.command(current.argv)]
    if execution:
        parts.append(html.P(f"Latest execution {execution}: " + ", ".join(f"{count} {name}" for name, count in counts.items() if count), className="hint"))
    if current is not None:
        parts.append(html.Pre("\n".join(cli_bridge.tail_log(current.log_path, n_lines=60)) or "(no output yet)", className="log"))
    return parts


def _run_state(config_path: Path) -> str | None:
    current = _CURRENT.get(str(config_path))
    return None if current is None else f"{current.started_at}:{cli_bridge.exit_code(current)}"


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("run-recordings", "options"), Output("run-recordings", "value"),
        Output("run-steps", "options"), Output("run-steps", "value"),
        Output("run-more", "options"), Output("run-more", "value"),
        Output("run-more-summary", "children"), Output("run-summary", "children"),
        Input("url", "pathname"), Input("run-state", "data"),
        State("run-recordings", "value"), State("run-steps", "value"), State("run-more", "value"),
    )
    def fill(pathname, _state, recordings, chosen, more):
        if (pathname or "/").rstrip("/") not in ("", PATH):
            return (no_update,) * 8
        config = resolve_config(config_path)
        every = resolve_recordings(config, None)
        recordings = [rid for rid in recordings if rid in every] if recordings is not None else every
        try:
            order, project, counts, needs = step_states(config_path, every)
        except Exception as error:   # the list still shows; the run itself reports what is wrong
            order, project, counts, needs = steps(), default_steps(config), {}, {}
            message = f"Could not work out which steps are up to date: {error}"
        else:
            message = None
        others = [name for name in order if name not in project]
        if chosen is None:
            default = set(default_steps(config))
            chosen, more = [name for name in project if name in default], [name for name in others if name in default]

        def options(names):
            return [{"label": _step_label(name, counts.get(name, Counter()), needs.get(name, []), len(every)), "value": name} for name in names]

        n_tasks = sum(sum(counts.get(name, Counter()).values()) for name in project)
        n_current = sum(counts.get(name, Counter()).get("current", 0) for name in project)
        summary = message or (
            f"{len(project)} steps for {len(every)} recording{'s' if len(every) != 1 else ''}, "
            + ("all up to date" if n_tasks and n_current == n_tasks else f"{n_current} of {n_tasks} tasks up to date")
        )
        return (
            every, recordings, options(project), [name for name in chosen if name in project],
            options(others), [name for name in more or [] if name in others],
            f"Steps this project does not run ({len(others)})", summary,
        )

    @app.callback(
        Output("run-command", "children"),
        Input("run-recordings", "value"), Input("run-steps", "value"), Input("run-more", "value"),
    )
    def show_command(recordings, chosen, more):
        config = resolve_config(config_path)
        argv = run_command(config_path, recordings or [], [*(chosen or []), *(more or [])], resolve_recordings(config, None), default_steps(config))
        return [html.Span("The same in a terminal", className="hint"), ui.command(argv)]

    @app.callback(
        Output("run-message", "children"),
        Input("run-preview", "n_clicks"), Input("run-start", "n_clicks"), Input("run-stop", "n_clicks"),
        State("run-recordings", "value"), State("run-steps", "value"), State("run-more", "value"),
        prevent_initial_call=True,
    )
    def act(_preview, _start, _stop, recordings, chosen, more):
        key = str(config_path)
        chosen = [*(chosen or []), *(more or [])]
        current = _CURRENT.get(key)
        if ctx.triggered_id == "run-stop":
            if current is None or not cli_bridge.is_running(current):
                return ui.note("Nothing is running.")
            cli_bridge.cancel(current)
            return ui.note("Stopped, with every process it started.", "warn")
        if not recordings or not chosen:
            return ui.note("Choose at least one recording and one step.", "warn")
        config = resolve_config(config_path)
        argv = run_command(config_path, recordings, chosen, resolve_recordings(config, None), default_steps(config))
        if ctx.triggered_id == "run-preview":
            return ui.card("Preview", ui.output(*cli_bridge.run_now([*argv, "--dry-run"], cwd=config_path.parent)))
        if current is not None and cli_bridge.is_running(current):
            return ui.note("A run is still going; stop it first.", "warn")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log = layout_for(config, create=False).work_dir / "gui" / f"run_{stamp}.log"
        _CURRENT[key] = cli_bridge.launch(argv, log, cwd=config_path.parent)
        return ui.note(f"Started; the full output goes to {log}.", "ok")

    @app.callback(Output("run-progress", "children"), Output("run-state", "data"), Input("run-poll", "n_intervals"), State("run-state", "data"))
    def poll(_, known):
        state = _run_state(config_path)
        return progress(config_path), state if state != known else no_update
