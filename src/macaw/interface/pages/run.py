"""Choose recordings and steps, start `macaw run`, and watch it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dash import Input, Output, State, ctx, dcc, html

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.extractors import get_extractor, list_extractors
from macaw.extractors.base import Gate
from macaw.interface import cli_bridge, ui
from macaw.project.layout import layout_for

LABEL = "Run"
# one run at a time per project, as the run lock enforces anyway
_CURRENT: dict[str, cli_bridge.LaunchedCommand] = {}


def steps() -> list[str]:
    return [name for name in list_extractors() if not issubclass(get_extractor(name), Gate)]


def run_command(config_path: Path, recordings: list[str], chosen: list[str], every_recording: list[str]) -> list[str]:
    narrowed = recordings if set(recordings) != set(every_recording) else None
    return cli_bridge.build_command(["run"], config_path=config_path, recordings=narrowed, extractors=chosen)


def layout(config_path: Path):
    config = resolve_config(config_path)
    recordings = resolve_recordings(config, None)
    names = steps()
    default = [name for name in names if name in config.pipeline.run] or names
    return html.Div(
        [
            ui.section(
                "What to run",
                html.Div(
                    [
                        html.Div([html.H3("Recordings"), dcc.Checklist(recordings, recordings, id="run-recordings", className="checks")]),
                        html.Div([html.H3("Steps"), dcc.Checklist(names, default, id="run-steps", className="checks")]),
                    ],
                    className="two-columns",
                ),
                html.Div(id="run-command"),
                html.Div(
                    [
                        html.Button("Preview", id="run-preview", className="button"),
                        html.Button("Run", id="run-start", className="button primary"),
                        html.Button("Stop", id="run-stop", className="button"),
                    ],
                    className="buttons",
                ),
                html.Div(id="run-message"),
                desc="Steps a chosen step needs run too, and finished ones are not run again. Preview shows the plan without running it.",
            ),
            ui.section("Progress", html.Div(id="run-progress"), dcc.Interval(id="run-poll", interval=2000)),
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
        parts.append(html.P(f"Latest execution {execution}: " + ", ".join(f"{count} {name}" for name, count in counts.items() if count)))
    if current is not None:
        parts.append(html.Pre("\n".join(cli_bridge.tail_log(current.log_path, n_lines=60)) or "(no output yet)", className="log"))
    return parts


def register(app, config_path: Path) -> None:
    @app.callback(Output("run-command", "children"), Input("run-recordings", "value"), Input("run-steps", "value"))
    def show_command(recordings, chosen):
        every = resolve_recordings(resolve_config(config_path), None)
        return ui.command(run_command(config_path, recordings or [], chosen or [], every))

    @app.callback(
        Output("run-message", "children"),
        Input("run-preview", "n_clicks"), Input("run-start", "n_clicks"), Input("run-stop", "n_clicks"),
        State("run-recordings", "value"), State("run-steps", "value"),
        prevent_initial_call=True,
    )
    def act(_preview, _start, _stop, recordings, chosen):
        key = str(config_path)
        if not recordings or not chosen:
            return ui.note("Choose at least one recording and one step.", "warn")
        argv = run_command(config_path, recordings, chosen, resolve_recordings(resolve_config(config_path), None))
        current = _CURRENT.get(key)
        if ctx.triggered_id == "run-preview":
            return ui.output(*cli_bridge.run_now([*argv, "--dry-run"], cwd=config_path.parent))
        if ctx.triggered_id == "run-stop":
            if current is None or not cli_bridge.is_running(current):
                return ui.note("Nothing is running.")
            cli_bridge.cancel(current)
            return ui.note("Stopped, with every process it started.", "warn")
        if current is not None and cli_bridge.is_running(current):
            return ui.note("A run is still going; stop it first.", "warn")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / f"run_{stamp}.log"
        _CURRENT[key] = cli_bridge.launch(argv, log, cwd=config_path.parent)
        return ui.note(f"Started; the full output goes to {log}.", "ok")

    @app.callback(Output("run-progress", "children"), Input("run-poll", "n_intervals"))
    def poll(_):
        return progress(config_path)
