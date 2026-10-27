"""Look at what a run produced: the QC page, and any measure over time per person."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.project.layout import layout_for
from macaw.qc.checks import RecordingResults
from macaw.qc.downsample import min_max_downsample
from macaw.store.schema import Scale

LABEL = "Results"
# colours of the QC report figures, one per person in order
COLOURS = ["#1d4e89", "#d1495b", "#2a9d8f", "#e9a03b", "#6c5b7b", "#57606a"]


def layout(config_path: Path):
    recordings = resolve_recordings(resolve_config(config_path), None)
    return html.Div(
        [
            html.Div(
                [dcc.Dropdown(recordings, recordings[0] if recordings else None, id="results-recording", clearable=False, className="picker")],
                className="toolbar",
            ),
            ui.section(
                "Quality review",
                html.Div(id="results-report"),
                html.Div(html.Button("Write the QC page", id="results-write-report", className="button"), className="buttons"),
                html.Div(id="results-report-message"),
            ),
            ui.section(
                "Measures",
                html.Div(
                    [
                        dcc.Dropdown(id="results-source", placeholder="step", className="picker"),
                        dcc.Dropdown(id="results-feature", placeholder="measure", className="picker wide"),
                    ],
                    className="toolbar",
                ),
                dcc.Loading(html.Div(id="results-view")),
                desc="A measure over time per person, or its value per person when it is one number for the recording.",
            ),
        ]
    )


def sources(config_path: Path, rid: str | None) -> list[str]:
    if rid is None:
        return []
    directory = layout_for(resolve_config(config_path), create=False).long_table_path(rid)
    return sorted(path.stem for path in directory.glob("*.parquet"))


def view(config_path: Path, rid: str, source: str, feature: str):
    config = resolve_config(config_path)
    rows = RecordingResults(layout_for(config, create=False), rid).long_table(source, features=[feature])
    if rows.empty:
        return html.P("No values.", className="none")
    who = rows["subject_id"].astype(object).where(rows["subject_id"].notna(), "everyone")
    pair = rows["subject_id_b"].notna()
    who[pair] = rows.loc[pair, "subject_id"].astype(str) + " & " + rows.loc[pair, "subject_id_b"].astype(str)
    rows = rows.assign(who=who)
    if set(rows["scale"].astype(str)) == {Scale.RECORDING.value}:
        return ui.table(["Person", feature], [[name, f"{value:.4g}"] for name, value in zip(rows["who"], rows["value"])], numeric={feature})
    figure = go.Figure()
    for index, (name, group) in enumerate(rows.sort_values("t_start").groupby("who", sort=True)):
        t, y = min_max_downsample(group["t_start"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float), config.qc.report_downsample_px)
        mode = "lines" if np.isfinite(y).sum() > 200 else "lines+markers"
        figure.add_trace(go.Scattergl(x=t, y=y, name=str(name), mode=mode, line={"width": 1, "color": COLOURS[index % len(COLOURS)]}))
    figure.update_layout(
        template="plotly_white", height=420, margin={"l": 50, "r": 20, "t": 20, "b": 40},
        xaxis_title="recording time (s)", yaxis_title=feature, legend={"orientation": "h", "y": 1.08},
    )
    return dcc.Graph(figure=figure, config={"displaylogo": False})


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("results-report", "children"),
        Output("results-report-message", "children"),
        Input("results-recording", "value"),
        Input("results-write-report", "n_clicks"),
    )
    def report(rid, _clicks):
        if rid is None:
            return html.P("No recordings found.", className="none"), None
        message = None
        if ctx.triggered_id == "results-write-report":
            argv = cli_bridge.build_command(["qc"], config_path=config_path, recordings=[rid], extra_flags=["--report"])
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        page = layout_for(resolve_config(config_path), create=False).qc_report_path(rid)
        link = (
            html.P([
                html.A("Open the QC page", href=file_url(page), target="_blank"),
                html.Span(f" written {datetime.fromtimestamp(page.stat().st_mtime):%Y-%m-%d %H:%M}", className="hint"),
            ])
            if page.is_file() else html.P("No QC page yet for this recording.", className="none")
        )
        return link, message

    @app.callback(Output("results-source", "options"), Output("results-source", "value"), Input("results-recording", "value"))
    def pick_source(rid):
        found = sources(config_path, rid)
        return found, found[0] if found else None

    @app.callback(
        Output("results-feature", "options"), Output("results-feature", "value"),
        Input("results-source", "value"), State("results-recording", "value"),
    )
    def pick_feature(source, rid):
        if not source or not rid:
            return [], None
        rows = RecordingResults(layout_for(resolve_config(config_path), create=False), rid).long_table(source)
        features = sorted(rows["feature"].astype(str).unique())
        return features, features[0] if features else None

    @app.callback(
        Output("results-view", "children"),
        Input("results-feature", "value"), State("results-source", "value"), State("results-recording", "value"),
    )
    def show(feature, source, rid):
        if not (feature and source and rid):
            return html.P("Nothing to show yet.", className="none")
        return view(config_path, rid, source, feature)
