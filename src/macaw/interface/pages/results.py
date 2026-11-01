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
            ui.page_head("Results", "What a run produced: the quality page, and any measure per person."),
            html.Div(
                ui.field("Recording", dcc.Dropdown(recordings, recordings[0] if recordings else None, id="results-recording", clearable=False, className="picker")),
                className="toolbar",
            ),
            ui.card(
                "Quality review",
                html.Div(id="results-report"),
                html.Div(id="results-report-message"),
                desc="The QC page lists what the checks flagged, the decisions the software made and where to look.",
                actions=ui.button("Write the QC page", "results-write-report", icon_name="refresh"),
            ),
            ui.card(
                "Measures",
                html.Div(
                    [
                        ui.field("Step", dcc.Dropdown(id="results-source", placeholder="step", className="picker")),
                        ui.field("Measure", dcc.Dropdown(id="results-feature", placeholder="measure", className="picker")),
                    ],
                    className="toolbar",
                ),
                dcc.Loading(html.Div(id="results-view")),
                desc="A measure over time per person, or its value per person when it is one number for the recording.",
            ),
            ui.card(
                "In 3D",
                html.Div(dcc.Slider(id="results-3d-time", min=0, max=1, value=0, step=None, marks=None, tooltip={"placement": "bottom"}), id="results-3d-slider"),
                dcc.Loading(html.Div(id="results-3d")),
                desc=(
                    "Each person's skeleton from the cameras together, and where the cameras stand; drag to turn it, move the "
                    "slider to go through the recording. Bones that stretch or people in the wrong place mean a point was misplaced."
                ),
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
    # transparent with mid-grey text and grid, so the figure reads on the light and the dark page alike
    axis = {"gridcolor": "rgba(128,128,128,0.2)", "zerolinecolor": "rgba(128,128,128,0.35)", "linecolor": "rgba(128,128,128,0.35)"}
    figure.update_layout(
        template="plotly_white", height=420, margin={"l": 50, "r": 20, "t": 20, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90"},
        xaxis={"title": "recording time (s)", **axis}, yaxis={"title": feature, **axis}, legend={"orientation": "h", "y": 1.08},
    )
    return dcc.Graph(figure=figure, config={"displaylogo": False})


def points_3d(config_path: Path, rid: str):
    """The recording's triangulated points, or None before pose.triangulate has run."""
    from macaw.extractors.video.pose.triangulate import points_file
    from macaw.store.keypoints import read_keypoints

    path = points_file(resolve_config(config_path), rid)
    return read_keypoints(path, recording_id=rid) if path.is_file() else None


def _shown(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Room axes as the first camera sees them (x right, y down, z ahead) turned so up is up on the page."""
    return xyz[..., 0], xyz[..., 2], -xyz[..., 1]


def view_3d(config_path: Path, rid: str, at_s: float, points=None):
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES, read_setup
    from macaw.extractors.video.pose.skeleton import SKELETON_EDGES
    from macaw.extractors.video.pose.triangulate import calibration_path

    config = resolve_config(config_path)
    points = points_3d(config_path, rid) if points is None else points
    if points is None or points.empty:
        return html.P("No 3D for this recording: it needs pose3d.calibration and the pose.triangulate step.", className="none")
    times = np.sort(points["t"].unique())
    moment = points[points["t"] == times[np.abs(times - at_s).argmin()]]
    figure = go.Figure()
    for index, (subject_id, own) in enumerate(moment.groupby("subject_id", sort=True)):
        colour = COLOURS[index % len(COLOURS)]
        joint = {row.joint: np.array([row.x, row.y, row.z]) for row in own.itertuples()}
        segments = [np.vstack([joint[a], joint[b], [np.nan] * 3]) for a, b in [*SKELETON_EDGES, ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")] if a in joint and b in joint]
        if segments:
            x, y, z = _shown(np.vstack(segments))
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": colour, "width": 6}, name=str(subject_id), legendgroup=str(subject_id)))
        x, y, z = _shown(np.vstack(list(joint.values())))
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={"size": 3, "color": colour}, showlegend=False, legendgroup=str(subject_id), text=list(joint), hoverinfo="text"))
    try:
        setup = read_setup(calibration_path(config), config.pose3d.units)
        where = np.vstack([camera.position() for camera in setup.cameras]) * UNITS_TO_METRES[setup.units]
        x, y, z = _shown(where)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers+text", text=[camera.name for camera in setup.cameras], textposition="top center",
            marker={"size": 5, "symbol": "diamond", "color": "#8b8b90"}, name="cameras",
        ))
    except Exception:   # the cameras are context; the people still show without them
        pass
    axis = {"gridcolor": "rgba(128,128,128,0.25)", "zerolinecolor": "rgba(128,128,128,0.4)", "backgroundcolor": "rgba(0,0,0,0)"}
    figure.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90"},
        legend={"orientation": "h", "y": 1.02},
        scene={
            "aspectmode": "data", "xaxis": {"title": "across (m)", **axis}, "yaxis": {"title": "ahead of the first camera (m)", **axis},
            "zaxis": {"title": "up (m)", **axis},
        },
        uirevision=rid,
    )
    return dcc.Graph(figure=figure, config={"displaylogo": False})


def register(app, config_path: Path) -> None:
    @app.callback(Output("results-recording", "options"), Input("url", "pathname"))
    def recordings(_):
        return resolve_recordings(resolve_config(config_path), None)

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
            html.Div([
                html.A([ui.icon("open"), "Open the QC page"], href=file_url(page), target="_blank", className="button"),
                html.Span(f"written {datetime.fromtimestamp(page.stat().st_mtime):%Y-%m-%d %H:%M}", className="hint"),
            ], className="buttons")
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

    @app.callback(
        Output("results-3d-time", "min"), Output("results-3d-time", "max"), Output("results-3d-time", "value"),
        Output("results-3d-slider", "style"), Input("results-recording", "value"),
    )
    def span_3d(rid):
        points = points_3d(config_path, rid) if rid else None
        if points is None or points.empty:
            return 0, 1, 0, {"display": "none"}
        start, end = float(points["t"].min()), float(points["t"].max())
        return start, end, start, {}

    @app.callback(Output("results-3d", "children"), Input("results-3d-time", "value"), State("results-recording", "value"))
    def show_3d(at_s, rid):
        if not rid:
            return html.P("No recordings found.", className="none")
        return view_3d(config_path, rid, float(at_s or 0.0))
