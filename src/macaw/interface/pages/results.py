"""Look at what a run produced: the QC page, and any measure over time per person."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.project.layout import layout_for
from macaw.project.manifest import read_manifest
from macaw.project.room import CM_PER_UNIT
from macaw.project.scale import CM_PER_METRE, shown_in
from macaw.project.segments import segments_of, summaries
from macaw.qc.checks import RecordingResults
from macaw.qc.downsample import min_max_downsample
from macaw.qc.report import changed_since_page
from macaw.store.schema import Scale

LABEL = "Results"
PATH = "/results"
STRIP_3D, TIME_3D, PLAYER_3D, ACT_3D = "results-3d-strip", "results-3d-time", "results-3d-player", "results-3d-act"
# colours of the QC report figures, one per person in order
COLOURS = ["#1d4e89", "#d1495b", "#2a9d8f", "#e9a03b", "#6c5b7b", "#57606a"]


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head("Results", "What a run produced: the quality page, and any measure per person."),
            ui.card(
                "Quality review",
                html.Div(id="results-report"),
                html.Div(id="results-report-message"),
                desc="What the checks flagged, what was decided, and where to look.",
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
                desc="Each measure per person, over time or as one number.",
            ),
            ui.card(
                "In 3D",
                html.Div(
                    [html.Div(id="results-3d-strip-box", className="three-d-play"), dcc.Store(id=TIME_3D),
                     dcc.Loading(html.Div(id="results-3d"), parent_className="three-d-view")],
                    className="three-d",
                ),
                html.Div(id="results-3d-message"),
                desc="Drag to turn it; stretching bones mean misplaced points. The 3D follows the video when it stops.",
            ),
        ]
    )


def sources(config_path: Path, rid: str | None) -> list[str]:
    if rid is None:
        return []
    from macaw.store.longtable import has_rows

    directory = layout_for(resolve_config(config_path), create=False).long_table_path(rid)
    return sorted(path.stem for path in directory.glob("*.parquet") if has_rows(path))


def chosen_source(found: list[str], linked: str | None, current: str | None) -> str | None:
    """The step a link asked for, else the one already shown, else the first; a step without measures is passed over."""
    for name in (linked, current):
        if name in found:
            return name
    return found[0] if found else None


def view(config_path: Path, rid: str, source: str, feature: str):
    config = resolve_config(config_path)
    rows = RecordingResults(layout_for(config, create=False), rid).long_table(source, features=[feature])
    if rows.empty:
        return html.P("No values.", className="none")
    rows = rows.assign(who=_who(rows))
    if set(rows["scale"].astype(str)) == {Scale.RECORDING.value}:
        return per_person(config, rid, source, feature, rows)
    factor, unit = measure_unit(config, feature)
    rows = rows.assign(value=rows["value"].astype(float) * factor)
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
        xaxis={"title": "recording time (s)", **axis}, yaxis={"title": _titled(feature, unit), **axis}, legend={"orientation": "h", "y": 1.08},
    )
    return dcc.Graph(figure=figure, config={"displaylogo": False})


def measure_unit(config, feature: str) -> tuple[float, str | None]:
    """The factor and unit a measure is shown in, from the manifest: a length in metres in the project's unit."""
    path = layout_for(config, create=False).manifest_path
    spec = read_manifest(path).specs.get(feature) if path.is_file() else None
    return shown_in(spec.unit if spec else None, config.project.units)


def _titled(name: str, unit: str | None) -> str:
    return name if unit is None else f"{name} ({unit})"


def _who(rows):
    who = rows["subject_id"].astype(object).where(rows["subject_id"].notna(), "everyone")
    pair = rows["subject_id_b"].notna()
    who[pair] = rows.loc[pair, "subject_id"].astype(str) + " & " + rows.loc[pair, "subject_id_b"].astype(str)
    return who


def per_person(config, rid: str, source: str, feature: str, rows):
    """One number per person, and beside it the same within each named segment once the segments are summarised."""
    parts = summaries(config, rid)
    parts = parts[(parts["source"].astype(str) == source) & (parts["feature"].astype(str) == feature)] if not parts.empty else parts
    named = [segment.name for segment in segments_of(config, rid) if (parts["segment"].astype(str) == segment.name).any()]
    factor, unit = measure_unit(config, feature)
    if not named:
        head = _titled(feature, unit)
        return ui.table(["Person", head], [[name, f"{value * factor:.4g}"] for name, value in zip(rows["who"], rows["value"])], numeric={head}, fit=True)
    within = {name: dict(zip(_who(own), own["value"] * factor)) for name, own in parts.groupby("segment", observed=True)}
    columns = ["Person", _titled("Whole recording", unit), *named]
    table = [
        [name, f"{value * factor:.4g}", *(_shown_value(within.get(segment, {}).get(name)) for segment in named)]
        for name, value in zip(rows["who"], rows["value"])
    ]
    return ui.table(columns, table, numeric=set(columns[1:]), fit=True)


def _shown_value(value) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.4g}"


def points_3d(config_path: Path, rid: str):
    """The recording's points in 3D, or None before the step that places them has run."""
    from macaw.extractors.video.pose.triangulate import points_file
    from macaw.store.keypoints import read_keypoints

    path = points_file(resolve_config(config_path, recording_id=rid), rid)
    return read_keypoints(path, recording_id=rid) if path.is_file() else None


def fullest_moment(points) -> float:
    """The time at which the most points were placed, where the view opens."""
    counts = points.groupby("t").size()
    return float(counts.idxmax())


def strip_3d(config_path: Path, rid: str, points):
    """The recording's copy beside the 3D and its time strip under both, opening where the most points were placed."""
    from macaw.interface.pages.segments import own_times, played_clock, recording_length, window_of

    config = resolve_config(config_path)
    own = resolve_config(config_path, recording_id=rid)
    preview = layout_for(config, create=False).preview_path(rid)
    # the copy's own seconds, which the strip turns into the recording's, where the 3D points' times are
    clock = played_clock(own, rid)
    length = max(recording_length(config, own, rid), *own_times([points["t"].max()], clock))
    if preview.is_file():
        player = html.Video(src=f"{file_url(preview)}?v={int(preview.stat().st_mtime)}", id=PLAYER_3D, preload="metadata", className="player")
    else:
        player = html.Div(
            [ui.note("The recording is made into a small copy with its sound first, once; a long recording takes a minute or two."),
             ui.button("Make a copy to play here", {"type": ACT_3D, "action": "prepare"}, primary=True, icon_name="play")],
            className="three-d-copy",
        )
    strip = ui.time_strip(
        STRIP_3D, video=PLAYER_3D if preview.is_file() else None, offset=clock[0], scale=clock[1], length=length,
        start=own_times([fullest_moment(points)], clock)[0], window=own_times(window_of(own.segment, clock[0] + length * clock[1]), clock),
        timeStore=TIME_3D,
    )
    return [player, strip]


def in_first_camera(setup, xyz: np.ndarray) -> np.ndarray:
    """Room points in metres as the first camera sees them; a calibration file from another tool may set its room's
    axes any way, so the view and its labels always follow the first camera."""
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES

    per_unit = UNITS_TO_METRES[setup.units]
    return setup.cameras[0].to_camera(np.asarray(xyz, dtype=float) / per_unit) * per_unit


def _shown(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The first camera's axes (x right, y down, z ahead) turned so up is up on the page."""
    return xyz[..., 0], xyz[..., 2], -xyz[..., 1]


def view_3d(config_path: Path, rid: str, at_s: float, points=None):
    from macaw.extractors.video.pose.cameras import UNITS_TO_METRES, read_setup
    from macaw.extractors.video.pose.skeleton import SKELETON_EDGES
    from macaw.extractors.video.pose.triangulate import calibration_path

    config = resolve_config(config_path, recording_id=rid)
    points = points_3d(config_path, rid) if points is None else points
    if points is None or points.empty:
        return html.P("No 3D for this recording: it needs calibrated cameras, a camera that records depth, or pose3d.one_camera.", className="none")
    try:
        setup = read_setup(calibration_path(config), config.pose3d.units)
    except Exception:   # the cameras are context; the people still show without them, in the file's own axes
        setup = None
    # people placed from different cameras are each in their own camera's frame, so one view shows one camera's
    cameras = sorted(set(points["video"].astype(str)) - {"3d"}) if "video" in points.columns else []
    if cameras:
        points = points[points["video"].astype(str) == cameras[0]]
    units = config.project.units
    per = CM_PER_METRE / CM_PER_UNIT[units]
    place = (lambda xyz: _shown(in_first_camera(setup, xyz) * per)) if setup else (lambda xyz: _shown(np.asarray(xyz) * per))
    times = np.sort(points["t"].unique())
    moment = points[points["t"] == times[np.abs(times - at_s).argmin()]]
    figure = go.Figure()
    for index, (subject_id, own) in enumerate(moment.groupby("subject_id", sort=True)):
        colour = COLOURS[index % len(COLOURS)]
        joint = {row.joint: np.array([row.x, row.y, row.z]) for row in own.itertuples()}
        segments = [np.vstack([joint[a], joint[b], [np.nan] * 3]) for a, b in [*SKELETON_EDGES, ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")] if a in joint and b in joint]
        if segments:
            x, y, z = place(np.vstack(segments))
            figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line={"color": colour, "width": 6}, name=str(subject_id), legendgroup=str(subject_id)))
        x, y, z = place(np.vstack(list(joint.values())))
        figure.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker={"size": 3, "color": colour}, showlegend=False, legendgroup=str(subject_id), text=list(joint), hoverinfo="text"))
    if setup is not None:
        x, y, z = place(np.vstack([camera.position() for camera in setup.cameras]) * UNITS_TO_METRES[setup.units])
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers+text", text=[camera.name for camera in setup.cameras], textposition="top center",
            marker={"size": 5, "symbol": "diamond", "color": "#8b8b90"}, name="cameras",
        ))
    elif cameras:
        # points from one camera are in that camera's frame, which stands at the origin
        x, y, z = place(np.zeros((1, 3)))
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers+text", text=[cameras[0]], textposition="top center",
            marker={"size": 5, "symbol": "diamond", "color": "#8b8b90"}, name="cameras",
        ))
    axis = {"gridcolor": "rgba(128,128,128,0.25)", "zerolinecolor": "rgba(128,128,128,0.4)", "backgroundcolor": "rgba(0,0,0,0)"}
    figure.update_layout(
        height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", font={"color": "#8b8b90"},
        legend={"orientation": "h", "y": 1.02},
        scene={
            # a little further out than the default, so cameras at the edge of the room stay in view
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.1}},
            **axes_3d(units, axis),
        },
        uirevision=rid,
    )
    graph = dcc.Graph(figure=figure, config={"displaylogo": False})
    if len(cameras) > 1:
        return html.Div([html.P(f"People placed from {cameras[0]}; those placed from {', '.join(cameras[1:])} are each in their own camera's frame.", className="hint"), graph])
    return graph


def axes_3d(units: str, axis: dict) -> dict:
    return {
        "aspectmode": "data", "xaxis": {"title": f"across ({units})", **axis},
        "yaxis": {"title": f"ahead of the first camera ({units})", **axis}, "zaxis": {"title": f"up ({units})", **axis},
    }


def out_of_date(steps: list[str], decided: list[str]) -> list:
    """What changed since the QC page was written, beside its time; nothing when it is current."""
    parts = [f"{len(steps)} step{'s' if len(steps) > 1 else ''} ran"] if steps else []
    parts += [f"{len(decided)} decision{'s' if len(decided) > 1 else ''} changed"] if decided else []
    if not parts:
        return []
    return [ui.pill("out of date", "warn"), html.Span(f"{' and '.join(parts)} since", className="hint", title=", ".join(steps + decided))]


def register(app, config_path: Path) -> None:
    @app.callback(
        Output("results-report", "children"),
        Output("results-report-message", "children"),
        Input(ui.SHOWN["results"], "data"),
        Input("results-write-report", "n_clicks"),
    )
    def report(rid, _clicks):
        if rid is None:
            return html.P("No recordings found.", className="none"), None
        message = None
        if ctx.triggered_id == "results-write-report":
            argv = cli_bridge.build_command(["qc"], config_path=config_path, recordings=[rid], extra_flags=["--report"])
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        files = layout_for(resolve_config(config_path), create=False)
        page = files.qc_report_path(rid)
        link = (
            html.Div([
                html.A([ui.icon("open"), "Open the QC page"], href=file_url(page), target="_blank", className="button"),
                html.Span(f"written {datetime.fromtimestamp(page.stat().st_mtime):%Y-%m-%d %H:%M}", className="hint"),
                *out_of_date(*changed_since_page(files, rid)),
            ], className="buttons")
            if page.is_file() else html.P("No QC page yet for this recording.", className="none")
        )
        return link, message

    @app.callback(
        Output("results-source", "options"), Output("results-source", "value"),
        Input(ui.SHOWN["results"], "data"), Input("url", "search"), State("url", "pathname"), State("results-source", "value"),
    )
    def pick_source(rid, search, pathname, current):
        if ctx.triggered_id == "url" and (pathname or "/").rstrip("/") != PATH:
            return no_update, no_update
        found = sources(config_path, rid)
        return found, chosen_source(found, ui.asked(search).get("step"), current)

    @app.callback(
        Output("results-feature", "options"), Output("results-feature", "value"),
        Input("results-source", "value"), State(ui.RECORDING, "value"),
    )
    def pick_feature(source, rid):
        if not source or not rid:
            return [], None
        rows = RecordingResults(layout_for(resolve_config(config_path), create=False), rid).long_table(source)
        features = sorted(rows["feature"].astype(str).unique())
        return features, features[0] if features else None

    @app.callback(
        Output("results-view", "children"),
        Input("results-feature", "value"), State("results-source", "value"), State(ui.RECORDING, "value"),
    )
    def show(feature, source, rid):
        if not (feature and source and rid):
            return html.P("Nothing to show yet.", className="none")
        return view(config_path, rid, source, feature)

    @app.callback(
        Output("results-3d-strip-box", "children"), Output("results-3d-message", "children"),
        Input(ui.SHOWN["results"], "data"), Input({"type": ACT_3D, "action": ALL}, "n_clicks"), State(ui.RECORDING, "value"),
    )
    def span_3d(shown, _acted, rid):
        trigger = ctx.triggered_id
        # a button appearing fires this too, with no click behind it
        if isinstance(trigger, dict) and not ctx.triggered[0]["value"]:
            return no_update, no_update
        rid = shown or rid
        points = points_3d(config_path, rid) if rid else None
        if points is None or points.empty:
            return None, None
        if isinstance(trigger, dict):
            from macaw.interface.pages.segments import played
            from macaw.media.ffmpeg import make_preview

            config = resolve_config(config_path)
            try:
                make_preview(played(resolve_config(config_path, recording_id=rid), rid), layout_for(config).preview_path(rid))
            except MacawError as error:
                return no_update, ui.note(str(error), "bad")
        return strip_3d(config_path, rid, points), None

    @app.callback(Output("results-3d", "children"), Input(TIME_3D, "data"), Input(ui.SHOWN["results"], "data"), State(ui.RECORDING, "value"))
    def show_3d(at, _, rid):
        if not rid:
            return html.P("No recordings found.", className="none")
        points = points_3d(config_path, rid)
        if points is None or points.empty:
            return view_3d(config_path, rid, 0.0, points)
        at_s = at["t"] if at and ctx.triggered_id == TIME_3D else fullest_moment(points)
        return view_3d(config_path, rid, float(at_s), points)
