"""Name parts of a recording while playing it, so every measure is worked out again within each, or set the part of
the recording that is analysed. The segment buttons run the macaw command shown with them."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, project_file, ui
from macaw.interface.files import file_url
from macaw.media.ffmpeg import make_preview
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.segments import seconds_text, segments_of, segments_path, summaries

LABEL = "Segments"
PLAYER = "segments-player"
ACT, REMOVE, FIELD, HERE = "segments-act", "segments-remove", "segments-field", "segments-here"


def layout(config_path: Path):
    recordings = resolve_recordings(resolve_config(config_path), None)
    return html.Div(
        [
            ui.page_head(
                "Segments",
                "Name parts of a recording, such as free play, to have every measure within each.",
            ),
            html.Div(
                ui.field("Recording", dcc.Dropdown(recordings, recordings[0] if recordings else None, id="segments-recording", clearable=False, className="picker")),
                className="toolbar",
            ),
            html.Div(id="segments-message"),
            dcc.Loading(html.Div(id="segments-body")),
        ]
    )


def played(config, rid: str) -> Path:
    """The file a preview is made of: the recording's own clock, so the player's time is the recording's time."""
    return build_recording(rid, config).reference_media().path


def body(config_path: Path, rid: str | None):
    if rid is None:
        return ui.note("No recordings found under the project's raw folder.", "info")
    config = resolve_config(config_path)
    own = resolve_config(config_path, recording_id=rid)
    preview = layout_for(config, create=False).preview_path(rid)
    try:
        source = played(config, rid)
    except MacawError as error:
        return ui.note(str(error), "bad")
    if preview.is_file():
        player = html.Video(src=file_url(preview), id=PLAYER, controls=True, preload="metadata", className="player")
    else:
        player = html.Div(
            [ui.note(f"{source.name} is made into a small copy with its sound first, once; a long recording takes a minute or two."),
             ui.button("Make a copy to play here", {"type": ACT, "action": "prepare"}, primary=True, icon_name="play")],
        )
    marking = html.Div(
        [
            html.Div(
                [
                    ui.field("Name", dcc.Input(id={"type": FIELD, "name": "name"}, placeholder="free_play")),
                    ui.field("Starts at (s)", dcc.Input(id={"type": FIELD, "name": "start"}, type="number", min=0, step="any")),
                    ui.button("Start here", {"type": HERE, "name": "start"}),
                    ui.field("Ends at (s)", dcc.Input(id={"type": FIELD, "name": "end"}, type="number", min=0, step="any")),
                    ui.button("End here", {"type": HERE, "name": "end"}),
                ],
                className="toolbar",
            ),
            ui.note("Saving a segment runs nothing again; changing the analysed part runs the recording again."),
        ]
    )
    return [
        ui.card("Play", player, desc=f"{source.name}, on the recording's clock."),
        html.Div([
            ui.card(
                "Mark a part", marking,
                actions=[ui.button("Save as a segment", {"type": ACT, "action": "save"}, primary=True),
                         ui.button("Analyse only this part", {"type": ACT, "action": "window"})],
            ),
            ui.card("This recording's segments", _listed(config, own, rid), desc=f"Kept in {segments_path(config).name}, which can be edited by hand."),
        ], className="card-pair"),
    ]


def _listed(config, own, rid: str):
    analysed = analysed_text(own.segment)
    found = segments_of(config, rid)
    if not found:
        return html.Div([ui.note(analysed), html.P("No segments yet.", className="none")])
    done = set(summaries(config, rid)["segment"].astype(str))
    rows = [
        [segment.name, f"{segment.start_s:.2f}", f"{segment.end_s:.2f}", f"{segment.end_s - segment.start_s:.1f}",
         ui.pill("summarised", "ok") if segment.name in done else ui.pill("after the next run", "info"),
         ui.button("Remove", {"type": REMOVE, "name": segment.name}, small=True)]
        for segment in found
    ]
    return html.Div([ui.note(analysed), ui.table(["Segment", "From (s)", "To (s)", "Length (s)", "", ""], rows, numeric={"From (s)", "To (s)", "Length (s)"})])


def analysed_text(window) -> str:
    end = f"{seconds_text(window.task_end_s)} s" if window.task_end_s is not None else "the end"
    if window.keep_last_s is not None:
        return f"Analysed part: the last {seconds_text(window.keep_last_s)} s before {end}."
    if window.task_start_s is not None or window.task_end_s is not None:
        return f"Analysed part: {seconds_text(window.task_start_s or 0)} s to {end}."
    return "The whole recording is analysed."


def segment_command(config_path: Path, rid: str, name: str, start: float | None = None, end: float | None = None) -> list[str]:
    flags = ["--name", name, *(["--clear"] if start is None else ["--start", seconds_text(start), "--end", seconds_text(end)])]
    return cli_bridge.build_command(["set-segment"], config_path=config_path, recordings=[rid], extra_flags=flags)


def set_window(config_path: Path, rid: str, start: float, end: float) -> Path:
    """Writes the analysed part into the recording's own settings file, which then has to read as before."""
    folder = Path(resolve_config(config_path).project.recording_config_dir)
    path = folder / f"{rid}.yaml"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    folder.mkdir(parents=True, exist_ok=True)
    project_file.save(path, project_file.with_values(text, {"segment.task_start_s": float(start), "segment.task_end_s": float(end), "segment.keep_last_s": None}))
    resolve_config(config_path, recording_id=rid)
    return path


def _time_from_player() -> str:
    # the player's own clock, to the hundredth of a second
    return (
        "function (clicks) { const player = document.getElementById('" + PLAYER + "');"
        " if (!clicks || !player) { return window.dash_clientside.no_update; }"
        " return Math.round(player.currentTime * 100) / 100; }"
    )


def register(app, config_path: Path) -> None:
    @app.callback(Output("segments-recording", "options"), Input("url", "pathname"))
    def recordings(_):
        return resolve_recordings(resolve_config(config_path), None)

    app.clientside_callback(
        _time_from_player(), Output({"type": FIELD, "name": MATCH}, "value"), Input({"type": HERE, "name": MATCH}, "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("segments-body", "children"),
        Output("segments-message", "children"),
        Input("segments-recording", "value"),
        Input({"type": REMOVE, "name": ALL}, "n_clicks"),
        Input({"type": ACT, "action": ALL}, "n_clicks"),
        State({"type": FIELD, "name": ALL}, "value"),
        State({"type": FIELD, "name": ALL}, "id"),
    )
    def update(rid, _removed, _acted, values, ids):
        trigger = ctx.triggered_id
        # a button appearing fires this too, with no click behind it
        if isinstance(trigger, dict) and not ctx.triggered[0]["value"]:
            return no_update, no_update
        fields = {item["name"]: value for item, value in zip(ids, values)}
        name, start, end = fields.get("name"), fields.get("start"), fields.get("end")
        action = trigger.get("action") if isinstance(trigger, dict) and trigger.get("type") == ACT else None
        message = None
        if isinstance(trigger, dict) and trigger.get("type") == REMOVE:
            argv = segment_command(config_path, rid, trigger["name"])
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        elif action in ("save", "window"):
            if start is None or end is None or end <= start or (action == "save" and not (name or "").strip()):
                needed = "a name, a start and an end after it" if action == "save" else "a start and an end after it"
                return no_update, ui.note(f"Give {needed} first.", "warn")
            if action == "save":
                argv = segment_command(config_path, rid, name.strip(), float(start), float(end))
                message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
            else:
                try:
                    path = set_window(config_path, rid, float(start), float(end))
                except MacawError as error:
                    return no_update, ui.note(str(error), "bad")
                message = ui.note(f"The analysed part is {seconds_text(start)} to {seconds_text(end)} s, in {path}. The next run works on this recording again from the start.", "ok")
        elif action == "prepare":
            config = resolve_config(config_path)
            try:
                make_preview(played(config, rid), layout_for(config).preview_path(rid))
            except MacawError as error:
                return no_update, ui.note(str(error), "bad")
        return body(config_path, rid), message
