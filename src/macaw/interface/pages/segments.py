"""Name parts of a recording on its time strip, so every measure is worked out again within each, or set the part of
the recording that is analysed."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import ConfigError, MacawError
from macaw.interface import cli_bridge, project_file, ui
from macaw.interface.files import file_url
from macaw.media.ffmpeg import make_preview
from macaw.media.probe import probe
from macaw.project.layout import layout_for
from macaw.project.previews import loudness
from macaw.project.recording import build_recording
from macaw.project.segments import Segment, remove_segment, seconds_text, segments_of, segments_path, set_segment, summaries

LABEL = "Segments"
PLAYER = "segments-player"
ACT, STRIP, EDITS, SHOWN, TABLE = "segments-act", "segments-strip", "segments-edits", "segments-shown", "segments-table"


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head(
                "Segments",
                "Name parts of a recording, such as free play, to have every measure within each.",
            ),
            html.Div(id="segments-message"),
            dcc.Loading(html.Div(id="segments-body")),
            dcc.Store(id=EDITS),
            dcc.Store(id=SHOWN),
        ]
    )


def played(config, rid: str) -> Path:
    """The file a preview is made of: the recording's own clock, so the player's time is the recording's time."""
    return build_recording(rid, config).reference_media().path


def window_of(segment, length: float) -> list[float]:
    end = segment.task_end_s if segment.task_end_s is not None else length
    start = end - segment.keep_last_s if segment.keep_last_s is not None else segment.task_start_s or 0.0
    return [max(float(start), 0.0), float(end)]


def recording_length(config, own, rid: str) -> float:
    try:
        return probe(played(config, rid)).duration_s or 0.0
    except Exception:   # a file that cannot be read still gets a strip as long as what is marked on it
        ends = [segment.end_s for segment in segments_of(config, rid)] + [own.segment.task_end_s or 0.0, 1.0]
        return max(ends)


def strip_segments(config, rid: str) -> list[dict]:
    done = set(summaries(config, rid)["segment"].astype(str))
    return [{"name": s.name, "a": s.start_s, "b": s.end_s, "done": s.name in done} for s in segments_of(config, rid)]


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
    length = recording_length(config, own, rid)
    loud = None
    if preview.is_file():
        player = html.Video(src=file_url(preview), id=PLAYER, preload="metadata", className="player")
        try:
            loud = loudness(preview)
        except MacawError:   # a copy without sound still plays
            pass
    else:
        player = html.Div(
            [ui.note(f"{source.name} is made into a small copy with its sound first, once; a long recording takes a minute or two."),
             ui.button("Make a copy to play here", {"type": ACT, "action": "prepare"}, primary=True, icon_name="play")],
        )
    strip = ui.time_strip(
        STRIP, video=PLAYER if preview.is_file() else None, length=length, window=window_of(own.segment, length), edit=True,
        segments=strip_segments(config, rid), loudness=loud, editStore=EDITS,
    )
    listed = html.Details([html.Summary("Every segment as a table"), html.Div(_listed(config, own, rid), id=TABLE)], className="more")
    return ui.card("Segments", player, strip, listed, desc=f"{source.name}, on the recording's clock; kept in {segments_path(config).name}.")


def _listed(config, own, rid: str):
    analysed = analysed_text(own.segment)
    found = segments_of(config, rid)
    if not found:
        return html.Div([ui.note(analysed), html.P("No segments yet.", className="none")])
    done = set(summaries(config, rid)["segment"].astype(str))
    rows = [
        [segment.name, f"{segment.start_s:.2f}", f"{segment.end_s:.2f}", f"{segment.end_s - segment.start_s:.1f}",
         "summarised" if segment.name in done else "after the next run"]
        for segment in found
    ]
    return html.Div([ui.note(analysed), ui.table(["Segment", "From (s)", "To (s)", "Length (s)", ""], rows, numeric={"From (s)", "To (s)", "Length (s)"})])


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
    return project_file.save_for_recording(
        config_path, rid, {"segment.task_start_s": float(start), "segment.task_end_s": float(end), "segment.keep_last_s": None}
    )


def apply_edit(config_path: Path, rid: str, edit: dict) -> str:
    """Saves one change made on the strip; what the strip says after it."""
    config = resolve_config(config_path)
    if edit["op"] == "window":
        set_window(config_path, rid, edit["start"], edit["end"])
        return f"Analysed part {seconds_text(edit['start'])} to {seconds_text(edit['end'])} s; the next run starts this recording again."
    if edit["op"] == "remove":
        remove_segment(config, rid, edit["name"])
        return f"{edit['name']} removed."
    name, old = edit["name"].strip(), edit.get("old")
    if name != old and any(segment.name == name for segment in segments_of(config, rid)):
        raise ConfigError(f"{rid} has a segment named {name} already")
    set_segment(config, rid, Segment(name, float(edit["start"]), float(edit["end"])))
    if old and old != name:
        remove_segment(config, rid, old)
    return "saved"


def register(app, config_path: Path) -> None:
    app.clientside_callback(
        "function (data) { if (data && window.macawStrip) { window.macawStrip.shown('" + STRIP + "', data); }"
        " return window.dash_clientside.no_update; }",
        Output(STRIP, "title"), Input(SHOWN, "data"), prevent_initial_call=True,
    )

    @app.callback(
        Output("segments-body", "children"),
        Output("segments-message", "children"),
        Input(ui.SHOWN["segments"], "data"),
        Input({"type": ACT, "action": ALL}, "n_clicks"),
    )
    def update(rid, _acted):
        trigger = ctx.triggered_id
        # a button appearing fires this too, with no click behind it
        if isinstance(trigger, dict) and not ctx.triggered[0]["value"]:
            return no_update, no_update
        if isinstance(trigger, dict) and trigger.get("action") == "prepare":
            config = resolve_config(config_path)
            try:
                make_preview(played(config, rid), layout_for(config).preview_path(rid))
            except MacawError as error:
                return no_update, ui.note(str(error), "bad")
        return body(config_path, rid), None

    @app.callback(
        Output(TABLE, "children"), Output(SHOWN, "data"), Output("segments-message", "children", allow_duplicate=True),
        Input(EDITS, "data"), State(ui.SHOWN["segments"], "data"),
        prevent_initial_call=True,
    )
    def edited(edit, rid):
        if not edit or not rid:
            return no_update, no_update, no_update
        try:
            said, message = apply_edit(config_path, rid, edit), None
        except MacawError as error:
            said, message = "not saved", ui.note(str(error), "bad")
        config = resolve_config(config_path)
        own = resolve_config(config_path, recording_id=rid)
        length = recording_length(config, own, rid)
        shown = {"segments": strip_segments(config, rid), "window": window_of(own.segment, length), "said": said, "op": edit["op"]}
        return _listed(config, own, rid), shown, message
