"""Name parts of a recording on its time strip, so every measure is worked out again within each, or set the part of
the recording that is analysed."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import ConfigError, MacawError
from macaw.interface import cli_bridge, project_file, ui
from macaw.interface.files import file_url
from macaw.media.probe import probe
from macaw.project.previews import current_recording_preview, loudness, make_recording_preview, played, recording_clock
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


def played_clock(config, rid: str) -> tuple[float, float]:
    """Where the copy's own seconds fall on the recording's clock, its time 0 and clock scale, once prep.media has run;
    the file played need not be the one the clock was set by."""
    try:
        return recording_clock(config, rid)
    except (MacawError, OSError, StopIteration):   # without its files the strip reads the recording's clock as is
        return 0.0, 1.0


def own_times(times, clock: tuple[float, float]) -> list[float]:
    offset, scale = clock
    return [round((float(t) - offset) / scale, 3) for t in times]


def on_own_clock(segments: list[dict], clock: tuple[float, float]) -> list[dict]:
    return [{**segment, **dict(zip(("a", "b"), own_times((segment["a"], segment["b"]), clock)))} for segment in segments]


def on_recording_clock(edit: dict, clock: tuple[float, float]) -> dict:
    """An edit made on the strip, whose times are the played file's own, with times on the recording's clock."""
    offset, scale = clock
    return {**edit, **{key: round(float(edit[key]) * scale + offset, 3) for key in ("start", "end") if edit.get(key) is not None}}


def window_of(segment, length: float) -> list[float]:
    end = segment.task_end_s if segment.task_end_s is not None else length
    start = end - segment.keep_last_s if segment.keep_last_s is not None else segment.task_start_s or 0.0
    return [max(float(start), 0.0), float(end)]


def recording_length(config, own, rid: str) -> float:
    """The played file's length on its own clock."""
    try:
        return probe(played(own, rid)).duration_s or 0.0
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
    try:
        source, clock = played(own, rid), played_clock(own, rid)
        preview = current_recording_preview(own, rid)
    except MacawError as error:
        return ui.note(str(error), "bad")
    length = recording_length(config, own, rid)
    loud = None
    if preview is not None:
        player = html.Video(src=f"{file_url(preview)}?v={int(preview.stat().st_mtime)}", id=PLAYER, preload="metadata", className="player")
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
        STRIP, video=PLAYER if preview is not None else None, offset=clock[0], scale=clock[1], length=length,
        window=own_times(window_of(own.segment, clock[0] + length * clock[1]), clock), edit=True,
        segments=on_own_clock(strip_segments(config, rid), clock), loudness=loud, editStore=EDITS,
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
            try:
                make_recording_preview(resolve_config(config_path, recording_id=rid), rid)
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
        clock = played_clock(resolve_config(config_path, recording_id=rid), rid)
        try:
            said, message = apply_edit(config_path, rid, on_recording_clock(edit, clock)), None
        except MacawError as error:
            said, message = "not saved", ui.note(str(error), "bad")
        config = resolve_config(config_path)
        own = resolve_config(config_path, recording_id=rid)
        length = recording_length(config, own, rid)
        shown = {
            "segments": on_own_clock(strip_segments(config, rid), clock),
            "window": own_times(window_of(own.segment, clock[0] + length * clock[1]), clock), "said": said, "op": edit["op"],
        }
        return _listed(config, own, rid), shown, message
