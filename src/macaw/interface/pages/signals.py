"""Other signals, such as fNIRS, under the recording's video: where each file sits on its clock, and stretches a person marks."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import ConfigError, MacawError
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.interface.pages.segments import own_times, played_clock, recording_length, window_of
from macaw.media.align import MANUAL, UNALIGNED
from macaw.project.layout import layout_for
from macaw.project.previews import current_recording_preview, loudness, make_recording_preview, played

LABEL = "Signals"
PATH = "/signals"
PLAYER, STRIP, EDITS, SHOWN = "signals-player", "signals-strip", "signals-edits", "signals-shown"
ACT, OFFSET, BY = "signals-act", "signals-offset", "signals-by"
CARPET, LINES, MEASURES, CODING = "signals-carpet", "signals-lines", "signals-measures", "signals-coding"
STRIP_BOX = "signals-strip-box"
# channel pairs drawn as lines at once; more would be a carpet again
MAX_LINES = 6
# a source-detector pair, then what the channel holds: a wavelength or hbo, hbr, hbt
_CHANNEL = re.compile(r"^(?P<pair>\S+)\s+(?P<group>\S+)$")


def wanted(config) -> bool:
    from macaw.signals.table import read_signals

    try:
        return bool(read_signals(config))
    except MacawError:   # a table with a mistake still gets its tab, which says what is wrong
        return True


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head("Signals", "Other signals, such as fNIRS, under the video: check where each file sits, and mark stretches."),
            html.Div(
                [ui.field("Your name", dcc.Input(id=BY, value="", placeholder="needed to save", persistence=True, persistence_type="local"))],
                className="toolbar",
            ),
            html.Div(id="signals-message"),
            dcc.Loading(html.Div(id="signals-body"), delay_show=300, target_components={"signals-body": "children"},
                        overlay_style={"visibility": "visible", "opacity": 0.5, "pointerEvents": "none"}),
            dcc.Store(id=EDITS),
            dcc.Store(id=SHOWN),
        ]
    )


def group_of(channel: str) -> tuple[str, str]:
    """(pair, what it holds) of a channel named like S1_D1 760 or S1_D1 hbo; a name of another shape is its own pair."""
    found = _CHANNEL.match(channel)
    return (found["pair"], found["group"]) if found else (channel, "")


def group_label(group: str) -> str:
    return {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}.get(group.lower(), f"{group} nm" if group.isdigit() else group or "channels")


def file_clock(result) -> tuple[float, float]:
    from macaw.signals.align import signal_clock

    return signal_clock(result) if result is not None and result.method != UNALIGNED else (result.offset_s if result else 0.0, 1.0)


def to_played(times, file_clock_: tuple[float, float], clock: tuple[float, float]) -> np.ndarray:
    """A file's own times as the played copy's own times, through the recording's clock."""
    recorded = file_clock_[0] + np.asarray(times, dtype=float) * file_clock_[1]
    return (recorded - clock[0]) / clock[1]


def _choices(own, rid: str, files) -> dict:
    """What the dropdowns offer for this recording."""
    from macaw.signals.read import load_info
    from macaw.store.annotations import read_annotations
    from macaw.store.longtable import read_long_table
    from macaw.store.schema import Scale

    groups, pairs = [], []
    for item in files:
        try:
            info = load_info(own, rid, item)
        except FileNotFoundError:
            continue
        for channel in info["channels"]:
            pair, group = group_of(channel["name"])
            groups += [group] if group not in groups else []
            pairs += [pair] if pair not in pairs else []
    layout = layout_for(own, create=False)
    path = layout.long_table_path(rid)
    measures = []
    if path.exists():
        rows = read_long_table(path, scale=Scale.FRAME)
        measures = sorted(rows["feature"].dropna().astype(str).unique()) if not rows.empty else []
    coded = layout.annotations_path(rid)
    tiers = sorted(read_annotations(coded)["tier"].dropna().astype(str).unique()) if coded.exists() else []
    return {"groups": groups, "pairs": pairs, "measures": measures, "tiers": tiers}


def body(config_path: Path, rid: str | None):
    from macaw.signals.table import signal_files, signals_path

    if rid is None:
        return ui.note("No recordings found under the project's raw folder.", "info")
    own = resolve_config(config_path, recording_id=rid)
    try:
        files = signal_files(own, rid)
    except MacawError as error:
        return ui.note(str(error), "bad")
    if not files:
        return ui.note(f"No signal files for {rid}. List them in {signals_path(own).name}, with the columns recording, person, stage and file.")
    try:
        source, clock = played(own, rid), played_clock(own, rid)
        preview = current_recording_preview(own, rid)
    except MacawError as error:
        return ui.note(str(error), "bad")
    if preview is not None:
        player = html.Video(src=f"{file_url(preview)}?v={int(preview.stat().st_mtime)}", id=PLAYER, preload="metadata", className="player")
    else:
        player = html.Div([
            ui.note(f"{source.name} is made into a small copy with its sound first, once; a long recording takes a minute or two."),
            ui.button("Make a copy to play here", {"type": ACT, "action": "prepare", "person": ""}, primary=True, icon_name="play"),
        ])
    offered = _choices(own, rid, files)

    def picker(id_, label, options, multi, hint):
        return ui.field(label, dcc.Dropdown(id=id_, options=options, multi=multi, className="picker", persistence=True,
                                            persistence_type="local", placeholder=hint))

    controls = html.Div([
        picker(CARPET, "Carpet", [{"label": group_label(g), "value": g} for g in offered["groups"]], False, "each file's first"),
        picker(LINES, "Channels as lines", offered["pairs"], True, "none"),
        picker(MEASURES, "Measures from the video", offered["measures"], True, "none"),
        picker(CODING, "Coding", offered["tiers"], True, "none"),
    ], className="toolbar")
    signals_card = ui.card(
        "Signals", player, controls, html.Div(id=STRIP_BOX),
        desc=f"{source.name}, on the recording's clock. Blood oxygen follows behaviour by a few seconds. Drag on the marks lane to "
             "mark a stretch, such as a child touching the cap.",
    )
    return html.Div([signals_card, offsets_card(own, rid, files)])


def strip(config_path: Path, rid: str, carpet_group: str | None, pairs: list[str], measures: list[str], tiers: list[str]):
    own = resolve_config(config_path, recording_id=rid)
    config = resolve_config(config_path)
    clock = played_clock(own, rid)
    preview = current_recording_preview(own, rid)
    length = recording_length(config, own, rid)
    loud = None
    if preview is not None:
        try:
            loud = loudness(preview)
        except MacawError:   # a copy without sound still plays
            pass
    return ui.time_strip(
        STRIP, video=PLAYER if preview is not None else None, offset=clock[0], scale=clock[1], length=length,
        window=own_times(window_of(own.segment, clock[0] + length * clock[1]), clock), edit=True, editWindow=False,
        noun="mark", editName="Marks", notes=True, segments=strip_marks(own, rid, clock), loudness=loud, editStore=EDITS,
        lanes=lanes(own, rid, clock, length, carpet_group, pairs or [], measures or [], tiers or []),
    )


def strip_marks(own, rid: str, clock: tuple[float, float]) -> list[dict]:
    from macaw.signals.marks import marks_of

    return [
        {"name": mark.name, "a": a, "b": b, "note": mark.note, "done": True}
        for mark in marks_of(own, rid) for a, b in [own_times((mark.start_s, mark.end_s), clock)]
    ]


def lanes(own, rid: str, clock, length: float, carpet_group, pairs, measures, tiers) -> list[dict]:
    """Per file a carpet and its chosen channels as lines, per person with a cap IMU its turning against the head's on
    the video, then the chosen measures and coding."""
    from macaw.signals.align import head_turning, signal_offsets
    from macaw.signals.imu import gyro_speed
    from macaw.signals.lanes import CARPET_Z, carpet, channel_z, lines, spans
    from macaw.signals.read import load_aux, load_samples
    from macaw.signals.table import signal_files

    offsets = signal_offsets(own, rid)
    out, turning_done = [], set()
    for item in signal_files(own, rid):
        result = offsets.get(item.key)
        own_clock = file_clock(result)
        where = "not aligned yet: set its offset below" if result is None or result.method == UNALIGNED else f"{own_clock[0]:+.2f} s"
        try:
            samples = load_samples(own, rid, item)
        except FileNotFoundError:
            continue
        at = to_played(samples["time"], own_clock, clock)
        names = [column for column in samples.columns if column != "time"]
        groups = [group_of(name)[1] for name in names]
        group = carpet_group if carpet_group in groups else (groups[0] if groups else None)
        chosen = [name for name, g in zip(names, groups) if g == group]
        drawn = carpet(samples, chosen, at, length)
        if drawn:
            out.append({"kind": "carpet", "name": f"{item.person or 'all'} {item.stage}",
                        "title": f"{item.path.name}, {group_label(group or '')}, each channel less its trend in its own SDs; {where}", **drawn})
        wanted_pairs = [name for name in chosen if group_of(name)[0] in pairs[:MAX_LINES]]
        if wanted_pairs:
            drawn = lines([(group_of(name)[0], at, channel_z(samples, name, at, length)) for name in wanted_pairs], length,
                          bounds=(-CARPET_Z, CARPET_Z), unit=" SD")
            if drawn:
                out.append({"kind": "lines", "name": f"{item.person or 'all'} {item.stage}",
                            "title": f"{item.path.name}, {group_label(group or '')}, each channel less its trend in its own SDs, as in the carpet", **drawn})
        if item.person and item.person not in turning_done:
            imu = gyro_speed(load_aux(own, rid, item), own.signals.gyro)
            if imu is not None:
                turning_done.add(item.person)
                series = []
                head = head_turning(own, rid, item.person)
                if head is not None:
                    series.append(("head on the video", (head[0] - clock[0]) / clock[1], head[1]))
                series.append(("cap", to_played(imu[0], own_clock, clock), imu[1]))
                drawn = lines(series, length, scaled=True)
                if drawn:
                    out.append({"kind": "lines", "name": f"{item.person} turning",
                                "title": "How fast the head turns on the video and the cap's gyroscope, each scaled to its own range; "
                                         "where the file sits right, they rise together", **drawn})
    out += measure_lanes(own, rid, clock, length, measures)
    out += coding_lanes(own, rid, clock, tiers, spans)
    return out


def measure_lanes(own, rid: str, clock, length: float, measures: list[str]) -> list[dict]:
    from macaw.signals.lanes import lines
    from macaw.store.longtable import read_long_table
    from macaw.store.schema import Scale

    path = layout_for(own, create=False).long_table_path(rid)
    if not measures or not path.exists():
        return []
    rows = read_long_table(path, scale=Scale.FRAME, features=list(measures))
    out = []
    for feature in measures:
        mine = rows[rows["feature"].astype(str) == feature]
        series = []
        for who, part in mine.groupby(mine["subject_id"].astype(str).where(mine["subject_id"].notna(), "everyone"), observed=True):
            part = part.sort_values("t_start")
            series.append((str(who), (part["t_start"].to_numpy(dtype=float) - clock[0]) / clock[1], part["value"].to_numpy(dtype=float)))
        drawn = lines(series, length)
        if drawn:
            out.append({"kind": "lines", "name": feature, "title": feature, **drawn})
    return out


def coding_lanes(own, rid: str, clock, tiers: list[str], spans) -> list[dict]:
    from macaw.store.annotations import read_annotations

    path = layout_for(own, create=False).annotations_path(rid)
    if not tiers or not path.exists():
        return []
    coded = read_annotations(path)
    out = []
    for tier in tiers:
        rows = coded[coded["tier"].astype(str) == tier]
        found = [
            (float(row.t_start), float(row.t_end), f"{row.subject_id}.{row.label}" if pd.notna(row.subject_id) else str(row.label))
            for row in rows.itertuples(index=False) if pd.notna(row.label) and str(row.label).strip()
        ]
        if found:
            out.append({"kind": "bars", "name": tier, "title": f"{tier}, as coded", "spans": spans(found, lambda t: (t - clock[0]) / clock[1])})
    return out


def offsets_card(own, rid: str, files):
    from macaw.signals.align import describe, needs_person, signal_offsets
    from macaw.signals.table import EVERYONE

    offsets = signal_offsets(own, rid)
    people: dict[str, list] = {}
    for item in files:
        people.setdefault(item.person or EVERYONE, []).append(item)
    rows = []
    for person, items in people.items():
        result = offsets.get(items[0].key)
        waiting = "not aligned yet" if result is None else needs_person(result, own.signals)
        if waiting:
            state = ui.pill(waiting, "warn")
        elif result.method == MANUAL:
            state = ui.pill(result.note or "set by hand", "ok")
        else:
            state = ui.pill("kept by " + result.accepted.split(" ")[0] if result.accepted else "automatic", "ok" if result.accepted else "info")
        value = round(file_clock(result)[0], 3) if result is not None and result.method != UNALIGNED else None
        keep = (ui.button("Keep", {"type": ACT, "action": "keep", "person": person}, small=True)
                if result is not None and result.method not in (MANUAL, UNALIGNED) and not result.accepted else "")
        rows.append([
            person, ", ".join(item.stage for item in items),
            dcc.Input(id={"type": OFFSET, "person": person}, type="number", step="any", value=value, debounce=True, placeholder="seconds", style={"width": "11rem"},
                      className="number"),
            describe(result) if result is not None else "signals.align has not run", state, keep,
        ])
    return ui.card(
        "Where each file sits",
        ui.table(["Person", "Files", "Offset (s)", "Found", "", ""], rows),
        desc="Recording time = the file's own time + the offset. A person's files share one clock, so one offset sets them all; "
             "type another to set it by hand.",
    )


def mark_edit(own, rid: str, edit: dict, clock: tuple[float, float]) -> str:
    """Saves one change made on the marks lane, whose times are the played copy's own; what the strip says after it."""
    from macaw.signals.marks import Mark, remove_mark, set_mark

    def recorded(t):
        return round(float(t) * clock[1] + clock[0], 3)

    if edit["op"] == "remove":
        remove_mark(own, rid, Mark(edit["name"], recorded(edit["start"]), recorded(edit["end"])))
        return f"{edit['name']} removed."
    was = edit.get("was")
    old = Mark(was["name"], recorded(was["a"]), recorded(was["b"])) if was else None
    set_mark(own, rid, Mark(edit["name"].strip(), recorded(edit["start"]), recorded(edit["end"]), edit.get("note") or ""), old)
    return "saved"


def action_command(config_path: Path, rid: str, action: str, person: str, by: str, offset: float | None = None) -> list[str]:
    by_flag = ["--by", by] if by else []
    if action == "keep":
        return cli_bridge.build_command(["accept-signal-offset"], config_path=config_path, recordings=[rid],
                                        extra_flags=["--person", person, *by_flag])
    return cli_bridge.build_command(["set-signal-offset"], config_path=config_path, recordings=[rid],
                                    extra_flags=["--person", person, "--offset", f"{offset:g}", *by_flag])


def register(app, config_path: Path) -> None:
    app.clientside_callback(
        "function (data) { if (data && window.macawStrip) { window.macawStrip.shown('" + STRIP + "', data); }"
        " return window.dash_clientside.no_update; }",
        Output(STRIP, "title"), Input(SHOWN, "data"), prevent_initial_call=True,
    )

    @app.callback(
        Output("signals-body", "children"), Output("signals-message", "children"),
        Input(ui.SHOWN["signals"], "data"), Input({"type": ACT, "action": ALL, "person": ALL}, "n_clicks"),
        Input({"type": OFFSET, "person": ALL}, "value"), State(BY, "value"),
    )
    def update(rid, _acted, _offsets, by):
        trigger = ctx.triggered_id
        message = None
        if isinstance(trigger, dict) and rid:
            value = ctx.triggered[0]["value"]
            if trigger["type"] == ACT and not value:   # a button appearing fires this too, with no click behind it
                return no_update, no_update
            if trigger["type"] == ACT and trigger["action"] == "prepare":
                try:
                    make_recording_preview(resolve_config(config_path, recording_id=rid), rid)
                except MacawError as error:
                    return no_update, ui.note(str(error), "bad")
            else:
                if trigger["type"] == OFFSET and (value is None or _same_offset(config_path, rid, trigger["person"], value)):
                    return no_update, no_update
                if not (by or "").strip():
                    return no_update, ui.note("Type your name first; it is saved with what you decide.", "warn")
                action = "keep" if trigger["type"] == ACT else "set"
                argv = action_command(config_path, rid, action, trigger["person"], by.strip(), value if action == "set" else None)
                message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        return body(config_path, rid), message

    @app.callback(
        Output(STRIP_BOX, "children"),
        Input(CARPET, "value"), Input(LINES, "value"), Input(MEASURES, "value"), Input(CODING, "value"),
        State(ui.SHOWN["signals"], "data"),
    )
    def drawn(group, pairs, measures, tiers, rid):
        if not rid:
            return no_update
        try:
            return strip(config_path, rid, group, pairs, measures, tiers)
        except MacawError as error:
            return ui.note(str(error), "bad")

    @app.callback(
        Output(SHOWN, "data"), Output("signals-message", "children", allow_duplicate=True),
        Input(EDITS, "data"), State(ui.SHOWN["signals"], "data"), prevent_initial_call=True,
    )
    def edited(edit, rid):
        if not edit or not rid:
            return no_update, no_update
        own = resolve_config(config_path, recording_id=rid)
        clock = played_clock(own, rid)
        try:
            said, message = mark_edit(own, rid, edit, clock), None
        except (MacawError, ConfigError) as error:
            said, message = "not saved", ui.note(str(error), "bad")
        return {"segments": strip_marks(own, rid, clock), "said": said, "op": edit["op"]}, message


def _same_offset(config_path: Path, rid: str, person: str, value: float) -> bool:
    """Whether a typed offset is the one the person's files have already, as a redrawn box reports its value again."""
    from macaw.signals.align import signal_offsets
    from macaw.signals.table import EVERYONE, signal_files

    own = resolve_config(config_path, recording_id=rid)
    offsets = signal_offsets(own, rid)
    for item in signal_files(own, rid):
        if (item.person or EVERYONE) == person and item.key in offsets and offsets[item.key].method != UNALIGNED:
            return abs(file_clock(offsets[item.key])[0] - float(value)) < 5e-4
    return False
