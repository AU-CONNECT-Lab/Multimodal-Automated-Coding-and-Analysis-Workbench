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
from macaw.signals.coherence import MEAN
from macaw.signals.read import COHERENCE
from macaw.project.layout import layout_for
from macaw.project.previews import current_recording_preview, loudness, make_recording_preview, played

LABEL = "Signals"
PATH = "/signals"
PLAYER, STRIP, EDITS, SHOWN = "signals-player", "signals-strip", "signals-edits", "signals-shown"
ACT, OFFSET, BY = "signals-act", "signals-offset", "signals-by"
CARPET, LINES, MEASURES, CODING = "signals-carpet", "signals-lines", "signals-measures", "signals-coding"
COHERENT = "signals-coherence"
STRIP_BOX, OFFSETS_BOX = "signals-strip-box", "signals-offsets-box"
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

    groups, pairs, coherent = [], [], []
    for item in files:
        try:
            info = load_info(own, rid, item)
        except FileNotFoundError:
            continue
        if info.get("format") == COHERENCE:
            coherent += [channel["name"] for channel in info["channels"] if channel["name"] != MEAN and channel["name"] not in coherent]
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
    return {"groups": groups, "pairs": pairs, "measures": measures, "tiers": tiers, "coherent": coherent}


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
        html.Div(picker(COHERENT, "Coherence pairs", offered["coherent"], True, "their mean only"),
                 style=None if offered["coherent"] else {"display": "none"}),
    ], className="toolbar")
    signals_card = ui.card(
        "Signals", player, controls, html.Div(id=STRIP_BOX),
        desc=f"{source.name}, on the recording's clock. Blood oxygen follows behaviour by a few seconds. Drag on the marks lane to "
             "mark a stretch, such as a child touching the cap; a mark named as one of a file's events places the file by it.",
    )
    return html.Div([signals_card, html.Div(offsets_card(own, rid, files), id=OFFSETS_BOX)])


def strip(config_path: Path, rid: str, carpet_group: str | None, pairs: list[str], measures: list[str], tiers: list[str],
          coherent: list[str] | None = None):
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
        lanes=lanes(own, rid, clock, length, carpet_group, pairs or [], measures or [], tiers or [], coherent or []),
        names=event_names(own, rid),
    )


def event_names(own, rid: str) -> list[str]:
    """The names of the events the recording's signal files marked, offered when a mark is named."""
    from macaw.signals.read import load_info
    from macaw.signals.table import signal_files

    found = []
    for item in signal_files(own, rid):
        try:
            markers = load_info(own, rid, item).get("markers", [])
        except FileNotFoundError:
            continue
        for marker in markers:
            if str(marker["name"]) not in found:
                found.append(str(marker["name"]))
    return found


def strip_marks(own, rid: str, clock: tuple[float, float]) -> list[dict]:
    from macaw.signals.marks import marks_of

    return [
        {"name": mark.name, "a": a, "b": b, "note": mark.note, "done": True}
        for mark in marks_of(own, rid) for a, b in [own_times((mark.start_s, mark.end_s), clock)]
    ]


def lanes(own, rid: str, clock, length: float, carpet_group, pairs, measures, tiers, coherent=()) -> list[dict]:
    """Per file a carpet, its chosen channels as lines and the events it marked, per person with a cap IMU its turning
    against the head's on the video, then the chosen measures and coding."""
    from macaw.signals.align import head_turning, signal_offsets
    from macaw.signals.imu import gyro_speed
    from macaw.signals.lanes import CARPET_Z, carpet, channel_z, lines, spans
    from macaw.signals.read import load_aux, load_info, load_samples
    from macaw.signals.table import signal_files

    offsets = signal_offsets(own, rid)
    out, turning_done, events_drawn = [], set(), set()
    for item in signal_files(own, rid):
        result = offsets.get(item.key)
        own_clock = file_clock(result)
        where = "not aligned yet: set its offset below" if result is None or result.method == UNALIGNED else f"{own_clock[0]:+.2f} s"
        try:
            samples = load_samples(own, rid, item)
        except FileNotFoundError:
            continue
        at = to_played(samples["time"], own_clock, clock)
        info = load_info(own, rid, item)
        if info.get("format") == COHERENCE:
            out += coherence_lane(info["coherence"], samples, at, length, coherent, where)
            continue
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
        events = [(float(m["onset"]), float(m["onset"]) + float(m.get("duration") or 0.0), str(m["name"]))
                  for m in info.get("markers", [])]
        # a person's files that share a clock carry the same events; one lane shows them
        if events and (item.person, tuple(events), own_clock) not in events_drawn:
            events_drawn.add((item.person, tuple(events), own_clock))
            out.append({"kind": "bars", "name": f"{item.person or 'all'} events",
                        "title": f"Events {item.path.name} marked, where its offset puts them; a mark on the video named as one places the file by it",
                        "spans": spans(events, lambda t: float(to_played([t], own_clock, clock)[0]))})
        if item.person and item.person not in turning_done:
            try:
                imu = gyro_speed(load_aux(own, rid, item), own.signals.gyro)
            except ConfigError:   # several gyroscopes and none named: signals.align says which to write
                imu = None
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


def coherence_lane(about: dict, samples: pd.DataFrame, at, length: float, coherent, where: str) -> list[dict]:
    """The coherence between two people over time: the mean of every pair, and the pairs chosen, on 0 to 1."""
    from macaw.signals.lanes import lines

    names = [MEAN] + [name for name in coherent if name in samples.columns][:MAX_LINES]
    drawn = lines([("mean of all pairs" if name == MEAN else name, at, samples[name].to_numpy(dtype=float)) for name in names],
                  length, bounds=(0.0, 1.0))
    if not drawn:
        return []
    low, high = about["band_hz"]
    return [{"kind": "lines", "name": "coherence", "title": (
        f"Wavelet coherence of {' and '.join(about['members'])} at {low:g} to {high:g} Hz, from nirspipe; {where}. "
        f"The wavelet spreads each moment over {1 / high:.0f} to {1 / low:.0f} s and blood oxygen follows behaviour by a few seconds, "
        "so read it by the minute, not frame by frame"), **drawn}]


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
    from macaw.signals.align import FOLLOWS, describe, needs_person, signal_offsets
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
                if result is not None and result.method not in (MANUAL, UNALIGNED, FOLLOWS) and not result.accepted else "")
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


def names_an_event(own, rid: str, edit: dict) -> bool:
    """Whether a mark edited, before or after, bears the name of an event a signal file marked."""
    from macaw.signals.markers import marker_key

    events = {marker_key(name) for name in event_names(own, rid)}
    named = [edit.get("name"), (edit.get("was") or {}).get("name")]
    return any(name and marker_key(name) in events for name in named)


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
        Input(CARPET, "value"), Input(LINES, "value"), Input(MEASURES, "value"), Input(CODING, "value"), Input(COHERENT, "value"),
        State(ui.SHOWN["signals"], "data"),
    )
    def drawn(group, pairs, measures, tiers, coherent, rid):
        if not rid:
            return no_update
        try:
            return strip(config_path, rid, group, pairs, measures, tiers, coherent)
        except MacawError as error:
            return ui.note(str(error), "bad")

    @app.callback(
        Output(SHOWN, "data"), Output("signals-message", "children", allow_duplicate=True),
        Output(OFFSETS_BOX, "children", allow_duplicate=True), Output(STRIP_BOX, "children", allow_duplicate=True),
        Input(EDITS, "data"), State(ui.SHOWN["signals"], "data"),
        State(CARPET, "value"), State(LINES, "value"), State(MEASURES, "value"), State(CODING, "value"), State(COHERENT, "value"),
        prevent_initial_call=True,
    )
    def edited(edit, rid, group, pairs, measures, tiers, coherent):
        if not edit or not rid:
            return no_update, no_update, no_update, no_update
        own = resolve_config(config_path, recording_id=rid)
        clock = played_clock(own, rid)
        try:
            said, message = mark_edit(own, rid, edit, clock), None
        except (MacawError, ConfigError) as error:
            said, message = "not saved", ui.note(str(error), "bad")
        shown = {"segments": strip_marks(own, rid, clock), "said": said, "op": edit["op"]}
        if message is not None or not names_an_event(own, rid, edit):
            return shown, message, no_update, no_update
        # a mark named as a file's event places that file: align again at once
        argv = cli_bridge.build_command(["run"], config_path=config_path, recordings=[rid], extractors=["signals.align"],
                                        extra_flags=["--no-report"])
        code, text = cli_bridge.run_now(argv, cwd=config_path.parent)
        message = None if code == 0 else [ui.command(argv), ui.output(code, text)]
        own = resolve_config(config_path, recording_id=rid)
        from macaw.signals.table import signal_files

        return (shown, message, offsets_card(own, rid, signal_files(own, rid)),
                strip(config_path, rid, group, pairs, measures, tiers, coherent))


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
