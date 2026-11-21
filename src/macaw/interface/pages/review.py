"""Where a person clears the gates and checks what the software decided, one recording at a time.
Every button stands for a `macaw` command, shown next to it."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.interop.coding_files import SUFFIXES as CODING_SUFFIXES
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.scale import HAND, calibration_clips, clip_frame, read_measurements, recording_scale
from macaw.qc.checks import RecordingResults
from macaw.qc.look import RIGHT, WRONG, describe, read_checks, read_look, summary, to_check, verdict
from macaw.store.annotations import read_annotations
from macaw.qc.align_clip import clip_path
from macaw.qc.report import _alignment, _cameras, _waiting

LABEL = "Review"
ACT = "review-act"
NO_ONE = "none"
PICK = "act-pick"
ACTS_DECIDED = "acts-decided"
SCALE_VIDEO, SCALE_AT, SCALE_GRAPH = "scale-video", "scale-at", "scale-graph"
SCALE_POINTS, SCALE_FRAME, SCALE_NOTE = "scale-points", "scale-frame", "scale-note"
SCALE_LENGTH, SCALE_OBJECT, SCALE_PERSON = "scale-length", "scale-object", "scale-person"
GAZE_RANDOM, GAZE_GRAPH = "gaze-random", "gaze-graph"
SCALE_AT_WRAP = "scale-at-wrap"
# the scale is measured where the player stands, unless a calibration clip is chosen
PLAYING = "playing"
# a frame's graph takes no room until there is a frame
HIDDEN = {"display": "none"}
GAZE_T, GAZE_NOTE, GAZE_PICK = "gaze-t", "gaze-note", "gaze-pick"
OBJECT_GRAPH, OBJECT_NOTE, OBJECT_PICK = "object-graph", "object-note", "object-pick"
# the point clicked on a thing, and the camera and time of the frame it was clicked on
OBJECT_POINT, OBJECT_T = "object-point", "object-t"
OBJECT_ACTIONS = {"object-start", "object-not-here", "object-keep", "object-clear"}
# a thing keeps one colour wherever it is drawn
_THING_COLOURS = ("#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#76b7b2", "#edc948", "#ff9da7", "#9c755f")
CLIP = "clip:"
UPLOAD = "review-upload"
# what each upload carries, and the files it takes
TRANSCRIPT_UPLOAD, CODING_UPLOAD = "transcript", "coding"
# the actions that record who decided, so they wait for a name
NAMED = {
    "accept-offset", "accept-owner", "accept-voices", "apply-review", "set-scale", "label-gaze", "look-right", "look-wrong",
    "name-people", "add-note", "further-nobody", "lines-suggested", "lines-apply", "lines-as-they-are",
    "object-start", "object-not-here", "object-keep",
}
VOICES_MIXED = "voices-mixed"
LINE_PICK, LINE_PERSON, LINE_HERE, LINE_AT = "line-pick", "line-split-person", "line-split-here", "line-split-at"
# more lines than this are changed from a csv with macaw line-speakers --apply
_MAX_LINES = 400
EVERYONE = "everyone"
RECORDING, PREVIOUS, NEXT = ui.RECORDING, ui.PREVIOUS, ui.NEXT
PLAYER, SEEK, SEEKED = "review-player", "review-seek", "review-seeked"
# parts only some recordings have, so callbacks reach them by pattern and find none where they are missing
CAMERA, PLAYER_BOX, TIMING, NAMES_BOX = "review-camera", "review-player-box", "review-timing", "review-names-box"
# where the player stands, {camera, own, t}; the frame cards follow it
TIME, STRIP = "review-time", "review-strip"
NOTE_TAGS, NOTE_TEXT = "review-note-tags", "review-note-text"
_MARK = "#d9534f"
# a page of dropdowns this long is still usable; the rest waits for the next save or the csv route
_MAX_ACTS = 200


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head("Review", "What the software decided, for a person to check."),
            html.Div(
                [
                    # kept in this browser, so a person types it once
                    ui.field("Your name", dcc.Input(id="review-by", value="", placeholder="needed to save", persistence=True, persistence_type="local")),
                ],
                className="toolbar",
            ),
            html.Div(id="review-message"),
            # the page stays in view, faded, while it is drawn again after a save
            dcc.Loading(
                html.Div(id="review-body"), delay_show=300, target_components={"review-body": "children"},
                overlay_style={"visibility": "visible", "opacity": 0.5, "pointerEvents": "none"},
            ),
            dcc.Store(id=SEEKED),
        ]
    )


def recording_label(config, rid: str, waiting: int | None = None) -> str:
    """The recording with what still waits on it and the tags of its notes, as the picker lists it."""
    from macaw.project.review_notes import note_tags, read_notes

    if waiting is None:
        try:
            from macaw.execution.runner import recording_configs

            own = recording_configs(config, [rid])[rid]
            waiting = len(_waiting(own, build_recording(rid, own)))
        except Exception:   # a recording that cannot be read is listed by its name alone
            waiting = 0
    tags = note_tags(read_notes(layout_for(config, create=False).review_notes_path(rid)))
    return " · ".join([rid, *([f"{waiting} waiting"] if waiting else []), *tags])


def stepped(options: list, current: str | None, step: int) -> str | None:
    values = [option["value"] if isinstance(option, dict) else option for option in options or []]
    if not values:
        return None
    if current not in values:
        return values[0]
    return values[min(max(values.index(current) + step, 0), len(values) - 1)]


def _button(label: str, action: str, target: str = "", primary: bool = False, small: bool = False):
    return ui.button(label, {"type": ACT, "action": action, "target": target}, primary=primary, small=small)


def _take_back(target: str):
    return _button("Take back", "take-back-click", target, small=True)


def _take_back_area(target: str):
    return _button("Take back", "take-back-area", target, small=True)


def _upload(label: str, kind: str, accept: str):
    return dcc.Upload(ui.button(label, icon_name="upload"), id={"type": UPLOAD, "kind": kind}, accept=accept)


def _offsets(config, layout, rid):
    rows, _ = _alignment(config, layout, rid, layout.output_dir / "page.html")
    if not rows:
        return None
    table_rows = [
        [
            row["target"], row["offset"], row["confidence"], row["agreement"],
            ui.pill(row["flag"], "warn") if row["flag"] else ui.pill("kept by " + row["accepted"].split(" ")[0], "ok") if row["accepted"] else ui.pill("automatic"),
            _button("Keep", "accept-offset", row["target"], small=True) if not row["accepted"] and row["method"] != "manual" else "",
        ]
        for row in rows
    ]
    figures = sorted((layout.renders_dir(rid) / "align").glob("*.png"))
    clip = clip_path(layout, rid)
    # the clip is remade under the same name when an offset changes, so its address carries its time
    player = html.Video(
        src=f"{file_url(clip)}?v={int(clip.stat().st_mtime)}", controls=True, preload="metadata", className="player",
    ) if clip.is_file() else None
    return ui.card(
        "Offsets between files",
        ui.table(["File", "Offset (s)", "Confidence", "Segments agreeing", "Status", ""], table_rows, numeric={"Offset (s)", "Confidence"}),
        player,
        html.Div([html.A(html.Img(src=file_url(path)), href=file_url(path), target="_blank") for path in figures], className="figures"),
        desc="Play the clip, where a file out of step sounds doubled, and look at its figure before keeping its offset.",
    )


def _one(kind: str) -> dict:
    return {"type": kind, "key": "one"}


def seek(t: float, label: str | None = None):
    """A time that moves the player there."""
    return html.Button(label or f"{t:.1f}", id={"type": SEEK, "t": f"{t:.2f}"}, className="seek", title="Play from here")


def quiet_pick(word: str, picker):
    return html.Label([word, picker], className="quiet-pick")


def _player(config, recording, camera: str | None):
    cameras = [item.name for item in recording.cameras()]
    if not cameras:
        return None
    camera = camera if camera in cameras else cameras[0]
    rid = recording.recording_id
    picker = dcc.Dropdown(cameras, camera, id=_one(CAMERA), clearable=False, searchable=False, className="quiet")
    # the slot keeps the card's place while it is pinned small under the tabs (assets/timeline.js)
    return html.Div(ui.card(
        "Play",
        html.Div(player_box(config, rid, camera), id=_one(PLAYER_BOX)),
        dcc.Store(id={"type": TIME, "key": rid}),
        desc="Every time on this page plays from here.",
        actions=quiet_pick("camera", picker) if len(cameras) > 1 else html.Div(picker, hidden=True),
        id="review-play",
    ), className="player-slot")


def player_box(config, rid: str, camera: str, at_s: float | None = None):
    """The camera's copy and its time strip; without a copy the strip still picks the frame the cards show."""
    from macaw.interface.pages.naming import first_moment
    from macaw.project.previews import current_preview, playing

    try:
        found = playing(config, rid, camera)
    except Exception as error:   # a file that cannot be read is said, not drawn
        return ui.note(str(error), "bad")
    window = [found.own_time(t) for t in found.window] if found.window else None
    start = found.own_time(at_s) if at_s is not None else first_moment(tuple(window) if window else None, found.duration_s)
    path = current_preview(config, rid, camera)
    strip = ui.time_strip(
        STRIP, video=PLAYER if path else None, camera=camera, offset=found.offset_s, scale=found.clock_scale,
        length=found.duration_s, start=start, window=window, ticks="#review-body .seek", timeStore={"type": TIME, "key": rid},
    )
    timing = dcc.Store(id=_one(TIMING), data={"offset_s": found.offset_s, "clock_scale": found.clock_scale})
    if path is None:
        return html.Div([
            timing,
            html.Div([
                ui.button("Make a copy to play here", {"type": ACT, "action": "prepare", "target": camera}, primary=True, icon_name="play"),
                _button("Make copies of every recording", "prepare-all"),
            ], className="buttons"),
            strip,
        ])
    notes = [] if found.sound == camera else [f"Sound from {found.sound}."]
    if not found.aligned:
        notes.append("Not aligned yet: times are the camera's own until prep.media has run.")
    return html.Div([
        timing,
        html.Video(src=f"{file_url(path)}?v={int(path.stat().st_mtime)}", id=PLAYER, preload="metadata", className="player"),
        strip,
        *([ui.note(" ".join(notes))] if notes else []),
    ], className="player-wrap")


def names_box(config, rid: str, camera: str, people: list[str]):
    from macaw.interface.pages import naming

    if not people:
        return ui.note(f"Nobody is read on {camera}; play another camera to name people.")
    side = html.Div([
        _button("Save the names", "name-people", primary=True),
        _button("Clear this camera's names", "clear-names"),
        _button("Ignore this area", "ignore-area"),
    ], className="buttons")
    return naming.editor(config, rid, camera, people, _take_back, _take_back_area, side=[side])


def _people_on_camera(config, layout, results, recording, camera: str | None):
    from macaw.interface.pages import naming

    rid = recording.recording_id
    people = naming.cameras_of(config, rid)
    if not people:
        return None
    roles = list(config.project.roles)
    blocks = []
    for found in _cameras(layout, results, rid):
        status = (
            ui.pill("kept by " + found["accepted"].split(" ")[0], "ok") if found["accepted"]
            else ui.pill("in doubt", "warn") if found["doubt"] else ui.pill("named", "ok")
        )
        blocks.append(
            html.Div(
                [
                    html.P([html.B(found["file"]), " ", status, " ", found["doubt"] or ""]),
                    html.P([html.Span([naming.swatch(person["subject"], roles), f": tracks {person['tracks']}"], className="tracks") for person in found["people"]]),
                    html.Div(_button("Keep as it stands", "accept-owner", found["file"], small=True), className="buttons"),
                ],
                className="camera",
            )
        )
    camera = camera if camera in people else next(iter(people))
    editor = html.Div(names_box(config, rid, camera, people[camera]), id=_one(NAMES_BOX))
    index = layout.renders_dir(rid) / "index.csv"
    shots = pd.read_csv(index) if index.is_file() else pd.DataFrame(columns=["file", "reason", "t", "video"])
    thumbs = [
        html.Figure([html.A(html.Img(src=file_url(layout.renders_dir(rid) / row.file)), href=file_url(layout.renders_dir(rid) / row.file), target="_blank"),
                     html.Figcaption([seek(float(row.t), f"{row.t:.1f} s"), f", {row.video}, {str(row.reason).replace('_', ' ')}"])])
        for row in shots.itertuples()
    ]
    return ui.card(
        "Who is on camera",
        *blocks,
        editor,
        html.Div([html.Span("Review frames", className="small-caps"), _button("Draw review frames", "render", small=True)], className="sub-head"),
        html.Div(thumbs, className="thumbs"),
        desc="Stop the player where everyone is in view, then click each person.",
    )


def _model_look(config, layout, rid):
    look = read_look(layout, rid)
    if not look and not config.qc.look.model:
        return None
    ask = html.Div(_button("Ask the model about the review frames", "look"), className="buttons")
    desc = "A model checks the people drawn on each frame; you check the model."
    if not look:
        return ui.card("What a model saw on the review frames", ask, desc=desc)
    checks = read_checks(layout, rid)
    renders = layout.renders_dir(rid)
    shots = []
    # what it flagged first, as in the report
    for entry in sorted(to_check(look), key=lambda item: not item["problems"]):
        said = "; ".join(item["says"] for item in entry["problems"]) or "nothing wrong, says the model; drawn at random to check"
        decided = verdict(entry, checks)
        shots.append(html.Figure([
            html.A(html.Img(src=file_url(renders / entry["file"])), href=file_url(renders / entry["file"]), target="_blank"),
            html.Figcaption([
                f"{entry['t']:.1f} s, {entry['video']}: {said} ",
                html.Div(
                    [_button("Right", "look-right", entry["file"], small=True), _button("Wrong", "look-wrong", entry["file"], small=True)],
                    className="buttons verdict" + {RIGHT: " said-right", WRONG: " said-wrong"}.get(decided, ""), title="Was the model right?",
                ),
            ]),
        ]))
    return ui.card(
        "What a model saw on the review frames",
        ui.note(describe(summary(layout, rid))),
        ask,
        html.Div(shots, className="thumbs"),
        desc=desc,
    )


def _voices(config, layout, recording):
    from macaw.extractors.audio.content.diarization import DiarizationExtractor, others_take_turns, read_voices, voice_owners
    from macaw.extractors.audio.content.speaker_id import offered_suggestion, read_voice_material, suggested_people, telling_lines

    rid = recording.recording_id
    voices = read_voices(config, rid)
    if not voices:
        return None
    named = voice_owners(config, rid)
    try:
        suggestion = offered_suggestion(config, recording) or {}
    except Exception:   # a suggestion that cannot be made leaves the choice to the person
        suggestion = {}
    suggested, why = suggestion.get("people", {}), suggested_people(suggestion) if suggestion else {}
    lines, pitch = read_voice_material(config, rid)
    subjects = [subject.subject_id for subject in recording.subjects]
    directory = layout.artifacts_dir(rid, DiarizationExtractor.name)
    rows = []
    for label, voice in sorted(voices.items()):
        current = named.get("people", {}).get(label) or suggested.get(label)
        median = pitch.get(label, {}).get("median_hz")
        own = lines[lines["subject_id"].astype(str) == label].sort_values("t_start")
        every = (
            html.Details([html.Summary(f"all {len(own)} lines"), html.Div([html.Div(f"{row.t_start:.1f} s  {row.text}") for row in own.itertuples()])])
            if len(own) > 3 else None
        )
        rows.append(
            [
                label,
                f"{voice.get('speech_s', 0.0):.0f} s",
                "" if median is None else f"{median:.0f} Hz",
                html.Div([*(html.Div(f"“{line}”") for line in telling_lines(lines, label)), every]),
                html.Div([_clip(directory, clip) for clip in voice.get("clips", [])], className="clips"),
                why.get(label, ""),
                dcc.Dropdown([*subjects, NO_ONE], current, id={"type": "voice-pick", "label": label}, clearable=False, className="picker small"),
            ]
        )
    by = named.get("by")
    status = ui.note(f"Named by {by.split(' ')[0]}." if by else "Not named yet; the choices shown are the suggestion.", "ok" if by else "warn")
    return ui.card(
        "Voices on the shared microphone",
        *_further_voice(config, layout, rid, directory),
        ui.table(["Voice", "Speech", "Pitch", "What it says", "Clips", "Suggested", "Person"], rows, numeric={"Pitch"}),
        status,
        _mouth_doubts(config, recording) if by else None,
        dcc.Checklist(
            [{"label": "One voice holds two people here: I will say who speaks each line", "value": "mixed"}],
            ["mixed"] if named.get("mixed") else [], id=_one(VOICES_MIXED), className="tag-picks",
        ),
        html.Div(_button("Save these names", "accept-voices", primary=True), className="buttons"),
        desc="Pick whose voice each is" + ("; none counts as other in turn-taking." if others_take_turns(config) else "; none leaves it out."),
    )


def _clip(directory: Path, clip: dict):
    """A voice's clip with a play button (assets/clips.js) and the same moment on the video."""
    return html.Div([
        html.Audio(src=file_url(directory / clip["file"]), preload="none"),
        html.Button(className="clip-play", title="Play this clip"),
        seek(clip["t_start"], "on video"),
    ], className="clip")


def _mouth_doubts(config, recording):
    """How many lines the faces speak against, as a hint that a voice may hold two people."""
    from macaw.extractors.audio.content.diarization import turn_takers
    from macaw.extractors.audio.content.line_speakers import evidence, speak_against, spoken_lines, with_evidence

    try:
        kind, found = evidence(config, recording.recording_id)
        lines = with_evidence(spoken_lines(config, recording), found, kind, everyone=turn_takers(config, recording)) if found else None
    except Exception:   # no shared transcript or no faces yet: no hint
        return None
    if lines is None or lines.empty:
        return None
    doubted = int((lines["doubt"] != "").sum())
    return ui.note(f"{speak_against(kind)} the voice of {doubted} of {len(lines)} lines.", "warn" if doubted else "ok")


def _further_voice(config, layout, rid: str, directory: Path) -> list:
    """What to show where one more voice than there are people was found: its clips and the three ways on."""
    import json

    from macaw.extractors.audio.content.diarization import describe_further_voice, read_further_voice, unheard_further_voice

    found = read_further_voice(config, rid)
    if not found or config.text.extra_voices:
        return []
    if not unheard_further_voice(config, rid):
        heard = json.loads(layout.further_voice_path(rid).read_text(encoding="utf-8"))
        return [ui.note(f"A further voice was found ({describe_further_voice(found)}); {heard['by'].split(' ')[0]} heard nobody else.", "ok")]
    return [
        ui.note(f"Someone else may speak here: {describe_further_voice(found)}. Listen, then choose.", "warn"),
        html.Div([_clip(directory, clip) for clip in found.get("clips", [])], className="clips"),
        html.Div([
            dcc.Link("Cut that part out", href=f"/segments?{urlencode({'recording': rid})}", className="button"),
            _button("Keep it as a voice of its own", "further-own"),
            _button("Nobody else speaks", "further-nobody"),
        ], className="buttons"),
    ]


def _lines(config, layout, recording):
    from macaw.extractors.audio.content.diarization import read_line_speakers, voices_mixed
    from macaw.extractors.audio.content.line_speakers import FACES, MOUTH, describe_kept, speak_against, suggested_lines

    rid = recording.recording_id
    if not voices_mixed(config, rid):
        return None
    title = "Who speaks each line"
    try:
        lines = suggested_lines(config, recording)
    except Exception as error:   # the shared transcript is not there yet, or cannot be read
        return ui.card(title, ui.note(str(error), "bad"))
    people = [*(subject.subject_id for subject in recording.subjects), NO_ONE]

    def row(line):
        doubt = html.Span([ui.pill("sure", "warn") if line.sure else ui.pill("doubt", "info"), f" {line.suggested or 'not ' + line.base}: {line.doubt}"]) if line.doubt else ""
        pick = dcc.Dropdown(people, line.person, id={"type": LINE_PICK, "t": f"{line.t_start:.3f}"}, clearable=False, className="picker small")
        return [seek(line.t_start), line.base, line.text, doubt, pick]

    kind = lines.attrs.get("evidence", MOUTH)
    columns = ["At (s)", "Voice of", "Words", "Faces" if kind == FACES else "Mouth movement", "Speaker"]
    shown = lines.head(_MAX_LINES)
    doubted, rest = shown[shown["doubt"] != ""], shown[shown["doubt"] == ""]
    kept = read_line_speakers(config, rid)
    sure = int((lines["sure"] & lines["suggested"].notna() & (lines["suggested"] != lines["person"])).sum())
    parts = [ui.note(f"{describe_kept(kept)}. Saved by {kept['by'].split(' ')[0]}." if kept else "Not checked yet.", "ok" if kept else "warn")]
    if doubted.empty:
        parts.append(ui.note(f"{speak_against(kind)} no line, or no face was read here."))
    else:
        parts.append(ui.table(columns, [row(line) for line in doubted.itertuples(index=False)], numeric={"At (s)"}))
    if not rest.empty:
        table = ui.table(columns, [row(line) for line in rest.itertuples(index=False)], numeric={"At (s)"})
        summary = f"the other {len(rest)} lines" + (f"; {kind} {'miss' if kind == FACES else 'misses'} some misplaced lines, so listen to these too" if not doubted.empty else "")
        parts.append(html.Details([html.Summary(summary), table], open=doubted.empty))
    if len(lines) > _MAX_LINES:
        parts.append(ui.note(f"Showing the first {_MAX_LINES} of {len(lines)} lines; the rest are changed from a csv."))
    parts.append(html.Div([
        ui.field("From the player's moment to the line's end", dcc.Dropdown(people, None, id=_one(LINE_PERSON), placeholder="speaker", className="picker small")),
        ui.button("Split there", _one(LINE_HERE)),
        dcc.Store(id=_one(LINE_AT)),
    ], className="toolbar"))
    parts.append(html.Div([
        *([_button(f"Give the {sure} sure ones to the person suggested", "lines-suggested")] if sure else []),
        _button("Nothing more to change", "lines-as-they-are"),
        _button("Save these speakers", "lines-apply", primary=True),
    ], className="buttons"))
    return ui.card(title, *parts, desc="Each line shows whose voice it fell in; pick who really speaks it. A sure suggestion needs both faces and two seconds.")


def lines_file(config_path: Path, rid: str, picks: dict[str, str]) -> Path:
    """The speakers picked, written where macaw line-speakers --apply reads them."""
    path = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / f"lines_{rid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t_start": [float(t) for t in picks], "person": list(picks.values())}).to_csv(path, index=False)
    return path


def _scale(config, layout, recording):
    from macaw.project.room import CM_PER_UNIT, UNIT_NAMES

    rid, units = recording.recording_id, config.project.units
    videos = sorted(item.name for item in recording.media if item.kind == "video")
    clips = calibration_clips(config)
    if not videos and not clips:
        return None
    own = read_measurements(layout.scale_path(rid))
    clip = config.scale.calibration
    from_clip = read_measurements(layout.calibration_scale_path(clip)) if clip else []
    scale = recording_scale(config, layout, rid)
    rows = [
        [
            where, f"{item.t:.1f}", item.person or EVERYONE, item.object or "",
            f"{item.length:g} {item.units}", f"{item.length_px:.1f}", f"{item.cm_per_px / CM_PER_UNIT[units]:.4g}", item.by or "",
        ]
        for where, item in [*((item.video, item) for item in own), *((f"calibration clip {item.video}", item) for item in from_clip)]
    ]
    if scale is None:
        status = ui.note("No scale yet: results and exports are in pixels.")
    elif scale.source == HAND:
        status = ui.note(f"Exports use {scale.describe()}" + (", not the measurements listed." if rows else "."), "ok")
    else:
        status = ui.note(f"Exports use {scale.describe()}.", "ok")
    if clip is None and clips:
        status = [status, ui.note(f"To use a calibration clip for this recording, write scale: {{calibration: name}} in project.yaml, or in recordings/{rid}.yaml for this one.")]
    key = {"key": rid}
    sources = ([{"label": "the camera playing", "value": PLAYING}] if videos else []) + [{"label": f"calibration clip {name}", "value": CLIP + name} for name in clips]
    chosen = CLIP + clip if not own and clip in clips else sources[0]["value"]
    picker = dcc.Dropdown(sources, chosen, id={"type": SCALE_VIDEO, **key}, clearable=False, searchable=False, className="quiet")
    remove = [_button("Remove this recording's measurements", "clear-scale")] if own else []
    remove += [_button(f"Remove the measurements of {clip}", "clear-scale", CLIP + clip)] if from_clip else []
    return ui.card(
        f"Pixels to {UNIT_NAMES[units]}",
        ui.table(
            ["Measured on", "At (s)", "Whose place", "What", "Length", "Pixels", f"{units} per pixel", "By"], rows,
            numeric={"At (s)", "Length", "Pixels", f"{units} per pixel"},
        ) if rows else None,
        html.Div([
            dcc.Graph(id={"type": SCALE_GRAPH, **key}, figure=scale_figure(None, []), config={"displayModeBar": False}, style=HIDDEN),
            html.Div([
                ui.note("A scale holds only at the object's distance from the camera, so measure where each person sits.", "warn"),
                *(status if isinstance(status, list) else [status]),
                html.Div(ui.field("At (s) of the clip", dcc.Input(id={"type": SCALE_AT, **key}, type="number", min=0, value=0, debounce=True)),
                         id={"type": SCALE_AT_WRAP, **key}, className="toolbar", style=None if chosen.startswith(CLIP) else HIDDEN),
                html.Div(id={"type": SCALE_NOTE, **key}),
                html.Div(
                    [
                        ui.field(f"Length ({units})", dcc.Input(id={"type": SCALE_LENGTH, **key}, type="number", min=0)),
                        ui.field("What it is", dcc.Input(id={"type": SCALE_OBJECT, **key}, placeholder="ruler")),
                        ui.field("Whose place", dcc.Dropdown([EVERYONE, *config.project.roles], EVERYONE, id={"type": SCALE_PERSON, **key}, clearable=False, className="picker small")),
                    ],
                    className="toolbar",
                ),
                html.Div([_button("Save this measurement", "set-scale", primary=True), *remove], className="buttons"),
            ], className="frame-controls"),
        ], className="frame-side"),
        dcc.Store(id={"type": SCALE_POINTS, **key}, data=[]),
        dcc.Store(id={"type": SCALE_FRAME, **key}, data=None),
        desc="Click the two ends of something of known length.",
        actions=quiet_pick("on", picker) if len(sources) > 1 else html.Div(picker, hidden=True),
    )


def _gaze(config, layout, recording):
    from macaw.project.gaze_labels import UNSURE, read_labels
    from macaw.qc.checks import check_gaze_labels

    rid = recording.recording_id
    videos = sorted(item.name for item in recording.media if item.kind == "video")
    if not videos:
        return None
    labels = read_labels(layout.gaze_labels_path(rid))
    roles = list(config.project.roles)
    counts = {(label.subject, label.target): 0 for label in labels}
    for label in labels:
        counts[(label.subject, label.target)] += 1
    rows = [[person, target, str(n)] for (person, target), n in sorted(counts.items())]
    checked = check_gaze_labels(recording, config, RecordingResults(layout, rid)).warnings if labels else []
    key = {"key": rid}
    choices = {person: [role for role in roles if role != person] + list(config.objects.names) + ["elsewhere", UNSURE] for person in roles}
    return ui.card(
        "Where people look",
        ui.table(["Who", "Looks at", "Frames"], rows, numeric={"Frames"}, fit=True) if rows else None,
        *[ui.note(f"{item.subject_id}: {item.message}", "warn" if item.severity == "warn" else "ok") for item in checked],
        html.Div([
            dcc.Graph(id={"type": GAZE_GRAPH, **key}, figure=scale_figure(None, []), config={"displayModeBar": False}, style=HIDDEN),
            html.Div([
                html.Div(id={"type": GAZE_NOTE, **key}),
                html.Div(
                    [ui.field(f"{person} looks at", dcc.Dropdown(choices[person], None, id={"type": GAZE_PICK, "key": person}, className="picker small")) for person in roles],
                    className="toolbar",
                ),
                html.Div([_button("Save what they look at", "label-gaze", primary=True)], className="buttons"),
            ], className="frame-controls"),
        ], className="frame-side"),
        dcc.Store(id={"type": GAZE_T, **key}, data=None),
        desc="Say whom or what each person looks at.",
        actions=html.Button(id={"type": GAZE_RANDOM, **key}, className="icon-button icon-dice", title="Go to a moment at random"),
    )


def gaze_frame(config_path: Path, rid: str, video: str, at_s: float):
    """The frame with each person's head box and name drawn on it, and a note; or why not."""
    from macaw.extractors.prep import apply_prep_timing
    from macaw.extractors.video.gaze.heads import head_boxes

    uri, problem = frame_uri(config_path, rid, video, at_s)
    if problem:
        return scale_figure(None, []), ui.note(problem, "warn")
    config = resolve_config(config_path, recording_id=rid)
    keypoints = RecordingResults(layout_for(config, create=False), rid).keypoints()
    recording = build_recording(rid, config)
    try:
        apply_prep_timing(config, recording)
    except Exception:   # before prep.media there are no people to draw anyway
        pass
    item = next(item for item in recording.media if item.name == video)
    figure = scale_figure(uri, [])
    if keypoints.empty or not item.fps:
        return figure, ui.note("Run pose first to see whose head is whose on the frame.")
    frame = round((at_s - item.offset_s) * item.fps)
    boxes = head_boxes(keypoints[(keypoints["video"] == video) & (keypoints["frame"] == frame) & keypoints["subject_id"].notna()])
    for box in boxes.itertuples():
        figure.add_shape(type="rect", x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1, line={"color": _MARK, "width": 2})
        figure.add_annotation(x=box.x0, y=box.y0, text=box.subject_id, showarrow=False, yshift=10, xanchor="left", font={"color": _MARK})
    return figure, ui.note(f"Frame at {at_s:.2f} s. Say for each person whom or what they look at, then save.")




def _objects(config, layout, recording):
    from macaw.extractors.video.objects.decisions import ABSENT, KEPT, STARTS, read_decisions
    from macaw.extractors.video.objects.review import waiting
    from macaw.extractors.video.objects.track import FINDER, load_starts

    names = list(config.objects.names)
    if not names:
        return None
    rid = recording.recording_id
    starts = load_starts(config, rid)
    said = read_decisions(layout.objects_path(rid))
    open_items = {(item.camera, item.name): item.why for item in waiting(config, rid)}
    rows, found_by_name = [], False
    for camera, entries in starts.items():
        for name, entry in entries.items():
            kept = said[KEPT].get(camera, {}).get(name, {}).get("by")
            spoken = bool(kept) or name in said[STARTS].get(camera, {}) or name in said[ABSENT].get(camera, {})
            if (camera, name) in open_items:
                status = ui.pill(open_items[camera, name], "warn")
                found_by_name |= entry["from"] == FINDER
            else:
                status = ui.pill(f"kept by {kept.split(' ')[0]}" if kept else "clear", "ok")
            share = f"{entry['found'] / entry['frames']:.0%}" if entry.get("frames") else ""
            gap = f"{entry['longest_gap_s']:g}" if entry.get("frames") else ""
            rows.append([
                camera, name, seek(entry["t"]) if "t" in entry else "", entry["from"], share, gap, status,
                _button("Forget", "object-clear", f"{camera}|{name}", small=True) if spoken else "",
            ])
    key = {"key": rid}
    keep_all = [_button("Keep the starts found by name", "object-keep")] if found_by_name else []
    return ui.card(
        "Things followed",
        ui.table(["Camera", "Thing", "Starts at (s)", "From", "In view", "Longest out of view (s)", "Status", ""], rows, numeric={"In view", "Longest out of view (s)"})
        if rows else ui.note("Not followed yet: the next run finds them by name."),
        html.Div([
            dcc.Graph(id={"type": OBJECT_GRAPH, **key}, figure=scale_figure(None, []), config={"displayModeBar": False}, style=HIDDEN),
            html.Div([
                html.Div(id={"type": OBJECT_NOTE, **key}),
                html.Div([ui.field("Thing", dcc.Dropdown(names, names[0] if len(names) == 1 else None, id={"type": OBJECT_PICK, **key}, className="picker small"))], className="toolbar"),
                html.Div([_button("Start it here", "object-start", primary=True), _button("Not on this camera", "object-not-here"), *keep_all], className="buttons"),
            ], className="frame-controls"),
        ], className="frame-side"),
        dcc.Store(id={"type": OBJECT_POINT, **key}, data=None),
        dcc.Store(id={"type": OBJECT_T, **key}, data=None),
        desc="Check where each thing starts; to start one elsewhere, click on it.",
    )


def object_frame(config_path: Path, rid: str, camera: str, at_s: float):
    """The frame with each thing's outline as followed there and each start found by name on it, and a note; or why not."""
    import plotly.graph_objects as go

    from macaw.extractors.prep import apply_prep_timing
    from macaw.extractors.video.objects.outline import polygons
    from macaw.extractors.video.objects.track import FINDER, load_starts, objects_file

    uri, problem = frame_uri(config_path, rid, camera, at_s)
    if problem:
        return scale_figure(None, []), ui.note(problem, "warn")
    config = resolve_config(config_path, recording_id=rid)
    recording = build_recording(rid, config)
    try:
        apply_prep_timing(config, recording)
    except Exception:   # before prep.media there is nothing followed to draw anyway
        pass
    item = next(item for item in recording.media if item.name == camera)
    figure = scale_figure(uri, [])
    path = objects_file(config, rid)
    if not item.fps or not path.is_file():
        return figure, ui.note("Nothing followed yet. Click on a thing, pick its name and press Start it here.")
    frame = round((at_s - item.offset_s) * item.fps)
    shown = pd.read_parquet(path, filters=[("video", "==", camera), ("frame", "==", frame), ("in_view", "==", True)])
    names = list(config.objects.names)
    for row in shown.itertuples():
        colour = _THING_COLOURS[names.index(row.label) % len(_THING_COLOURS)] if row.label in names else _MARK
        for part in polygons(row.outline):
            closed = [*part.tolist(), part[0].tolist()]
            figure.add_trace(go.Scatter(
                x=[x for x, _ in closed], y=[y for _, y in closed], mode="lines", fill="toself", line={"color": colour, "width": 2},
                opacity=0.45, hoverinfo="skip",
            ))
        figure.add_annotation(x=row.x0, y=row.y0, text=row.label, showarrow=False, yshift=10, xanchor="left", font={"color": colour}, bgcolor="rgba(0,0,0,0.6)")
    for name, entry in load_starts(config, rid).get(camera, {}).items():
        if entry.get("from") == FINDER and entry.get("frame") == frame:
            x0, y0, x1, y1 = entry["box"]
            figure.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line={"color": _MARK, "width": 2, "dash": "dash"})
            figure.add_annotation(x=x0, y=y1, text=f"{name} found here", showarrow=False, yshift=-10, xanchor="left", font={"color": _MARK}, bgcolor="rgba(0,0,0,0.6)")
    seen = ", ".join(sorted(set(shown["label"]))) or "none of the things"
    return figure, ui.note(f"Frame at {at_s:.2f} s: {seen} in view. To start a thing here, click on it, pick its name and press Start it here.")


def with_point(figure: dict, point: list[float]) -> dict:
    """The frame's figure with the clicked point on it, in place of one clicked before."""
    data = [trace for trace in figure.get("data", []) if trace.get("name") != OBJECT_POINT]
    marker = {"type": "scatter", "name": OBJECT_POINT, "x": [point[0]], "y": [point[1]], "mode": "markers", "hoverinfo": "skip",
              "marker": {"size": 14, "color": _MARK, "symbol": "x"}}
    return {**figure, "data": [*data, marker]}


def object_command(config_path: Path, rid: str, action: str, target: str, by: str, name: str | None, point, shown) -> list[str] | str:
    """The macaw objects command a button stands for, or what is missing for it."""
    def command(*extra):
        return cli_bridge.build_command(["objects"], config_path=config_path, recordings=[rid], extra_flags=[*extra, *(["--by", by] if by else [])])

    if action == "object-keep":
        return command("--keep")
    if action == "object-clear":
        camera, _, thing = target.partition("|")
        return cli_bridge.build_command(["objects"], config_path=config_path, recordings=[rid], extra_flags=["--clear", thing, "--camera", camera])
    if not shown:
        return "Play or seek to a frame of the camera first."
    if not name:
        return "Pick the thing first."
    if action == "object-not-here":
        return command("--not-here", name, "--camera", shown["camera"])
    if not point:
        return "Click on the thing in the frame first."
    return command("--start", name, "--camera", shown["camera"], "--at", f"{shown['t']:g}", "--point", f"{point[0]:g},{point[1]:g}")


def frame_uri(config_path: Path, rid: str, video: str, at_s: float) -> tuple[str | None, str | None]:
    """The frame the video steps read at a recording time, or a calibration clip's frame at its own time, as an
    image the page can show; or why not."""
    from macaw.extractors.prep import apply_prep_timing
    from macaw.media.ffmpeg import extract_frame

    config = resolve_config(config_path, recording_id=rid)
    if video.startswith(CLIP):
        clip = video[len(CLIP):]
        dest = layout_for(config, create=False).work_dir / "gui" / "scale" / f"clip_{Path(clip).stem}.jpg"
        try:
            clip_frame(config, clip, at_s, dest)
        except Exception as error:
            return None, f"No frame at {at_s:g} s of {clip}: {error}"
        return "data:image/jpeg;base64," + base64.b64encode(dest.read_bytes()).decode(), None
    recording = build_recording(rid, config)
    try:
        apply_prep_timing(config, recording)
    except Exception:   # before prep.media the file's own clock is all there is
        pass
    item = next((item for item in recording.media if item.name == video), None)
    if item is None:
        return None, f"{video} is not a video of {rid}."
    if config.video.normalise and item.prepared_video is None:
        return None, "Run prep.video first: frames are shown from its copy of the video, the one people are measured on."
    folder = layout_for(config, create=False).work_dir / "gui" / "scale"
    folder.mkdir(parents=True, exist_ok=True)
    # a file of its own: the scale, gaze and object cards ask for a frame at the same moment, each in a thread of its own
    handle, name = tempfile.mkstemp(prefix=f"{rid}_{Path(video).stem}_", suffix=".jpg", dir=folder)
    os.close(handle)
    dest = Path(name)
    try:
        extract_frame(item.frames_path, item.own_time(at_s), dest)
        return "data:image/jpeg;base64," + base64.b64encode(dest.read_bytes()).decode(), None
    except Exception as error:
        return None, f"No frame at {at_s:g} s: {error}"
    finally:
        dest.unlink(missing_ok=True)


def graph_style(figure):
    return None if figure.data else HIDDEN


def scale_figure(uri: str | None, points: list):
    import plotly.graph_objects as go

    figure = go.Figure()
    if uri:
        figure.add_trace(go.Image(source=uri, hovertemplate="x %{x}, y %{y}<extra></extra>"))
    if points:
        figure.add_trace(go.Scatter(
            x=[point[0] for point in points], y=[point[1] for point in points], mode="lines+markers",
            marker={"size": 10, "color": _MARK}, line={"color": _MARK, "width": 2}, hoverinfo="skip",
        ))
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False, dragmode=False,
        xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=520 if uri else 40,
    )
    return figure


def _transcript(config, layout, rid, path):
    from macaw.extractors.audio.content.asr import read_transcript
    from macaw.qc.transcript_check import KINDS, describe, read_check, still_there

    stamp = layout.corrected_transcript_stamp_path(rid)
    state = ui.note("The transcript in use is the one corrected by hand.", "ok") if stamp.is_file() else ui.note("The transcript in use is the recogniser's.")
    found = read_check(layout.transcript_check_path(rid))
    doubts = []
    if found:
        transcript = read_transcript(path)
        marked, _ = still_there(found, transcript)
        doubts.append(ui.note(describe(found, transcript), "warn" if not marked.empty else "ok"))
        if not marked.empty:
            rows = [[seek(row.t_start), row.subject_id, row.text, KINDS[row.kind], row.why] for row in marked.itertuples(index=False)]
            doubts.append(ui.table(["At (s)", "Who", "Words", "Kind", "Why"], rows, numeric={"At (s)"}))
    return ui.card(
        "Transcript",
        state,
        *doubts,
        html.Div(
            [
                _button("Export to ELAN", "transcript-export"),
                _upload("Import a corrected transcript (.eaf)", TRANSCRIPT_UPLOAD, ".eaf"),
                *([_button("Back to the recogniser's", "transcript-reset")] if layout.corrected_transcript_path(rid).is_file() else []),
                *([_button("Ask the model for doubtful lines", "transcript-check")] if config.coding.llm.model else []),
            ],
            className="buttons",
        ),
        desc="Export it, correct it in ELAN, then import it here.",
    )


def _act_key(subject, t_start, tier) -> str:
    return f"{subject}|{float(t_start):.3f}|{tier}"


def _waiting_acts(config, rid) -> pd.DataFrame:
    from macaw.coding.review_queue import export_review

    return export_review(config, [rid])


def _speech_acts(config, layout, recording):
    from macaw.coding.codebook import read_codebook
    from macaw.coding.llm.judge import NO_USABLE_ANSWER, SAMPLES_NAME, JudgementExtractor, coding_units

    rid = recording.recording_id
    if not layout.step_annotations_path(rid, JudgementExtractor.name).is_file():
        return None
    queue = _waiting_acts(config, rid)
    desc = "Pick the right code for each unit; the model's is kept beside yours."
    if queue.empty:
        return ui.card("Speech acts", ui.note("No coded unit is waiting for a person.", "ok"), desc=desc)
    try:
        codebook = read_codebook(config.coding.codebook)
        codes = codebook.labels()
        units = coding_units(config, recording, codebook.unit)
        words = {_act_key(row.subject_id, row.t_start, codebook.name): row.text for row in units.itertuples()}
    except Exception:   # the codebook moved or the transcript was redone; the times still identify each unit
        codes, words = sorted(set(queue["label"].dropna())), {}
    samples_path = layout.artifacts_dir(rid, JudgementExtractor.name) / SAMPLES_NAME
    answers = {}
    if samples_path.is_file():
        samples = pd.read_parquet(samples_path)
        for (subject, t_start), group in samples.groupby(["subject_id", "t_start"]):
            counted = group["label"].fillna(NO_USABLE_ANSWER).value_counts()
            text = ", ".join(f"{label} {count}" for label, count in counted.items())
            # the reason given with the commonest answer, where the model was asked for one
            given = group.loc[group["label"] == counted.index[0], "reason"] if "reason" in group else pd.Series(dtype=str)
            given = [reason for reason in given.fillna("").astype(str) if reason]
            answers[(str(subject), round(float(t_start), 3))] = text + (f": {given[0]}" if given else "")
    rows = []
    for row in queue.head(_MAX_ACTS).itertuples():
        key = _act_key(row.subject_id, row.t_start, row.tier)
        rows.append([
            seek(row.t_start), row.subject_id, words.get(key, ""),
            answers.get((str(row.subject_id), round(float(row.t_start), 3)), "" if pd.isna(row.label) else str(row.label)),
            ui.pill(str(row.reason).replace("_", " "), "warn"),
            dcc.Dropdown(codes, None, id={"type": PICK, "key": key}, placeholder="code", className="picker small"),
        ])
    more = [ui.note(f"Showing the first {_MAX_ACTS} of {len(queue)}; save these to see the rest.")] if len(queue) > _MAX_ACTS else []
    return ui.card(
        "Speech acts",
        ui.table(["At (s)", "Who", "Words", "Model's answers", "Why", "Code"], rows, numeric={"At (s)"}),
        *more,
        html.Div(_button("Save decisions", "apply-review", "foot", primary=True), className="buttons"),
        desc=desc,
        actions=[html.Span(decided_text(0, len(rows)), id=ACTS_DECIDED, className="hint"), _button("Save decisions", "apply-review", "head", primary=True)],
    )


def decided_text(decided: int, shown: int) -> str:
    return f"{decided} of {shown} decided"


def review_file(config_path: Path, rid: str, by: str, picks: dict[str, str]) -> Path:
    """The queue with the person's picks filled in, written where macaw review --apply reads it."""
    config = resolve_config(config_path, recording_id=rid)
    queue = _waiting_acts(config, rid)
    keys = [_act_key(row.subject_id, row.t_start, row.tier) for row in queue.itertuples()]
    queue["decision"] = [picks.get(key) or pd.NA for key in keys]
    queue["reviewer"] = by or pd.NA
    path = layout_for(config, create=False).work_dir / "gui" / f"review_{rid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(path, index=False)
    return path


def _notes(config, layout, rid: str):
    from macaw.project.review_notes import read_notes

    rows = [
        [html.Span([ui.pill(tag, "warn") for tag in note["tags"]], className="tags"), note["text"], note["by"], note["at"][:10],
         _button("Remove", "remove-note", note["id"], small=True)]
        for note in read_notes(layout.review_notes_path(rid))
    ]
    return ui.card(
        "Notes",
        ui.table(["Tags", "Note", "By", "On", ""], rows) if rows else None,
        dcc.Checklist(list(config.qc.note_tags), [], id=_one(NOTE_TAGS), inline=True, className="tag-picks"),
        dcc.Textarea(id=_one(NOTE_TEXT), placeholder="what to do with this recording, and why", className="note-text"),
        html.Div(_button("Save the note", "add-note", primary=True), className="buttons"),
        desc="For the people who review and run the project; a note changes nothing by itself. Tags are set in qc.note_tags.",
    )


def body(config_path: Path, rid: str | None, camera: str | None = None):
    if rid is None:
        return html.P("No recordings found.", className="none")
    config = resolve_config(config_path)
    own = resolve_config(config_path, recording_id=rid)
    layout = layout_for(config, create=False)
    # the recording as its own settings make it: its setup's devices, its checks and its codebook
    recording = build_recording(rid, own)
    results = RecordingResults(layout, rid)
    waiting = _waiting(own, recording)
    parts = [
        _player(own, build_recording(rid, own), camera),
        # two short cards side by side rather than each across the page
        html.Div([
            ui.card(
                "Waiting for a person",
                html.Ul([html.Li([ui.pill("waiting", "warn"), html.Span([html.B(gate), html.Span(ui.with_code(message), className="value")], className="what")]) for gate, message in waiting], className="status-list")
                if waiting else ui.note("Nothing is waiting on this recording.", "ok"),
            ),
            _notes(config, layout, rid),
        ], className="card-pair"),
        _offsets(config, layout, rid),
        _people_on_camera(own, layout, results, recording, camera),
        _model_look(config, layout, rid),
        _voices(own, layout, recording),
        _lines(own, layout, recording),
        _speech_acts(own, layout, recording),
        _coding(layout, rid),
        _scale(resolve_config(config_path, recording_id=rid), layout, recording),
        _gaze(resolve_config(config_path, recording_id=rid), layout, recording),
        _objects(own, layout, recording),
    ]
    if results.artifact("text.asr", "transcript.parquet") is not None:
        parts.append(_transcript(config, layout, rid, results.artifact("text.asr", "transcript.parquet")))
    return [part for part in parts if part is not None]


def _coding(layout, rid: str):
    path = layout.imported_codes_path(rid)
    upload = html.Div(_upload("Import your coding (.eaf, .csv, .xlsx)", CODING_UPLOAD, ",".join(CODING_SUFFIXES)), className="buttons")
    desc = "Kept under your name; importing again under it replaces your tiers."
    if not path.is_file():
        return ui.card("Your coding", html.P("No coding imported for this recording.", className="none"), upload, desc=desc)
    codes = read_annotations(path)
    counts = codes.groupby(["annotator", "subject_id", "tier"], dropna=False, sort=True).size().reset_index(name="n")
    rows = [[row.annotator, row.subject_id if isinstance(row.subject_id, str) else "everyone", row.tier, int(row.n)] for row in counts.itertuples()]
    first = counts.iloc[0]
    compare = f"/agreement?{urlencode({'tier': str(first.tier), 'units': str(first.annotator)})}"
    return ui.card(
        "Your coding",
        ui.table(["Coder", "Person", "Tier", "Codes"], rows, numeric={"Codes"}, fit=True),
        ui.note(["To compare it with the machine's codes or another coder's on the same tier, open ",
                 dcc.Link("the Agreement page", href=compare), ", or in a terminal:"]),
        ui.command(["macaw", "agreement", "--tier", str(first.tier), "--units", str(first.annotator)]),
        upload,
        desc=desc,
    )


def action_command(
    config_path: Path, rid: str, action: str, target: str, by: str, voices: dict, scale: dict | None = None,
    gaze: dict | None = None, mixed: bool = False,
) -> list[str]:
    def command(*sub, extra=()):
        return cli_bridge.build_command(list(sub), config_path=config_path, recordings=[rid], extra_flags=list(extra))

    by_flag = ["--by", by] if by else []
    if action == "label-gaze":
        return command("label-gaze", extra=["--at", f"{gaze['t']:g}", "--person", target, "--looks-at", gaze["looks_at"], *by_flag])
    if action in ("set-scale", "clear-scale"):
        source = scale["video"] if action == "set-scale" else target
        on_clip = bool(source) and source.startswith(CLIP)

        def measure(*extra):
            if on_clip:
                return cli_bridge.build_command(["set-scale"], config_path=config_path, extra_flags=["--calibration", source[len(CLIP):], *extra])
            return command("set-scale", extra=[*(["--camera", source] if source else []), *extra])

        if action == "clear-scale":
            return measure("--clear")
        (x1, y1), (x2, y2) = scale["points"]
        named = ["--object", scale["object"]] if scale.get("object") else []
        person = ["--person", scale["person"]] if scale.get("person") not in (None, EVERYONE) else []
        return measure(
            "--points", f"{x1:g},{y1:g},{x2:g},{y2:g}", "--length", f"{scale['length']:g}", "--at", f"{scale['at']:g}",
            *person, *named, *by_flag,
        )
    if action == "accept-offset":
        return command("accept-offset", extra=["--device", target, *by_flag])
    if action == "accept-owner":
        return command("accept-owner", extra=["--camera", target, *by_flag])
    if action == "accept-voices":
        names = ",".join(f"{label}={who}" for label, who in sorted(voices.items()))
        return command("accept-voices", extra=["--rename", names, *(["--mixed"] if mixed else []), *by_flag])
    if action == "further-nobody":
        return command("accept-voices", extra=["--nobody-else", *by_flag])
    if action in ("lines-suggested", "lines-as-they-are"):
        return command("line-speakers", extra=[f"--{action.removeprefix('lines-')}", *by_flag])
    if action == "lines-apply":
        return command("line-speakers", extra=["--apply", target, *by_flag])
    if action == "lines-split":
        return command("line-speakers", extra=["--at", target, "--person", voices["person"], *by_flag])
    if action == "render":
        return command("qc", extra=["--render"])
    if action == "look":
        return command("qc", extra=["--look"])
    if action in ("look-right", "look-wrong"):
        return command("check-look", extra=["--frame", target, "--verdict", action.removeprefix("look-"), *by_flag])
    if action == "clear-names":
        return command("name-people", extra=["--camera", target, "--clear"])
    if action == "remove-note":
        return command("note", extra=["--remove", target])
    if action == "prepare":
        return command("previews")
    if action == "transcript-reset-sure":
        return ["macaw", "transcript", "reset", "-c", str(config_path), "-r", rid]
    if action == "transcript-check":
        return ["macaw", "transcript", "check", "-c", str(config_path), "-r", rid]
    raise ValueError(f"unknown action {action}")


def note_command(config_path: Path, rid: str, tags: list[str], text: str, by: str) -> list[str]:
    flags = [flag for tag in tags for flag in ("--tag", tag)] + (["--text", text.strip()] if text.strip() else [])
    return cli_bridge.build_command(["note"], config_path=config_path, recordings=[rid], extra_flags=[*flags, "--by", by])


def _transcript_command(config_path: Path, rid: str, action: str, upload: Path | None, by: str) -> list[str]:
    if action == "transcript-export":
        return ["macaw", "transcript", "export", "-c", str(config_path), "-r", rid]
    return ["macaw", "transcript", "import", str(upload), "-c", str(config_path), "-r", rid, *(["--by", by] if by else [])]


def clicked(points: list, click: dict) -> list:
    """The ends clicked so far, in the frame's pixels; a click after two starts a new pair."""
    point = click["points"][0]
    kept = points if points and len(points) < 2 else []
    return [*kept, [float(point["x"]), float(point["y"])]]


def register(app, config_path: Path) -> None:
    from macaw.interface.pages import naming
    from macaw.interface.pages.overview import people_waiting

    naming.register(app, config_path, RECORDING, {"type": TIME, "key": ALL})

    @app.callback(Output(RECORDING, "options"), Input("url", "pathname"))
    def recordings(_):
        config = resolve_config(config_path)
        try:
            gates, _ = people_waiting(config)
        except Exception:   # the picker still lists the recordings, without what waits on them
            gates = {}
        return [{"label": recording_label(config, rid, len(gates.get(rid, []))), "value": rid} for rid in resolve_recordings(config, None)]

    @app.callback(Output(RECORDING, "value"), Input("url", "pathname"), Input("url", "search"))
    def linked_recording(pathname, search):
        rid = ui.asked(search).get("recording")
        page = (pathname or "/").strip("/").lower()
        if page not in ui.ONE_RECORDING_PAGES or rid not in resolve_recordings(resolve_config(config_path), None):
            return no_update
        return rid

    @app.callback(
        Output(RECORDING, "value", allow_duplicate=True),
        Input(PREVIOUS, "n_clicks"), Input(NEXT, "n_clicks"), State(RECORDING, "value"), State(RECORDING, "options"),
        prevent_initial_call=True,
    )
    def step(_previous, _next, rid, options):
        return stepped(options, rid, -1 if ctx.triggered_id == PREVIOUS else 1) or no_update

    @app.callback(
        Output({"type": PLAYER_BOX, "key": ALL}, "children"), Output({"type": NAMES_BOX, "key": ALL}, "children"),
        Input({"type": CAMERA, "key": ALL}, "value"), State(RECORDING, "value"),
        State({"type": PLAYER_BOX, "key": ALL}, "id"), State({"type": NAMES_BOX, "key": ALL}, "id"), State({"type": TIME, "key": ALL}, "data"),
        prevent_initial_call=True,
    )
    def camera_chosen(cameras, rid, players, boxes, times):
        # the picker appearing with the page fires this too, with nothing chosen by hand
        if not cameras or not any(item["value"] is not None for item in ctx.triggered) or len(ctx.triggered) != 1:
            return [no_update] * len(players), [no_update] * len(boxes)
        own = resolve_config(config_path, recording_id=rid)
        camera = cameras[0]
        people = naming.cameras_of(own, rid)
        at = (times or [None])[0]
        # the new camera opens at the moment the last one stood
        return [player_box(own, rid, camera, at["t"] if at else None) for _ in players], [names_box(own, rid, camera, people.get(camera, [])) for _ in boxes]

    app.clientside_callback(
        """function (clicks, timing) {
            const trigger = window.dash_clientside.callback_context.triggered_id;
            const video = document.getElementById('""" + PLAYER + """');
            if (!trigger || !(clicks || []).some(Boolean)) { return window.dash_clientside.no_update; }
            if (window.macawStrip && window.macawStrip.seek(parseFloat(trigger.t), !!video)) { return parseFloat(trigger.t); }
            if (!video) { return window.dash_clientside.no_update; }
            const t = parseFloat(trigger.t), own = timing && timing.length ? (t - timing[0].offset_s) / timing[0].clock_scale : t;
            video.currentTime = Math.max(own, 0);
            video.play();
            return t;
        }""",
        Output(SEEKED, "data"), Input({"type": SEEK, "t": ALL}, "n_clicks"), State({"type": TIMING, "key": ALL}, "data"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """function (clicks, timing) {
            const video = document.getElementById('""" + PLAYER + """');
            if (!clicks || !video) { return window.dash_clientside.no_update; }
            const t = timing && timing.length ? video.currentTime * timing[0].clock_scale + timing[0].offset_s : video.currentTime;
            return {t: Math.round(t * 100) / 100, n: clicks};
        }""",
        Output({"type": LINE_AT, "key": MATCH}, "data"), Input({"type": LINE_HERE, "key": MATCH}, "n_clicks"),
        State({"type": TIMING, "key": ALL}, "data"), prevent_initial_call=True,
    )

    @app.callback(
        Output({"type": SCALE_GRAPH, "key": MATCH}, "figure"),
        Output({"type": SCALE_GRAPH, "key": MATCH}, "style"),
        Output({"type": SCALE_POINTS, "key": MATCH}, "data"),
        Output({"type": SCALE_FRAME, "key": MATCH}, "data"),
        Output({"type": SCALE_NOTE, "key": MATCH}, "children"),
        Output({"type": SCALE_AT_WRAP, "key": MATCH}, "style"),
        Input({"type": SCALE_GRAPH, "key": MATCH}, "clickData"),
        Input({"type": SCALE_VIDEO, "key": MATCH}, "value"),
        Input({"type": SCALE_AT, "key": MATCH}, "value"),
        Input({"type": TIME, "key": MATCH}, "data"),
        State({"type": SCALE_POINTS, "key": MATCH}, "data"),
        State({"type": SCALE_FRAME, "key": MATCH}, "data"),
    )
    def scale_frame(click, video, at, time, points, uri):
        figure, *rest = _scale_frame(click, video, at, time, points, uri)
        return figure, no_update if figure is no_update else graph_style(figure), *rest, None if video.startswith(CLIP) else HIDDEN

    def _scale_frame(click, video, at, time, points, uri):
        trigger = ctx.triggered_id
        playing = video == PLAYING
        if trigger is not None and trigger["type"] == TIME and not playing:
            return no_update, no_update, no_update, no_update
        if trigger is None or trigger["type"] in (SCALE_VIDEO, SCALE_AT, TIME):
            if playing and not time:
                return scale_figure(None, []), [], None, None
            source, moment = (time["camera"], time["t"]) if playing else (video, float(at or 0.0))
            uri, problem = frame_uri(config_path, ctx.outputs_list[0]["id"]["key"], source, moment)
            note = ui.note(problem, "warn") if problem else ui.note("Click one end of the object, then the other.")
            return scale_figure(uri, []), [], uri, note
        if not uri or not click:
            return no_update, no_update, no_update, no_update
        points = clicked(points, click)
        if len(points) == 1:
            return scale_figure(uri, points), points, no_update, ui.note("Now click the other end.")
        (x1, y1), (x2, y2) = points
        note = ui.note(f"{((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5:.1f} pixels between the two ends. Type the length and save, or click again to start over.")
        return scale_figure(uri, points), points, no_update, note

    @app.callback(
        Output({"type": GAZE_GRAPH, "key": MATCH}, "figure"),
        Output({"type": GAZE_GRAPH, "key": MATCH}, "style"),
        Output({"type": GAZE_T, "key": MATCH}, "data"),
        Output({"type": GAZE_NOTE, "key": MATCH}, "children"),
        Input({"type": TIME, "key": MATCH}, "data"),
    )
    def show_gaze_frame(time):
        if not time:
            return no_update, no_update, no_update, no_update
        figure, note = gaze_frame(config_path, ctx.outputs_list[0]["id"]["key"], time["camera"], time["t"])
        return figure, graph_style(figure), time["t"], note

    @app.callback(
        Output({"type": OBJECT_GRAPH, "key": MATCH}, "figure"),
        Output({"type": OBJECT_GRAPH, "key": MATCH}, "style"),
        Output({"type": OBJECT_POINT, "key": MATCH}, "data"),
        Output({"type": OBJECT_T, "key": MATCH}, "data"),
        Output({"type": OBJECT_NOTE, "key": MATCH}, "children"),
        Input({"type": TIME, "key": MATCH}, "data"),
        Input({"type": OBJECT_GRAPH, "key": MATCH}, "clickData"),
        State({"type": OBJECT_GRAPH, "key": MATCH}, "figure"),
        State({"type": OBJECT_T, "key": MATCH}, "data"),
    )
    def show_object_frame(time, click, figure, shown):
        trigger = ctx.triggered_id
        if trigger is not None and trigger["type"] == OBJECT_GRAPH:
            if not click or not shown:
                return no_update, no_update, no_update, no_update, no_update
            point = [round(float(click["points"][0]["x"]), 1), round(float(click["points"][0]["y"]), 1)]
            note = ui.note(f"Clicked at {point[0]:g}, {point[1]:g} on the frame at {shown['t']:.2f} s. Pick the thing and press Start it here.")
            return with_point(figure, point), no_update, point, no_update, note
        if not time:
            return no_update, no_update, no_update, no_update, no_update
        figure, note = object_frame(config_path, ctx.outputs_list[0]["id"]["key"], time["camera"], time["t"])
        return figure, graph_style(figure), None, {"camera": time["camera"], "t": time["t"]}, note

    app.clientside_callback(
        "function (clicks) { if (clicks && window.macawStrip) { window.macawStrip.random(); } return window.dash_clientside.no_update; }",
        Output({"type": GAZE_RANDOM, "key": MATCH}, "title"), Input({"type": GAZE_RANDOM, "key": MATCH}, "n_clicks"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("review-body", "children"),
        Output("review-message", "children"),
        Output(RECORDING, "options", allow_duplicate=True),
        Input(ui.SHOWN["review"], "data"),
        Input({"type": ACT, "action": ALL, "target": ALL}, "n_clicks"),
        Input({"type": UPLOAD, "kind": ALL}, "contents"),
        Input({"type": LINE_AT, "key": ALL}, "data"),
        State({"type": UPLOAD, "kind": ALL}, "filename"),
        State("review-by", "value"),
        State({"type": "voice-pick", "label": ALL}, "value"),
        State({"type": "voice-pick", "label": ALL}, "id"),
        State({"type": PICK, "key": ALL}, "value"),
        State({"type": PICK, "key": ALL}, "id"),
        State({"type": GAZE_T, "key": ALL}, "data"),
        State({"type": GAZE_PICK, "key": ALL}, "value"),
        State({"type": GAZE_PICK, "key": ALL}, "id"),
        State({"type": SCALE_VIDEO, "key": ALL}, "value"),
        State({"type": SCALE_AT, "key": ALL}, "value"),
        State({"type": SCALE_POINTS, "key": ALL}, "data"),
        State({"type": SCALE_LENGTH, "key": ALL}, "value"),
        State({"type": SCALE_OBJECT, "key": ALL}, "value"),
        State({"type": SCALE_PERSON, "key": ALL}, "value"),
        State({"type": CAMERA, "key": ALL}, "value"),
        State({"type": naming.MARKS["type"], "key": ALL}, "data"),
        State({"type": naming.FRAME["type"], "key": ALL}, "data"),
        State({"type": naming.CORNERS["type"], "key": ALL}, "data"),
        State({"type": NOTE_TAGS, "key": ALL}, "value"),
        State({"type": NOTE_TEXT, "key": ALL}, "value"),
        State({"type": VOICES_MIXED, "key": ALL}, "value"),
        State({"type": LINE_PERSON, "key": ALL}, "value"),
        State({"type": LINE_PICK, "t": ALL}, "value"),
        State({"type": LINE_PICK, "t": ALL}, "id"),
        State({"type": TIME, "key": ALL}, "data"),
        State({"type": OBJECT_PICK, "key": ALL}, "value"),
        State({"type": OBJECT_POINT, "key": ALL}, "data"),
        State({"type": OBJECT_T, "key": ALL}, "data"),
        State(RECORDING, "options"),
        prevent_initial_call="initial_duplicate",
    )
    def update(rid, *values):
        *values, options = values
        content, message = _acted(rid, *values)
        if content is no_update:
            return no_update, message, no_update
        return content, message, relabelled(options, rid)

    def _acted(rid, clicks, uploads, split, filenames, by, picks, pick_ids, acts, act_ids, gaze_t, looks, look_ids, *rest):
        *scale, cameras, marks, shown, corners, tags, text, mixed, split_person, line_picks, line_ids, times, things, thing_points, thing_frames = rest
        camera = cameras[0] if cameras else None
        marks, shown, corners = (marks or [None])[0], (shown or [None])[0], (corners or [None])[0]
        trigger = ctx.triggered_id
        message = None
        by = (by or "").strip()
        if isinstance(trigger, dict) and trigger.get("type") == ACT:
            # a button appearing fires this too, with no click behind it
            if not any(value for value in clicks or []) or not ctx.triggered[0]["value"]:
                return no_update, no_update
            action, target = trigger["action"], trigger["target"]
            if action in NAMED and not by:
                return no_update, ui.note("Type your name first: each decision is kept with who made it.", "warn")
            measured = dict(zip(("video", "at", "points", "length", "object", "person"), (values[0] if values else None for values in scale)))
            if measured["video"] == PLAYING:
                at = (times or [None])[0] or {}
                measured["video"], measured["at"] = at.get("camera"), at.get("t")
            if action == "set-scale" and (len(measured["points"] or []) != 2 or not measured["length"]):
                return no_update, ui.note("Click both ends on the frame and type the length first.", "warn")
            if action == "label-gaze":
                looked = {item["key"]: value for item, value in zip(look_ids, looks) if value}
                if not gaze_t or gaze_t[0] is None or not looked:
                    return no_update, ui.note("Show a frame and say whom or what someone looks at first.", "warn")
                message = []
                for person, value in looked.items():
                    argv = action_command(config_path, rid, action, person, by, {}, gaze={"t": gaze_t[0], "looks_at": value})
                    message += [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
                return body(config_path, rid, camera), message
            if action == "name-people":
                if not shown or not marks:
                    return no_update, ui.note("Show a frame and click on the people first.", "warn")
                argv = naming.name_command(config_path, shown, marks, by)
            elif action in OBJECT_ACTIONS:
                argv = object_command(
                    config_path, rid, action, target, by, (things or [None])[0], (thing_points or [None])[0], (thing_frames or [None])[0],
                )
                if isinstance(argv, str):
                    return no_update, ui.note(argv, "warn")
            elif action == "add-note":
                if not (tags and tags[0]) and not (text and (text[0] or "").strip()):
                    return no_update, ui.note("Tick a tag or write the note first.", "warn")
                argv = note_command(config_path, rid, list(tags[0] or []), text[0] or "", by)
            elif action == "clear-names":
                argv = action_command(config_path, rid, action, camera, by, {})
            elif action == "take-back-click":
                argv = naming.take_back_command(config_path, rid, camera, target)
            elif action == "ignore-area":
                if not shown or len(corners or []) != 2:
                    return no_update, ui.note("Choose an area to ignore under Click on, then click two opposite corners of it on the frame.", "warn")
                argv = naming.area_command(config_path, rid, camera, corners)
            elif action == "take-back-area":
                argv = naming.area_back_command(config_path, rid, camera, target)
            elif action == "prepare-all":
                argv = cli_bridge.build_command(["previews"], config_path=config_path)
                log = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / "previews.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                cli_bridge.launch(argv, log, cwd=config_path.parent)
                return no_update, [ui.command(argv), ui.note(f"Making the copies in the background; {log} says how far it got. A recording's player appears once its copy is made.", "ok")]
            elif action == "transcript-export":
                argv = _transcript_command(config_path, rid, action, None, by)
            elif action == "transcript-reset":
                # hours of correction in ELAN go with one click, so the click only asks
                return no_update, ui.note([
                    "This deletes the corrected transcript; to have it back, import its .eaf again. ",
                    _button("Drop the corrected transcript", "transcript-reset-sure"),
                ], "warn")
            elif action == "apply-review":
                chosen = {item["key"]: value for item, value in zip(act_ids, acts) if value}
                argv = ["macaw", "review", "-c", str(config_path), "--apply", str(review_file(config_path, rid, by, chosen))]
            elif action == "further-own":
                from macaw.interface import project_file

                path = project_file.save_for_recording(config_path, rid, {"text.extra_voices": 1})
                return body(config_path, rid, camera), ui.note(f"text.extra_voices is 1 for this recording, in {path}. The next run separates its voices again; name them here after it.", "ok")
            elif action == "lines-apply":
                chosen = {item["t"]: value for item, value in zip(line_ids, line_picks) if value}
                argv = action_command(config_path, rid, action, str(lines_file(config_path, rid, chosen)), by, {})
            else:
                voices = {item["label"]: value for item, value in zip(pick_ids, picks)}
                measured["at"] = float(measured["at"] or 0.0)
                argv = action_command(config_path, rid, action, target, by, voices, measured, mixed=bool(mixed and mixed[0]))
            # a model asked about every frame or line, or a copy of a camera, takes longer than a click's command
            slow = {"timeout_s": 3600} if action in ("look", "transcript-check", "prepare") else {}
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent, **slow))]
        elif isinstance(trigger, dict) and trigger.get("type") == LINE_AT:
            at = ctx.triggered[0]["value"]
            if not at:   # the card was drawn again, nothing was clicked
                return no_update, no_update
            if not by or not (split_person and split_person[0]):
                return no_update, ui.note("Type your name and pick the speaker first.", "warn")
            argv = action_command(config_path, rid, "lines-split", f"{at['t']:g}", by, {"person": split_person[0]})
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        elif isinstance(trigger, dict) and trigger.get("type") == UPLOAD:
            contents = ctx.triggered[0]["value"]
            if not contents:   # the card was drawn again, nothing was uploaded
                return no_update, no_update
            if not by:
                return no_update, ui.note("Type your name first: what you import is kept with who made it.", "warn")
            kinds = [item["id"]["kind"] for item in ctx.inputs_list[2]]
            filename = filenames[kinds.index(trigger["kind"])] if trigger["kind"] in kinds else None
            path = _uploaded(config_path, contents, filename or f"{rid}.eaf")
            if trigger["kind"] == TRANSCRIPT_UPLOAD:
                argv = _transcript_command(config_path, rid, "transcript-import", path, by)
            else:
                argv = ["macaw", "import-codes", str(path), "-c", str(config_path), "-r", rid, "--coder", by]
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        return body(config_path, rid, camera), message

    def relabelled(options: list, rid: str | None) -> list:
        config = resolve_config(config_path)
        return [
            {**option, "label": recording_label(config, rid)} if isinstance(option, dict) and option.get("value") == rid else option
            for option in options or []
        ]

    @app.callback(Output(ACTS_DECIDED, "children"), Input({"type": PICK, "key": ALL}, "value"), prevent_initial_call=True)
    def decided(picks):
        return decided_text(sum(1 for value in picks if value), len(picks))


def _uploaded(config_path: Path, contents: str, filename: str) -> Path:
    directory = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(filename).name
    path.write_bytes(base64.b64decode(contents.split(",", 1)[1]))
    return path
