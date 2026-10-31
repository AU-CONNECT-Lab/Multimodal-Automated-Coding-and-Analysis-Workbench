"""Where a person clears the gates and checks what the software decided, one recording at a time.
Every button stands for a `macaw` command, shown next to it."""

from __future__ import annotations

import base64
import getpass
from pathlib import Path

import pandas as pd
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.project.identity import pairing, read_confirmation
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.scale import HAND, read_measurements, recording_scale
from macaw.qc.checks import RecordingResults
from macaw.qc.report import _alignment, _cameras, _waiting

LABEL = "Review"
ACT = "review-act"
NO_ONE = "none"
PICK = "act-pick"
SCALE_VIDEO, SCALE_AT, SCALE_SHOW, SCALE_GRAPH = "scale-video", "scale-at", "scale-show", "scale-graph"
SCALE_POINTS, SCALE_FRAME, SCALE_NOTE = "scale-points", "scale-frame", "scale-note"
SCALE_LENGTH, SCALE_OBJECT = "scale-length", "scale-object"
_MARK = "#d9534f"
# a page of dropdowns this long is still usable; the rest waits for the next save or the csv route
_MAX_ACTS = 200


def layout(config_path: Path):
    recordings = resolve_recordings(resolve_config(config_path), None)
    return html.Div(
        [
            ui.page_head(
                "Review",
                "What the software decided, for a person to check, one recording at a time. Every button runs the macaw command shown under it.",
                [
                    ui.button("Refresh", "review-refresh", icon_name="refresh"),
                    dcc.Upload(ui.button("Import a corrected transcript (.eaf)", icon_name="upload"), id="review-upload", accept=".eaf"),
                ],
            ),
            html.Div(
                [
                    ui.field("Recording", dcc.Dropdown(recordings, recordings[0] if recordings else None, id="review-recording", clearable=False, className="picker")),
                    ui.field("Your name", dcc.Input(id="review-by", value=getpass.getuser(), placeholder="your name", debounce=True)),
                ],
                className="toolbar",
            ),
            html.Div(id="review-message"),
            dcc.Loading(html.Div(id="review-body")),
        ]
    )


def _button(label: str, action: str, target: str = "", primary: bool = False, small: bool = False):
    return ui.button(label, {"type": ACT, "action": action, "target": target}, primary=primary, small=small)


def _identity(config, recording):
    entry = pairing(recording)
    confirmation = read_confirmation(layout_for(config, create=False).identity_path)
    rows = [[subject, ", ".join(files) or "(none)"] for subject, files in entry["subjects"].items()]
    rows += [["shared", ", ".join(entry["shared"])]] if entry["shared"] else []
    if config.identity.confirm == "skip":
        status = ui.note("identity.confirm is skip: file names are trusted without a person confirming.", "warn")
    elif confirmation:
        status = ui.note(f"Confirmed for the project by {confirmation.get('confirmed_by')} on {str(confirmation.get('confirmed_at', ''))[:10]}.", "ok")
    else:
        status = ui.note("Nobody has confirmed this yet.", "warn")
    return ui.card(
        "Which file is whose", ui.table(["Person", "Files"], rows), status,
        html.Div(_button("Confirm for the whole project", "identify", primary=True), className="buttons"),
        desc="Read from the file names by project.media_pattern. Confirming covers every recording in the project.",
    )


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
    return ui.card(
        "Offsets between files",
        ui.table(["File", "Offset (s)", "Confidence", "Segments agreeing", "Status", ""], table_rows, numeric={"Offset (s)", "Confidence"}),
        html.Div([html.A(html.Img(src=file_url(path)), href=file_url(path), target="_blank") for path in figures], className="figures"),
        desc="Look at a file's figure before keeping its offset; set one by hand with macaw set-offset.",
    )


def _people_on_camera(layout, results, rid):
    cameras = _cameras(layout, results, rid)
    if not cameras:
        return None
    blocks = []
    for camera in cameras:
        people = "; ".join(f"{person['subject']}: tracks {person['tracks']}" for person in camera["people"])
        status = (
            ui.pill("kept by " + camera["accepted"].split(" ")[0], "ok") if camera["accepted"]
            else ui.pill("in doubt", "warn") if camera["doubt"] else ui.pill("automatic")
        )
        blocks.append(
            html.Div(
                [
                    html.P([html.B(camera["file"]), " ", status, " ", camera["doubt"] or ""]),
                    html.P(people),
                    html.Div(
                        [
                            _button("Keep", "accept-owner", camera["file"], small=True),
                            dcc.Input(id={"type": "review-rename", "target": camera["file"]}, placeholder="SHOWN=RIGHT, ...", className="rename"),
                            _button("Rename", "rename-owner", camera["file"], small=True),
                        ],
                        className="buttons",
                    ),
                ],
                className="camera",
            )
        )
    index = layout.renders_dir(rid) / "index.csv"
    shots = pd.read_csv(index) if index.is_file() else pd.DataFrame(columns=["file", "reason", "t", "video"])
    thumbs = [
        html.Figure([html.A(html.Img(src=file_url(layout.renders_dir(rid) / row.file)), href=file_url(layout.renders_dir(rid) / row.file), target="_blank"),
                     html.Figcaption(f"{row.t:.1f} s, {row.video}, {str(row.reason).replace('_', ' ')}")])
        for row in shots.itertuples()
    ]
    return ui.card(
        "Who is on camera",
        *blocks,
        html.Div(_button("Draw review frames", "render"), className="buttons"),
        html.Div(thumbs, className="thumbs"),
        desc="Each box in the review frames carries the name it was given and its track id. Rename corrects who is who on one camera: P1=P3,P3=P1.",
    )


def _voices(config, layout, recording):
    from macaw.extractors.audio.content.diarization import DiarizationExtractor, others_take_turns, read_voice_owners, read_voices
    from macaw.extractors.audio.content.speaker_id import read_voice_material, suggest_voices, suggested_people, telling_lines

    rid = recording.recording_id
    voices = read_voices(config, rid)
    if not voices:
        return None
    named = read_voice_owners(layout.voice_owners_path(rid))
    try:
        suggestion = suggest_voices(config, recording) or {}
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
                html.Div([html.Audio(src=file_url(directory / clip["file"]), controls=True) for clip in voice.get("clips", [])], className="clips"),
                why.get(label, ""),
                dcc.Dropdown([*subjects, NO_ONE], current, id={"type": "voice-pick", "label": label}, clearable=False, className="picker small"),
            ]
        )
    by = named.get("by")
    status = ui.note(f"Named by {by.split(' ')[0]}." if by else "Not named yet; the choices shown are the suggestion.", "ok" if by else "warn")
    return ui.card(
        "Voices on the shared microphone",
        ui.table(["Voice", "Speech", "Pitch", "What it says", "Clips", "Suggested", "Person"], rows, numeric={"Pitch"}),
        status,
        html.Div(_button("Save these names", "accept-voices", primary=True), className="buttons"),
        desc=(
            "Read what each voice says and listen to the clips, then pick whose voice each is; none leaves a voice out"
            + (", and its words count as one further speaker, other, in turn-taking." if others_take_turns(config) else ".")
        ),
    )


def _scale(config, layout, recording):
    rid = recording.recording_id
    videos = sorted(item.path.name for item in recording.media if item.kind == "video")
    if not videos:
        return None
    measurements = read_measurements(layout.scale_path(rid))
    scale = recording_scale(config, layout, rid)
    rows = [
        [item.video, f"{item.t:.1f}", item.object or "", f"{item.length_cm:g}", f"{item.length_px:.1f}", f"{item.cm_per_px:.4g}", item.by or ""]
        for item in measurements
    ]
    if scale is None:
        status = ui.note("No scale yet: results and exports are in pixels.")
    elif scale.source == HAND:
        status = ui.note(f"scale.cm_per_px is {scale.cm_per_px:g} in the settings, so exports use it" + (" instead of these measurements." if rows else "."), "ok")
    else:
        status = ui.note(f"Exports use {scale.describe()}.", "ok")
    key = {"key": rid}
    return ui.card(
        "Pixels to centimetres",
        ui.table(["Video", "At (s)", "What", "Length (cm)", "Pixels", "cm per pixel", "By"], rows, numeric={"At (s)", "Length (cm)", "Pixels", "cm per pixel"}) if rows else None,
        status,
        html.Div(
            [
                ui.field("Video", dcc.Dropdown(videos, measurements[0].video if measurements else videos[0], id={"type": SCALE_VIDEO, **key}, clearable=False, className="picker")),
                ui.field("At (s)", dcc.Input(id={"type": SCALE_AT, **key}, type="number", min=0, value=10)),
                ui.button("Show frame", {"type": SCALE_SHOW, **key}),
            ],
            className="toolbar",
        ),
        html.Div(id={"type": SCALE_NOTE, **key}),
        dcc.Graph(id={"type": SCALE_GRAPH, **key}, figure=scale_figure(None, []), config={"displayModeBar": False}),
        dcc.Store(id={"type": SCALE_POINTS, **key}, data=[]),
        dcc.Store(id={"type": SCALE_FRAME, **key}, data=None),
        html.Div(
            [
                ui.field("Length (cm)", dcc.Input(id={"type": SCALE_LENGTH, **key}, type="number", min=0)),
                ui.field("What it is", dcc.Input(id={"type": SCALE_OBJECT, **key}, placeholder="table edge")),
            ],
            className="toolbar",
        ),
        html.Div(
            [_button("Save this measurement", "set-scale", primary=True), *([_button("Remove the measurements", "clear-scale")] if rows else [])],
            className="buttons",
        ),
        desc=(
            "Show a frame, click the two ends of something whose length you know and that sits about as far from the camera "
            "as the people, type its length and save. Measure two or three times to see how much the clicks vary. "
            "Results stay in pixels; exports add centimetres."
        ),
    )


def frame_uri(config_path: Path, rid: str, video: str, at_s: float) -> tuple[str | None, str | None]:
    """The frame the video steps read at a recording time, as an image the page can show; or why not."""
    from macaw.extractors.prep import apply_prep_timing
    from macaw.media.ffmpeg import extract_frame

    config = resolve_config(config_path, recording_id=rid)
    recording = build_recording(rid, config)
    try:
        apply_prep_timing(config, recording)
    except Exception:   # before prep.media the file's own clock is all there is
        pass
    item = next((item for item in recording.media if item.path.name == video), None)
    if item is None:
        return None, f"{video} is not a video of {rid}."
    if config.video.normalise and item.prepared_video is None:
        return None, "Run prep.video first: people are measured on its copy of the video, and the two ends must be clicked on the same picture."
    dest = layout_for(config, create=False).work_dir / "gui" / "scale" / f"{rid}_{Path(video).stem}.jpg"
    try:
        extract_frame(item.frames_path, at_s - item.offset_s, dest)
    except Exception as error:
        return None, f"No frame at {at_s:g} s: {error}"
    return "data:image/jpeg;base64," + base64.b64encode(dest.read_bytes()).decode(), None


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


def _transcript(layout, rid):
    stamp = layout.corrected_transcript_stamp_path(rid)
    state = ui.note("The transcript in use is the one corrected by hand.", "ok") if stamp.is_file() else ui.note("The transcript in use is the recogniser's.")
    return ui.card(
        "Transcript",
        state,
        html.Div(
            [
                _button("Export to ELAN", "transcript-export"),
                _button("Back to the recogniser's", "transcript-reset"),
            ],
            className="buttons",
        ),
        desc="Correct words, boundaries or who said what in ELAN, then import the file with the button at the top. Utterances that were left out sit on tiers of their own and go back in when moved to a transcript tier.",
    )


def _act_key(subject, t_start, tier) -> str:
    return f"{subject}|{float(t_start):.3f}|{tier}"


def _waiting_acts(config, rid) -> pd.DataFrame:
    from macaw.coding.review_queue import export_review

    return export_review(config, [rid])


def _speech_acts(config, layout, recording):
    from macaw.coding.codebook import read_codebook
    from macaw.coding.llm.judge import LAYER, SAMPLES_NAME, JudgementExtractor, coding_units

    rid = recording.recording_id
    if not (layout.annotations_path(rid) / f"{LAYER}.parquet").is_file():
        return None
    queue = _waiting_acts(config, rid)
    desc = "Units the model was unsure about. Pick the right code for each and save; your choice is kept beside the model's, which agreement then compares."
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
            counted = group["label"].value_counts()
            answers[(str(subject), round(float(t_start), 3))] = ", ".join(f"{label} {count}" for label, count in counted.items())
    rows = []
    for row in queue.head(_MAX_ACTS).itertuples():
        key = _act_key(row.subject_id, row.t_start, row.tier)
        rows.append([
            f"{row.t_start:.1f}", row.subject_id, words.get(key, ""),
            answers.get((str(row.subject_id), round(float(row.t_start), 3)), "" if pd.isna(row.label) else str(row.label)),
            ui.pill(str(row.reason).replace("_", " "), "warn"),
            dcc.Dropdown(codes, None, id={"type": PICK, "key": key}, placeholder="code", className="picker small"),
        ])
    more = [ui.note(f"Showing the first {_MAX_ACTS} of {len(queue)}; save these to see the rest.")] if len(queue) > _MAX_ACTS else []
    return ui.card(
        "Speech acts",
        ui.table(["At (s)", "Who", "Words", "Model's answers", "Why", "Code"], rows, numeric={"At (s)"}),
        *more,
        html.Div(_button("Save decisions", "apply-review", primary=True), className="buttons"),
        desc=desc,
    )


def review_file(config_path: Path, rid: str, by: str, picks: dict[str, str]) -> Path:
    """The queue with the person's picks filled in, written where macaw review --apply reads it."""
    config = resolve_config(config_path)
    queue = _waiting_acts(config, rid)
    keys = [_act_key(row.subject_id, row.t_start, row.tier) for row in queue.itertuples()]
    queue["decision"] = [picks.get(key) or pd.NA for key in keys]
    queue["reviewer"] = by or pd.NA
    path = layout_for(config, create=False).work_dir / "gui" / f"review_{rid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(path, index=False)
    return path


def body(config_path: Path, rid: str | None):
    if rid is None:
        return html.P("No recordings found.", className="none")
    config = resolve_config(config_path)
    layout = layout_for(config, create=False)
    recording = build_recording(rid, config)
    results = RecordingResults(layout, rid)
    waiting = _waiting(config, recording)
    parts = [
        ui.card(
            "Waiting for a person",
            html.Ul([html.Li([ui.pill("waiting", "warn"), html.Span([html.B(gate), html.Span(message, className="value")], className="what")]) for gate, message in waiting], className="status-list")
            if waiting else ui.note("Nothing is waiting on this recording.", "ok"),
        ),
        _identity(config, recording),
        _offsets(config, layout, rid),
        _people_on_camera(layout, results, rid),
        _voices(config, layout, recording),
        _speech_acts(config, layout, recording),
        _scale(resolve_config(config_path, recording_id=rid), layout, recording),
    ]
    if results.artifact("text.asr", "transcript.parquet") is not None:
        parts.append(_transcript(layout, rid))
    return [part for part in parts if part is not None]


def action_command(
    config_path: Path, rid: str, action: str, target: str, by: str, voices: dict, renames: dict, scale: dict | None = None
) -> list[str]:
    def command(*sub, extra=()):
        return cli_bridge.build_command(list(sub), config_path=config_path, recordings=[rid], extra_flags=list(extra))

    by_flag = ["--by", by] if by else []
    if action == "set-scale":
        (x1, y1), (x2, y2) = scale["points"]
        named = ["--object", scale["object"]] if scale.get("object") else []
        return command("set-scale", extra=[
            "--media", scale["video"], "--points", f"{x1:g},{y1:g},{x2:g},{y2:g}", "--length-cm", f"{scale['length']:g}",
            "--at", f"{scale['at']:g}", *named, *by_flag,
        ])
    if action == "clear-scale":
        return command("set-scale", extra=["--clear"])
    if action == "identify":
        return command("identify", extra=["--yes", *by_flag])
    if action == "accept-offset":
        return command("accept-offset", extra=["--media", target, *by_flag])
    if action == "accept-owner":
        return command("accept-owner", extra=["--media", target, *by_flag])
    if action == "rename-owner":
        return command("accept-owner", extra=["--media", target, "--rename", renames.get(target) or "", *by_flag])
    if action == "accept-voices":
        return command("accept-voices", extra=["--rename", ",".join(f"{label}={who}" for label, who in sorted(voices.items())), *by_flag])
    if action == "render":
        return command("qc", extra=["--render"])
    if action == "transcript-reset":
        return ["macaw", "transcript", "reset", "-c", str(config_path), "-r", rid]
    raise ValueError(f"unknown action {action}")


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
    @app.callback(Output("review-recording", "options"), Input("url", "pathname"))
    def recordings(_):
        return resolve_recordings(resolve_config(config_path), None)

    @app.callback(
        Output({"type": SCALE_GRAPH, "key": MATCH}, "figure"),
        Output({"type": SCALE_POINTS, "key": MATCH}, "data"),
        Output({"type": SCALE_FRAME, "key": MATCH}, "data"),
        Output({"type": SCALE_NOTE, "key": MATCH}, "children"),
        Input({"type": SCALE_SHOW, "key": MATCH}, "n_clicks"),
        Input({"type": SCALE_GRAPH, "key": MATCH}, "clickData"),
        State({"type": SCALE_VIDEO, "key": MATCH}, "value"),
        State({"type": SCALE_AT, "key": MATCH}, "value"),
        State({"type": SCALE_POINTS, "key": MATCH}, "data"),
        State({"type": SCALE_FRAME, "key": MATCH}, "data"),
        prevent_initial_call=True,
    )
    def scale_frame(_show, click, video, at, points, uri):
        trigger = ctx.triggered_id
        if trigger["type"] == SCALE_SHOW:
            uri, problem = frame_uri(config_path, trigger["key"], video, float(at or 0.0))
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
        Output("review-body", "children"),
        Output("review-message", "children"),
        Input("review-recording", "value"),
        Input("review-refresh", "n_clicks"),
        Input({"type": ACT, "action": ALL, "target": ALL}, "n_clicks"),
        Input("review-upload", "contents"),
        State("review-upload", "filename"),
        State("review-by", "value"),
        State({"type": "voice-pick", "label": ALL}, "value"),
        State({"type": "voice-pick", "label": ALL}, "id"),
        State({"type": "review-rename", "target": ALL}, "value"),
        State({"type": "review-rename", "target": ALL}, "id"),
        State({"type": PICK, "key": ALL}, "value"),
        State({"type": PICK, "key": ALL}, "id"),
        State({"type": SCALE_VIDEO, "key": ALL}, "value"),
        State({"type": SCALE_AT, "key": ALL}, "value"),
        State({"type": SCALE_POINTS, "key": ALL}, "data"),
        State({"type": SCALE_LENGTH, "key": ALL}, "value"),
        State({"type": SCALE_OBJECT, "key": ALL}, "value"),
    )
    def update(rid, _refresh, clicks, upload, filename, by, picks, pick_ids, renames, rename_ids, acts, act_ids, *scale):
        trigger = ctx.triggered_id
        message = None
        if isinstance(trigger, dict) and trigger.get("type") == ACT:
            # a button appearing fires this too, with no click behind it
            if not any(value for value in clicks or []) or not ctx.triggered[0]["value"]:
                return no_update, no_update
            action, target = trigger["action"], trigger["target"]
            measured = dict(zip(("video", "at", "points", "length", "object"), (values[0] if values else None for values in scale)))
            if action == "set-scale" and (len(measured["points"] or []) != 2 or not measured["length"]):
                return no_update, ui.note("Click both ends on the frame and type the length first.", "warn")
            if action == "transcript-export":
                argv = _transcript_command(config_path, rid, action, None, by)
            elif action == "apply-review":
                chosen = {item["key"]: value for item, value in zip(act_ids, acts) if value}
                argv = ["macaw", "review", "-c", str(config_path), "--apply", str(review_file(config_path, rid, by, chosen))]
            else:
                voices = {item["label"]: value for item, value in zip(pick_ids, picks)}
                named = {item["target"]: value for item, value in zip(rename_ids, renames)}
                measured["at"] = float(measured["at"] or 0.0)
                argv = action_command(config_path, rid, action, target, by, voices, named, measured)
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        elif trigger == "review-upload" and upload:
            directory = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / "uploads"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / Path(filename or f"{rid}.eaf").name
            path.write_bytes(base64.b64decode(upload.split(",", 1)[1]))
            argv = _transcript_command(config_path, rid, "transcript-import", path, by)
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        return body(config_path, rid), message
