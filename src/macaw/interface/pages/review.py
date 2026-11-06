"""Where a person clears the gates and checks what the software decided, one recording at a time.
Every button stands for a `macaw` command, shown next to it."""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.interop.coding_files import SUFFIXES as CODING_SUFFIXES
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.scale import CALIBRATION, HAND, calibration_clips, clip_frame, read_measurements, recording_scale
from macaw.qc.checks import RecordingResults
from macaw.qc.look import RIGHT, describe, read_checks, read_look, summary, to_check, verdict
from macaw.store.annotations import read_annotations
from macaw.qc.report import _alignment, _cameras, _waiting

LABEL = "Review"
ACT = "review-act"
NO_ONE = "none"
PICK = "act-pick"
ACTS_DECIDED = "acts-decided"
SCALE_VIDEO, SCALE_AT, SCALE_SHOW, SCALE_GRAPH = "scale-video", "scale-at", "scale-show", "scale-graph"
SCALE_POINTS, SCALE_FRAME, SCALE_NOTE = "scale-points", "scale-frame", "scale-note"
SCALE_LENGTH, SCALE_OBJECT, SCALE_PERSON = "scale-length", "scale-object", "scale-person"
GAZE_VIDEO, GAZE_AT, GAZE_SHOW, GAZE_RANDOM, GAZE_GRAPH = "gaze-video", "gaze-at", "gaze-show", "gaze-random", "gaze-graph"
GAZE_T, GAZE_NOTE, GAZE_PICK = "gaze-t", "gaze-note", "gaze-pick"
CLIP = "clip:"
UPLOAD = "review-upload"
# what each upload carries, and the files it takes
TRANSCRIPT_UPLOAD, CODING_UPLOAD = "transcript", "coding"
# the actions that record who decided, so they wait for a name
NAMED = {"accept-offset", "accept-owner", "accept-voices", "apply-review", "set-scale", "label-gaze", "look-right", "look-wrong"}
EVERYONE = "everyone"
_MARK = "#d9534f"
# a page of dropdowns this long is still usable; the rest waits for the next save or the csv route
_MAX_ACTS = 200


def layout(config_path: Path):
    recordings = resolve_recordings(resolve_config(config_path), None)
    return html.Div(
        [
            ui.page_head("Review", "What the software decided, for a person to check."),
            html.Div(
                [
                    ui.field("Recording", dcc.Dropdown(recordings, recordings[0] if recordings else None, id="review-recording", clearable=False, className="picker")),
                    # kept in this browser, so a person types it once
                    ui.field("Your name", dcc.Input(id="review-by", value="", placeholder="needed to save", persistence=True, persistence_type="local")),
                ],
                className="toolbar",
            ),
            html.Div(id="review-message"),
            dcc.Loading(html.Div(id="review-body")),
        ]
    )


def _button(label: str, action: str, target: str = "", primary: bool = False, small: bool = False):
    return ui.button(label, {"type": ACT, "action": action, "target": target}, primary=primary, small=small)


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
            else ui.pill("in doubt", "warn") if camera["doubt"] else ui.pill("named", "ok")
        )
        blocks.append(
            html.Div(
                [
                    html.P([html.B(camera["file"]), " ", status, " ", camera["doubt"] or ""]),
                    html.P(people),
                    html.Div(
                        [
                            _button("Keep as it stands", "accept-owner", camera["file"], small=True),
                            dcc.Link("Click the people again on the Setup page", href=f"/setup?recording={rid}&camera={camera['file']}", className="button small"),
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
        desc="A wrong name is put right on the Setup page.",
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
                ui.pill("the model was right", "ok") if decided == RIGHT else ui.pill("the model was wrong", "warn") if decided else "",
                html.Div([_button("Right", "look-right", entry["file"], small=True), _button("Wrong", "look-wrong", entry["file"], small=True)], className="buttons"),
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
    from macaw.extractors.audio.content.diarization import DiarizationExtractor, others_take_turns, read_voice_owners, read_voices
    from macaw.extractors.audio.content.speaker_id import offered_suggestion, read_voice_material, suggested_people, telling_lines

    rid = recording.recording_id
    voices = read_voices(config, rid)
    if not voices:
        return None
    named = read_voice_owners(layout.voice_owners_path(rid))
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
        desc="Pick whose voice each is" + ("; none counts as other in turn-taking." if others_take_turns(config) else "; none leaves it out."),
    )


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
    sources = [{"label": name, "value": name} for name in videos] + [{"label": f"calibration clip {name}", "value": CLIP + name} for name in clips]
    chosen = own[0].video if own else (CLIP + clip if clip in clips else sources[0]["value"])
    key = {"key": rid}
    remove = [_button("Remove this recording's measurements", "clear-scale")] if own else []
    remove += [_button(f"Remove the measurements of {clip}", "clear-scale", CLIP + clip)] if from_clip else []
    return ui.card(
        f"Pixels to {UNIT_NAMES[units]}",
        ui.note(
            "A scale holds only at the object's distance from the camera, so measure where each person sits.",
            "warn",
        ),
        ui.table(
            ["Measured on", "At (s)", "Whose place", "What", "Length", "Pixels", f"{units} per pixel", "By"], rows,
            numeric={"At (s)", "Length", "Pixels", f"{units} per pixel"},
        ) if rows else None,
        status,
        html.Div(
            [
                ui.field("Measure on", dcc.Dropdown(sources, chosen, id={"type": SCALE_VIDEO, **key}, clearable=False, className="picker")),
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
                ui.field(f"Length ({units})", dcc.Input(id={"type": SCALE_LENGTH, **key}, type="number", min=0)),
                ui.field("What it is", dcc.Input(id={"type": SCALE_OBJECT, **key}, placeholder="ruler")),
                ui.field("Whose place", dcc.Dropdown([EVERYONE, *config.project.roles], EVERYONE, id={"type": SCALE_PERSON, **key}, clearable=False, className="picker small")),
            ],
            className="toolbar",
        ),
        html.Div([_button("Save this measurement", "set-scale", primary=True), *remove], className="buttons"),
        desc=f"Click the two ends of something of known length; exports add {UNIT_NAMES[units]}.",
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
    choices = {person: [role for role in roles if role != person] + list(config.gaze.targets) + ["elsewhere", UNSURE] for person in roles}
    return ui.card(
        "Where people look",
        ui.table(["Who", "Looks at", "Frames"], rows, numeric={"Frames"}, fit=True) if rows else None,
        *[ui.note(f"{item.subject_id}: {item.message}", "warn" if item.severity == "warn" else "ok") for item in checked],
        html.Div(
            [
                ui.field("Video", dcc.Dropdown(videos, videos[0], id={"type": GAZE_VIDEO, **key}, clearable=False, className="picker")),
                ui.field("At (s)", dcc.Input(id={"type": GAZE_AT, **key}, type="number", min=0, value=10)),
                ui.button("Show frame", {"type": GAZE_SHOW, **key}),
                ui.button("A frame at random", {"type": GAZE_RANDOM, **key}),
            ],
            className="toolbar",
        ),
        html.Div(id={"type": GAZE_NOTE, **key}),
        dcc.Graph(id={"type": GAZE_GRAPH, **key}, figure=scale_figure(None, []), config={"displayModeBar": False}),
        dcc.Store(id={"type": GAZE_T, **key}, data=None),
        html.Div(
            [ui.field(f"{person} looks at", dcc.Dropdown(choices[person], None, id={"type": GAZE_PICK, "key": person}, className="picker small")) for person in roles],
            className="toolbar",
        ),
        html.Div([_button("Save what they look at", "label-gaze", primary=True)], className="buttons"),
        desc="Say whom or what each person looks at, on frames taken at random.",
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


def random_time(config_path: Path, rid: str, video: str) -> float:
    from macaw.extractors.prep import apply_prep_timing

    config = resolve_config(config_path, recording_id=rid)
    recording = build_recording(rid, config)
    try:
        prep = apply_prep_timing(config, recording)
        start, length = prep.t_origin_s, prep.duration_s
    except Exception:   # before prep.media only the file's own length is known
        item = next(item for item in recording.media if item.name == video)
        start, length = item.offset_s, item.duration_s or 0.0
    return round(start + float(np.random.default_rng().uniform(0, max(length, 0.0))), 2)


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
            rows = [[f"{row.t_start:.1f}", row.subject_id, row.text, KINDS[row.kind], row.why] for row in marked.itertuples(index=False)]
            doubts.append(ui.table(["At (s)", "Who", "Words", "Kind", "Why"], rows, numeric={"At (s)"}))
    return ui.card(
        "Transcript",
        state,
        *doubts,
        html.Div(
            [
                _button("Export to ELAN", "transcript-export"),
                _upload("Import a corrected transcript (.eaf)", TRANSCRIPT_UPLOAD, ".eaf"),
                _button("Back to the recogniser's", "transcript-reset"),
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
    from macaw.coding.llm.judge import LAYER, SAMPLES_NAME, JudgementExtractor, coding_units

    rid = recording.recording_id
    if not (layout.annotations_path(rid) / f"{LAYER}.parquet").is_file():
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
            counted = group["label"].value_counts()
            text = ", ".join(f"{label} {count}" for label, count in counted.items())
            # the reason given with the commonest answer, where the model was asked for one
            given = group.loc[group["label"] == counted.index[0], "reason"] if "reason" in group else pd.Series(dtype=str)
            given = [reason for reason in given.fillna("").astype(str) if reason]
            answers[(str(subject), round(float(t_start), 3))] = text + (f": {given[0]}" if given else "")
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
        html.Div(_button("Save decisions", "apply-review", "foot", primary=True), className="buttons"),
        desc=desc,
        actions=[html.Span(decided_text(0, len(rows)), id=ACTS_DECIDED, className="hint"), _button("Save decisions", "apply-review", "head", primary=True)],
    )


def decided_text(decided: int, shown: int) -> str:
    return f"{decided} of {shown} decided"


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
            html.Ul([html.Li([ui.pill("waiting", "warn"), html.Span([html.B(gate), html.Span(ui.with_code(message), className="value")], className="what")]) for gate, message in waiting], className="status-list")
            if waiting else ui.note("Nothing is waiting on this recording.", "ok"),
        ),
        _offsets(config, layout, rid),
        _people_on_camera(layout, results, rid),
        _model_look(config, layout, rid),
        _voices(config, layout, recording),
        _speech_acts(config, layout, recording),
        _coding(layout, rid),
        _scale(resolve_config(config_path, recording_id=rid), layout, recording),
        _gaze(resolve_config(config_path, recording_id=rid), layout, recording),
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
    gaze: dict | None = None,
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
        return command("accept-voices", extra=["--rename", ",".join(f"{label}={who}" for label, who in sorted(voices.items())), *by_flag])
    if action == "render":
        return command("qc", extra=["--render"])
    if action == "look":
        return command("qc", extra=["--look"])
    if action in ("look-right", "look-wrong"):
        return command("check-look", extra=["--frame", target, "--verdict", action.removeprefix("look-"), *by_flag])
    if action == "transcript-reset":
        return ["macaw", "transcript", "reset", "-c", str(config_path), "-r", rid]
    if action == "transcript-check":
        return ["macaw", "transcript", "check", "-c", str(config_path), "-r", rid]
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

    @app.callback(Output("review-recording", "value"), Input("url", "pathname"), Input("url", "search"))
    def linked_recording(pathname, search):
        rid = ui.asked(search).get("recording")
        if (pathname or "/").rstrip("/") != f"/{LABEL.lower()}" or rid not in resolve_recordings(resolve_config(config_path), None):
            return no_update
        return rid

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
        Output({"type": GAZE_GRAPH, "key": MATCH}, "figure"),
        Output({"type": GAZE_T, "key": MATCH}, "data"),
        Output({"type": GAZE_AT, "key": MATCH}, "value"),
        Output({"type": GAZE_NOTE, "key": MATCH}, "children"),
        Input({"type": GAZE_SHOW, "key": MATCH}, "n_clicks"),
        Input({"type": GAZE_RANDOM, "key": MATCH}, "n_clicks"),
        State({"type": GAZE_VIDEO, "key": MATCH}, "value"),
        State({"type": GAZE_AT, "key": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def show_gaze_frame(_show, _random, video, at):
        rid = ctx.triggered_id["key"]
        at = random_time(config_path, rid, video) if ctx.triggered_id["type"] == GAZE_RANDOM else float(at or 0.0)
        figure, note = gaze_frame(config_path, rid, video, at)
        return figure, at, at, note

    @app.callback(
        Output("review-body", "children"),
        Output("review-message", "children"),
        Input("review-recording", "value"),
        Input({"type": ACT, "action": ALL, "target": ALL}, "n_clicks"),
        Input({"type": UPLOAD, "kind": ALL}, "contents"),
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
    )
    def update(rid, clicks, uploads, filenames, by, picks, pick_ids, acts, act_ids, gaze_t, looks, look_ids, *scale):
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
                return body(config_path, rid), message
            if action == "transcript-export":
                argv = _transcript_command(config_path, rid, action, None, by)
            elif action == "apply-review":
                chosen = {item["key"]: value for item, value in zip(act_ids, acts) if value}
                argv = ["macaw", "review", "-c", str(config_path), "--apply", str(review_file(config_path, rid, by, chosen))]
            else:
                voices = {item["label"]: value for item, value in zip(pick_ids, picks)}
                measured["at"] = float(measured["at"] or 0.0)
                argv = action_command(config_path, rid, action, target, by, voices, measured)
            # a model asked about every frame or line takes longer than a click's command
            slow = {"timeout_s": 3600} if action in ("look", "transcript-check") else {}
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent, **slow))]
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
        return body(config_path, rid), message

    @app.callback(Output(ACTS_DECIDED, "children"), Input({"type": PICK, "key": ALL}, "value"), prevent_initial_call=True)
    def decided(picks):
        return decided_text(sum(1 for value in picks if value), len(picks))


def _uploaded(config_path: Path, contents: str, filename: str) -> Path:
    directory = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(filename).name
    path.write_bytes(base64.b64decode(contents.split(",", 1)[1]))
    return path
