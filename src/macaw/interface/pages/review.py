"""Where a person clears the gates and checks what the software decided, one recording at a time.
Every button stands for a `macaw` command, shown next to it."""

from __future__ import annotations

import base64
import getpass
from pathlib import Path

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.execution.runner import resolve_recordings
from macaw.interface import cli_bridge, ui
from macaw.interface.files import file_url
from macaw.project.identity import pairing, read_confirmation
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.checks import RecordingResults
from macaw.qc.report import _alignment, _cameras, _waiting

LABEL = "Review"
ACT = "review-act"
NO_ONE = "none"


def layout(config_path: Path):
    recordings = resolve_recordings(resolve_config(config_path), None)
    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(recordings, recordings[0] if recordings else None, id="review-recording", clearable=False, className="picker"),
                    dcc.Input(id="review-by", value=getpass.getuser(), placeholder="your name", className="name"),
                    html.Button("Refresh", id="review-refresh", className="button"),
                    dcc.Upload(html.Button("Import a corrected transcript (.eaf)", className="button"), id="review-upload", accept=".eaf"),
                ],
                className="toolbar",
            ),
            html.Div(id="review-message"),
            dcc.Loading(html.Div(id="review-body")),
        ]
    )


def _button(label: str, action: str, target: str = "", primary: bool = False):
    return html.Button(label, id={"type": ACT, "action": action, "target": target}, className="button primary" if primary else "button")


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
    return ui.section(
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
            _button("Keep", "accept-offset", row["target"]) if not row["accepted"] and row["method"] != "manual" else "",
        ]
        for row in rows
    ]
    figures = sorted((layout.renders_dir(rid) / "align").glob("*.png"))
    return ui.section(
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
                            _button("Keep", "accept-owner", camera["file"]),
                            dcc.Input(id={"type": "review-rename", "target": camera["file"]}, placeholder="SHOWN=RIGHT, ...", className="rename"),
                            _button("Rename", "rename-owner", camera["file"]),
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
    return ui.section(
        "Who is on camera",
        *blocks,
        html.Div(_button("Draw review frames", "render"), className="buttons"),
        html.Div(thumbs, className="thumbs"),
        desc="Each box in the review frames carries the name it was given and its track id. Rename corrects who is who on one camera: P1=P3,P3=P1.",
    )


def _voices(config, layout, recording):
    from macaw.extractors.audio.content.diarization import DiarizationExtractor, read_voice_owners, read_voices
    from macaw.extractors.audio.content.speaker_id import suggest_voices

    rid = recording.recording_id
    voices = read_voices(config, rid)
    if not voices:
        return None
    named = read_voice_owners(layout.voice_owners_path(rid))
    try:
        suggested = (suggest_voices(config, recording) or {}).get("people", {})
    except Exception:   # a suggestion that cannot be made leaves the choice to the person
        suggested = {}
    subjects = [subject.subject_id for subject in recording.subjects]
    directory = layout.artifacts_dir(rid, DiarizationExtractor.name)
    rows = []
    for label, voice in sorted(voices.items()):
        current = named.get("people", {}).get(label) or suggested.get(label)
        rows.append(
            [
                label,
                f"{voice.get('speech_s', 0.0):.0f} s",
                html.Div([html.Audio(src=file_url(directory / clip["file"]), controls=True) for clip in voice.get("clips", [])], className="clips"),
                dcc.Dropdown([*subjects, NO_ONE], current, id={"type": "voice-pick", "label": label}, clearable=False, className="picker small"),
            ]
        )
    by = named.get("by")
    status = ui.note(f"Named by {by.split(' ')[0]}." if by else "Not named yet; the choices shown are the suggestion.", "ok" if by else "warn")
    return ui.section(
        "Voices on the shared microphone",
        ui.table(["Voice", "Speech", "Clips", "Person"], rows),
        status,
        html.Div(_button("Save these names", "accept-voices", primary=True), className="buttons"),
        desc="Listen to the clips, then pick whose voice each is; none leaves a voice out.",
    )


def _transcript(layout, rid):
    stamp = layout.corrected_transcript_stamp_path(rid)
    state = ui.note("The transcript in use is the one corrected by hand.", "ok") if stamp.is_file() else ui.note("The transcript in use is the recogniser's.")
    return ui.section(
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


def body(config_path: Path, rid: str | None):
    if rid is None:
        return html.P("No recordings found.", className="none")
    config = resolve_config(config_path)
    layout = layout_for(config, create=False)
    recording = build_recording(rid, config)
    results = RecordingResults(layout, rid)
    waiting = _waiting(config, recording)
    parts = [
        ui.section(
            "Waiting for a person",
            *[html.P([ui.pill("waiting", "warn"), " ", html.B(gate), ": ", message]) for gate, message in waiting]
            or [html.P("Nothing is waiting on this recording.", className="none")],
        ),
        _identity(config, recording),
        _offsets(config, layout, rid),
        _people_on_camera(layout, results, rid),
        _voices(config, layout, recording),
    ]
    if results.artifact("text.asr", "transcript.parquet") is not None:
        parts.append(_transcript(layout, rid))
    return [part for part in parts if part is not None]


def action_command(config_path: Path, rid: str, action: str, target: str, by: str, voices: dict, renames: dict) -> list[str]:
    def command(*sub, extra=()):
        return cli_bridge.build_command(list(sub), config_path=config_path, recordings=[rid], extra_flags=list(extra))

    by_flag = ["--by", by] if by else []
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


def register(app, config_path: Path) -> None:
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
    )
    def update(rid, _refresh, clicks, upload, filename, by, picks, pick_ids, renames, rename_ids):
        trigger = ctx.triggered_id
        message = None
        if isinstance(trigger, dict) and trigger.get("type") == ACT:
            # a button appearing fires this too, with no click behind it
            if not any(value for value in clicks or []) or not ctx.triggered[0]["value"]:
                return no_update, no_update
            action, target = trigger["action"], trigger["target"]
            if action == "transcript-export":
                argv = _transcript_command(config_path, rid, action, None, by)
            else:
                voices = {item["label"]: value for item, value in zip(pick_ids, picks)}
                named = {item["target"]: value for item, value in zip(rename_ids, renames)}
                argv = action_command(config_path, rid, action, target, by, voices, named)
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        elif trigger == "review-upload" and upload:
            directory = layout_for(resolve_config(config_path), create=False).work_dir / "gui" / "uploads"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / Path(filename or f"{rid}.eaf").name
            path.write_bytes(base64.b64decode(upload.split(",", 1)[1]))
            argv = _transcript_command(config_path, rid, "transcript-import", path, by)
            message = [ui.command(argv), ui.output(*cli_bridge.run_now(argv, cwd=config_path.parent))]
        return body(config_path, rid), message
