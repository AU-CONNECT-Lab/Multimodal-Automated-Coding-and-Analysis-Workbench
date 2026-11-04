"""The setup of the recordings: the people, cameras and microphones, whom each device records and where everyone and
everything is in the room. Saved as setups/<name>.yaml; nothing runs before it is."""

from __future__ import annotations

import re
from pathlib import Path

from dash import Input, Output, State, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.config.schema import DEVICE_NAME
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import ui
from macaw.interface.pages import cameras, naming, room
from macaw.project.recording import build_recording
from macaw.project.setup import setup_path

LABEL = "Setup"
PATH = "/setup"


def setup_names(config) -> list[str]:
    folder = Path(config.project.setups_dir)
    saved = sorted(path.stem for path in folder.glob("*.yaml")) if folder.is_dir() else []
    return sorted({*saved, config.setup.name})


def layout(config_path: Path):
    config = resolve_config(config_path)
    three_d = cameras.section(config_path)
    sections = [("The setup", "setup-room"), ("What each recording holds", "setup-holds"), ("Who is who", "setup-names")]
    return html.Div(
        [
            ui.page_head(
                "Setup",
                "Who and what is recorded, and where. Nothing runs until the setup is saved.",
            ),
            ui.jumps([*sections, *([("Cameras for three dimensions", "setup-3d")] if three_d is not None else [])]),
            html.Div(
                [
                    ui.field("Setup", dcc.Dropdown(setup_names(config), config.setup.name, id="setup-pick", clearable=False, className="picker")),
                    ui.field("New setup", dcc.Input(id="setup-new-name", placeholder="home_visits")),
                    ui.button("Start it", "setup-new"),
                    html.Div(id="setup-status"),
                ],
                className="toolbar",
            ),
            html.Div(id="setup-message"),
            html.Div(room.card(config), id="setup-room"),
            ui.card(
                "What each recording holds",
                dcc.Loading(html.Div(holdings(config_path), id="setup-files")),
                desc="The file found in each device's folder.",
                id="setup-holds",
            ),
            html.Div(naming.card(config), id="setup-names"),
            *([html.Div(three_d, id="setup-3d")] if three_d is not None else []),
        ]
    )


def holdings(config_path: Path):
    """Each recording's devices as found on disk, by the setup it uses."""
    config = resolve_config(config_path)
    rows = []
    for rid in resolve_recordings(config, None):
        try:
            own = resolve_config(config_path, recording_id=rid)
            recording = build_recording(rid, own)
        except MacawError as error:
            rows.append([rid, "", ui.pill("cannot be read", "bad"), str(error)])
            continue
        if not setup_path(own).is_file():
            rows.append([rid, own.setup.name, ui.pill("setup not saved", "warn"), ""])
            continue
        found = [html.Div(f"{item.name}: {item.path.relative_to(recording.root / item.name).as_posix()}") for item in recording.media]
        notes = recording.metadata.get("setup_notes", [])
        rows.append([rid, own.setup.name, found or ui.pill("nothing found", "warn"), html.Div([html.Div(note) for note in notes])])
    if not rows:
        return html.P("No recordings found under the project's raw folder.", className="none")
    return ui.table(["Recording", "Setup", "Devices found", "Notes"], rows)


def register(app, config_path: Path) -> None:
    naming.register(app, config_path)
    room.register(app, config_path)
    cameras.register(app, config_path)

    @app.callback(Output("setup-status", "children"), Input("setup-pick", "value"), Input("room-saved", "data"))
    def status(name, _saved):
        config = resolve_config(config_path)
        if (Path(config.project.setups_dir) / f"{name}.yaml").is_file():
            return ui.pill(f"saved as {Path(config.project.setups_dir).name}/{name}.yaml", "ok")
        return ui.pill(f"{name} is not saved yet: nothing runs until it is", "warn")

    @app.callback(
        Output("setup-pick", "options"),
        Output("setup-pick", "value"),
        Output("setup-message", "children", allow_duplicate=True),
        Input("setup-new", "n_clicks"),
        State("setup-new-name", "value"),
        prevent_initial_call=True,
    )
    def start(clicks, name):
        name = (name or "").strip()
        if not clicks:
            return no_update, no_update, no_update
        if not re.fullmatch(DEVICE_NAME, name):
            return no_update, no_update, ui.note("A setup's name takes letters, digits, - and _ only.", "bad")
        config = resolve_config(config_path)
        note = ui.note(f"Recordings made with {name} say setup: {{name: {name}}} in recordings/<id>.yaml.")
        return sorted({*setup_names(config), name}), name, note
