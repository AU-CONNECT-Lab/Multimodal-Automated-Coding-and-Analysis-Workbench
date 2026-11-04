"""The cameras and microphones of the recordings: what each device folder of a recording holds, whom each camera is read
for and whose voices each device carries. Saved as setups/<name>.yaml; nothing runs before it is."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from pydantic import ValidationError

from macaw.config.loader import resolve_config
from macaw.config.schema import DEVICE_NAME, MacawConfig, SetupSection
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import ui
from macaw.interface.pages import naming
from macaw.project.recording import build_recording
from macaw.project.setup import setup_path, write_setup

LABEL = "Setup"
PATH = "/setup"
NAME, USE, ADD, REMOVE = "setup-name", "setup-use", "setup-add", "setup-remove"
KINDS = {"cameras": ("body", "face", "gaze", "hears"), "microphones": ("hears",)}
COLUMNS = {"body": "Body", "face": "Face", "gaze": "Gaze", "hears": "Voices"}


def setup_names(config) -> list[str]:
    folder = Path(config.project.setups_dir)
    saved = sorted(path.stem for path in folder.glob("*.yaml")) if folder.is_dir() else []
    return sorted({*saved, config.setup.name})


def layout(config_path: Path):
    config = resolve_config(config_path)
    return html.Div(
        [
            ui.page_head(
                "Setup",
                "The cameras and microphones of the recordings. Each recording folder holds one folder per device, named "
                "as here, with that device's file in it; nothing is read from the files' own names. Nothing runs before the "
                "setup is saved.",
            ),
            html.Div(
                [
                    ui.field("Setup", dcc.Dropdown(setup_names(config), config.setup.name, id="setup-pick", clearable=False, className="picker")),
                    ui.field("New setup", dcc.Input(id="setup-new-name", placeholder="home_visits")),
                    ui.button("Start it", "setup-new", small=True),
                ],
                className="toolbar",
            ),
            html.Div(id="setup-message"),
            html.Div(id="setup-form"),
            naming.card(config),
            ui.card(
                "What each recording holds",
                dcc.Loading(html.Div(holdings(config_path), id="setup-files")),
                desc="The file found in each device folder, by the setup each recording uses. A recording uses another "
                "setup when its own file, recordings/<id>.yaml, says setup: {name: ...}.",
            ),
        ]
    )


def rows_of(config, name: str) -> dict[str, list[dict]]:
    """The saved setup of that name as form rows, read as written so a file that no longer checks can be put right here."""
    path = Path(config.project.setups_dir) / f"{name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    data = data if isinstance(data, dict) else {}
    rows: dict[str, list[dict]] = {}
    for kind, uses in KINDS.items():
        entries = data.get(kind) if isinstance(data.get(kind), dict) else {}
        rows[kind] = [
            {"name": str(key), **{use: list((value or {}).get(use) or []) for use in uses}}
            for key, value in entries.items()
        ]
    return rows


def form(config, name: str, rows: dict[str, list[dict]]):
    roles = list(config.project.roles)
    saved = (Path(config.project.setups_dir) / f"{name}.yaml").is_file()
    cards = [
        ui.note(
            f"Setup {name} is not saved yet: nothing runs until it is." if not saved else f"Saved as {Path(config.project.setups_dir).name}/{name}.yaml.",
            "warn" if not saved else "ok",
        ),
        ui.card("People", html.P(", ".join(roles)), desc="The roles of the people in each recording, set under Settings."),
    ]
    for kind, uses in KINDS.items():
        listed = [
            [
                dcc.Input(value=row.get("name", ""), id={"type": NAME, "kind": kind, "index": index}, placeholder="folder name"),
                *[
                    dcc.Checklist(roles, row.get(use, []), id={"type": USE, "kind": kind, "index": index, "use": use}, inline=True)
                    for use in uses
                ],
                ui.button("Remove", {"type": REMOVE, "kind": kind, "index": index}, small=True),
            ]
            for index, row in enumerate(rows.get(kind, []))
        ]
        noun = "camera" if kind == "cameras" else "microphone"
        desc = (
            "One row per camera: the name of its folder in each recording, the people whose body, face and gaze are read "
            "from it, and whose voices its own sound carries (none leaves its sound to lining the files up)."
            if kind == "cameras" else
            "One row per microphone or sound recorder: the name of its folder and whose voices it carries. A device that "
            "carries one person's voice alone is their own microphone; the voices on one that carries several are separated."
        )
        cards.append(
            ui.card(
                kind.capitalize(),
                ui.table(["Folder", *[COLUMNS[use] for use in uses], ""], listed) if listed else html.P(f"No {noun} yet.", className="none"),
                desc=desc,
                actions=[ui.button(f"Add a {noun}", {"type": ADD, "kind": kind}, small=True)],
            )
        )
    cards.append(html.Div(ui.button("Save the setup", "setup-save", primary=True), className="buttons"))
    return cards


def rows_from(names: list, name_ids: list[dict], uses: list, use_ids: list[dict]) -> dict[str, list[dict]]:
    """The form as it stands, read back from its fields."""
    rows: dict[str, list[dict]] = {kind: [] for kind in KINDS}
    by_index: dict[tuple[str, int], dict] = {}
    for value, identity in zip(names, name_ids):
        row = {"name": (value or "").strip(), **{use: [] for use in KINDS[identity["kind"]]}}
        by_index[identity["kind"], identity["index"]] = row
        rows[identity["kind"]].append(row)
    for value, identity in zip(uses, use_ids):
        by_index[identity["kind"], identity["index"]][identity["use"]] = list(value or [])
    return rows


def checked_setup(config, name: str, rows: dict[str, list[dict]]) -> SetupSection:
    """The setup the form describes, refused with a reason a person can act on."""
    if not re.fullmatch(DEVICE_NAME, name or ""):
        raise MacawError(f"{name!r}: a setup's name takes letters, digits, - and _ only")
    named = [row["name"] for kind in KINDS for row in rows[kind]]
    if "" in named:
        raise MacawError("every device needs the name of its folder")
    twice = sorted({device for device in named if named.count(device) > 1})
    if twice:
        raise MacawError(f"two devices are called {', '.join(twice)}; each has a folder of its own")
    data = {
        "name": name,
        "cameras": {row["name"]: {use: row[use] for use in KINDS["cameras"]} for row in rows["cameras"]},
        "microphones": {row["name"]: {"hears": row["hears"]} for row in rows["microphones"]},
    }
    try:
        return MacawConfig.model_validate({**config.model_dump(), "setup": data}).setup
    except ValidationError as error:
        raise MacawError("; ".join(item["msg"].removeprefix("Value error, ") for item in error.errors())) from None


def save(config_path: Path, name: str, rows: dict[str, list[dict]]) -> Path:
    config = resolve_config(config_path)
    setup = checked_setup(config, name, rows)
    return write_setup(Path(config.project.setups_dir) / f"{name}.yaml", setup)


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

    @app.callback(
        Output("setup-form", "children"),
        Input("setup-pick", "value"),
        Input({"type": ADD, "kind": ALL}, "n_clicks"),
        Input({"type": REMOVE, "kind": ALL, "index": ALL}, "n_clicks"),
        State({"type": NAME, "kind": ALL, "index": ALL}, "value"),
        State({"type": NAME, "kind": ALL, "index": ALL}, "id"),
        State({"type": USE, "kind": ALL, "index": ALL, "use": ALL}, "value"),
        State({"type": USE, "kind": ALL, "index": ALL, "use": ALL}, "id"),
    )
    def edit(name, _added, _removed, names, name_ids, uses, use_ids):
        config = resolve_config(config_path)
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            return form(config, name, rows_of(config, name))
        # buttons drawn afresh report themselves without a click
        if not ctx.triggered[0]["value"]:
            return no_update
        rows = rows_from(names, name_ids, uses, use_ids)
        if trigger["type"] == ADD:
            rows[trigger["kind"]].append({"name": "", **{use: [] for use in KINDS[trigger["kind"]]}})
        else:
            del rows[trigger["kind"]][trigger["index"]]
        return form(config, name, rows)

    @app.callback(
        Output("setup-message", "children"),
        Output("setup-files", "children"),
        Output("setup-form", "children", allow_duplicate=True),
        Input("setup-save", "n_clicks"),
        State("setup-pick", "value"),
        State({"type": NAME, "kind": ALL, "index": ALL}, "value"),
        State({"type": NAME, "kind": ALL, "index": ALL}, "id"),
        State({"type": USE, "kind": ALL, "index": ALL, "use": ALL}, "value"),
        State({"type": USE, "kind": ALL, "index": ALL, "use": ALL}, "id"),
        prevent_initial_call=True,
    )
    def keep(clicks, name, names, name_ids, uses, use_ids):
        if not clicks:
            return no_update, no_update, no_update
        rows = rows_from(names, name_ids, uses, use_ids)
        try:
            path = save(config_path, name, rows)
        except MacawError as error:
            return ui.note(str(error), "bad"), no_update, no_update
        config = resolve_config(config_path)
        return ui.note(f"Saved {path.name}.", "ok"), holdings(config_path), form(config, name, rows_of(config, name))

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
        note = ui.note(
            f"Recordings filmed with setup {name} name it in their own file, recordings/<id>.yaml, as setup: {{name: {name}}}; "
            f"setup.name in project.yaml chooses the one every other recording uses."
        )
        return sorted({*setup_names(config), name}), name, note
