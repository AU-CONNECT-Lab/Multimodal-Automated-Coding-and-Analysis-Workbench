"""The project's settings: the common ones as a form, every one as the text of project.yaml.
Nothing is written before the new settings pass the same check a run makes."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.interface import project_file, ui

LABEL = "Settings"
PATH = "/settings"
SETTING = "setting"
WRAP = "setting-wrap"
# variable, what it is for, program looked for on PATH when it is not set, whether its value is secret
ENVIRONMENT = (
    ("MACAW_FFMPEG", "ffmpeg, which reads and cuts the media", "ffmpeg", False),
    ("MACAW_OPENFACE", "OpenFace's folder, for face.backend openface", "FaceLandmarkVidMulti", False),
    ("MACAW_YOLO", "the yolo program, for pose.backend yolo", "yolo", False),
    ("MACAW_FACE_MODEL", "face_landmarker.task, for face.backend mediapipe", None, False),
    ("MACAW_POSE_MODEL", "pose_landmarker task file, for pose.backend mediapipe", None, False),
    ("OPENAI_API_KEY", "for coding.llm.provider openai", None, True),
    ("ANTHROPIC_API_KEY", "for coding.llm.provider anthropic", None, True),
)
THEMES = [
    {"label": "Light", "value": "light"},
    {"label": "Dark", "value": "dark"},
    {"label": "Same as the system", "value": "system"},
]


def _control(dotted: str, value):
    kind = project_file.control(dotted)
    identity = {"type": SETTING, "key": dotted}
    shown = project_file.to_form(dotted, value)
    if kind["kind"] in ("choice", "multi"):
        options = list(kind["options"])
        # a value the file names but nothing registers still shows, so the check can say what is wrong with it
        for item in shown if isinstance(shown, list) else [shown]:
            if item is not None and item not in options:
                options.append(item)
        return dcc.Dropdown(options, shown, id=identity, multi=kind["kind"] == "multi", clearable=kind["kind"] == "multi" or kind.get("optional", False))
    if kind["kind"] == "number":
        return dcc.Input(id=identity, type="number", value=shown, step=1 if kind["integer"] else "any", debounce=True)
    return dcc.Input(id=identity, type="text", value=shown, debounce=True)


def _hint(dotted: str, hints: dict[str, str]):
    return [html.Code(dotted), " ", hints.get(dotted, "")]


def _form(config_path: Path):
    try:
        values = project_file.file_values(config_path)
    except (MacawError, ValueError):   # a file that does not read is mended as text below
        return []
    hints = project_file.hints()
    return [
        ui.card(
            title,
            html.Div(
                [
                    html.Div(
                        ui.field(project_file.LABELS[dotted], _control(dotted, values[dotted]), _hint(dotted, hints)),
                        id={"type": WRAP, "key": dotted},
                    )
                    for dotted in keys
                ],
                className="field-grid",
            ),
            desc=desc,
        )
        for title, desc, keys in project_file.FORM
    ]


def _environment():
    items = [
        html.Li([ui.pill("in use", "ok"), html.Span(["Python", html.Span(sys.executable, className="value"),
                 html.Span("Every button runs macaw with this interpreter.", className="value")], className="what")])
    ]
    for variable, purpose, program, secret in ENVIRONMENT:
        value = os.environ.get(variable)
        found = shutil.which(program) if program and not value else None
        state = ui.pill("set", "ok") if value else ui.pill("on PATH", "ok") if found else ui.pill("not set")
        shown = ("(hidden)" if secret else value) if value else found or ""
        items.append(html.Li([state, html.Span([html.Code(variable), html.Span(purpose, className="value"),
                                               *([html.Span(shown, className="value")] if shown else [])], className="what")]))
    return html.Ul(items, className="status-list")


def layout(config_path: Path):
    return html.Div(
        [
            ui.page_head("Settings", ["Saved in ", html.Code(str(config_path))]),
            html.Div(
                [
                    html.Div(
                        [
                            *_form(config_path),
                            ui.card(
                                "project.yaml",
                                dcc.Textarea(id="settings-yaml", value=config_path.read_text(encoding="utf-8"), className="yaml", spellCheck=False),
                                html.Div(ui.button("Save and check", "settings-yaml-save", icon_name="check", primary=True), className="buttons"),
                                desc="Every setting, as the file has it. A key left out takes the package default; macaw config show lists them all.",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            ui.card(
                                "Only for you",
                                dcc.RadioItems(THEMES, "system", id="theme-choice", className="segmented", inline=True, persistence=True, persistence_type="local"),
                                dcc.Store(id="theme-applied"),
                                desc="Kept in this browser; nothing in the project changes.",
                            ),
                            ui.card("This computer", _environment(), desc="Set these in the environment the GUI is started from."),
                            html.Div(
                                ui.card(
                                    "Changes",
                                    html.Div(id="settings-changes"),
                                    html.Div(
                                        [
                                            ui.button("Discard", "settings-discard", icon_name="undo"),
                                            ui.button("Save and check", "settings-save", icon_name="check", primary=True),
                                        ],
                                        className="buttons",
                                    ),
                                    html.Div(id="settings-message"),
                                ),
                                className="sticky",
                            ),
                        ]
                    ),
                ],
                className="columns",
            ),
        ]
    )


def _form_values(values, ids) -> dict:
    return {item["key"]: value for item, value in zip(ids, values)}


def _describe(value) -> str:
    if value in (None, "", []):
        return "(empty)"
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value)


def _runs_again_note(config_path: Path, text: str):
    """What the new text would do: an error, or the steps whose finished results it makes out of date."""
    try:
        after = project_file.check(config_path, text)
    except (MacawError, ValueError) as error:
        return None, ui.note(str(error), "bad")
    try:
        names = project_file.runs_again(resolve_config(config_path), after)
    except (MacawError, ValueError):
        return after, ui.note("The settings are valid. What the file had before could not be read, so which results change is unknown.", "warn")
    if not names:
        return after, ui.note("The settings are valid, and no finished result goes out of date.", "ok")
    return after, ui.note(f"The settings are valid. Out of date afterwards, so the next run makes them again: {', '.join(names)}.", "warn")


def register(app, config_path: Path) -> None:
    app.clientside_callback(
        """function (theme) {
            const root = document.documentElement;
            if (theme === "light" || theme === "dark") { root.dataset.theme = theme; } else { delete root.dataset.theme; }
            return theme;
        }""",
        Output("theme-applied", "data"),
        Input("theme-choice", "value"),
    )

    @app.callback(
        Output("settings-changes", "children"),
        Output({"type": WRAP, "key": ALL}, "className"),
        Input({"type": SETTING, "key": ALL}, "value"),
        State({"type": SETTING, "key": ALL}, "id"),
    )
    def preview(values, ids):
        found = project_file.changes(config_path, _form_values(values, ids))
        marks = ["changed" if item["key"] in found else "" for item in ids]
        if not found:
            return html.P("No unsaved changes.", className="none"), marks
        items = html.Ul(
            [
                html.Li([project_file.LABELS[dotted], html.Span(dotted, className="key"), f"{_describe(old)} → {_describe(new)}"])
                for dotted, (old, new) in found.items()
            ],
            className="changes",
        )
        text = project_file.with_values(config_path.read_text(encoding="utf-8"), {dotted: new for dotted, (_, new) in found.items()})
        return [items, _runs_again_note(config_path, text)[1]], marks

    @app.callback(
        Output({"type": SETTING, "key": ALL}, "value"),
        Output("settings-yaml", "value"),
        Output("settings-message", "children"),
        Input("settings-save", "n_clicks"), Input("settings-discard", "n_clicks"), Input("settings-yaml-save", "n_clicks"),
        State({"type": SETTING, "key": ALL}, "value"), State({"type": SETTING, "key": ALL}, "id"), State("settings-yaml", "value"),
        prevent_initial_call=True,
    )
    def save(_save, _discard, _save_text, values, ids, text):
        message = None
        if ctx.triggered_id == "settings-save":
            found = project_file.changes(config_path, _form_values(values, ids))
            if not found:
                return [no_update] * len(ids), no_update, ui.note("Nothing to save.")
            text = project_file.with_values(config_path.read_text(encoding="utf-8"), {dotted: new for dotted, (_, new) in found.items()})
        if ctx.triggered_id in ("settings-save", "settings-yaml-save"):
            after, message = _runs_again_note(config_path, text)
            if after is None:
                return [no_update] * len(ids), no_update, message
            project_file.save(config_path, text)
            message = html.Div([ui.note(f"Saved to {config_path.name}.", "ok"), message])
        current = project_file.file_values(config_path)
        return [project_file.to_form(item["key"], current[item["key"]]) for item in ids], config_path.read_text(encoding="utf-8"), message
