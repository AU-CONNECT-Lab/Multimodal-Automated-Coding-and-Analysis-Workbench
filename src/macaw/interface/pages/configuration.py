"""The project's settings as they resolve, checked; editing stays in project.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from dash import Input, Output, html

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.interface import ui

LABEL = "Configuration"


def layout(config_path: Path):
    return html.Div(
        [
            html.Div(html.Button("Check again", id="config-check", className="button"), className="toolbar"),
            html.Div(id="config-body"),
        ]
    )


def body(config_path: Path):
    try:
        config = resolve_config(config_path)
    except (MacawError, ValueError) as error:
        return ui.section("project.yaml", ui.note(str(error), "bad"), desc=str(config_path))
    resolved = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return [
        ui.section(
            "project.yaml",
            ui.note("The settings are valid.", "ok"),
            html.P(["Edit ", html.Code(str(config_path)), " in any text editor, then check again. ",
                    html.Code("macaw config diff"), " says which steps a change makes run again."]),
            desc="Every setting as it resolves: the file over the package defaults.",
        ),
        html.Pre(resolved, className="config"),
    ]


def register(app, config_path: Path) -> None:
    @app.callback(Output("config-body", "children"), Input("config-check", "n_clicks"))
    def check(_):
        return body(config_path)
