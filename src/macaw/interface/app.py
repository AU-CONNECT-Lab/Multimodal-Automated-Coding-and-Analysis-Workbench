"""Web app. Batch work goes through cli_bridge; no pipeline logic lives in this package."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, dcc, html
from flask import abort, send_file

from macaw.config.loader import resolve_config
from macaw.interface.files import PREFIX, path_from_url
from macaw.interface.pages import configuration, results, review, run

PAGES = (run, review, results, configuration)


def create_app(config_path: Path) -> Dash:
    config_path = Path(config_path).resolve()
    app = Dash(
        __name__,
        title="macaw",
        assets_folder=str(Path(__file__).parent / "assets"),
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout(config_path)
    for page in PAGES:
        page.register(app, config_path)
    _register_address(app)

    @app.server.route(PREFIX + "<path:rest>")
    def serve(rest):
        path = path_from_url(rest, resolve_config(config_path))
        if path is None:
            abort(404)
        return send_file(path)

    return app


def build_layout(config_path: Path):
    name = resolve_config(config_path).project.name
    return html.Div(
        [
            dcc.Location(id="url"),
            html.Header([html.Span("macaw", className="brand"), html.Span(name, className="chip")], id="topbar"),
            html.Main(
                dcc.Tabs(
                    [
                        dcc.Tab(page.layout(config_path), label=page.LABEL, value=page.LABEL.lower(), className="tab", selected_className="tab-selected")
                        for page in PAGES
                    ],
                    id="tabs",
                    value=PAGES[0].LABEL.lower(),
                    className="tabs",
                )
            ),
        ]
    )


def _register_address(app: Dash) -> None:
    """/review opens the Review tab, so a page can be bookmarked or linked to."""
    names = [page.LABEL.lower() for page in PAGES]

    @app.callback(Output("tabs", "value"), Input("url", "pathname"))
    def open_tab(pathname):
        wanted = (pathname or "/").strip("/").lower()
        return wanted if wanted in names else names[0]
