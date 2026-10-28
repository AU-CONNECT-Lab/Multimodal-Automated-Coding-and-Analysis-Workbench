"""Web app. Batch work goes through cli_bridge; no pipeline logic lives in this package."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, dcc, html
from flask import abort, send_file

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interface import ui
from macaw.interface.files import PREFIX, path_from_url
from macaw.interface.pages import results, review, run, settings

PAGES = (run, review, results, settings)
# the tabs along the top; settings is reached by the gear
TABS = (run, review, results)


def create_app(config_path: Path) -> Dash:
    config_path = Path(config_path).resolve()
    app = Dash(
        __name__,
        title="macaw",
        assets_folder=str(Path(__file__).parent / "assets"),
        suppress_callback_exceptions=True,
    )
    # a function, so reloading the page reads the project afresh
    app.layout = lambda: build_layout(config_path)
    for page in PAGES:
        page.register(app, config_path)
    _register_navigation(app, config_path)

    @app.server.route(PREFIX + "<path:rest>")
    def serve(rest):
        path = path_from_url(rest, resolve_config(config_path))
        if path is None:
            abort(404)
        return send_file(path)

    return app


def _header(title: str, subtitle: str):
    return html.Header(
        [
            html.Div(
                [
                    html.Span(title, className="title"),
                    html.Span(subtitle, className="subtitle"),
                    html.Div(
                        [
                            ui.button("Refresh", "app-refresh", icon_name="refresh"),
                            dcc.Link(ui.icon("settings"), href=settings.PATH, id="nav-settings", className="gear", title="Settings"),
                        ],
                        className="actions",
                    ),
                ],
                className="bar",
            ),
            html.Nav(
                [
                    dcc.Link([page.LABEL, *([html.Span(id="review-badge", className="badge")] if page is review else [])],
                             href=f"/{page.LABEL.lower()}", id=f"nav-{page.LABEL.lower()}")
                    for page in TABS
                ],
                className="tabs",
            ),
        ],
        id="topbar",
    )


def build_layout(config_path: Path):
    try:
        config = resolve_config(config_path)
    except (MacawError, ValueError) as error:
        # only the settings can be shown until the file reads again
        return html.Div(
            [
                dcc.Location(id="url"),
                _header(config_path.parent.name, str(config_path)),
                html.Main([ui.note(str(error), "bad"), html.Div(settings.layout(config_path), id="page-settings", className="page shown")]),
            ]
        )
    count = len(resolve_recordings(config, None))
    return html.Div(
        [
            dcc.Location(id="url"),
            _header(config.project.name, f"{count} recording{'s' if count != 1 else ''} · {config_path}"),
            html.Main([html.Div(page.layout(config_path), id=f"page-{page.LABEL.lower()}", className="page") for page in PAGES]),
        ]
    )


def page_for(pathname: str | None) -> str:
    """/review opens the Review page, so a page can be bookmarked or linked to; anything else opens Run."""
    wanted = (pathname or "/").strip("/").lower()
    names = [page.LABEL.lower() for page in PAGES]
    return wanted if wanted in names else names[0]


def waiting_count(config_path: Path) -> int:
    """Gates waiting for a person, plus coded units waiting for one, across the project."""
    from macaw.coding.review_queue import export_review
    from macaw.project.recording import build_recording
    from macaw.qc.report import _waiting

    config = resolve_config(config_path)
    recordings = resolve_recordings(config, None)
    count = 0
    for rid in recordings:
        try:
            count += len(_waiting(config, build_recording(rid, config)))
        except MacawError:
            continue
    try:
        count += len(export_review(config, recordings))
    except Exception:   # no codebook or nothing coded: no units are waiting
        pass
    return count


def _register_navigation(app: Dash, config_path: Path) -> None:
    names = [page.LABEL.lower() for page in PAGES]

    @app.callback(
        [Output(f"page-{name}", "className") for name in names]
        + [Output(f"nav-{name}", "className") for name in names],
        Input("url", "pathname"),
    )
    def show(pathname):
        shown = page_for(pathname)
        pages = ["page shown" if name == shown else "page" for name in names]
        links = [("gear active" if shown == name else "gear") if name == "settings" else ("active" if shown == name else "") for name in names]
        return pages + links

    @app.callback(Output("review-badge", "children"), Input("url", "pathname"))
    def badge(_):
        try:
            count = waiting_count(config_path)
        except Exception:   # a badge that cannot be counted is left off rather than wrong
            return ""
        return str(count) if count else ""

    app.clientside_callback(
        "function (clicks) { if (clicks) { window.location.reload(); } return window.dash_clientside.no_update; }",
        Output("app-refresh", "title"),
        Input("app-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
