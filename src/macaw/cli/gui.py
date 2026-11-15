from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config

app = typer.Typer()


@app.command("gui", rich_help_panel="Project")
def gui(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    browser: bool = typer.Option(False, "--browser", help="Open in the default browser."),
    server: bool = typer.Option(False, "--server", help="Listen on 0.0.0.0 for remote access."),
    host: str = typer.Option(None, "--host", help="Address to listen on. Default: gui.host, or 0.0.0.0 with --server."),
    port: int = typer.Option(None, "--port", help="Port to listen on. Default: gui.port."),
    debug: bool = typer.Option(False, "--debug", help="Turn on the page's developer tools, for development."),
) -> None:
    """Launch the workbench. Default is gui.mode from the config, a native window unless set otherwise."""
    from macaw.interface.run_gui import launch_browser, launch_server, launch_window

    settings = resolve_config(config).gui
    mode = "server" if server else "browser" if browser else settings.mode
    port = port or settings.port
    if mode == "server":
        typer.echo(f"listening on {host or '0.0.0.0'}:{port}; from your machine: ssh -L {port}:localhost:{port} <server>, then http://localhost:{port}")
        launch_server(config, host=host or "0.0.0.0", port=port)
    elif mode == "browser":
        launch_browser(config, host=host or settings.host, port=port, debug=debug)
    else:
        launch_window(config, port=port, debug=debug)
