from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("gui")
def gui(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    browser: bool = typer.Option(False, "--browser", help="Open in the default browser."),
    server: bool = typer.Option(False, "--server", help="Listen on 0.0.0.0 for remote access."),
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Launch the workbench. Default is a native window."""
    raise NotImplementedError
