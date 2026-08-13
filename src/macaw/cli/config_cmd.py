from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Inspect and validate configuration.")


@app.command("show")
def show(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    section: str = typer.Option(None, "--section"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Print the fully merged configuration; what is actually in effect."""
    raise NotImplementedError


@app.command("validate")
def validate(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
) -> None:
    """Check the schema and reject unknown keys."""
    raise NotImplementedError


@app.command("diff")
def diff(
    a: Path = typer.Argument(..., help="First resolved config."),
    b: Path = typer.Argument(..., help="Second resolved config."),
) -> None:
    """Show which sections differ, and therefore what a rerun would recompute."""
    raise NotImplementedError
