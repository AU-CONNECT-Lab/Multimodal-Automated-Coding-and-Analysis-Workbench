from __future__ import annotations

from pathlib import Path

import typer

from macaw.exceptions import MacawError
from macaw.extractors import list_extractors
from macaw.plugin_template import scaffold_extractor

app = typer.Typer()


@app.command("new-extractor")
def new_extractor(
    name: str = typer.Argument(..., help="Namespaced extractor name, e.g. mylab.f0_spread."),
    directory: Path = typer.Option(Path("."), "--dir", help="Where to create the package."),
    group: str = typer.Option("custom", "--group", help="Your own modality label, kept in the manifest."),
) -> None:
    """Write a package holding a working extractor to edit, its entry point and a test."""
    if name in list_extractors():
        raise MacawError(f"An extractor named '{name}' is already installed")
    typer.echo(str(scaffold_extractor(name, directory, group=group)))
