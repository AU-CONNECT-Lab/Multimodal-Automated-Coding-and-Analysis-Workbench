from __future__ import annotations

from pathlib import Path

import typer

from macaw.backend_template import scaffold_backend
from macaw.exceptions import MacawError
from macaw.extractors import list_backend_groups, list_backends, list_extractors
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


@app.command("new-backend")
def new_backend(
    name: str = typer.Argument(..., help="Namespaced backend name, e.g. mylab.openpose_csv."),
    directory: Path = typer.Option(Path("."), "--dir", help="Where to create the package."),
    kind: str = typer.Option("pose", "--kind", help="pose, asr or prosody."),
) -> None:
    """Write a package holding a working backend to edit, its entry point and a test."""
    if any(name in list_backends(group) for group in list_backend_groups()):
        raise MacawError(f"A backend named '{name}' is already installed")
    typer.echo(str(scaffold_backend(name, directory, kind=kind)))
