from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("init")
def init_project(
    directory: Path = typer.Argument(Path("."), help="Where to create the project."),
    name: str = typer.Option(..., "--name", help="Project name."),
    raw_dir: Path = typer.Option(..., "--raw-dir", help="Directory holding the recordings."),
    n_subjects: int = typer.Option(2, "--n-subjects"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing project.yaml."),
) -> None:
    """Scaffold project.yaml and the output directory structure."""
    raise NotImplementedError
