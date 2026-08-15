from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import scaffold_project_config
from macaw.logging_setup import get_logger
from macaw.project.layout import make_layout

app = typer.Typer()
logger = get_logger("init")

DERIVATIVES_DIRNAME = "derivatives"


@app.command("init")
def init_project(
    directory: Path = typer.Argument(Path("."), help="Where to create the project."),
    name: str = typer.Option(..., "--name", help="Project name."),
    raw_dir: Path = typer.Option(..., "--raw-dir", help="Directory holding the recordings."),
    n_subjects: int = typer.Option(2, "--n-subjects"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing project.yaml."),
) -> None:
    """Scaffold project.yaml and the output directory structure."""
    directory = directory.resolve()
    # BIDS puts processed output under derivatives/; keeping to it means the raw tree stays
    # untouched and a whole project can be moved as one directory
    output_dir = directory / DERIVATIVES_DIRNAME
    config_path = scaffold_project_config(
        directory / "project.yaml",
        force=force,
        name=name,
        raw_dir=raw_dir.resolve(),
        output_dir=output_dir,
        n_subjects=n_subjects,
    )
    make_layout(output_dir)
    (directory / "recordings").mkdir(parents=True, exist_ok=True)

    logger.info("scaffolded %s", config_path)
    typer.echo(str(config_path))
    typer.echo(str(output_dir))
