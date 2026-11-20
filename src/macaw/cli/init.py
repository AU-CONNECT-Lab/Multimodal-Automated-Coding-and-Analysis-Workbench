from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import scaffold_project_config
from macaw.logging_setup import get_logger
from macaw.project.layout import make_layout

app = typer.Typer()
logger = get_logger("init")

DERIVATIVES_DIRNAME = "derivatives"


@app.command("init", rich_help_panel="Project")
def init_project(
    directory: Path = typer.Argument(Path("."), help="Where to create the project."),
    name: str = typer.Option(..., "--name", help="Project name."),
    raw_dir: Path = typer.Option(..., "--raw-dir", help="Directory holding the recordings."),
    work_dir: Path = typer.Option(
        None, "--work", "-w", help="Logs, job records and cache. Defaults to under the output dir."
    ),
    roles: str = typer.Option(
        "subject_1,subject_2", "--roles", help="What each person in a recording is called, one name per person, e.g. parent,child."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing project.yaml."),
) -> None:
    """Create a project: a project.yaml to fill in and the folders the results go to."""
    directory = directory.resolve()
    # BIDS puts processed output under derivatives/; keeping to it means the raw tree stays
    # untouched and a whole project can be moved as one directory
    output_dir = directory / DERIVATIVES_DIRNAME
    work_dir = work_dir.resolve() if work_dir is not None else None
    config_path = scaffold_project_config(
        directory / "project.yaml",
        force=force,
        name=name,
        raw_dir=raw_dir.resolve(),
        output_dir=output_dir,
        work_dir=work_dir,
        roles=[role.strip() for role in roles.split(",") if role.strip()],
    )
    make_layout(output_dir, work_dir)
    (directory / "recordings").mkdir(parents=True, exist_ok=True)

    logger.info("scaffolded %s", config_path)
    typer.echo(str(config_path))
    typer.echo(str(output_dir))
