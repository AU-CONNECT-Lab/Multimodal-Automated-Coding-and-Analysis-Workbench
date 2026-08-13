from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Generate cluster job scripts. Submission stays manual.")


@app.command("generate")
def generate(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    session: list[str] = typer.Option(None, "--session", "-s"),
    extractor: list[str] = typer.Option(None, "--extractor", "-e"),
    scheduler: str = typer.Option("slurm", "--scheduler", help="slurm | pbs"),
    out: Path = typer.Option(Path("submit_macaw.sh"), "--out"),
) -> None:
    """Write a job-array script. Run it yourself with sbatch or qsub."""
    raise NotImplementedError


@app.command("rerun-script")
def rerun_script(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    run_id: str = typer.Option(..., "--run-id"),
    out: Path = typer.Option(Path("rerun_macaw.sh"), "--out"),
) -> None:
    """Generate a script covering only the failed array indices."""
    raise NotImplementedError


@app.command("prefetch-models")
def prefetch_models(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    dest: Path = typer.Option(..., "--dest", help="Shared directory to point HF_HOME at."),
) -> None:
    """Download model weights on the login node -- compute nodes often have no internet."""
    raise NotImplementedError
