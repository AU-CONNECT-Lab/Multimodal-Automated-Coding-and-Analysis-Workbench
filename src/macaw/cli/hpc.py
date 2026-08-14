from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Generate cluster job scripts. Submission stays manual.")


@app.command("generate")
def generate(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
    extractor: list[str] = typer.Option(None, "--extractor", "-e"),
    hpc_config: Path = typer.Option(None, "--hpc-config", help="Defaults to $MACAW_HPC_CONFIG, then ./hpc.yaml, then the packaged file."),
    out: Path = typer.Option(Path("submit_macaw.sh"), "--out"),
) -> None:
    """Write a chained job-array script. Run it yourself with sbatch.

    Stops at any step needing a human and prints which command to run locally before
    submitting the next segment. Cloud LLM providers are excluded with a warning, since
    compute nodes usually have no outbound network.
    """
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
    extractor: list[str] = typer.Option(None, "--extractor", "-e"),
    verify_only: bool = typer.Option(False, "--verify"),
) -> None:
    """Download model weights on the login node before submitting anything."""
    raise NotImplementedError


@app.command("preflight")
def preflight(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
) -> None:
    """Check resource requests, model availability and interactive gates before submitting."""
    raise NotImplementedError
