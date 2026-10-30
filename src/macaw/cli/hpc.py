from __future__ import annotations

import os
from pathlib import Path

import typer

from macaw.config.hpc import load_hpc_config
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution import runner, slurm
from macaw.execution.array_map import read_map, recording_for_index
from macaw.execution.job_db import JobDB
from macaw.project.layout import layout_for

app = typer.Typer(help="Run on a Slurm cluster: job arrays you submit with sbatch, and model weights fetched beforehand.")


@app.command("generate")
def generate(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    extractor: list[str] = typer.Option(None, "--extractor", "-e", help="Extractor or group name, repeatable. Defaults to pipeline.run."),
    hpc_config: Path = typer.Option(None, "--hpc-config", help="Defaults to $MACAW_HPC_CONFIG, then hpc.yaml beside project.yaml, then the packaged file."),
) -> None:
    """Write job arrays for the steps still to run, one element per recording, and say how to submit them.
    Run it again after a person has cleared what a run stopped at, or after failures: finished steps are left out."""
    resolved = resolve_config(config)
    hpc = load_hpc_config(hpc_config, config)
    jobs = slurm.generate(resolved, hpc, runner.resolve_recordings(resolved, recordings), list(extractor or ()))
    for note in jobs.notes:
        typer.echo(f"note: {note}")
    if not jobs.segments:
        typer.echo("nothing left to run")
        return
    for number, segment in enumerate(jobs.segments, start=1):
        typer.echo(f"{number}  {'gpu' if segment.gpu else 'cpu'}  {segment.profile}  {', '.join(segment.extractors)}")
    typer.echo(f"{len(jobs.recording_ids)} recording(s), one array element each")
    typer.echo(f"submit from a login node: bash {jobs.directory.as_posix()}/submit.sh")


@app.command("prefetch-models")
def prefetch_models(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    extractor: list[str] = typer.Option(None, "--extractor", "-e", help="Extractor or group name, repeatable. Defaults to pipeline.run."),
    hpc_config: Path = typer.Option(None, "--hpc-config", help="Where hf_home is set; found as for generate."),
) -> None:
    """Download the model weights the steps would otherwise fetch while running, into hpc.yaml's hf_home.
    Run it where there is internet and that folder is the one the compute nodes see, e.g. a login node;
    the jobs then run offline."""
    from macaw.execution.dag import build_dag
    from macaw.execution.models import fetch_models

    hpc = load_hpc_config(hpc_config, config)
    if hpc.hf_home is not None:
        os.environ["HF_HOME"] = str(hpc.hf_home)
    resolved = resolve_config(config)
    dag = build_dag(resolved, list(extractor or ()) or None)
    failed = fetch_models(
        config, dag.topological_order(), runner.resolve_recordings(resolved, None), strict=True, echo=typer.echo
    )
    if slurm.JUDGE in dag.nodes and resolved.coding.llm.is_local():
        typer.echo(f"{slurm.JUDGE}: pull {resolved.coding.llm.model} into the model server's own folder yourself, e.g. ollama pull")
    if failed:
        raise typer.Exit(code=1)


@app.command("task", hidden=True)
def task(
    config: Path = typer.Option(..., "--config", "-c"),
    map_path: Path = typer.Option(..., "--map"),
    execution_id: str = typer.Option(..., "--execution-id"),
    extractor: list[str] = typer.Option(..., "--extractor", "-e"),
) -> None:
    """One array element: the recording at $SLURM_ARRAY_TASK_ID, through these steps."""
    index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if index is None:
        raise MacawError("SLURM_ARRAY_TASK_ID is not set: this runs inside a job array written by `macaw hpc generate`")
    recording = recording_for_index(read_map(map_path), int(index))
    resolved = resolve_config(config, recording_id=recording)
    summary = runner.run_recording(resolved, config, recording, list(extractor), execution_id)
    typer.echo(f"{recording}: done {summary.n_done}  cached {summary.n_cached}  failed {summary.n_failed}")
    for wait in summary.waiting:
        typer.echo(f"{recording} waits at {wait.gate}: {wait.message}")
    if summary.n_failed:
        raise typer.Exit(code=1)


@app.command("finish", hidden=True)
def finish(
    config: Path = typer.Option(..., "--config", "-c"),
    map_path: Path = typer.Option(..., "--map"),
    execution_id: str = typer.Option(..., "--execution-id"),
    extractor: list[str] = typer.Option(..., "--extractor", "-e"),
) -> None:
    """The closing job: merges the feature manifest and lists what failed and who waits for a person."""
    resolved = resolve_config(config)
    summary = runner.finish(resolved, sorted(read_map(map_path)), list(extractor), execution_id)
    typer.echo(f"execution_id {execution_id}")
    typer.echo(f"tasks {summary.n_tasks}  done {summary.n_done}  cached {summary.n_cached}  failed {summary.n_failed}")
    for record in JobDB(layout_for(resolved, create=False).job_db_path).failed(execution_id):
        typer.echo(f"failed {record.task_id}: {record.message or ''}  log: {record.log_path or ''}")
    if summary.waiting:
        typer.echo(f"waiting for a person ({len({wait.recording_id for wait in summary.waiting})} recording(s)):")
        for wait in summary.waiting:
            typer.echo(f"  {wait.recording_id}  {wait.gate}: {wait.message}")
    if summary.n_failed or summary.waiting:
        typer.echo("clear those, or fix what failed, then `macaw hpc generate` again; finished steps are left out")
