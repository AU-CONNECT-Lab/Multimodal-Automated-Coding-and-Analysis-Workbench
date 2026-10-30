from __future__ import annotations

import json
from pathlib import Path

import typer

from macaw.coding.agreement import (
    agreement_report,
    format_report,
    load_annotations,
    report_to_json,
)
from macaw.coding.describe import describe_codes, format_description
from macaw.coding.review_queue import apply_review, export_review, queue_summary, read_queue, write_queue
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution import runner
from macaw.execution.runner import resolve_recordings

app = typer.Typer()


_LAYER_STEPS = {"A": "coding.layer_a", "B": "coding.judge"}


@app.command("code")
def code(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    layer: list[str] = typer.Option(None, "--layer", "-l", help="A or B; defaults to coding.layers."),
    codebook: Path = typer.Option(None, "--codebook", help="Overrides coding.codebook; required for layer B."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Apply rule-based (A) and LLM (B) coding. Low-confidence items go to the review queue."""
    resolved = resolve_config(config, cli_overrides=set_ or ())
    if codebook is not None:
        resolved.coding.codebook = codebook
    layers = [name.upper() for name in (layer or resolved.coding.layers)]
    unknown = [name for name in layers if name not in _LAYER_STEPS]
    if unknown:
        raise MacawError(f"Layer(s) {', '.join(unknown)} cannot be coded; available: {', '.join(_LAYER_STEPS)}")
    summary = runner.run(
        resolved,
        recordings=resolve_recordings(resolved, recordings),
        extractors=[_LAYER_STEPS[name] for name in layers],
    )
    typer.echo(f"tasks {summary.n_tasks}  done {summary.n_done}  cached {summary.n_cached}  failed {summary.n_failed}")
    if "B" in layers:
        pending = len(export_review(resolved, resolve_recordings(resolved, recordings)))
        typer.echo(f"{pending} unit(s) wait for review: macaw review --export queue.csv")
    if summary.n_failed:
        raise typer.Exit(code=1)


@app.command("review")
def review(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    export_to: Path = typer.Option(None, "--export", help="Write the queue to a file (.csv to edit by hand)."),
    apply_from: Path = typer.Option(None, "--apply", help="Merge back reviewed decisions."),
) -> None:
    """Export or re-import the human review queue. Fill in decision (a code) and reviewer, then --apply."""
    if (export_to is None) == (apply_from is None):
        raise MacawError("Give one of --export <file> or --apply <file>.")
    resolved = resolve_config(config)
    if export_to is not None:
        queue = export_review(resolved, resolve_recordings(resolved, recordings))
        write_queue(queue, export_to)
        summary = queue_summary(queue)
        reasons = ", ".join(f"{count} {reason}" for reason, count in summary["by_reason"].items())
        typer.echo(f"{summary['n_items']} unit(s) to review ({reasons or 'none'}) written to {export_to}")
        return
    counts = apply_review(resolved, read_queue(apply_from))
    if not counts:
        typer.echo("No decisions filled in; nothing applied.")
    for recording_id, count in counts.items():
        typer.echo(f"{recording_id}: {count}")


@app.command("agreement")
def agreement(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    tier: str = typer.Option(None, "--tier", help="Restrict to one annotation tier."),
    layer: str = typer.Option(None, "--layer", "-l", help="A, B, or C."),
    grid_s: float = typer.Option(None, "--grid-s", help="Compare on a fixed time grid instead of on exact intervals."),
    out: Path = typer.Option(None, "--out", help="Write the report as json."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Inter-annotator agreement over the annotations table: Krippendorff alpha and Cohen kappa."""
    resolved = resolve_config(config, cli_overrides=set_ or ())
    annotations = load_annotations(resolved, resolve_recordings(resolved, recordings))
    report = agreement_report(annotations, tier=tier, layer=layer, grid_s=grid_s)
    typer.echo(format_report(report))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_to_json(report), indent=2), encoding="utf-8")


@app.command("describe-codes")
def describe_codes_command(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    tier: str = typer.Option(None, "--tier", help="Restrict to one annotation tier."),
    layer: str = typer.Option(None, "--layer", "-l", help="A, B, or C."),
    out: Path = typer.Option(None, "--out", help="Write the table as csv."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """How each code was used in each recording: units, time, runs of one code, and how many runs last one unit."""
    resolved = resolve_config(config, cli_overrides=set_ or ())
    annotations = load_annotations(resolved, resolve_recordings(resolved, recordings))
    if tier is not None:
        annotations = annotations[annotations["tier"] == tier]
    if layer is not None:
        annotations = annotations[annotations["layer"] == layer]
    table = describe_codes(annotations)
    typer.echo(format_description(table))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
