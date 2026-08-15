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
from macaw.config.loader import resolve_config
from macaw.project.recording import detect_recordings, parse_recordings_input

app = typer.Typer()


def _select_recordings(config, recordings: str | None) -> list[str]:
    ids = (
        parse_recordings_input(recordings)
        if recordings
        else detect_recordings(
            config.project.raw_dir,
            config.project.recording_pattern,
            config.project.recording_prefix,
        )
    )
    excluded = set(config.project.excluded_recordings)
    return [recording_id for recording_id in ids if recording_id not in excluded]


@app.command("code")
def code(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    layer: list[str] = typer.Option(None, "--layer", "-l", help="A, B, or C."),
    codebook: Path = typer.Option(None, "--codebook", help="Required for layer B."),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Apply rule-based (A) and LLM (B) coding. Low-confidence items go to the review queue."""
    raise NotImplementedError


@app.command("review")
def review(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    export_to: Path = typer.Option(None, "--export", help="Write the queue to a file."),
    apply_from: Path = typer.Option(None, "--apply", help="Merge back reviewed decisions."),
) -> None:
    """Export or re-import the human review queue."""
    raise NotImplementedError


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
    annotations = load_annotations(resolved, _select_recordings(resolved, recordings))
    report = agreement_report(annotations, tier=tier, layer=layer, grid_s=grid_s)
    typer.echo(format_report(report))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_to_json(report), indent=2), encoding="utf-8")


@app.command("refine-codebook")
def refine_codebook(
    codebook: Path = typer.Option(..., "--codebook"),
    out: Path = typer.Option(..., "--out"),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    examples: int = typer.Option(50, "--examples", help="Utterances sampled as context."),
) -> None:
    """Turn a vague codebook into operational criteria and boundary cases. Output is advisory."""
    raise NotImplementedError
