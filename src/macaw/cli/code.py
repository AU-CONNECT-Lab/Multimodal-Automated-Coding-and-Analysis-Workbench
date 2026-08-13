from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("code")
def code(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
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


@app.command("refine-codebook")
def refine_codebook(
    codebook: Path = typer.Option(..., "--codebook"),
    out: Path = typer.Option(..., "--out"),
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    examples: int = typer.Option(50, "--examples", help="Utterances sampled as context."),
) -> None:
    """Turn a vague codebook into operational criteria and boundary cases. Output is advisory."""
    raise NotImplementedError
