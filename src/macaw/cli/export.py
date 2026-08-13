from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer()


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    sessions: str = typer.Option(None, "--sessions", "-s", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option("csv", "--format", "-f", help="csv | parquet | eaf"),
    scale: str = typer.Option(None, "--scale", help="frame | event | window | session"),
    wide: bool = typer.Option(True, "--wide/--long"),
    out: Path = typer.Option(None, "--out"),
    rmd: bool = typer.Option(False, "--rmd", help="Also write an .Rmd starter script."),
) -> None:
    """Export features for downstream analysis, or annotations as .eaf."""
    raise NotImplementedError


@app.command("import-eaf")
def import_eaf(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    session: str = typer.Option(..., "--session", "-s"),
    path: Path = typer.Argument(..., help="The .eaf file to import."),
    replace: bool = typer.Option(False, "--replace", help="Replace existing tiers instead of merging."),
) -> None:
    """Bring externally edited annotations back into the project."""
    raise NotImplementedError
