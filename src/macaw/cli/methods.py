from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.project.manifest import read_manifest
from macaw.qc.boilerplate import write_methods

app = typer.Typer()


@app.command("methods")
def methods(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    out: Path = typer.Option(None, "--out", help="Directory to write into; output_dir/methods by default."),
) -> None:
    """Write a Methods section and its BibTeX for the steps this project has run."""
    layout = layout_for(resolve_config(config), create=False)
    if not layout.manifest_path.exists():
        raise MacawError(f"No feature manifest in {layout.output_dir}; run the pipeline first.")
    # the settings the features were made with, not whatever project.yaml says now
    settings = layout.resolved_config_path if layout.resolved_config_path.exists() else config
    text, bib = write_methods(read_manifest(layout.manifest_path), resolve_config(settings), out or layout.output_dir / "methods")
    typer.echo(f"methods {text}")
    typer.echo(f"references {bib}")
