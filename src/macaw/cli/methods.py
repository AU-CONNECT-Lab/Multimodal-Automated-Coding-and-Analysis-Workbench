from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.project.layout import layout_for
from macaw.execution.runner import resolve_recordings
from macaw.project.dictionary import NOTES_BY, merged_notes
from macaw.project.manifest import read_manifest, recording_specs
from macaw.qc.boilerplate import (
    drift_corrected_files, recordings_scaled, settings_by_recording, transcripts_corrected, write_methods,
)
from macaw.qc.look import project_summary

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
    scaled = recordings_scaled(config, resolve_config(config))
    looked = project_summary(resolve_config(config))
    corrected = transcripts_corrected(resolve_config(config))
    drifted = drift_corrected_files(resolve_config(config))
    ran = [rid for rid in resolve_recordings(resolve_config(config), None) if layout.manifest_fragment_path(rid, "x").parent.is_dir()]
    specs = merged_notes({rid: recording_specs(layout, rid) for rid in ran})
    own = settings_by_recording(specs)
    manifest = read_manifest(layout.manifest_path)
    # the project's manifest keeps one recording's note by entity; the merged one has every value's
    for name, spec in manifest.specs.items():
        merged = [theirs[name] for theirs in specs.values() if name in theirs and NOTES_BY in theirs[name].params]
        if merged and NOTES_BY in spec.params:
            spec.params = {**spec.params, NOTES_BY: merged[0].params[NOTES_BY]}
    text, bib = write_methods(
        manifest, resolve_config(settings), out or layout.output_dir / "methods", scaled=scaled, looked=looked,
        corrected=corrected, drifted=drifted, own=own,
    )
    typer.echo(f"methods {text}")
    typer.echo(f"references {bib}")
