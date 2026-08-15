from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from macaw.coding.codebook import read_codebook
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interop.eaf import export_eaf
from macaw.logging_setup import get_logger
from macaw.project.layout import make_layout
from macaw.project.recording import build_recording
from macaw.store.annotations import read_annotations
from macaw.store.longtable import concat_long_tables, read_long_table
from macaw.store.schema import Scale
from macaw.store.sequence import export_state_sequence, to_state_sequence
from macaw.store.wide import export_csv, export_rmd_template, to_wide

app = typer.Typer()
logger = get_logger("export")

FORMATS = ("csv", "parquet", "eaf", "sequence")


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option("csv", "--format", "-f", help="csv | parquet | eaf | sequence"),
    scale: str = typer.Option(None, "--scale", help="frame | event | window | recording"),
    wide: bool = typer.Option(True, "--wide/--long"),
    step: float = typer.Option(1.0, "--step", help="Sequence format only: sampling step in seconds."),
    out: Path = typer.Option(None, "--out"),
    rmd: bool = typer.Option(False, "--rmd", help="Also write an .Rmd starter script."),
) -> None:
    """Export features for downstream analysis, annotations as .eaf, or a nominal state sequence."""
    if format_ not in FORMATS:
        raise MacawError(f"Unknown format '{format_}'. Choose one of: {', '.join(FORMATS)}")
    if scale is not None and scale not in {member.value for member in Scale}:
        known = ", ".join(member.value for member in Scale)
        raise MacawError(f"Unknown scale '{scale}'. Choose one of: {known}")

    resolved = resolve_config(config)
    layout = make_layout(resolved.project.output_dir, create=False)
    recording_ids = resolve_recordings(resolved, recordings)

    if format_ == "eaf":
        _export_eaf(resolved, layout, recording_ids, out)
        return
    if format_ == "sequence":
        _export_sequence(layout, recording_ids, step, out)
        return

    frames = []
    for recording_id in recording_ids:
        path = layout.long_table_path(recording_id)
        if not path.exists():
            logger.info("no features yet for %s", recording_id)
            continue
        frames.append(read_long_table(path, scale=scale))
    table = concat_long_tables(frames)
    if table.empty:
        raise MacawError(
            f"No feature rows under {layout.recordings_dir}. Run `macaw run` before exporting."
        )

    out = out or (layout.output_dir / "exports" / f"features.{format_}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if format_ == "csv":
        export_csv(table, out, wide=wide)
    else:
        (to_wide(table) if wide else table).to_parquet(out, index=False)

    logger.info("%d rows from %d recordings", len(table), len(frames))
    typer.echo(str(out))
    if rmd:
        typer.echo(str(export_rmd_template(out.parent, out.name, layout.manifest_path.name)))


def _annotation_frames(layout, recording_ids: list[str]) -> list[tuple[str, pd.DataFrame]]:
    found = []
    for recording_id in recording_ids:
        path = layout.annotations_path(recording_id)
        if not path.exists():
            logger.info("no annotations yet for %s", recording_id)
            continue
        frame = read_annotations(path)
        if not frame.empty:
            found.append((recording_id, frame))
    return found


def _export_eaf(resolved, layout, recording_ids: list[str], out: Path | None) -> None:
    """One .eaf per recording: ELAN opens a file against one set of media, not a corpus."""
    found = _annotation_frames(layout, recording_ids)
    if not found:
        raise MacawError(
            f"No annotations under {layout.recordings_dir}. Run a coding extractor before exporting."
        )

    vocabulary = None
    if resolved.coding.codebook is not None:
        codebook = read_codebook(resolved.coding.codebook)
        # every tier gets the same list; the codebook does not say which codes belong where
        vocabulary = {tier: codebook.labels() for tier in _tiers(found)}

    directory = out or (layout.output_dir / "exports" / "eaf")
    directory.mkdir(parents=True, exist_ok=True)
    for recording_id, frame in found:
        target = directory / f"{recording_id}.eaf"
        export_eaf(
            frame,
            build_recording(recording_id, resolved),
            target,
            controlled_vocabulary=vocabulary,
        )
        typer.echo(str(target))


def _tiers(found: list[tuple[str, pd.DataFrame]]) -> set[str]:
    return {str(tier) for _, frame in found for tier in frame["tier"].dropna().unique()}


def _export_sequence(layout, recording_ids: list[str], step: float, out: Path | None) -> None:
    """Annotations resampled onto a fixed grid, the shape recurrence analysis wants."""
    found = _annotation_frames(layout, recording_ids)
    if not found:
        raise MacawError(
            f"No annotations under {layout.recordings_dir}. Run a coding extractor before exporting."
        )

    sequence = to_state_sequence(
        pd.concat([frame for _, frame in found], ignore_index=True), step_s=step
    )
    if sequence.empty:
        raise MacawError("Nothing to sample: every annotation interval was empty or unlabelled.")

    out = out or (layout.output_dir / "exports" / "state_sequence.csv")
    export_state_sequence(sequence, out)
    logger.info("%d steps of %.3f s", len(sequence), step)
    typer.echo(str(out))
