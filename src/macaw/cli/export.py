from __future__ import annotations

from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.logging_setup import get_logger
from macaw.project.layout import make_layout
from macaw.store.longtable import concat_long_tables, read_long_table
from macaw.store.schema import Scale
from macaw.store.wide import export_csv, export_rmd_template, to_wide

app = typer.Typer()
logger = get_logger("export")

FORMATS = ("csv", "parquet", "eaf")


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option("csv", "--format", "-f", help="csv | parquet | eaf"),
    scale: str = typer.Option(None, "--scale", help="frame | event | window | recording"),
    wide: bool = typer.Option(True, "--wide/--long"),
    out: Path = typer.Option(None, "--out"),
    rmd: bool = typer.Option(False, "--rmd", help="Also write an .Rmd starter script."),
) -> None:
    """Export features for downstream analysis, or annotations as .eaf."""
    if format_ not in FORMATS:
        raise MacawError(f"Unknown format '{format_}'. Choose one of: {', '.join(FORMATS)}")
    if format_ == "eaf":
        raise MacawError("eaf export is not implemented yet; it waits on interop/eaf.py")
    if scale is not None and scale not in {member.value for member in Scale}:
        known = ", ".join(member.value for member in Scale)
        raise MacawError(f"Unknown scale '{scale}'. Choose one of: {known}")

    resolved = resolve_config(config)
    layout = make_layout(resolved.project.output_dir, create=False)
    recording_ids = resolve_recordings(resolved, recordings)

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
