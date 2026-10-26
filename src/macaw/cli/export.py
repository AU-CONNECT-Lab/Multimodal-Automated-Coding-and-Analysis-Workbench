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
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.store.annotations import read_annotations
from macaw.store.longtable import concat_long_tables, read_long_table
from macaw.store.schema import Scale
from macaw.store.sequence import (
    export_state_sequence,
    EMPTY_STATE,
    floor_intervals,
    speaking_intervals,
    to_state_sequence,
)
from macaw.store.wide import export_csv, export_rmd_template, to_wide

app = typer.Typer()
logger = get_logger("export")

FORMATS = ("csv", "parquet", "eaf", "sequence")
SEQUENCES = ("speech", "floor", "annotations")


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option("csv", "--format", "-f", help="csv | parquet | eaf | sequence"),
    scale: str = typer.Option(None, "--scale", help="frame | event | window | recording"),
    wide: bool = typer.Option(True, "--wide/--long"),
    step: float = typer.Option(1.0, "--step", help="Sequence format only: sampling step in seconds."),
    of: str = typer.Option(
        "speech", "--of",
        help="Sequence format only: speech (one speaking/none series per person), floor (whose turn, or overlap) "
        "or annotations.",
    ),
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
    layout = layout_for(resolved, create=False)
    recording_ids = resolve_recordings(resolved, recordings)

    if format_ == "eaf":
        _export_eaf(resolved, layout, recording_ids, out)
        return
    if format_ == "sequence":
        if of not in SEQUENCES:
            raise MacawError(f"Unknown sequence '{of}'. Choose one of: {', '.join(SEQUENCES)}")
        if of == "annotations":
            _export_sequence(layout, recording_ids, step, out)
        else:
            _export_talk_sequence(resolved, layout, recording_ids, of, step, out)
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


def _export_talk_sequence(config, layout, recording_ids: list[str], of: str, step: float, out: Path | None) -> None:
    """Talk resampled over the whole task window, so silence before the first word counts too."""
    from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, read_transcript
    from macaw.extractors.prep import load_prep
    from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, read_turns

    source, name, read = (
        (ASRExtractor.name, TRANSCRIPT_NAME, read_transcript) if of == "speech"
        else (TurnStructureExtractor.name, TURNS_NAME, read_turns)
    )
    pieces = []
    for recording_id in recording_ids:
        path = layout.artifacts_dir(recording_id, source) / name
        if not path.is_file():
            logger.warning("%s: no %s output, left out", recording_id, source)
            continue
        table = read(path)
        table["recording_id"] = recording_id
        intervals = speaking_intervals(table) if of == "speech" else floor_intervals(table)
        prep = load_prep(config, recording_id)
        subjects = [subject.subject_id for subject in build_recording(recording_id, config).subjects]
        series = to_state_sequence(
            intervals, step_s=step, state_column="label",
            series_columns=("subject_id",) if of == "speech" else ("series",),
            t_start=prep.t_origin_s, t_end=prep.t_origin_s + prep.duration_s,
        )
        if of == "speech":
            series = _with_silent_people(series, subjects, recording_id)
        pieces.append(series)
    if not pieces:
        raise MacawError(f"No {source} output to export. Run it first.")
    sequence = pd.concat(pieces, ignore_index=True)
    out = out or (layout.output_dir / "exports" / f"{of}_sequence.csv")
    export_state_sequence(sequence, out)
    logger.info("%d steps of %.3f s", len(sequence), step)
    typer.echo(str(out))


def _with_silent_people(sequence: pd.DataFrame, subjects: list[str], recording_id: str) -> pd.DataFrame:
    """Someone who never spoke still gets their series, all none."""
    grid = sequence["t"].drop_duplicates().to_numpy()
    missing = [subject for subject in subjects if subject not in set(sequence["series"])]
    filler = [
        pd.DataFrame({"recording_id": recording_id, "series": subject, "t": grid, "state": EMPTY_STATE})
        for subject in missing
    ]
    return pd.concat([sequence, *filler], ignore_index=True) if filler else sequence
