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
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, load_prep
from macaw.project.recording import build_recording
from macaw.project.segments import summaries
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
from macaw.store.wide import export_csv, export_rmd_template, to_wide, with_segment

app = typer.Typer()
logger = get_logger("export")

FORMATS = ("csv", "parquet", "eaf", "sequence", "skeleton")
SEQUENCES = ("speech", "floor", "annotations")


@app.command("export")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option("csv", "--format", "-f", help="csv | parquet | eaf | sequence | skeleton"),
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
    """Export features for downstream analysis, annotations as .eaf, a nominal state sequence, or each person's body points."""
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
    if format_ == "skeleton":
        if not wide:
            raise MacawError("The skeleton export has one layout, a row per person and frame; leave out --long")
        _export_skeleton(layout, recording_ids, out)
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
        frames.append(_whole_and_segments(resolved, recording_id, read_long_table(path, scale=scale), scale))
    table = with_segment(pd.concat(frames, ignore_index=True)) if frames else with_segment(concat_long_tables([]))
    if table.empty:
        raise MacawError(
            f"No feature rows under {layout.recordings_dir}. Run `macaw run` before exporting."
        )

    out = out or (layout.output_dir / "exports" / f"features.{format_}")
    out.parent.mkdir(parents=True, exist_ok=True)
    table = _in_project_units(config, resolved.project.units, layout, table, out.parent)
    if format_ == "csv":
        export_csv(table, out, wide=wide)
    else:
        (to_wide(table) if wide else table).to_parquet(out, index=False)

    logger.info("%d rows from %d recordings", len(table), len(frames))
    typer.echo(str(out))
    if rmd:
        typer.echo(str(export_rmd_template(out.parent, out.name, layout.manifest_path.name)))


def _whole_and_segments(config, recording_id: str, rows: pd.DataFrame, scale: str | None) -> pd.DataFrame:
    """A recording's rows and its per-segment ones, each summary spanning what it covers, so a person in a segment is one wide row."""
    whole = rows.copy()
    prepared = layout_for(config, create=False).artifacts_dir(recording_id, MediaPrepExtractor.name) / PREP_INFO_NAME
    if prepared.is_file():
        prep = load_prep(config, recording_id)
        summary = whole["scale"].astype(str).to_numpy() == Scale.RECORDING.value
        whole.loc[summary, "t_start"] = prep.t_origin_s
        whole.loc[summary, "t_end"] = prep.t_origin_s + prep.duration_s
    parts = summaries(config, recording_id)
    if scale is not None:
        parts = parts[parts["scale"].astype(str) == scale]
    return with_segment(pd.concat([with_segment(whole), parts], ignore_index=True))


def _in_project_units(config_path: Path, units: str, layout, table: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Features in metres again in the project's unit, and pixel features too for each recording with a scale;
    scale.csv says which scale."""
    from macaw.project.manifest import read_manifest
    from macaw.project.room import CM_PER_UNIT
    from macaw.project.scale import DEPTH_NOTE, recording_scale, suffix, unit_rows, unusable
    from macaw.qc.checks import RecordingResults

    if not layout.manifest_path.exists():
        return table
    features = {name: spec.unit for name, spec in read_manifest(layout.manifest_path).specs.items()}
    pieces, used, converted = [table], [], 0
    for recording_id, rows in table.groupby("recording_id", observed=True):
        recording_id = str(recording_id)
        own = resolve_config(config_path, recording_id=recording_id)
        scale = recording_scale(own, layout, recording_id)
        problem = unusable(scale, own, recording_id, RecordingResults(layout, recording_id).keypoints()) if scale is not None else None
        if problem:
            logger.warning("%s: %s; left in pixels", recording_id, problem)
        usable = scale is not None and problem is None
        roles = {subject.subject_id: subject.role or subject.subject_id for subject in build_recording(recording_id, own).subjects}
        factor = (lambda first, second: scale.factor(roles.get(first, first), roles.get(second, second))) if usable else None
        found = unit_rows(rows, features, units, factor)
        converted += not found.empty
        pieces.append(found)
        if not usable:
            continue
        objects = ", ".join(sorted({item.object for item in scale.measurements if item.object})) or None
        used += [
            {
                "recording_id": recording_id, "person": role or "everyone", f"{units}_per_px": value / CM_PER_UNIT[units],
                "spread": None if scale.spread.get(role) is None else scale.spread[role] / CM_PER_UNIT[units],
                "source": scale.source, "measured_on": scale.video,
                "measurements": sum(item.person == role for item in scale.measurements), "objects": objects, "holds": DEPTH_NOTE,
            }
            for role, value in scale.by_role.items()
        ]
    if used:
        path = out_dir / "scale.csv"
        pd.DataFrame(used).to_csv(path, index=False)
        logger.info("features in pixels also given in %s, named *%s, where a recording has a scale; see %s", units, suffix(units), path)
    if not converted:
        return table
    return with_segment(pd.concat(pieces, ignore_index=True))


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


def _export_skeleton(layout, recording_ids: list[str], out: Path | None) -> None:
    """One csv per recording of the points as the measures read them, and one of the points in 3D."""
    from macaw.extractors.video.pose.keypoints import PoseExtractor
    from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
    from macaw.extractors.video.pose.triangulate import TriangulationExtractor
    from macaw.store.keypoints import read_keypoints, to_wide

    directory = out or (layout.output_dir / "exports" / "skeleton")
    written = 0
    for recording_id in recording_ids:
        folder = layout.keypoints_path(recording_id)
        if not (folder / f"{PoseExtractor.name}.parquet").is_file():
            logger.info("no body points yet for %s", recording_id)
            continue
        for step, suffix in ((PoseExtractor.name, ""), (TriangulationExtractor.name, "_3d")):
            path = folder / f"{step}.parquet"
            if not path.is_file():
                continue
            table = to_wide(read_keypoints(path, recording_id=recording_id), list(STANDARD_JOINTS))
            if table.empty:
                continue
            if suffix:
                table = table.drop(columns="video")
            table = table.round({"t": 4, **{column: 3 for column in table.columns if column.endswith(("_x", "_y", "_z", "_conf"))}})
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{recording_id}{suffix}.csv"
            table.to_csv(target, index=False, encoding="utf-8")
            typer.echo(str(target))
            written += 1
    if not written:
        raise MacawError(f"No body points under {layout.recordings_dir}. Run the pose steps (macaw run) before exporting.")


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
