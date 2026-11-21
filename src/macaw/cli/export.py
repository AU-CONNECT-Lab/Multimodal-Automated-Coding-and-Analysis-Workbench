from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from macaw.coding.codebook import read_codebook
from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.interop.eaf import export_eaf
from macaw.media.align import apply_offset_table
from macaw.logging_setup import get_logger
from macaw.project.dictionary import feature_dictionary, long_columns, write_dictionary
from macaw.project.review_notes import notes_table
from macaw.project.layout import layout_for
from macaw.project.manifest import recording_specs
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, load_prep
from macaw.project.recording import build_recording
from macaw.project.segments import summaries
from macaw.project.tables import BUILDERS, TABLES, write_table
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
from macaw.store.wide import to_wide, with_segment

app = typer.Typer()
logger = get_logger("export")

FORMATS = ("csv", "parquet", "eaf", "sequence", "skeleton", "video", "transcript", "turns", "voices", "judge", "warnings")
SEQUENCES = ("speech", "floor", "annotations")


@app.command("export", rich_help_panel="Results")
def export(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c", help="The project's settings file."),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    format_: str = typer.Option(
        "csv", "--format", "-f",
        help="csv | parquet | eaf | sequence | skeleton | video, or a folder of tables: transcript (utterances, words, what was "
        "left out and sentence embeddings), turns (turns, transitions and stretches of talk), voices (the voices heard on a shared microphone), judge "
        "(the language model's answers per unit) or warnings (the quality checks' warnings and moments to look at).",
    ),
    scale: str = typer.Option(None, "--scale", help="frame | event | window | recording"),
    wide: bool = typer.Option(True, "--wide/--long", help="One column per measure, or one row per measure and value."),
    step: float = typer.Option(1.0, "--step", help="Sequence format only: sampling step in seconds."),
    of: str = typer.Option(
        "speech", "--of",
        help="Sequence format only: speech (one speaking/none series per person), floor (whose turn, or overlap) "
        "or annotations.",
    ),
    hide: str = typer.Option(
        "faces", "--hide",
        help="Video format only: faces (every head found is blurred, the rest of the picture stays) or everything (only the "
        "skeletons, on a plain ground).",
    ),
    sound: bool = typer.Option(False, "--sound/--no-sound", help="Video format only: keep the video's own sound. A voice identifies a person."),
    out: Path = typer.Option(None, "--out", help="Where to write. Default: a file or folder under exports in the output directory."),
) -> None:
    """Export the measures for analysis, the codes as ELAN files, a state sequence, each person's body points, a copy of the videos with the people hidden, or the tables the steps keep."""
    if format_ not in FORMATS:
        raise MacawError(f"Unknown format '{format_}'. Choose one of: {', '.join(FORMATS)}")
    if scale is not None and scale not in {member.value for member in Scale}:
        known = ", ".join(member.value for member in Scale)
        raise MacawError(f"Unknown scale '{scale}'. Choose one of: {known}")

    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    recording_ids = resolve_recordings(resolved, recordings)

    if format_ == "eaf":
        _export_eaf(config, resolved, layout, recording_ids, out)
        return
    if format_ in TABLES:
        _export_tables(resolved, layout, recording_ids, format_, out)
        return
    if format_ == "skeleton":
        if not wide:
            raise MacawError("The skeleton export has one layout, a row per person and frame; leave out --long")
        _export_skeleton(layout, recording_ids, out)
        return
    if format_ == "video":
        _export_video(config, resolved, layout, recording_ids, hide, sound, out)
        return
    if format_ == "sequence":
        if of not in SEQUENCES:
            raise MacawError(f"Unknown sequence '{of}'. Choose one of: {', '.join(SEQUENCES)}")
        if of == "annotations":
            _export_sequence(layout, recording_ids, step, out)
        else:
            _export_talk_sequence(config, layout, recording_ids, of, step, out)
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
    units = resolved.project.units
    specs = {str(rid): recording_specs(layout, str(rid)) for rid in table["recording_id"].dropna().unique()}
    table = _in_project_units(config, units, layout, table, out.parent, specs)
    exported = to_wide(table) if wide else with_segment(table)
    if format_ == "csv":
        exported.to_csv(out, index=False, encoding="utf-8")
    else:
        exported.to_parquet(out, index=False)
    columns = list(exported.columns) if wide else long_columns(table)
    written = write_dictionary(feature_dictionary(table, columns, specs, units), out)
    notes = notes_table(layout, recording_ids)
    if not notes.empty:
        written = [*written, out.with_name(f"{out.stem}_review_notes.csv")]
        notes.to_csv(written[-1], index=False, encoding="utf-8")

    logger.info("%d rows from %d recordings", len(table), len(frames))
    for path in (out, *written):
        typer.echo(str(path))


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


def _in_project_units(config_path: Path, units: str, layout, table: pd.DataFrame, out_dir: Path, specs: dict) -> pd.DataFrame:
    """Features in metres again in the project's unit, and pixel features too for each recording with a scale;
    scale.csv says which scale. specs holds each recording's own feature specs."""
    from macaw.project.room import CM_PER_UNIT
    from macaw.project.scale import DEPTH_NOTE, converts, holds_on, pixel_power, recording_scale, suffix, unit_rows, unusable
    from macaw.qc.checks import RecordingResults

    pieces, used, converted = [table], [], 0
    for recording_id, rows in table.groupby("recording_id", observed=True):
        recording_id = str(recording_id)
        features = {name: spec.unit for name, spec in specs.get(recording_id, {}).items()}
        if not features:
            continue
        own = resolve_config(config_path, recording_id=recording_id)
        scale = recording_scale(own, layout, recording_id)
        keypoints = RecordingResults(layout, recording_id).keypoints() if scale is not None else None
        problem = unusable(scale, own, recording_id, keypoints) if scale is not None else None
        if problem:
            logger.warning("%s: %s; left in pixels", recording_id, problem)
        usable = scale is not None and problem is None
        roles = {subject.subject_id: subject.role or subject.subject_id for subject in build_recording(recording_id, own).subjects}
        cameras = holds_on(scale, keypoints) if usable else set()

        def factor(first, second, camera):
            return scale.factor(roles.get(first, first), roles.get(second, second)) if converts(cameras, camera) else None

        found = unit_rows(rows, features, units, factor if usable else None)
        if usable:
            pixels = rows[rows["feature"].isin([name for name, unit in features.items() if pixel_power(unit)])]
            left = sorted(set(pixels.loc[[not converts(cameras, camera) for camera in pixels["camera"]], "feature"].astype(str)))
            if left:
                logger.info(
                    "%s: %s stay in pixels, taken on another camera than %s or on no one camera", recording_id, ", ".join(left), scale.video,
                )
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


def _export_eaf(config_path: Path, resolved, layout, recording_ids: list[str], out: Path | None) -> None:
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
            apply_offset_table(build_recording(recording_id, resolve_config(config_path, recording_id=recording_id)), layout.offset_table_path(recording_id)),
            target,
            controlled_vocabulary=vocabulary,
        )
        typer.echo(str(target))


def _export_skeleton(layout, recording_ids: list[str], out: Path | None) -> None:
    """One csv per recording of the points as the measures read them, and one of the points in 3D."""
    from macaw.extractors.video.pose.keypoints import PoseExtractor
    from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS
    from macaw.extractors.video.pose.triangulate import DEPTH_STEP, ONE_CAMERA_STEP, TriangulationExtractor
    from macaw.store.keypoints import read_keypoints, to_wide

    directory = out or (layout.output_dir / "exports" / "skeleton")
    written = 0
    for recording_id in recording_ids:
        folder = layout.keypoints_path(recording_id)
        if not (folder / f"{PoseExtractor.name}.parquet").is_file():
            logger.info("no body points yet for %s", recording_id)
            continue
        # a project places people in 3D one way, so at most one of the last three is there
        for step, suffix in ((PoseExtractor.name, ""), *((name, "_3d") for name in (TriangulationExtractor.name, DEPTH_STEP, ONE_CAMERA_STEP))):
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


def _export_video(config_path: Path, resolved, layout, recording_ids: list[str], hide: str, sound: bool, out: Path | None) -> None:
    """A copy of each camera's video with the people hidden, and beside it the stretches in which a head was not."""
    from macaw.extractors.video.pose.keypoints import PoseExtractor
    from macaw.media.hide import HIDE, write_hidden
    from macaw.store.keypoints import read_keypoints

    if hide not in HIDE:
        raise MacawError(f"Unknown --hide '{hide}'. Choose one of: {', '.join(HIDE)}")
    directory = out or (layout.output_dir / "exports" / "video")
    suffix = "faces_hidden" if hide == "faces" else "skeletons"
    written = 0
    for recording_id in recording_ids:
        path = layout.keypoints_path(recording_id) / f"{PoseExtractor.name}.parquet"
        if not path.is_file():
            logger.info("no body points yet for %s", recording_id)
            continue
        keypoints = read_keypoints(path, recording_id=recording_id)
        recording = build_recording(recording_id, resolve_config(config_path, recording_id=recording_id))
        subjects = [subject.subject_id for subject in recording.subjects]
        for camera in recording.cameras():
            people = keypoints[keypoints["video"] == camera.name]
            # a camera nobody was found on is not copied, as nothing on it could be hidden
            if people.empty or not camera.frames_path.is_file():
                logger.info("%s %s: no body points on this camera, so no copy was made", recording_id, camera.name)
                continue
            target = directory / f"{recording_id}_{Path(camera.name).stem}_{suffix}.mp4"
            on_camera = [subject for subject in subjects if subject in set(people["subject_id"].dropna())]
            gaps = write_hidden(camera.frames_path, target, people, on_camera, hide=hide, sound=sound)
            typer.echo(str(target))
            written += 1
            if hide == "faces":
                listed = target.with_name(f"{target.stem}_not_hidden.csv")
                gaps.round({"t_start": 3, "t_end": 3}).to_csv(listed, index=False, encoding="utf-8")
                typer.echo(str(listed))
    if not written:
        raise MacawError(f"No body points under {layout.recordings_dir}. Run the pose steps (macaw run) before exporting.")
    if hide == "faces":
        typer.echo(
            "Heads are blurred only where a body was found. Watch each copy before sharing it; the _not_hidden.csv beside it "
            "lists when a named person had no blur, in or out of the picture. Bodies, clothes and the room stay visible."
        )


def _export_tables(resolved, layout, recording_ids: list[str], kind: str, out: Path | None) -> None:
    """Each table across the recordings as one csv, with its data dictionary, in a folder of its own."""
    folder = out or (layout.output_dir / "exports" / kind)
    written = []
    for name, (table, said, step) in BUILDERS[kind](resolved, recording_ids).items():
        if table.empty:
            logger.info("%s: no %s in these recordings", kind, name)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        written += write_table(folder, name, table, said, step)
    if not written:
        raise MacawError(f"Nothing to export as {kind} under {layout.recordings_dir}: the steps that keep these tables have not run.")
    for path in written:
        typer.echo(str(path))


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


def _export_talk_sequence(config_path: Path, layout, recording_ids: list[str], of: str, step: float, out: Path | None) -> None:
    """Talk resampled over the whole task window, so silence before the first word counts too."""
    from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, read_transcript
    from macaw.extractors.prep import load_prep
    from macaw.turns.segmentation import TURNS_NAME, TurnStructureExtractor, floor_turns, read_turns

    source, name, read = (
        (ASRExtractor.name, TRANSCRIPT_NAME, read_transcript) if of == "speech"
        else (TurnStructureExtractor.name, TURNS_NAME, read_turns)
    )
    pieces = []
    for recording_id in recording_ids:
        # each recording's talk by its own settings: its setup, segment and text
        config = resolve_config(config_path, recording_id=recording_id)
        path = layout.artifacts_dir(recording_id, source) / name
        if not path.is_file():
            logger.warning("%s: no %s output, left out", recording_id, source)
            continue
        table = read(path)
        if of == "speech":
            # on a shared microphone the voices say when each person talked, overlaps included
            from macaw.extractors.audio.content.asr import load_words
            from macaw.turns.talk import talk_spans, voices_time_talk

            words = load_words(config, recording_id) if voices_time_talk(config, recording_id) else None
            table = talk_spans(config, recording_id, table, words)[0]
        else:
            table = floor_turns(table)
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
