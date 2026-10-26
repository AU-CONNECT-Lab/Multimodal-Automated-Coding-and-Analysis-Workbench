from __future__ import annotations

import getpass
import json
from datetime import datetime
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.extractors.video.pose.keypoints import PoseExtractor, read_owner_file
from macaw.logging_setup import get_logger
from macaw.project.identity import identity_basis, pairing, write_confirmation
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.checks import POSE_SOURCE, RecordingResults, run_checks, write_candidates

app = typer.Typer()
logger = get_logger("qc")


@app.command("qc")
def qc(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    check: list[str] = typer.Option(None, "--check", help="Run only these checks."),
    render: bool = typer.Option(
        False, "--render/--no-render", help="Draw the frames worth reviewing into work_dir/renders."
    ),
    report: bool = typer.Option(False, "--report/--no-report"),
    open_report: bool = typer.Option(False, "--open"),
) -> None:
    """Run quality checks; write warnings and the moments worth reviewing, optionally rendered."""
    if report or open_report:
        raise MacawError(
            "The HTML report is not implemented yet; run without --report, the warnings and "
            "candidates are written either way"
        )

    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    for recording_id in resolve_recordings(resolved, recordings):
        recording = build_recording(recording_id, resolved)
        results = RecordingResults(layout, recording_id)
        outcome = run_checks(recording, resolved, results, only=check)

        directory = layout.qc_dir(recording_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "warnings.json").write_text(
            json.dumps([asdict(warning) for warning in outcome.warnings], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_candidates(outcome.candidates, directory / "candidates.parquet")
        severities = Counter(warning.severity for warning in outcome.warnings)
        logger.info(
            "%s: %d fail, %d warn, %d info, %d candidate span(s)",
            recording_id, severities["fail"], severities["warn"], severities["info"], len(outcome.candidates),
        )
        typer.echo(str(directory))

        if render:
            _render(resolved, layout, recording, results, outcome.candidates)


def _render(resolved, layout, recording, results, candidates) -> None:
    from macaw.extractors.prep import apply_prep_timing
    from macaw.qc.overlay import render_review, review_sample

    # a run that found nobody still renders: the bare frames show why
    if not (layout.keypoints_path(recording.recording_id) / f"{POSE_SOURCE}.parquet").is_file():
        logger.info("%s: %s has not run, nothing to render", recording.recording_id, POSE_SOURCE)
        return
    keypoints = results.keypoints()
    # frame numbers come from the same clock pose.keypoints used
    prep = apply_prep_timing(resolved, recording)
    cfg = resolved.qc
    sample = review_sample(
        candidates,
        # the whole task window, so a stretch where nobody was detected is sampled like any other
        (prep.t_origin_s, prep.t_origin_s + prep.duration_s),
        recording_id=recording.recording_id,
        max_candidates=cfg.review_max_candidates,
        n_baseline=cfg.review_baseline,
        seed=cfg.review_seed,
    )
    out_dir = layout.renders_dir(recording.recording_id)
    index = render_review(recording, keypoints, sample, out_dir)
    logger.info("%s: %d frame(s) rendered", recording.recording_id, len(index))
    typer.echo(str(out_dir))


@app.command("identify")
def identify_speakers(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Record the confirmation without asking."),
    by: str = typer.Option(None, "--by", help="Who is confirming. Default: the login name."),
) -> None:
    """Show which file belongs to whom across the project, then record that a person checked it."""
    resolved = resolve_config(config)
    basis = identity_basis(resolved)
    pairings = [
        pairing(build_recording(recording_id, resolved)) for recording_id in resolve_recordings(resolved, recordings)
    ]
    if not pairings:
        raise MacawError(f"No recordings found under {resolved.project.raw_dir}")

    typer.echo(f"which file belongs to whom, read from file names by media_pattern '{basis['media_pattern']}':")
    for entry in pairings:
        people = "   ".join(
            f"{subject}: {', '.join(files) if files else '(none)'}" for subject, files in entry["subjects"].items()
        )
        shared = ", ".join(entry["shared"]) or "-"
        typer.echo(f"  {entry['recording_id']}   {people}   shared: {shared}")
    empty = sorted({subject for entry in pairings for subject, files in entry["subjects"].items() if not files})
    if empty:
        typer.echo(f"no file of their own in some recordings: {', '.join(empty)}")

    if not yes and not typer.confirm("Is every file paired with the right person?", default=False):
        typer.echo("nothing recorded; fix project.media_pattern or the file names, then run this again")
        raise typer.Exit(code=1)
    layout = layout_for(resolved)
    path = write_confirmation(layout.identity_path, basis, pairings, by or getpass.getuser())
    typer.echo(f"confirmed for the project: {path}")


@app.command("accept-owner")
def accept_owner(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    media: list[str] = typer.Option(None, "--media", help="Camera file, repeatable. Default: every one in doubt."),
    tracks: str = typer.Option(None, "--tracks", help="Comma-separated track ids of the owner, for one --media."),
    by: str = typer.Option(None, "--by", help="Who looked. Default: the login name."),
) -> None:
    """Keep the automatic choice of whose camera it is, or name the owner's tracks instead."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    recording_ids = resolve_recordings(resolved, recordings)
    wanted = set(media or ())

    if tracks is not None:
        if len(wanted) != 1 or len(recording_ids) != 1:
            raise MacawError("--tracks names the owner of one camera: give one -r and one --media")
        recording_id, file_name = recording_ids[0], next(iter(wanted))
        named = [int(track) for track in tracks.split(",") if track.strip()]
        path = layout.camera_owners_path(recording_id)
        entries = read_owner_file(path)
        entries[file_name] = {"tracks": named, "by": stamp}
        _write_json(path, entries)
        typer.echo(
            f"{recording_id}: {file_name} belongs to tracks {named}; apply it with "
            f"`macaw run -r {recording_id} -e pose.kinematics --no-resume`"
        )
        return

    matched = set()
    for recording_id in recording_ids:
        owners = read_owner_file(layout.artifacts_dir(recording_id, PoseExtractor.name) / "owners")
        chosen = {
            name: owner for name, owner in owners.items() if (name in wanted if wanted else owner["doubt"])
        }
        matched.update(chosen)
        if not chosen:
            typer.echo(f"{recording_id}: nothing in doubt")
            continue
        path = layout.camera_owners_accepted_path(recording_id)
        entries = read_owner_file(path)
        for name, owner in chosen.items():
            entries[name] = {"tracks": owner["tracks"], "by": stamp}
        _write_json(path, entries)
        typer.echo(
            f"{recording_id}: kept " + ", ".join(f"{name} as tracks {owner['tracks']}" for name, owner in chosen.items())
        )
    unknown = wanted - matched
    if unknown:
        raise MacawError(f"no owner choice recorded for: {', '.join(sorted(unknown))}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
