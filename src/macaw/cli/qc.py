from __future__ import annotations

import getpass
import json
import webbrowser
from datetime import datetime
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.extractors.audio.content.diarization import NO_ONE, read_voices
from macaw.extractors.audio.content.speaker_id import suggest_voices
from macaw.extractors.video.pose.keypoints import PoseExtractor, read_owner_file, waiting_cameras
from macaw.logging_setup import get_logger
from macaw.project.identity import identity_basis, pairing, write_confirmation
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.checks import POSE_SOURCE, RecordingResults, run_checks, write_candidates
from macaw.qc.report import write_project_report, write_recording_report

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
    report: bool = typer.Option(False, "--report/--no-report", help="Write an HTML page per recording and one for the project."),
    open_report: bool = typer.Option(False, "--open", help="Write the pages and open the project page."),
) -> None:
    """Run quality checks; write warnings and the moments worth reviewing, optionally rendered."""

    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    pages = {}
    failed = None
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
        if report or open_report:
            pages[recording_id] = write_recording_report(resolved, recording_id)
        if severities["fail"] and resolved.qc.fail_fast:
            failed = recording_id
            break

    if pages:
        project_page = write_project_report(resolved, pages)
        typer.echo(str(project_page))
        if open_report:
            webbrowser.open(project_page.resolve().as_uri())
    if failed is not None:
        typer.echo(f"qc.fail_fast: {failed} failed a check, the remaining recordings were not checked", err=True)
        raise typer.Exit(code=1)


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
    index = render_review(recording, keypoints, sample, out_dir, read_shared=resolved.pose.read_shared)
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
    media: list[str] = typer.Option(None, "--media", help="Camera file, repeatable. Default: every one waiting for a person."),
    tracks: str = typer.Option(None, "--tracks", help="Comma-separated track ids of one person, for one --media."),
    subject: str = typer.Option(None, "--subject", help="Whose tracks --tracks names. Default: the camera's own subject."),
    rename: str = typer.Option(
        None, "--rename", help="SHOWN=RIGHT pairs, comma-separated: the person labelled SHOWN in the renders is RIGHT."
    ),
    by: str = typer.Option(None, "--by", help="Who looked. Default: the login name."),
) -> None:
    """Keep the automatic choice of who is on camera, or correct it."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    recording_ids = resolve_recordings(resolved, recordings)
    wanted = set(media or ())

    if tracks is not None or rename is not None:
        if tracks is not None and rename is not None:
            raise MacawError("give --tracks or --rename, not both")
        if len(wanted) != 1 or len(recording_ids) != 1:
            raise MacawError("--tracks and --rename correct one camera: give one -r and one --media")
        recording_id, file_name = recording_ids[0], next(iter(wanted))
        recording = build_recording(recording_id, resolved)
        if tracks is not None:
            people = {_tracks_owner(recording, file_name, subject): [int(track) for track in tracks.split(",") if track.strip()]}
        else:
            owners = read_owner_file(layout.artifacts_dir(recording_id, PoseExtractor.name) / "owners")
            people = _renamed(owners.get(file_name, {}).get("people", {}), rename, recording)
        path = layout.camera_owners_path(recording_id)
        entries = read_owner_file(path)
        claimed = {track for ids in people.values() for track in ids}
        earlier = {
            who: ids
            for who, ids in entries.get(file_name, {}).get("people", {}).items()
            if who not in people and not claimed & set(ids)
        }
        entries[file_name] = {"people": {**earlier, **people}, "by": stamp}
        _write_json(path, entries)
        named = "; ".join(f"{who} is tracks {_shortened(ids)}" for who, ids in people.items())
        typer.echo(
            f"{recording_id}: on {file_name}, {named}; apply it with "
            f"`macaw run -r {recording_id} -e pose.kinematics`"
        )
        return

    matched = set()
    for recording_id in recording_ids:
        owners = read_owner_file(layout.artifacts_dir(recording_id, PoseExtractor.name) / "owners")
        waiting = set(waiting_cameras(resolved, recording_id, every=resolved.pose.review == "always"))
        chosen = {
            name: owner for name, owner in owners.items() if (name in wanted if wanted else name in waiting)
        }
        matched.update(chosen)
        if not chosen:
            typer.echo(f"{recording_id}: nothing waiting")
            continue
        path = layout.camera_owners_accepted_path(recording_id)
        entries = read_owner_file(path)
        for name, owner in chosen.items():
            entries[name] = {"people": owner["people"], "by": stamp}
        _write_json(path, entries)
        kept = "; ".join(
            f"{name} as " + ", ".join(f"{who} tracks {_shortened(ids)}" for who, ids in owner["people"].items())
            for name, owner in chosen.items()
        )
        typer.echo(f"{recording_id}: kept {kept}")
    unknown = wanted - matched
    if unknown:
        raise MacawError(f"no choice recorded for: {', '.join(sorted(unknown))}")


def _shortened(ids: list[int], shown: int = 6) -> str:
    rest = f" and {len(ids) - shown} more" if len(ids) > shown else ""
    return ", ".join(str(track) for track in ids[:shown]) + rest


def _tracks_owner(recording, file_name: str, subject: str | None) -> str:
    camera = next((item for item in recording.media if item.path.name == file_name), None)
    if camera is None:
        raise MacawError(f"{recording.recording_id} has no file {file_name}")
    who = subject or camera.subject_id
    if who is None:
        raise MacawError(f"{file_name} is a shared camera: say whose tracks these are with --subject")
    if who not in [person.subject_id for person in recording.subjects]:
        raise MacawError(f"{who} is not a subject of {recording.recording_id}")
    return who


def _renamed(people: dict[str, list[int]], rename: str, recording) -> dict[str, list[int]]:
    """The tracks shown under one name, given to the person they really are."""
    if not people:
        raise MacawError("no automatic choice recorded for this camera; run pose.keypoints first")
    pairs = [pair.split("=", 1) for pair in rename.split(",") if pair.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise MacawError("--rename takes SHOWN=RIGHT pairs, e.g. P1=P3,P3=P1")
    subjects = [person.subject_id for person in recording.subjects]
    shown = [old.strip() for old, _ in pairs]
    right = [new.strip() for _, new in pairs]
    unknown = [name for name in shown if name not in people] + [name for name in right if name not in subjects]
    if unknown:
        raise MacawError(f"not a person on this camera or not a subject: {', '.join(unknown)}")
    if len(set(right)) != len(right):
        raise MacawError("--rename gives one person two sets of tracks")
    return {new: people[old] for old, new in zip(shown, right)}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command("accept-voices")
def accept_voices(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    rename: str = typer.Option(
        None, "--rename", help=f"LABEL=PERSON pairs for every voice, comma-separated; PERSON {NO_ONE} leaves a voice out."
    ),
    by: str = typer.Option(None, "--by", help="Who listened. Default: the login name."),
) -> None:
    """Keep the suggested names of the voices on a shared microphone, or name them."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    recording_ids = resolve_recordings(resolved, recordings)
    if rename is not None and len(recording_ids) != 1:
        raise MacawError("--rename names the voices of one recording: give one -r")
    for recording_id in recording_ids:
        recording = build_recording(recording_id, resolved)
        voices = read_voices(resolved, recording_id)
        if not voices:
            raise MacawError(f"{recording_id}: no voices found yet; run text.diarize first")
        if rename is not None:
            people = _named_voices(voices, rename, recording)
        else:
            suggestion = suggest_voices(resolved, recording)
            if suggestion is None:
                raise MacawError(
                    f"{recording_id}: no suggestion without each person's own mic or face.movement on their camera; name the voices with --rename"
                )
            people = suggestion["people"]
            undecided = sorted(label for label, who in people.items() if who is None)
            if undecided:
                raise MacawError(
                    f"{recording_id}: the suggestion leaves {', '.join(undecided)} undecided between "
                    f"{', '.join(suggestion['unheard'])}; listen to the clips and name all voices with --rename"
                )
        _write_json(layout.voice_owners_path(recording_id), {"people": people, "by": stamp})
        named = "; ".join(f"{label} is {who}" for label, who in sorted(people.items()))
        typer.echo(f"{recording_id}: {named}; apply it with `macaw run -r {recording_id}`")


def _named_voices(voices: dict, rename: str, recording) -> dict[str, str]:
    pairs = [pair.split("=", 1) for pair in rename.split(",") if pair.strip()]
    if any(len(pair) != 2 for pair in pairs):
        raise MacawError("--rename takes LABEL=PERSON pairs, e.g. SPEAKER_00=A,SPEAKER_01=none")
    people = {label.strip(): who.strip() for label, who in pairs}
    subjects = {person.subject_id for person in recording.subjects}
    unknown = [label for label in people if label not in voices] + [
        who for who in people.values() if who not in subjects and who != NO_ONE
    ]
    if unknown:
        raise MacawError(f"not a voice found or not a subject: {', '.join(unknown)}")
    missing = sorted(set(voices) - set(people))
    if missing:
        raise MacawError(f"say who every voice is, or {NO_ONE}: {', '.join(missing)} not named")
    return people
