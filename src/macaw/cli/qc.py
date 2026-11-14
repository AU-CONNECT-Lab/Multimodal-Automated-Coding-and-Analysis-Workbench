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
from macaw.extractors.audio.content.diarization import NO_ONE, further_voice_id, read_further_voice, read_voices, voices_mixed
from macaw.extractors.audio.content.speaker_id import offered_suggestion
from macaw.extractors.video.pose.keypoints import PoseExtractor, read_owner_file, waiting_cameras
from macaw.logging_setup import get_logger
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.checks import POSE_SOURCE, RecordingResults, run_checks, write_candidates
from macaw.qc.look import PLAIN
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
    look: bool = typer.Option(
        False, "--look", help="Draw the review frames and have the model in qc.look check the people drawn on them; a person checks the model."
    ),
    report: bool = typer.Option(False, "--report/--no-report", help="Write an HTML page per recording and one for the project."),
    open_report: bool = typer.Option(False, "--open", help="Write the pages and open the project page."),
) -> None:
    """Run quality checks; write warnings and the moments worth reviewing, optionally rendered."""

    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    pages = {}
    failed = None
    backend = None
    if look:
        from macaw.qc.look import backend_for, notice

        warning = notice(resolved.qc.look)
        if warning:
            typer.echo(warning, err=True)
        backend = backend_for(resolved.qc.look)
    for recording_id in resolve_recordings(resolved, recordings):
        # what this recording ran with, its own recordings/<id>.yaml included
        own = resolve_config(config, recording_id=recording_id)
        recording = build_recording(recording_id, own)
        results = RecordingResults(layout, recording_id)
        outcome = run_checks(recording, own, results, only=check)

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

        if render or look:
            index = _render(resolved, layout, recording, results, outcome.candidates, plain=look)
            if look and index is not None:
                _look(resolved, layout, recording, results, index, backend)
        if report or open_report:
            pages[recording_id] = write_recording_report(own, recording_id)
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


def _render(resolved, layout, recording, results, candidates, *, plain: bool = False):
    from macaw.extractors.prep import apply_prep_timing
    from macaw.qc.overlay import render_review, review_sample

    # a run that found nobody still renders: the bare frames show why
    if not (layout.keypoints_path(recording.recording_id) / f"{POSE_SOURCE}.parquet").is_file():
        logger.info("%s: %s has not run, nothing to render", recording.recording_id, POSE_SOURCE)
        return None
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
    index = render_review(
        recording, keypoints, sample, out_dir, plain_dir=out_dir / PLAIN if plain else None,
    )
    logger.info("%s: %d frame(s) rendered", recording.recording_id, len(index))
    typer.echo(str(out_dir))
    return index


def _look(resolved, layout, recording, results, index, backend) -> None:
    from macaw.qc.look import describe, look_at_recording, summary

    look_at_recording(backend, resolved.qc.look, layout, recording, results.keypoints(), index, seed=resolved.qc.review_seed)
    typer.echo(f"{recording.recording_id}: {describe(summary(layout, recording.recording_id))}")


@app.command("check-look")
def check_look(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    frame: str = typer.Option(..., "--frame", help="The review frame's file name, as macaw qc --look lists it."),
    verdict: str = typer.Option(None, "--verdict", help="right when the model was right about the frame, wrong when not."),
    clear: bool = typer.Option(False, "--clear", help="Take the verdict back."),
    by: str = typer.Option(None, "--by", help="Who checked. Default: the login name."),
) -> None:
    """Say whether the model in qc.look was right about one review frame, one it flagged or one it passed that was drawn at random."""
    from macaw.qc.look import describe, set_check, summary

    if clear == (verdict is not None):
        raise MacawError("give --verdict right or wrong, or --clear")
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    set_check(layout, recording, frame, None if clear else verdict, by or getpass.getuser())
    typer.echo(f"{recording}: {describe(summary(layout, recording))}")


@app.command("note")
def note(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    tag: list[str] = typer.Option(None, "--tag", help="One of qc.note_tags, repeatable."),
    text: str = typer.Option("", "--text", help="The note in your own words."),
    remove: str = typer.Option(None, "--remove", help="Take back the note with this id."),
    by: str = typer.Option(None, "--by", help="Who wrote it. Default: the login name."),
) -> None:
    """Note something about a recording while reviewing it, such as a part to cut or a step to run again; it changes nothing by itself."""
    from macaw.project.review_notes import add_note, remove_note

    resolved = resolve_config(config, recording_id=recording)
    if recording not in resolve_recordings(resolve_config(config), None):
        raise MacawError(f"{recording} is not one of this project's recordings")
    path = layout_for(resolved, create=False).review_notes_path(recording)
    if remove:
        if not remove_note(path, remove):
            raise MacawError(f"{recording} has no note {remove}")
        typer.echo(f"{recording}: note {remove} taken back")
        return
    tags = list(tag or [])
    unknown = [item for item in tags if item not in resolved.qc.note_tags]
    if unknown:
        raise MacawError(f"{', '.join(unknown)} not in qc.note_tags ({', '.join(resolved.qc.note_tags)}); add it there first")
    if not tags and not text.strip():
        raise MacawError("give --tag, --text or both")
    added = add_note(path, tags, text, by or getpass.getuser())
    typer.echo(f"{recording}: note {added['id']} kept in {path}")


@app.command("name-people")
def name_people(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    camera: str = typer.Option(..., "--camera", help="The camera, as its folder is named."),
    at: float = typer.Option(None, "--at", help="Seconds into the camera's file, from its first picture."),
    person: list[str] = typer.Option(
        None, "--person", help="ROLE=X,Y: where that person is in the picture, as shares of its width and height. Repeatable."
    ),
    clear: bool = typer.Option(False, "--clear", help="Take back every click on this camera."),
    remove: bool = typer.Option(
        False, "--remove", help="Take back the clicks at --at, or only those of each --person ROLE given."
    ),
    by: str = typer.Option(None, "--by", help="Who clicked. Default: the login name."),
) -> None:
    """Say who is who on a camera: at one moment of its file, where each person is in the picture. pose.keypoints
    names whoever is there and follows them through the recording; click again at another moment where it lost someone."""
    from macaw.extractors.video.pose.keypoints import analysed_part, camera_people
    from macaw.project.names import add_clicks, clear_clicks, remove_clicks

    resolved = resolve_config(config, recording_id=recording)
    built = build_recording(recording, resolved)
    item = next((item for item in built.cameras() if item.name == camera), None)
    if item is None:
        cameras = ", ".join(item.name for item in built.cameras()) or "none"
        raise MacawError(f"{camera} is not a camera of {recording} read for anyone; its cameras are {cameras}")
    path = layout_for(resolved).names_path(recording)
    if clear:
        if at is not None or person:
            raise MacawError("--clear takes back every click on the camera; give it alone")
        clear_clicks(path, camera)
        typer.echo(f"{recording}: no clicks on {camera} now; apply it with `macaw run -r {recording} -e pose.keypoints`")
        return
    if remove:
        if at is None or any("=" in given for given in person or []):
            raise MacawError("--remove takes --at, the moment of the clicks, and optionally a --person ROLE for each person whose click goes")
        taken = sum(remove_clicks(path, camera, at, given) for given in person) if person else remove_clicks(path, camera, at)
        if not taken:
            raise MacawError(f"no click on {camera} at {at:g} s{' of ' + ', '.join(person) if person else ''} to take back")
        typer.echo(f"{recording}: took back {taken} click{'s' if taken > 1 else ''} on {camera} at {at:g} s; apply it with `macaw run -r {recording} -e pose.keypoints`")
        return
    if at is None or at < 0 or not person:
        raise MacawError("give --at, the moment from the camera's first picture, and a --person ROLE=X,Y for each person clicked")
    points: dict[str, tuple[float, float]] = {}
    for given in person:
        role, _, place = given.partition("=")
        values = [part.strip() for part in place.split(",")]
        try:
            x, y = (float(value) for value in values)
        except ValueError:
            raise MacawError(f"--person {given}: write ROLE=X,Y with X and Y the shares of the picture's width and height") from None
        if role not in camera_people(item):
            raise MacawError(f"--person {given}: the setup reads {camera} for {', '.join(camera_people(item))}, not {role}")
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise MacawError(f"--person {given}: X and Y are shares of the picture, from 0 to 1")
        if role in points:
            raise MacawError(f"--person {role} is given twice at one moment")
        points[role] = (x, y)
    names = add_clicks(path, camera, at, points, by or getpass.getuser())
    clicked = ", ".join(f"{click['person']} at {click['at']:g} s" for click in names[camera])
    typer.echo(f"{recording}: on {camera}, {clicked}; apply it with `macaw run -r {recording} -e pose.keypoints`")
    part = analysed_part(resolved, recording, camera)
    if part is not None and not part[0] <= at <= part[1]:
        typer.echo(
            f"note: {at:g} s is outside the analysed part of {camera} ({part[0]:.1f} to {part[1]:.1f} s of its file), "
            "so pose.keypoints leaves these clicks out while the analysed part stays where it is"
        )


@app.command("accept-owner")
def accept_owner(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    media: list[str] = typer.Option(None, "--camera", help="Camera, repeatable. Default: every one waiting for a person."),
    by: str = typer.Option(None, "--by", help="Who looked. Default: the login name."),
) -> None:
    """Keep who is on camera as it stands, doubts and all, after looking at the review frames."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    wanted = set(media or ())
    matched = set()
    for recording_id in resolve_recordings(resolved, recordings):
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
            f"{name} as " + (", ".join(f"{who} tracks {_shortened(ids)}" for who, ids in owner["people"].items()) or "nobody named")
            for name, owner in chosen.items()
        )
        typer.echo(f"{recording_id}: kept {kept}")
    unknown = wanted - matched
    if unknown:
        raise MacawError(f"no naming recorded for: {', '.join(sorted(unknown))}")


def _shortened(ids: list[int], shown: int = 6) -> str:
    rest = f" and {len(ids) - shown} more" if len(ids) > shown else ""
    return ", ".join(str(track) for track in ids[:shown]) + rest


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.command("accept-voices")
def accept_voices(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    rename: str = typer.Option(
        None, "--rename", help=f"LABEL=PERSON pairs for every voice, comma-separated; PERSON {NO_ONE} leaves a voice out (with text.extra_voices it takes turns as other)."
    ),
    mixed: bool = typer.Option(
        False, "--mixed", help="One voice holds two people here: the lines then wait for `macaw line-speakers` to say who speaks each."
    ),
    nobody_else: bool = typer.Option(
        False, "--nobody-else", help="You listened to the further voice found and nobody but the people speaks; names nothing."
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
        if nobody_else:
            found = read_further_voice(resolved, recording_id)
            if not found:
                raise MacawError(f"{recording_id}: no further voice was found")
            _write_json(layout.further_voice_path(recording_id), {"found": further_voice_id(found), "by": stamp})
            typer.echo(f"{recording_id}: nobody else speaks; apply it with `macaw run -r {recording_id}`")
            continue
        if rename is not None:
            people = _named_voices(voices, rename, recording)
        else:
            suggestion = offered_suggestion(resolved, recording)
            if suggestion is None:
                raise MacawError(
                    f"{recording_id}: no suggestion can be made (nobody has a mic of their own or "
                    "face.movement on a camera); name the voices with --rename"
                )
            people = suggestion["people"]
            undecided = sorted(label for label, who in people.items() if who is None)
            if undecided:
                raise MacawError(
                    f"{recording_id}: the suggestion leaves {', '.join(undecided)} undecided between "
                    f"{', '.join(suggestion['unheard'])}; listen to the clips and name all voices with --rename"
                )
        _write_json(layout.voice_owners_path(recording_id), {"people": people, "by": stamp, **({"mixed": True} if mixed else {})})
        named = "; ".join(f"{label} is {who}" for label, who in sorted(people.items()))
        then = f"say who speaks each line with `macaw line-speakers -r {recording_id}`" if mixed else f"apply it with `macaw run -r {recording_id}`"
        typer.echo(f"{recording_id}: {named}; {then}")


@app.command("line-speakers")
def line_speakers(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recordings", "-r", help="One recording whose voices were marked as mixed."),
    suggested: bool = typer.Option(False, "--suggested", help="Give every line with a sure suggestion to the person suggested."),
    apply: Path = typer.Option(None, "--apply", help="A csv with t_start and person: who speaks the line starting then."),
    at: float = typer.Option(None, "--at", help="With --person: from the word said at this second to the end of its line, that person speaks."),
    person: str = typer.Option(None, "--person"),
    as_they_are: bool = typer.Option(False, "--as-they-are", help="You checked the lines and nothing more needs changing."),
    clear: bool = typer.Option(False, "--clear", help="Forget every change; the lines wait for a decision again."),
    by: str = typer.Option(None, "--by", help="Who decided. Default: the login name."),
) -> None:
    """On a shared microphone whose voices are mixed: list who speaks each line, or change it."""
    import pandas as pd

    from macaw.extractors.audio.content.line_speakers import (
        BY_HAND, SUGGESTED, keep_suggested, split_at, suggested_lines, with_picks, write_line_speakers,
    )

    resolved = resolve_config(config, recording_id=recording)
    layout = layout_for(resolved, create=False)
    if not voices_mixed(resolved, recording):
        raise MacawError(
            f"{recording}: the voices are not marked as mixed; name them with `macaw accept-voices -r {recording} --rename ... --mixed` first"
        )
    if clear:
        layout.line_speakers_path(recording).unlink(missing_ok=True)
        typer.echo(f"{recording}: changes forgotten")
        return
    subjects = build_recording(recording, resolved)
    lines = suggested_lines(resolved, subjects)
    name = by or getpass.getuser()
    people = {subject.subject_id for subject in subjects.subjects} | {NO_ONE}
    if person is not None and person not in people:
        raise MacawError(f"not a subject: {person}")
    if sum([suggested, apply is not None, at is not None, as_they_are]) > 1:
        raise MacawError("give one of --suggested, --apply, --at and --as-they-are")
    if suggested:
        lines, route = keep_suggested(lines), SUGGESTED
    elif apply is not None:
        table = pd.read_csv(apply).dropna(subset=["person"])
        unknown = sorted(set(table["person"].astype(str)) - people)
        if unknown:
            raise MacawError(f"not a subject: {', '.join(unknown)}")
        lines, route = with_picks(lines, {round(float(row.t_start), 3): str(row.person) for row in table.itertuples()}, name), BY_HAND
    elif at is not None:
        if person is None:
            raise MacawError("--at needs --person")
        lines, route = split_at(lines, at, person, name), BY_HAND
    elif as_they_are:
        route = BY_HAND
    else:
        for line in lines.itertuples(index=False):
            doubt = f"  [{'sure' if line.sure else 'doubt'}: {line.suggested or 'not ' + line.base}, {line.doubt}]" if line.doubt else ""
            changed = f" (was {line.base})" if line.person != line.base else ""
            typer.echo(f"  {line.t_start:.1f} s {line.person}{changed}: {line.text}{doubt}")
        typer.echo(f"{recording}: {len(lines)} lines, {int((lines['person'] != lines['base']).sum())} changed, {int((lines['doubt'] != '').sum())} doubted")
        return
    write_line_speakers(resolved, recording, lines, f"{name} {datetime.now().isoformat(timespec='seconds')}", route)
    changed = int((lines["person"] != lines["base"]).sum())
    typer.echo(f"{recording}: {changed} of {len(lines)} lines with another person than their voice's; apply it with `macaw run -r {recording}`")


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
