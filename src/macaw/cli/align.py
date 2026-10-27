from __future__ import annotations

import getpass
from datetime import datetime
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import BackendNotAvailableError, MacawError
from macaw.execution.runner import resolve_recordings
from macaw.logging_setup import get_logger
from macaw.media.align import (
    MANUAL,
    SAMPLE_RATE,
    UNALIGNED,
    AlignResult,
    align_recording,
    describe_agreement,
    describe_drift,
    drift_flagged,
    keep_accepted,
    keep_manual,
    low_confidence,
    needs_person,
    read_offset_table,
    write_offset_table,
)
from macaw.media.probe import probe
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording

app = typer.Typer()
logger = get_logger("align")


@app.command("align")
def align(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    recut: bool = typer.Option(False, "--recut", help="Physically re-encode instead of storing offsets."),
    drift_check: bool = typer.Option(True, "--drift-check/--no-drift-check"),
    set_: list[str] = typer.Option(None, "--set"),
) -> None:
    """Estimate inter-source offsets by cross-correlation and draw each pair for review."""
    if recut:
        raise MacawError("--recut is not supported yet; offsets are stored and added when files are read")
    overrides = list(set_ or ())
    if not drift_check:
        overrides.append("align.drift_check=false")
    resolved = resolve_config(config, cli_overrides=overrides)
    cfg = resolved.align
    if cfg.method == "none":
        typer.echo("align.method is none: every file is taken to start together, nothing to estimate")
        return
    if cfg.method == "manual":
        typer.echo("align.method is manual: set each offset with `macaw set-offset`")
        return

    layout = layout_for(resolved)
    for recording_id in resolve_recordings(resolved, recordings):
        recording = build_recording(recording_id, resolved)
        infos = _probe_all(recording)
        figures = layout.renders_dir(recording_id) / "align"
        results = align_recording(recording, cfg, infos, on_pair=_drawer(figures, cfg))

        path = layout.offset_table_path(recording_id)
        if path.is_file():
            previous = read_offset_table(path)
            results, stale = keep_manual(results, previous)
            for target in stale:
                typer.echo(f"  {target}: its manual offset was set against another reference and was replaced")
            results, dropped = keep_accepted(results, previous)
            for target in dropped:
                typer.echo(f"  {target}: the result changed since it was accepted, look at it again")
        write_offset_table(results, path)
        _summarise(recording_id, results, cfg)
        typer.echo(f"  table {path}")
        if any(figures.glob("*.png")):
            typer.echo(f"  figures {figures}")
        if cfg.review != "skip" and any(needs_person(result, cfg) for result in results):
            typer.echo(
                f"  files waiting for a person hold the run back: look at them, then `macaw accept-offset -r {recording_id}` "
                "to keep them or `macaw set-offset` to change one"
            )


@app.command("set-offset")
def set_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recording: str = typer.Option(..., "--recording", "-r"),
    media: str = typer.Option(..., "--media", help="Media file name to adjust."),
    offset: float = typer.Option(..., "--offset", help="Seconds."),
) -> None:
    """Manually override one offset after inspecting it; a later `macaw align` keeps it."""
    resolved = resolve_config(config)
    layout = layout_for(resolved)
    built = build_recording(recording, resolved)
    names = [item.path.name for item in built.media]
    if media not in names:
        raise MacawError(f"{media} is not a file of {recording}; its files are {', '.join(names)}")

    path = layout.offset_table_path(recording)
    results = read_offset_table(path) if path.is_file() else []
    if results:
        reference = results[0].reference
    else:
        _probe_all(built)
        reference = built.reference_media().path.name
        results = [AlignResult(reference=reference, target=reference, offset_s=0.0, confidence=None, method=MANUAL)]
    if media == reference:
        raise MacawError(f"{media} is the reference; every other offset is measured against it")

    stamp = datetime.now().isoformat(timespec="seconds")
    entry = AlignResult(
        reference=reference, target=media, offset_s=offset, confidence=None, method=MANUAL,
        note=f"set by hand {stamp}",
    )
    results = [result for result in results if result.target != media] + [entry]
    write_offset_table(results, path)
    typer.echo(f"{recording}: {media} set to {offset:+.3f} s against {reference}")


@app.command("accept-offset")
def accept_offset(
    config: Path = typer.Option(Path("project.yaml"), "--config", "-c"),
    recordings: str = typer.Option(None, "--recordings", "-r", help="Comma-separated ids or a txt file path."),
    media: list[str] = typer.Option(None, "--media", help="File to keep, repeatable. Default: every file waiting for a person."),
    by: str = typer.Option(None, "--by", help="Who looked at them. Default: the login name."),
) -> None:
    """Keep offsets waiting for review, after looking at their figures, so the run goes on."""
    resolved = resolve_config(config)
    layout = layout_for(resolved, create=False)
    wanted = set(media or ())
    stamp = f"{by or getpass.getuser()} {datetime.now().isoformat(timespec='seconds')}"
    matched = set()
    for recording_id in resolve_recordings(resolved, recordings):
        path = layout.offset_table_path(recording_id)
        if not path.is_file():
            typer.echo(f"{recording_id}: no offset table, run `macaw align` first")
            continue
        results = read_offset_table(path)
        kept = [
            result for result in results
            if (result.target in wanted if wanted else needs_person(result, resolved.align) is not None)
        ]
        matched.update(result.target for result in kept)
        if not kept:
            typer.echo(f"{recording_id}: nothing waiting")
            continue
        for result in kept:
            result.accepted = stamp
        write_offset_table(results, path)
        typer.echo(f"{recording_id}: kept {', '.join(result.target for result in kept)}")
    unknown = wanted - matched
    if unknown:
        raise MacawError(f"not in any offset table: {', '.join(sorted(unknown))}")


def _probe_all(recording) -> dict:
    """Durations decide the reference, and the streams decide what can be correlated."""
    infos = {}
    for item in recording.media:
        info = probe(item.path)
        item.duration_s = info.duration_s
        infos[item.path.name] = info
    return infos


def _drawer(out_dir: Path, cfg):
    try:
        from macaw.qc.figures import alignment_figure, require_matplotlib

        require_matplotlib()
    except BackendNotAvailableError as error:
        logger.warning("%s; no review figures drawn", error)
        return None

    def draw(reference, target, result):
        name = f"{Path(result.target).stem}.png"
        alignment_figure(reference, target, SAMPLE_RATE, result, cfg, out_dir / name)

    return draw


def _summarise(recording_id: str, results: list[AlignResult], cfg) -> None:
    typer.echo(f"{recording_id}  reference {results[0].reference if results else '-'}")
    for result in results:
        if result.target == result.reference:
            continue
        flags = []
        if result.method == UNALIGNED:
            hint = "" if result.metadata_offset_s is None else f", metadata says {result.metadata_offset_s:+.1f} s"
            line = f"not aligned ({result.note}{hint})"
            flags.append("CHECK")
        elif result.method == MANUAL:
            line = f"{result.offset_s:+.3f} s  set by hand"
        else:
            line = f"{result.offset_s:+.3f} s  confidence {result.confidence:.2f}, {describe_agreement(result)}"
            if result.drift_ppm is not None:
                line += f"  {describe_drift(result)}"
            if low_confidence(result, cfg):
                flags.append("LOW CONFIDENCE")
        if drift_flagged(result):
            flags.append("DRIFT")
        if flags and result.accepted:
            flags.append(f"(kept by {result.accepted})")
        typer.echo(f"  {result.target:<32} {line}  {' '.join(flags)}".rstrip())
