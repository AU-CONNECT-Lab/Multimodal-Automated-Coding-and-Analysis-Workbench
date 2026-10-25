from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import typer

from macaw.config.loader import resolve_config
from macaw.exceptions import MacawError
from macaw.execution.runner import resolve_recordings
from macaw.logging_setup import get_logger
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.qc.checks import RecordingResults, run_checks, write_candidates

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

    keypoints = results.keypoints()
    if keypoints.empty:
        logger.info("%s: no keypoints, nothing to render", recording.recording_id)
        return
    # frame numbers come from the same clock pose.keypoints used
    apply_prep_timing(resolved, recording)
    cfg = resolved.qc
    sample = review_sample(
        candidates,
        (float(keypoints["t"].min()), float(keypoints["t"].max())),
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
    interactive: bool = typer.Option(True, "--interactive/--accept-votes"),
) -> None:
    """Three-source speaker identity vote, then confirm each recording interactively."""
    raise NotImplementedError
