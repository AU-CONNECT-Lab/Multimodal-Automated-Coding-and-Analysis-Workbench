"""Slurm job arrays written out for a person to submit with sbatch; nothing here submits or watches jobs."""

from __future__ import annotations

import re
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from macaw.config.hpc import HPCConfig, ResourceProfile, resolve_profile
from macaw.config.schema import MacawConfig
from macaw.exceptions import ConfigValidationError
from macaw.execution.array_map import array_spec, build_map, write_map
from macaw.execution.dag import build_dag
from macaw.extractors import get_extractor
from macaw.project.layout import layout_for, new_execution_id

JUDGE = "coding.judge"
MAP_NAME = "recordings.tsv"
# the tier the closing job asks for; it only merges files and counts
FINISH_PROFILE = "data_manage"

_MEMORY = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)b?\s*$", re.IGNORECASE)


@dataclass
class Segment:
    """Consecutive steps run on the same kind of node, one array element per recording."""

    extractors: list[str]
    gpu: bool
    profile: str = ""


@dataclass
class Jobs:
    directory: Path | None
    segments: list[Segment]
    recording_ids: list[str]
    notes: list[str] = field(default_factory=list)


def needs_gpu(profile: ResourceProfile) -> bool:
    return bool(profile.gres) and "gpu" in profile.gres


def _megabytes(text: str | None) -> float:
    match = _MEMORY.match(text or "")
    if match is None:
        return 0.0
    # Slurm reads a bare number as megabytes
    return float(match[1]) * {"": 1, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 ** 2}[match[2].lower()]


def _seconds(text: str | None) -> int:
    """Slurm's time formats: minutes, minutes:seconds, hours:minutes:seconds, days-hours[:minutes[:seconds]]."""
    if not text:
        return 0
    days, _, rest = text.rpartition("-")
    parts = [int(part) for part in rest.split(":") if part.strip().isdigit()]
    if days:
        hours, minutes, seconds = (parts + [0, 0, 0])[:3]
    else:
        hours, minutes, seconds = {1: (0, *parts, 0), 2: (0, *parts), 3: tuple(parts)}.get(len(parts), (0, 0, 0))
    return ((int(days or 0) * 24 + hours) * 60 + minutes) * 60 + seconds


def _weight(profile: ResourceProfile) -> tuple:
    return (needs_gpu(profile), _megabytes(profile.memory), _seconds(profile.time), profile.cpus_per_task)


def segments(config: MacawConfig, hpc: HPCConfig, dag, steps: list[str]) -> list[Segment]:
    """GPU and CPU steps grouped into as few runs as the dependencies allow, so a recording queues a few times rather
    than once per step. Each run asks for the heaviest tier among its steps."""
    tiers = {name: get_extractor(name).resources(config) for name in steps}
    gpu = {name: needs_gpu(resolve_profile(hpc, tiers[name])) for name in steps}
    needs = {name: dag.ancestors(name) & set(steps) for name in steps}
    order = {name: index for index, name in enumerate(steps)}
    left = set(steps)
    found: list[Segment] = []
    while left:
        ready = sorted((name for name in left if not needs[name] & left), key=order.get)
        same = [name for name in ready if found and gpu[name] == found[-1].gpu]
        if not same:
            # CPU first when both are ready, since the GPU steps mostly read what the CPU ones make
            kind = all(gpu[name] for name in ready)
            found.append(Segment(extractors=[], gpu=kind))
            same = [name for name in ready if gpu[name] == kind]
        found[-1].extractors.extend(same)
        left -= set(same)
    for segment in found:
        segment.profile = max((tiers[name] for name in segment.extractors), key=lambda tier: _weight(resolve_profile(hpc, tier)))
    return found


def sbatch_flags(profile: ResourceProfile) -> list[str]:
    values = (
        ("partition", profile.partition), ("account", profile.account), ("nodes", profile.nodes), ("ntasks", profile.ntasks),
        ("cpus-per-task", profile.cpus_per_task), ("mem", profile.memory), ("mem-per-cpu", profile.memory_per_cpu),
        ("time", profile.time), ("gres", profile.gres),
    )
    return [f"--{option}={value}" for option, value in values if value is not None] + list(profile.additional_args)


def plan(config: MacawConfig, hpc: HPCConfig, recording_ids: list[str], wanted: list[str]) -> Jobs:
    """The segments with anything left to do, the recordings they run on, and notes for whoever submits them."""
    from macaw.coding.llm.backends import resolve_llm_backend
    from macaw.execution.runner import task_id_for, task_states

    dag = build_dag(config, wanted or None)
    gates = set(dag.interactive_gates())
    steps = [name for name in dag.topological_order() if name not in gates]
    notes = []
    if JUDGE in steps:
        llm = config.coding.llm
        if not llm.is_local():
            left_out = {JUDGE} | dag.descendants(JUDGE)
            steps = [name for name in steps if name not in left_out]
            notes.append(
                f"{JUDGE} left out: {llm.provider} is reached over the internet, which compute nodes usually lack; "
                f"run `macaw run -e {JUDGE}` where it can be reached"
            )
        elif not hpc.llm_server:
            notes.append(
                f"{JUDGE} expects a model server at {resolve_llm_backend(llm).base_url}, and hpc.yaml llm_server "
                "starts none in the job"
            )

    states = task_states(config, dag, recording_ids)
    todo = {
        (recording_id, name) for recording_id in recording_ids for name in steps
        if states.get(task_id_for(recording_id, name)) in ("stale", "new")
    }
    held = sorted({
        recording_id for recording_id in recording_ids for name in steps
        if states.get(task_id_for(recording_id, name)) == "held"
    })
    if held:
        notes.append(
            f"{len(held)} recording(s) wait for a person before some steps ({', '.join(held[:5])}"
            f"{', ...' if len(held) > 5 else ''}); `macaw run --dry-run` says what, and generating again once they "
            "are cleared picks them up"
        )
    if hpc.hf_home is None:
        notes.append("hpc.yaml hf_home is empty, so the jobs use the default model cache; `macaw hpc prefetch-models` fills it")

    kept = [
        segment for segment in segments(config, hpc, dag, steps)
        if any((recording_id, name) in todo for recording_id in recording_ids for name in segment.extractors)
    ]
    return Jobs(directory=None, segments=kept, recording_ids=sorted({recording_id for recording_id, _ in todo}), notes=notes)


def generate(config: MacawConfig, hpc: HPCConfig, recording_ids: list[str], wanted: list[str]) -> Jobs:
    """Writes the job folder under the work directory, unless nothing is left to run."""
    from macaw.config.loader import archive_resolved

    jobs = plan(config, hpc, recording_ids, wanted)
    if not jobs.segments:
        return jobs
    layout = layout_for(config)
    execution_id = new_execution_id()
    directory = layout.work_dir / "hpc" / execution_id
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    # the jobs read this copy, so a local run that archives a new config meanwhile changes nothing for them
    config_path = Path(shutil.copyfile(archive_resolved(config, layout.output_dir), directory / "config.yaml"))
    mapping = build_map(jobs.recording_ids)
    map_path = write_map(mapping, directory / MAP_NAME)

    run = ["--config", _quote(config_path), "--map", _quote(map_path), "--execution-id", execution_id]
    total = len(jobs.segments)
    for number, segment in enumerate(jobs.segments, start=1):
        _write(directory / f"segment_{number}.sh", segment_script(config, hpc, segment, f"{execution_id}, step {number} of {total}", run))
    steps = [name for segment in jobs.segments for name in segment.extractors]
    _write(directory / "finish.sh", finish_script(hpc, steps, execution_id, run))
    _write(directory / "submit.sh", submit_script(config, hpc, jobs.segments, mapping))
    jobs.directory = directory
    return jobs


def _write(path: Path, text: str) -> None:
    # bash on the cluster would take a carriage return for part of the command
    path.write_text(text, encoding="utf-8", newline="\n")


def _quote(value) -> str:
    return shlex.quote(Path(value).as_posix() if isinstance(value, Path) else str(value))


def _preamble(hpc: HPCConfig, comment: str) -> list[str]:
    lines = ["#!/bin/bash", f"# {comment}", "set -eo pipefail", *hpc.module_load, *hpc.env_activate]
    if hpc.hf_home is not None:
        lines.append(f"export HF_HOME={_quote(hpc.hf_home)}")
    return lines


def segment_script(config: MacawConfig, hpc: HPCConfig, segment: Segment, comment: str, run: list[str]) -> str:
    lines = _preamble(hpc, f"{comment}: {' '.join(segment.extractors)}")
    if JUDGE in segment.extractors and hpc.llm_server:
        from macaw.coding.llm.backends import resolve_llm_backend

        url = f"{resolve_llm_backend(config.coding.llm).base_url}/models"
        lines += [
            "# the model server lives only as long as this job",
            "trap 'kill $(jobs -p) 2>/dev/null || true' EXIT",
            *hpc.llm_server,
            f"for _ in $(seq {hpc.llm_ready_timeout_s}); do",
            f"  python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)' {_quote(url)} 2>/dev/null && break",
            "  sleep 1",
            "done",
        ]
    lines.append(" ".join(["python -m macaw hpc task", *run, *(f"-e {name}" for name in segment.extractors)]))
    return "\n".join(lines) + "\n"


def finish_script(hpc: HPCConfig, steps: list[str], execution_id: str, run: list[str]) -> str:
    lines = _preamble(hpc, f"{execution_id}: merges what the arrays wrote and lists what is left")
    lines.append(" ".join(["python -m macaw hpc finish", *run, *(f"-e {name}" for name in steps)]))
    return "\n".join(lines) + "\n"


def submit_script(config: MacawConfig, hpc: HPCConfig, found: list[Segment], mapping: dict[str, int]) -> str:
    prefix = config.execution.job_name_prefix
    lines = ["#!/bin/bash", "# submit from a login node: bash submit.sh", "set -eo pipefail", 'cd "$(dirname "$0")"']
    jobs = []
    for number, segment in enumerate(found, start=1):
        profile = resolve_profile(hpc, segment.profile)
        flags = [
            f"--job-name={prefix}-{number}", f"--array={array_spec(mapping, max_concurrent=profile.array_limit)}",
            f"--output=logs/{number}_%A_%a.out", *sbatch_flags(profile),
        ]
        # element N waits for element N of the step before, so each recording moves on by itself;
        # a recording that failed there is dropped here rather than left pending
        after = f" --dependency=aftercorr:${jobs[-1]} --kill-on-invalid-dep=yes" if jobs else ""
        lines.append(f"job{number}=$(sbatch --parsable{after} {' '.join(map(_quote, flags))} segment_{number}.sh)")
        lines.append(f"job{number}=${{job{number}%%;*}}")
        jobs.append(f"job{number}")
    try:
        closing = resolve_profile(hpc, FINISH_PROFILE)
    except ConfigValidationError:
        closing = hpc.defaults
    flags = [f"--job-name={prefix}-finish", "--output=logs/finish_%j.out", *sbatch_flags(closing.model_copy(update={"gres": None}))]
    waits = ":".join(f"${job}" for job in jobs)
    lines.append(f"sbatch --parsable --dependency=afterany:{waits} {' '.join(map(_quote, flags))} finish.sh > /dev/null")
    lines.append(f'echo "submitted {" ".join(f"${job}" for job in jobs)}; once they are done, logs/finish_*.out says what ran"')
    return "\n".join(lines) + "\n"
