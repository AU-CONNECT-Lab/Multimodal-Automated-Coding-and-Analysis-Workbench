from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app
from macaw.config.hpc import HPCConfig, ResourceProfile, load_hpc_config
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.execution import runner as runner_module
from macaw.execution import slurm
from macaw.execution.dag import build_dag

from conftest import SAMPLE_RATE

HPC = {
    "defaults": {"partition": "batch"},
    "resource_profiles": {
        "data_manage": {"memory": "2gb", "time": "00:20:00", "cpus_per_task": 2},
        "light_short": {"memory": "16gb", "time": "04:00:00", "cpus_per_task": 4},
        "standard": {"memory": "32gb", "time": "20:00:00", "cpus_per_task": 8, "array_limit": 10},
        "gpu_standard": {"partition": "gpu_p", "memory": "32gb", "time": "12:00:00", "gres": "gpu:L4:1", "array_limit": 4},
    },
    "module_load": ["ml FFmpeg"],
    "env_activate": ["source ~/env/bin/activate"],
    "hf_home": "/shared/hf",
}


@pytest.fixture
def project(tmp_path):
    """Two recordings of a parent and a child, each with their own microphone."""
    for recording in ("003", "004"):
        raw = tmp_path / "raw" / recording
        raw.mkdir(parents=True)
        for who in ("parent", "child"):
            (raw / f"{recording}_{who}.wav").write_bytes(b"")
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    hpc = tmp_path / "hpc.yaml"
    hpc.write_text(yaml.safe_dump(HPC), encoding="utf-8")

    def resolved(*overrides):
        return resolve_config(config, cli_overrides=list(overrides))

    return SimpleNamespace(config=config, hpc=lambda: load_hpc_config(None, config), resolved=resolved, root=tmp_path)


class TestSlurmValues:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [("90", 5400), ("05:30", 330), ("12:00:00", 43200), ("2-00", 172800), ("1-02:30", 95400), ("1-00:00:10", 86410), (None, 0)],
    )
    def test_every_time_format_slurm_takes(self, text, seconds):
        assert slurm._seconds(text) == seconds

    @pytest.mark.parametrize(("text", "megabytes"), [("2gb", 2048), ("32G", 32768), ("500mb", 500), ("4000", 4000), ("1t", 1048576)])
    def test_memory_in_slurm_units(self, text, megabytes):
        assert slurm._megabytes(text) == megabytes

    def test_flags_come_from_the_profile_and_its_extras(self):
        profile = ResourceProfile(partition="gpu_p", cpus_per_task=8, memory="32gb", time="12:00:00", gres="gpu:L4:1", additional_args=["--qos=short"])
        assert slurm.sbatch_flags(profile) == [
            "--partition=gpu_p", "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=32gb", "--time=12:00:00", "--gres=gpu:L4:1", "--qos=short",
        ]


def test_steps_on_the_same_kind_of_node_share_a_job(project):
    config = project.resolved("text.diarization=pyannote")
    dag = build_dag(config, ["turns.entropy", "prosody.frame"])
    steps = [name for name in dag.topological_order() if name not in dag.interactive_gates()]
    found = slurm.segments(config, project.hpc(), dag, steps)
    assert [(segment.gpu, segment.profile, segment.extractors) for segment in found] == [
        (False, "data_manage", ["prep.media"]),
        (True, "gpu_standard", ["text.diarize", "text.asr"]),
        (False, "standard", ["prosody.frame", "turns.structure", "turns.transitions", "turns.entropy"]),
    ]


def states_where(new: set[str], held: set[str] = frozenset()):
    def task_states(config, dag, recording_ids):
        return {
            f"{recording_id}:{name}": "new" if f"{recording_id}:{name}" in new else "held" if f"{recording_id}:{name}" in held else "current"
            for recording_id in recording_ids for name in dag.nodes
        }
    return task_states


class TestPlan:
    def test_only_recordings_and_steps_with_work_left_are_submitted(self, project, monkeypatch):
        monkeypatch.setattr(runner_module, "task_states", states_where({"004:text.asr", "004:turns.structure"}))
        jobs = slurm.plan(project.resolved(), project.hpc(), ["003", "004"], ["turns.structure"])
        assert jobs.recording_ids == ["004"]
        assert [segment.extractors for segment in jobs.segments] == [["text.asr"], ["turns.structure"]]

    def test_nothing_left_writes_nothing(self, project, monkeypatch):
        monkeypatch.setattr(runner_module, "task_states", states_where(set()))
        jobs = slurm.generate(project.resolved(), project.hpc(), ["003", "004"], ["turns.structure"])
        assert jobs.segments == [] and jobs.directory is None

    def test_a_recording_waiting_for_a_person_is_named(self, project, monkeypatch):
        monkeypatch.setattr(runner_module, "task_states", states_where({"004:prep.media"}, held={"003:turns.structure"}))
        jobs = slurm.plan(project.resolved(), project.hpc(), ["003", "004"], ["turns.structure"])
        assert jobs.recording_ids == ["004"] and any("1 recording(s) wait for a person" in note for note in jobs.notes)

    def test_a_cloud_language_model_stays_off_the_cluster(self, project, monkeypatch):
        monkeypatch.setattr(runner_module, "task_states", states_where({"003:coding.judge", "003:turns.structure"}))
        config = project.resolved("coding.layers=[A, B]", "coding.llm.provider=openai")
        jobs = slurm.plan(config, project.hpc(), ["003"], ["coding.judge"])
        assert all("coding.judge" not in segment.extractors for segment in jobs.segments)
        assert any("coding.judge left out" in note for note in jobs.notes)


class TestScripts:
    @pytest.fixture
    def written(self, project, monkeypatch):
        monkeypatch.setattr(runner_module, "task_states", states_where({"003:prep.media", "004:prep.media", "004:coding.judge"}))
        hpc = project.hpc().model_copy(update={"llm_server": ["ollama serve > ollama.log 2>&1 &"]})
        config = project.resolved("text.diarization=pyannote", "coding.layers=[A, B]", "coding.llm.model=qwen")
        jobs = slurm.generate(config, hpc, ["003", "004"], ["prep.media", "coding.judge"])
        return SimpleNamespace(jobs=jobs, text={path.name: path.read_bytes().decode() for path in jobs.directory.glob("*.sh")})

    def test_each_step_waits_for_the_same_recording_in_the_step_before(self, written):
        submit = written.text["submit.sh"]
        assert "job1=$(sbatch --parsable --job-name=macaw-1 --array=0-1 " in submit
        assert "--dependency=aftercorr:$job1 --kill-on-invalid-dep=yes" in submit
        assert "--array=0-1%4" in submit and "--gres=gpu:L4:1" in submit and "--partition=gpu_p" in submit
        last = len(written.jobs.segments)
        assert f"--dependency=afterany:{':'.join(f'$job{n}' for n in range(1, last + 1))}" in submit

    def test_a_job_sets_up_its_environment_then_runs_its_steps(self, written):
        first = written.text["segment_1.sh"].splitlines()
        assert first[:2] == ["#!/bin/bash", f"# {written.jobs.directory.name}, step 1 of {len(written.jobs.segments)}: prep.media"]
        assert first[3:6] == ["ml FFmpeg", "source ~/env/bin/activate", "export HF_HOME=/shared/hf"]
        assert first[-1].startswith("python -m macaw hpc task --config ") and first[-1].endswith("-e prep.media")

    def test_the_model_server_starts_only_where_the_judge_runs(self, written):
        judged = [name for name, text in written.text.items() if "-e coding.judge" in text and name.startswith("segment")]
        assert len(judged) == 1
        assert "ollama serve > ollama.log 2>&1 &" in written.text[judged[0]] and "http://localhost:11434/v1/models" in written.text[judged[0]]
        assert all("ollama" not in text for name, text in written.text.items() if name not in judged)

    def test_scripts_are_plain_bash(self, written):
        assert all("\r" not in text for text in written.text.values())
        if shutil.which("bash") is None:
            pytest.skip("no bash here")
        for path in written.jobs.directory.glob("*.sh"):
            assert subprocess.run(["bash", "-n", path.as_posix()], capture_output=True).returncode == 0, path.name


def test_an_element_without_its_index_refuses(project):
    result = CliRunner().invoke(app, ["hpc", "task", "-c", str(project.config), "--map", "m.tsv", "--execution-id", "x", "-e", "prep.media"])
    assert result.exit_code != 0 and "SLURM_ARRAY_TASK_ID" in str(result.exception)


def test_the_arrays_run_each_recording_through_and_a_second_generate_finds_nothing(project, write_wav, monkeypatch):
    from macaw.exceptions import ExternalToolError
    from macaw.media.probe import require_ffmpeg

    try:
        require_ffmpeg()
    except ExternalToolError:
        pytest.skip("needs ffmpeg")
    t = np.arange(4 * SAMPLE_RATE) / SAMPLE_RATE
    for recording in ("003", "004"):
        for who, pitch in (("parent", 180.0), ("child", 300.0)):
            write_wav(project.root / "raw" / recording / f"{recording}_{who}.wav", 0.3 * np.sin(2 * np.pi * pitch * t))
    # no person in the loop here; what waits for one is TestPlan's business
    data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
    data.update(align={"method": "none"}, identity={"confirm": "skip"})
    project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
    # prep.media on a GPU tier, so the run is two chained arrays
    hpc = yaml.safe_load(yaml.safe_dump(HPC))
    hpc["resource_profiles"]["data_manage"]["gres"] = "gpu:1"
    (project.root / "hpc.yaml").write_text(yaml.safe_dump(hpc), encoding="utf-8")

    invoke = lambda *args: CliRunner().invoke(app, list(args), catch_exceptions=False)  # noqa: E731
    result = invoke("hpc", "generate", "-c", str(project.config), "-e", "prosody.frame")
    assert result.exit_code == 0, result.output
    assert "1  gpu  data_manage  prep.media" in result.output and "2  cpu  " in result.output
    directory = Path(result.output.split("bash ")[-1].strip()).parent
    for number in (1, 2):
        command = shlex.split((directory / f"segment_{number}.sh").read_text(encoding="utf-8").splitlines()[-1])[3:]
        for index in ("0", "1"):
            monkeypatch.setenv("SLURM_ARRAY_TASK_ID", index)
            outcome = invoke(*command)
            assert outcome.exit_code == 0 and "done 1  " in outcome.output and "failed 0" in outcome.output, outcome.output
    closing = invoke(*shlex.split((directory / "finish.sh").read_text(encoding="utf-8").splitlines()[-1])[3:])
    assert "failed 0" in closing.output and (project.resolved().project.output_dir / "manifest.json").is_file()
    assert "nothing left to run" in invoke("hpc", "generate", "-c", str(project.config), "-e", "prosody.frame").output


def test_prefetch_loads_each_model_once_into_the_shared_cache(project, monkeypatch):
    import os

    from macaw.extractors.audio.content.asr import ASRExtractor
    from macaw.extractors.prep import MediaPrepExtractor

    monkeypatch.setenv("HF_HOME", "elsewhere")
    calls = []
    monkeypatch.setattr(MediaPrepExtractor, "check_available", lambda self: None)
    monkeypatch.setattr(ASRExtractor, "check_available", lambda self: None)
    monkeypatch.setattr(ASRExtractor, "prefetch", lambda self: calls.append((os.environ["HF_HOME"], self.config.text.device)))
    result = CliRunner().invoke(app, ["hpc", "prefetch-models", "-c", str(project.config), "-e", "text.asr"])
    assert result.exit_code == 0, result.output
    assert calls == [(str(Path("/shared/hf")), "cpu")] and result.output.splitlines() == ["text.asr: fetched"]
