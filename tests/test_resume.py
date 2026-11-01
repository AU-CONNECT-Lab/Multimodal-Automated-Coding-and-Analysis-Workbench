from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.execution import runner as runner_module
from macaw.exceptions import MacawError
from macaw.execution.dag import build_dag
from macaw.execution.lock import run_lock
from macaw.execution.runner import _write_stamp, completed_on_disk, task_states
from macaw.extractors.base import Phase
from macaw.project.layout import layout_for
from macaw.store import cache as cache_store


@pytest.fixture
def line(tmp_path, monkeypatch):
    """first -> second, both finished on disk under the key they were made with."""
    import macaw.extractors as extractors

    source = tmp_path / "input.txt"
    source.write_text("one", encoding="utf-8")

    def step(name, requires, phase):
        return type(name, (), {
            "name": name, "requires": requires, "phase": phase, "config_sections": (), "version": "0.1.0",
            "dependencies": classmethod(lambda cls, config: cls.requires), "runs_after": (), "wanted_by_default": classmethod(lambda cls, config: True),
            "__init__": lambda self, config: None,
            "inputs_for": lambda self, recording: [source],
        })

    registry = {"first": step("first", (), Phase.PREP), "second": step("second", ("first",), Phase.EXTRACT)}
    monkeypatch.setattr(extractors, "_REGISTRY", registry)
    monkeypatch.setattr(extractors, "_builtins_loaded", True)
    monkeypatch.setattr(
        runner_module, "build_recording", lambda recording_id, config: SimpleNamespace(recording_id=recording_id)
    )

    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    config = resolve_config(project)
    layout = layout_for(config)
    for name in registry:
        output = layout.long_table_path("rec1") / f"{name}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rows")
        key = cache_store.make_key(extractor=name, extractor_version="0.1.0", inputs=[source], config_hash="c")
        _write_stamp(layout.completion_path("rec1", name), key)
    monkeypatch.setattr(runner_module, "hash_extractor_config", lambda config, name: "c")

    def current():
        return completed_on_disk(config, build_dag(config, ["second"]), ["rec1"])

    return SimpleNamespace(config=config, layout=layout, source=source, registry=registry, current=current)


class TestResume:
    def test_outputs_made_from_what_is_there_now_are_kept(self, line):
        assert line.current() == {"rec1:first", "rec1:second"}

    def test_outputs_without_a_stamp_run_again(self, line):
        line.layout.completion_path("rec1", "second").unlink()
        assert line.current() == {"rec1:first"}

    def test_a_new_version_runs_again(self, line):
        line.registry["second"].version = "0.2.0"
        assert line.current() == {"rec1:first"}

    def test_a_changed_input_runs_again(self, line):
        line.source.write_text("two", encoding="utf-8")
        assert line.current() == set()

    def test_what_follows_a_step_that_runs_again_runs_again_too(self, line):
        line.registry["first"].version = "0.2.0"
        assert line.current() == set()

    def test_a_key_that_cannot_be_worked_out_runs_again(self, line):
        line.source.unlink()
        assert line.current() == set()

    def test_each_task_says_how_a_resumed_run_finds_it(self, line):
        dag = build_dag(line.config, ["second"])
        assert task_states(line.config, dag, ["rec1"]) == {"rec1:first": "current", "rec1:second": "current"}
        line.registry["second"].version = "0.2.0"
        assert task_states(line.config, dag, ["rec1"]) == {"rec1:first": "current", "rec1:second": "stale"}
        assert task_states(line.config, dag, ["rec2"]) == {"rec2:first": "new", "rec2:second": "new"}


class TestRunLock:
    def test_a_second_run_on_the_same_output_is_refused(self, tmp_path):
        with run_lock(tmp_path):
            with pytest.raises(MacawError, match="another macaw run"):
                with run_lock(tmp_path):
                    pass

    def test_the_lock_is_free_again_once_the_run_ends(self, tmp_path):
        with run_lock(tmp_path):
            pass
        with run_lock(tmp_path):
            pass


def test_a_recording_with_its_own_settings_is_judged_by_them(tmp_path, monkeypatch):
    """A recording's own file changes the settings its tasks hash, so a resumed run must hash them the same way."""
    from macaw.config.hashing import hash_extractor_config
    from macaw.execution.runner import recording_configs

    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    (tmp_path / "recordings").mkdir()
    (tmp_path / "recordings" / "rec1.yaml").write_text("segment:\n  task_start_s: 40.0\n", encoding="utf-8")
    config = resolve_config(project)
    monkeypatch.setattr(runner_module, "build_recording", lambda recording_id, config: SimpleNamespace(recording_id=recording_id))
    own = recording_configs(config, ["rec1", "rec2"])
    assert own["rec1"].segment.task_start_s == 40.0 and own["rec2"] is config
    assert own["rec1"] == resolve_config(project, recording_id="rec1")
    assert hash_extractor_config(own["rec1"], "prep.media") != hash_extractor_config(config, "prep.media")
    assert not list(layout_for(config).output_dir.glob(".*"))


def test_a_recording_file_may_not_need_a_step_the_project_plan_lacks(tmp_path):
    from macaw.exceptions import ConfigError
    from macaw.execution.runner import check_recording_steps

    project = tmp_path / "project.yaml"
    scaffold_project_config(project, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    own = tmp_path / "recordings" / "rec1.yaml"
    own.parent.mkdir()
    own.write_text("gaze:\n  targets: [bottle]\n", encoding="utf-8")
    config = resolve_config(project)
    with pytest.raises(ConfigError, match="gaze.objects.*project.yaml"):
        check_recording_steps(config, build_dag(config, ["gaze.target"]), ["rec1", "rec2"], ["gaze.target"])

    project.write_text(project.read_text(encoding="utf-8") + "gaze:\n  targets: [bottle, cup]\n", encoding="utf-8")
    config = resolve_config(project)
    check_recording_steps(config, build_dag(config, ["gaze.target"]), ["rec1", "rec2"], ["gaze.target"])
    own.write_text("gaze:\n  targets: []\n", encoding="utf-8")
    check_recording_steps(config, build_dag(config, ["gaze.target"]), ["rec1", "rec2"], ["gaze.target"])
