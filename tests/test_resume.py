from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.execution import runner as runner_module
from macaw.execution.dag import build_dag
from macaw.execution.runner import _write_stamp, completed_on_disk
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
