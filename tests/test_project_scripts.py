from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw import extractors, project_scripts
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError
from macaw.execution.runner import _make_key
from macaw.project_scripts import load_project_scripts, script_file_of, script_function

EXTRACTOR = """
from macaw.extractors import Extractor, ExtractorResult, register_extractor


@register_extractor
class Scripted(Extractor):
    name = "lab.scripted"
    version = "0.1.0"

    def check_available(self):
        return None

    def inputs_for(self, recording):
        return []

    def run(self, recording, workspace):
        return ExtractorResult()
"""


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(project_scripts, "_loaded", {})
    monkeypatch.setattr(project_scripts, "_failed", {})
    extractors.load_builtin_extractors()
    monkeypatch.setattr(extractors, "_REGISTRY", dict(extractors._REGISTRY))


def test_a_function_is_found_by_file_and_name(tmp_path):
    (tmp_path / "lab.py").write_text("def gap(a, b):\n    return 1\n", encoding="utf-8")
    assert script_function(tmp_path, "lab.gap")(None, None) == 1


def test_a_broken_file_is_reported_where_it_is_used_and_the_others_still_load(tmp_path):
    (tmp_path / "broken.py").write_text("raise RuntimeError('typo')\n", encoding="utf-8")
    (tmp_path / "fine.py").write_text("def gap(a, b):\n    return 2\n", encoding="utf-8")
    load_project_scripts(tmp_path)
    assert script_function(tmp_path, "fine.gap")(None, None) == 2
    with pytest.raises(ConfigError, match="failed to load: RuntimeError: typo"):
        script_function(tmp_path, "broken.gap")


def test_a_missing_function_is_named(tmp_path):
    (tmp_path / "lab.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="has no function gap"):
        script_function(tmp_path, "lab.gap")


def test_no_folder_is_no_scripts(tmp_path):
    load_project_scripts(tmp_path / "absent")
    load_project_scripts(None)


def test_an_extractor_in_a_script_registers_and_its_file_is_part_of_the_cache_key(tmp_path):
    script = tmp_path / "scripted.py"
    script.write_text(EXTRACTOR, encoding="utf-8")
    load_project_scripts(tmp_path)
    cls = extractors.get_extractor("lab.scripted")
    assert script_file_of(cls) == script.resolve()

    recording = SimpleNamespace(recording_id="r1")
    before = _make_key(None, cls, cls(None), recording, "config").as_dirname()
    script.write_text(EXTRACTOR + "\n# changed\n", encoding="utf-8")
    assert _make_key(None, cls, cls(None), recording, "config").as_dirname() != before


def test_an_extractor_without_the_decorator_is_registered_too(tmp_path):
    (tmp_path / "plain.py").write_text(EXTRACTOR.replace("@register_extractor\n", "").replace("lab.scripted", "lab.plain"), encoding="utf-8")
    load_project_scripts(tmp_path)
    assert "lab.plain" in extractors.list_extractors()


def test_a_script_taking_a_built_in_name_fails_on_its_own(tmp_path):
    (tmp_path / "clash.py").write_text(EXTRACTOR.replace("lab.scripted", "pose.kinematics"), encoding="utf-8")
    (tmp_path / "fine.py").write_text("def gap(a, b):\n    return 2\n", encoding="utf-8")
    load_project_scripts(tmp_path)
    assert script_file_of(extractors.get_extractor("pose.kinematics")) is None
    assert "already taken" in project_scripts._failed[(tmp_path / "clash.py").resolve()]
    assert script_function(tmp_path, "fine.gap")(None, None) == 2


def test_a_built_in_is_not_a_script():
    assert script_file_of(extractors.get_extractor("pose.kinematics")) is None


def test_reading_a_project_loads_its_scripts_folder(tmp_path):
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "scripted.py").write_text(EXTRACTOR, encoding="utf-8")
    resolved = resolve_config(config)
    assert resolved.project.scripts_dir == (tmp_path / "scripts").resolve()
    assert "lab.scripted" in extractors.list_extractors()
