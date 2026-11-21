from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw import extractors, project_scripts
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.exceptions import ConfigError
from macaw.execution.runner import _make_key
from macaw.project_scripts import load_project_scripts, code_file_of, script_function

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
    monkeypatch.setattr(project_scripts, "_read_at", {})
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
    assert code_file_of(cls) == script.resolve()

    recording = SimpleNamespace(recording_id="r1", subjects=[])
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
    assert code_file_of(extractors.get_extractor("pose.kinematics")) is None
    assert "already taken" in project_scripts._failed[(tmp_path / "clash.py").resolve()]
    assert script_function(tmp_path, "fine.gap")(None, None) == 2


def test_a_built_in_is_not_a_script():
    assert code_file_of(extractors.get_extractor("pose.kinematics")) is None


def test_reading_a_project_loads_its_scripts_folder(tmp_path):
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out", roles=("parent", "child"))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "scripted.py").write_text(EXTRACTOR, encoding="utf-8")
    resolved = resolve_config(config)
    assert resolved.project.scripts_dir == (tmp_path / "scripts").resolve()
    assert "lab.scripted" in extractors.list_extractors()


READER = """
from pathlib import Path

from macaw.extractors import register_backend
from macaw.extractors.base import Backend

KEPT = Path(__file__).with_name("kept.csv")


@register_backend("lab")
class Reader(Backend):
    name = "reader"
    version = "0.1.0"

    def check_available(self):
        return None

    def inputs_for(self, recording):
        return [KEPT] if KEPT.is_file() else []
"""


class Backed(extractors.Extractor):
    name, version = "lab.backed", "0.1.0"

    def check_available(self):
        return None

    def inputs_for(self, recording):
        return []

    def backends(self):
        return [extractors.get_backend("lab", "reader")()]

    def run(self, recording, workspace):
        return None


def _backed_key():
    return _make_key(None, Backed, Backed(None), SimpleNamespace(recording_id="r1", subjects=[]), "config").as_dirname()


def _load_reader(tmp_path, source):
    (tmp_path / "reader.py").write_text(source, encoding="utf-8")
    project_scripts._loaded.clear()
    extractors._BACKENDS.get("lab", {}).clear()
    load_project_scripts(tmp_path)


def test_a_backends_version_code_and_files_are_part_of_the_cache_key(tmp_path, monkeypatch):
    monkeypatch.setattr(extractors, "_BACKENDS", {group: dict(names) for group, names in extractors._BACKENDS.items()})
    _load_reader(tmp_path, READER)
    first = _backed_key()
    monkeypatch.setattr(extractors.get_backend("lab", "reader"), "version", "0.2.0")
    assert _backed_key() != first
    monkeypatch.undo()
    monkeypatch.setattr(extractors, "_BACKENDS", {group: dict(names) for group, names in extractors._BACKENDS.items()})
    _load_reader(tmp_path, READER)
    assert _backed_key() == first
    _load_reader(tmp_path, READER + "\n# changed\n")
    edited = _backed_key()
    assert edited != first
    (tmp_path / "kept.csv").write_text("frame,x\n0,1\n", encoding="utf-8")
    assert _backed_key() != edited


def test_a_step_whose_backend_cannot_be_had_still_has_a_key_and_a_different_one():
    from macaw.exceptions import BackendNotAvailableError

    class Missing(extractors.Extractor):
        name, version = "lab.missing", "0.1.0"

        def check_available(self):
            return None

        def inputs_for(self, recording):
            return []

        def backends(self):
            raise BackendNotAvailableError("not here")

        def run(self, recording, workspace):
            return None

    class Plain(Missing):
        def backends(self):
            return []

    recording = SimpleNamespace(recording_id="r1", subjects=[])
    assert _make_key(None, Missing, Missing(None), recording, "c").as_dirname() != _make_key(None, Plain, Plain(None), recording, "c").as_dirname()


def test_the_built_in_steps_name_their_backend_without_loading_it(monkeypatch):
    from macaw.config.schema import MacawConfig
    from macaw.extractors.video.pose import backends as pose_backends

    def refuse(self):
        raise AssertionError("checked")

    monkeypatch.setattr(pose_backends.MediaPipeBackend, "check_available", refuse)
    config = MacawConfig.model_validate({"project": {"name": "demo", "raw_dir": "raw", "output_dir": "out"}, "pose": {"backend": "mediapipe"}})
    [backend] = extractors.get_extractor("pose.detect")(config).backends()
    assert backend.name == "mediapipe" and backend.key_version() == backend.version


def test_an_llm_backend_keys_without_asking_its_server(monkeypatch):
    from macaw.coding.llm.backends import OllamaBackend
    from macaw.config.schema import LLMSection

    monkeypatch.setattr(OllamaBackend, "_served_version", lambda self: (_ for _ in ()).throw(AssertionError("asked")))
    assert OllamaBackend(LLMSection(provider="ollama", model="qwen3:8b")).key_version() == ""


def test_a_script_edited_while_the_interface_is_open_is_read_again(tmp_path):
    import os

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    measure = scripts / "lab.py"
    measure.write_text("def reach(:\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="failed to load"):
        script_function(scripts, "lab.reach")
    measure.write_text(EXTRACTOR + "\n\ndef reach(table):\n    return 1\n", encoding="utf-8")
    os.utime(measure, ns=(measure.stat().st_atime_ns, measure.stat().st_mtime_ns + 10**9))
    assert script_function(scripts, "lab.reach")(None) == 1
    first = extractors.get_extractor("lab.scripted")
    measure.write_text(EXTRACTOR + "\n\ndef reach(table):\n    return 2\n", encoding="utf-8")
    os.utime(measure, ns=(measure.stat().st_atime_ns, measure.stat().st_mtime_ns + 2 * 10**9))
    assert script_function(scripts, "lab.reach")(None) == 2 and extractors.get_extractor("lab.scripted") is not first
