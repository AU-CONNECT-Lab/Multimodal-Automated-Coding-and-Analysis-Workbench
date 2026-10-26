from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import macaw.extractors as registry
from macaw.backend_template import scaffold_backend
from macaw.cli.main import app
from macaw.config.schema import PoseSection, ProsodySection, TextSection
from macaw.exceptions import BackendNotAvailableError, MacawError
from macaw.extractors.video.pose.backends import BACKEND_GROUP, BACKEND_KEYPOINT_SCHEMA, PoseBackend, resolve_backend
from macaw.store.schema import validate_frame

NAME = "mylab.csv_pose"


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_BACKENDS", {group: dict(names) for group, names in registry._BACKENDS.items()})
    root = scaffold_backend(NAME, tmp_path / "plugins")
    monkeypatch.syspath_prepend(str(root / "src"))
    yield root
    for module in ("mylab_csv_pose.backend", "mylab_csv_pose"):
        importlib.sys.modules.pop(module, None)


def load_module():
    return importlib.import_module("mylab_csv_pose.backend")


class TestBackendFieldsTakeAnyName:
    """Built-in names are not the only choice; an unknown one fails when it is resolved, listing what exists."""

    def test_a_plugin_name_is_accepted_by_the_config(self):
        assert PoseSection(backend=NAME).backend == NAME
        assert TextSection(asr_backend="mylab.asr").asr_backend == "mylab.asr"
        assert ProsodySection(backend="mylab.pitch").backend == "mylab.pitch"

    def test_an_unknown_name_is_refused_when_resolved_with_the_choices(self):
        with pytest.raises(BackendNotAvailableError, match="yolo"):
            resolve_backend("mylab.nothing")


class TestScaffold:
    def test_writes_a_package(self, package):
        written = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
        assert written == {
            "pyproject.toml",
            "README.md",
            "src/mylab_csv_pose/__init__.py",
            "src/mylab_csv_pose/backend.py",
            "tests/test_backend.py",
        }

    def test_the_entry_point_names_the_module(self, package):
        text = (package / "pyproject.toml").read_text(encoding="utf-8")
        assert '[project.entry-points."macaw.backends"]' in text
        assert '"mylab.csv_pose" = "mylab_csv_pose.backend"' in text

    def test_refuses_to_overwrite(self, package):
        with pytest.raises(MacawError, match="already exists"):
            scaffold_backend(NAME, package.parent)

    def test_importing_it_registers_a_pose_backend(self, package):
        load_module()
        cls = registry.get_backend(BACKEND_GROUP, NAME)
        assert issubclass(cls, PoseBackend) and cls.name == NAME

    def test_the_generated_tests_pass(self, package, tmp_path):
        tests = (package / "tests" / "test_backend.py").read_text(encoding="utf-8")
        namespace: dict = {}
        exec(compile(tests, "test_backend.py", "exec"), namespace)
        for name, function in namespace.items():
            if name.startswith("test_"):
                folder = tmp_path / name
                folder.mkdir()
                function(folder)

    def test_what_it_reads_satisfies_the_backend_contract(self, package, tmp_path):
        path = tmp_path / "clip.keypoints.csv"
        pd.DataFrame(
            [(0, 1, "nose", 10.0, 20.0, 0.9), (1, 2, "chin", 11.0, 21.0, 0.8)],
            columns=["frame", "track_id", "joint", "x", "y", "confidence"],
        ).to_csv(path, index=False)
        table = load_module().read_keypoints(path, min_confidence=0.3, frames=None)
        validate_frame(table, BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


class TestEntryPoints:
    def test_an_installed_backend_registers(self, package, monkeypatch):
        entry = SimpleNamespace(name=NAME, load=load_module)
        monkeypatch.setattr(registry, "entry_points", lambda group: [entry] if group == "macaw.backends" else [])
        assert registry.load_entry_point_backends() == 1
        assert NAME in registry.list_backends(BACKEND_GROUP)

    def test_a_broken_backend_is_skipped(self, monkeypatch):
        def broken():
            raise ImportError("no module named lab_model")

        monkeypatch.setattr(registry, "entry_points", lambda group: [SimpleNamespace(name="lab.x", load=broken)])
        assert registry.load_entry_point_backends() == 0


class TestCommand:
    def test_writes_the_package_and_prints_where(self, tmp_path):
        result = CliRunner().invoke(app, ["new-backend", "mylab.openpose_csv", "--dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert Path(result.output.strip()) == tmp_path / "macaw-mylab-openpose-csv"

    def test_a_name_without_a_namespace_is_refused(self, tmp_path):
        result = CliRunner().invoke(app, ["new-backend", "yolo", "--dir", str(tmp_path)])
        assert result.exit_code != 0
        assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kind, group, base",
    [
        ("asr", "asr", "macaw.extractors.audio.content.asr:ASRBackend"),
        ("prosody", "prosody", "macaw.extractors.audio.prosody.backends:ProsodyBackend"),
    ],
)
class TestOtherKinds:
    def test_it_registers_under_its_own_group_and_its_tests_pass(self, kind, group, base, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "_BACKENDS", {g: dict(names) for g, names in registry._BACKENDS.items()})
        name = f"mylab.{kind}_example"
        root = scaffold_backend(name, tmp_path / "plugins", kind=kind)
        monkeypatch.syspath_prepend(str(root / "src"))
        module = f"mylab_{kind}_example"
        try:
            importlib.import_module(f"{module}.backend")
            module_name, cls_name = base.split(":")
            assert issubclass(registry.get_backend(group, name), getattr(importlib.import_module(module_name), cls_name))
            namespace: dict = {}
            exec(compile((root / "tests" / "test_backend.py").read_text(encoding="utf-8"), "test_backend.py", "exec"), namespace)
            for test_name, function in namespace.items():
                if test_name.startswith("test_"):
                    folder = tmp_path / test_name
                    folder.mkdir()
                    function(*([folder] if function.__code__.co_argcount else []))
        finally:
            for loaded in (f"{module}.backend", module):
                importlib.sys.modules.pop(loaded, None)

    def test_the_command_takes_the_kind(self, kind, group, base, tmp_path):
        result = CliRunner().invoke(app, ["new-backend", f"mylab.{kind}_cmd", "--kind", kind, "--dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "register_backend" in (Path(result.output.strip()) / "src" / f"mylab_{kind}_cmd" / "backend.py").read_text(encoding="utf-8")


def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(MacawError, match="pose, asr, prosody"):
        scaffold_backend("mylab.x", tmp_path, kind="face")
