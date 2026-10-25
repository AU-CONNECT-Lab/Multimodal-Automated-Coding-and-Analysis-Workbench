from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import macaw.extractors as registry
from macaw.cli.main import app
from macaw.config.hashing import hash_section
from macaw.config.schema import MacawConfig
from macaw.exceptions import MacawError
from macaw.extractors.base import Extractor, Phase
from macaw.plugin_template import names_for, scaffold_extractor
from macaw.project.layout import make_layout
from macaw.store.longtable import build_long_rows, write_long_table
from macaw.store.schema import LONG_TABLE_SCHEMA, Scale, validate_frame

pytestmark = pytest.mark.skipif(
    not hasattr(MacawConfig, "model_validate"), reason="real pydantic required"
)

NAME = "mylab.f0_spread"


def config(tmp_path, plugins=None, **sections):
    return MacawConfig.model_validate(
        {
            "project": {"name": "demo", "raw_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "out")},
            "plugins": plugins or {},
            **sections,
        }
    )


class TestPluginParametersInTheCacheKey:
    """Plugins are keyed by extractor name, which has dots of its own."""

    def test_a_plugins_own_parameters_move_its_hash(self, tmp_path):
        before = hash_section(config(tmp_path, {NAME: {"quantiles": [0.1, 0.9]}}), f"plugins.{NAME}")
        after = hash_section(config(tmp_path, {NAME: {"quantiles": [0.2, 0.8]}}), f"plugins.{NAME}")
        assert before != after

    def test_another_plugins_parameters_do_not(self, tmp_path):
        mine = {NAME: {"quantiles": [0.1, 0.9]}}
        before = hash_section(config(tmp_path, mine), f"plugins.{NAME}")
        after = hash_section(config(tmp_path, {**mine, "other.thing": {"k": 1}}), f"plugins.{NAME}")
        assert before == after

    def test_an_unset_plugin_hashes_as_nothing_rather_than_failing(self, tmp_path):
        assert hash_section(config(tmp_path), f"plugins.{NAME}") == hash_section(config(tmp_path), "no.such")

    def test_ordinary_dotted_sections_still_resolve(self, tmp_path):
        """coding.llm must not be invalidated by the codebook, as before."""
        plain = hash_section(config(tmp_path), "coding.llm")
        changed = hash_section(config(tmp_path, coding={"codebook": "elsewhere.yaml"}), "coding.llm")
        assert plain == changed
        assert plain != hash_section(config(tmp_path, coding={"llm": {"model": "other"}}), "coding.llm")


class TestNames:
    def test_derived_names(self):
        assert names_for(NAME) == {
            "name": NAME,
            "module": "mylab_f0_spread",
            "distribution": "macaw-mylab-f0-spread",
            "cls": "MylabF0SpreadExtractor",
        }

    @pytest.mark.parametrize("bad", ["f0_spread", "MyLab.f0", "mylab.", "mylab..x", "my-lab.x", "1lab.x"])
    def test_a_name_without_a_namespace_or_with_odd_characters_is_refused(self, bad):
        with pytest.raises(MacawError):
            names_for(bad)


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = scaffold_extractor(NAME, tmp_path / "plugins", group="vocal")
    monkeypatch.syspath_prepend(str(root / "src"))
    yield root
    for module in ("mylab_f0_spread.extractor", "mylab_f0_spread"):
        importlib.sys.modules.pop(module, None)


def load_class(root):
    return importlib.import_module("mylab_f0_spread.extractor").MylabF0SpreadExtractor


class TestScaffold:
    def test_writes_a_package(self, package):
        expected = {
            "pyproject.toml",
            "README.md",
            "src/mylab_f0_spread/__init__.py",
            "src/mylab_f0_spread/extractor.py",
            "tests/test_extractor.py",
        }
        written = {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()}
        assert written == expected

    def test_the_entry_point_names_the_class(self, package):
        text = (package / "pyproject.toml").read_text(encoding="utf-8")
        assert '[project.entry-points."macaw.extractors"]' in text
        assert '"mylab.f0_spread" = "mylab_f0_spread.extractor:MylabF0SpreadExtractor"' in text

    def test_refuses_to_overwrite(self, package):
        with pytest.raises(MacawError, match="already exists"):
            scaffold_extractor(NAME, package.parent)

    def test_the_generated_class_is_a_derive_extractor(self, package):
        cls = load_class(package)
        assert issubclass(cls, Extractor)
        assert (cls.name, cls.group, cls.phase) == (NAME, "vocal", Phase.DERIVE)
        assert cls.config_sections == (f"plugins.{NAME}",)
        assert cls.requires == ("prosody.frame",)

    def test_the_generated_tests_pass(self, package):
        tests = (package / "tests" / "test_extractor.py").read_text(encoding="utf-8")
        namespace: dict = {}
        exec(compile(tests, "test_extractor.py", "exec"), namespace)
        for name, function in namespace.items():
            if name.startswith("test_"):
                function()

    def test_the_example_runs_on_real_tables(self, package, tmp_path):
        """Parameters from project.yaml override the defaults, and the rows are valid."""
        cfg = config(tmp_path, {NAME: {"quantiles": [0.25, 0.75]}})
        layout = make_layout(cfg.project.output_dir)
        f0 = build_long_rows(
            recording_id="r1", source="prosody.frame", scale=Scale.FRAME, subject_id="parent",
            values=pd.DataFrame({"t_start": np.arange(101) * 0.01, "f0_hz": np.arange(100.0, 201.0)}),
        )
        write_long_table(f0, layout.long_table_path("r1"))
        recording = SimpleNamespace(recording_id="r1", entities=lambda: {})

        result = load_class(package)(cfg).run(recording, tmp_path)
        validate_frame(result.rows, LONG_TABLE_SCHEMA, "long table")
        assert float(result.rows["value"].iloc[0]) == pytest.approx(50.0)
        assert result.specs[0].params == {"quantiles": [0.25, 0.75]}


class TestEntryPoints:
    @pytest.fixture
    def fresh_registry(self, monkeypatch):
        monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))

    def test_an_installed_plugin_registers(self, package, fresh_registry, monkeypatch):
        entry = SimpleNamespace(name=NAME, load=lambda: load_class(package))
        monkeypatch.setattr(registry, "entry_points", lambda group: [entry])
        assert registry.load_entry_point_extractors() == 1
        assert registry.get_extractor(NAME) is load_class(package)

    def test_a_broken_plugin_is_skipped_with_a_warning(self, fresh_registry, monkeypatch, caplog):
        def broken():
            raise ImportError("no module named lab_secret_sauce")

        monkeypatch.setattr(registry, "entry_points", lambda group: [SimpleNamespace(name="lab.x", load=broken)])
        monkeypatch.setattr(logging.getLogger("macaw"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="macaw"):
            assert registry.load_entry_point_extractors() == 0
        assert "lab.x" in caplog.text

    def test_plugins_load_with_the_builtins(self, monkeypatch):
        """They were discoverable but never loaded before; nothing called the loader."""
        calls = []
        monkeypatch.setattr(registry, "_builtins_loaded", False)
        monkeypatch.setattr(registry, "BUILTIN_MODULES", ())
        monkeypatch.setattr(registry, "load_entry_point_extractors", lambda: calls.append(1))
        registry.load_builtin_extractors()
        assert calls == [1]


class TestCommand:
    def test_writes_the_package_and_prints_where(self, tmp_path):
        result = CliRunner().invoke(app, ["new-extractor", "mylab.speech_rate", "--dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert Path(result.output.strip()) == tmp_path / "macaw-mylab-speech-rate"

    def test_an_installed_name_is_refused(self, tmp_path):
        result = CliRunner().invoke(app, ["new-extractor", "pose.kinematics", "--dir", str(tmp_path)])
        assert result.exit_code != 0
        assert not any(tmp_path.iterdir())
