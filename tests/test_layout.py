"""The output_dir / work_dir split. The rule under test: deleting work_dir costs
recomputation and never a result, so nothing a result depends on may live there."""

from __future__ import annotations

from pathlib import Path

import pytest

from macaw.config.loader import (
    OUTPUT_DIR_ENV_VAR,
    WORK_DIR_ENV_VAR,
    resolve_config,
    scaffold_project_config,
)
from macaw.config.schema import MacawConfig
from macaw.project.layout import layout_for, make_layout

needs_pydantic = pytest.mark.skipif(
    not hasattr(MacawConfig, "model_validate"),
    reason="real pydantic required; the conftest stub cannot validate a config",
)

WORK_PATHS = ("cache_dir", "logs_dir", "slurm_logs_dir", "job_db_path", "array_map_path")
OUTPUT_PATHS = ("reports_dir", "recordings_dir", "manifest_path", "resolved_config_path")


def _is_under(path: Path, root: Path) -> bool:
    return root in path.parents or path == root


class TestDefaultsToOutputDir:
    """Everything used to live under output_dir; a project that never sets work_dir must
    keep finding its cache exactly where it left it."""

    def test_every_path_stays_under_output_dir(self, tmp_path):
        layout = make_layout(tmp_path / "derivatives", create=False)
        assert layout.work_dir == layout.output_dir
        for name in WORK_PATHS + OUTPUT_PATHS:
            assert _is_under(getattr(layout, name), layout.output_dir)

    def test_the_cache_keeps_its_old_place(self, tmp_path):
        layout = make_layout(tmp_path / "derivatives", create=False)
        assert layout.cache_dir == tmp_path / "derivatives" / ".cache"
        assert layout.job_db_path == tmp_path / "derivatives" / "jobs"


class TestSplitRoots:
    @pytest.fixture
    def layout(self, tmp_path):
        return make_layout(tmp_path / "derivatives", tmp_path / "scratch", create=False)

    @pytest.mark.parametrize("name", WORK_PATHS)
    def test_bookkeeping_moves_to_work_dir(self, layout, name):
        assert _is_under(getattr(layout, name), layout.work_dir)

    @pytest.mark.parametrize("name", OUTPUT_PATHS)
    def test_results_stay_in_output_dir(self, layout, name):
        path = getattr(layout, name)
        assert _is_under(path, layout.output_dir)
        assert not _is_under(path, layout.work_dir)

    @pytest.mark.parametrize(
        "name", ("recording_dir", "long_table_path", "keypoints_path", "annotations_path")
    )
    def test_the_per_recording_tree_is_a_result(self, layout, name):
        assert _is_under(getattr(layout, name)("rec1"), layout.output_dir)

    def test_artifacts_are_a_result(self, layout):
        assert _is_under(layout.artifacts_dir("rec1", "prep.media"), layout.output_dir)

    def test_slurm_logs_sit_with_the_task_logs(self, layout):
        assert layout.slurm_logs_dir.parent == layout.logs_dir

    def test_creating_makes_both_roots(self, tmp_path):
        layout = make_layout(tmp_path / "derivatives", tmp_path / "scratch")
        assert layout.output_dir.is_dir()
        assert layout.work_dir.is_dir()
        assert layout.logs_dir.is_dir()

    def test_a_string_is_accepted_for_either_root(self, tmp_path):
        layout = make_layout(str(tmp_path / "out"), str(tmp_path / "work"), create=False)
        assert layout.cache_dir == tmp_path / "work" / ".cache"


@needs_pydantic
class TestConfigResolution:
    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        monkeypatch.delenv(OUTPUT_DIR_ENV_VAR, raising=False)
        monkeypatch.delenv(WORK_DIR_ENV_VAR, raising=False)
        raw = tmp_path / "raw"
        raw.mkdir()
        config = tmp_path / "project.yaml"
        scaffold_project_config(
            config, name="demo", raw_dir=raw, output_dir=tmp_path / "derivatives"
        )
        return config

    def test_work_dir_is_null_by_default_and_layout_falls_back(self, project):
        resolved = resolve_config(project)
        assert resolved.project.work_dir is None
        layout = layout_for(resolved, create=False)
        assert layout.work_dir == layout.output_dir

    def test_the_scaffold_can_carry_a_work_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv(WORK_DIR_ENV_VAR, raising=False)
        raw = tmp_path / "raw"
        raw.mkdir()
        config = tmp_path / "project.yaml"
        scaffold_project_config(
            config,
            name="demo",
            raw_dir=raw,
            output_dir=tmp_path / "derivatives",
            work_dir=tmp_path / "scratch space",
        )
        resolved = resolve_config(config)
        assert resolved.project.work_dir == tmp_path / "scratch space"

    def test_the_argument_wins_over_the_file(self, project, tmp_path):
        resolved = resolve_config(project, work_dir=tmp_path / "elsewhere")
        assert resolved.project.work_dir == tmp_path / "elsewhere"

    def test_the_environment_is_read_when_no_argument_is_given(self, project, tmp_path, monkeypatch):
        monkeypatch.setenv(WORK_DIR_ENV_VAR, str(tmp_path / "from_env"))
        monkeypatch.setenv(OUTPUT_DIR_ENV_VAR, str(tmp_path / "out_from_env"))
        resolved = resolve_config(project)
        assert resolved.project.work_dir == tmp_path / "from_env"
        assert resolved.project.output_dir == tmp_path / "out_from_env"

    def test_the_argument_wins_over_the_environment(self, project, tmp_path, monkeypatch):
        monkeypatch.setenv(WORK_DIR_ENV_VAR, str(tmp_path / "from_env"))
        resolved = resolve_config(project, work_dir=tmp_path / "from_flag")
        assert resolved.project.work_dir == tmp_path / "from_flag"

    def test_a_relative_override_follows_the_shell_not_the_config(self, project, tmp_path, monkeypatch):
        """Paths written in project.yaml resolve against the file so the project stays
        portable; a flag typed at a prompt has to mean what the prompt is looking at."""
        elsewhere = tmp_path / "cwd_here"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        resolved = resolve_config(project, work_dir=Path("work"))
        assert resolved.project.work_dir == elsewhere / "work"

    def test_the_directories_do_not_reach_the_config_hash(self, project, tmp_path):
        """Absolute paths never enter a hash, so moving the output between machines must
        not invalidate a cache. work_dir is one more path with that same rule."""
        from macaw.config.hashing import hash_full

        baseline = resolve_config(project)
        moved = resolve_config(project, work_dir=tmp_path / "somewhere_else")
        assert hash_full(baseline) == hash_full(moved)
