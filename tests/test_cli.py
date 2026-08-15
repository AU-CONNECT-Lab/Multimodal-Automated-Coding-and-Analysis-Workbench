from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from macaw.cli.main import app, entrypoint
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.config.schema import MacawConfig
from macaw.execution.job_db import JobDB, JobRecord
from macaw.logging_setup import LOGGER_NAME, setup_logging
from macaw.project.layout import make_layout

runner = CliRunner()

EXECUTION = "20260815-120000-abc123"

# Every command below resolves a config, which the conftest pydantic stub cannot do: it only
# reads section defaults. Skipping beats 20 failures that all say the same thing.
needs_pydantic = pytest.mark.skipif(
    not hasattr(MacawConfig, "model_validate"),
    reason="real pydantic required; the conftest stub cannot validate a config",
)


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "raw"
    for name in ("003", "004"):
        (raw / name).mkdir(parents=True)
    output = tmp_path / "derivatives"
    config = tmp_path / "project.yaml"
    scaffold_project_config(config, name="demo", raw_dir=raw, output_dir=output)
    return SimpleNamespace(root=tmp_path, config=config, raw=raw, output=output)


def invoke(*args):
    return runner.invoke(app, list(args))


class TestInit:
    def test_scaffolds_config_and_output_tree(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        result = invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        assert result.exit_code == 0
        assert (tmp_path / "project.yaml").is_file()
        assert (tmp_path / "derivatives").is_dir()
        assert (tmp_path / "recordings").is_dir()

    @needs_pydantic
    def test_the_written_config_resolves(self, tmp_path):
        """A scaffold that its own loader rejects is worse than no scaffold."""
        raw = tmp_path / "raw"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        resolved = resolve_config(tmp_path / "project.yaml")
        assert resolved.project.name == "demo"
        assert resolved.project.raw_dir == raw.resolve()

    @needs_pydantic
    def test_a_path_with_spaces_survives_the_round_trip(self, tmp_path):
        """Windows paths carry backslashes and spaces; unquoted YAML mangles them."""
        raw = tmp_path / "my data dir"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        assert resolve_config(tmp_path / "project.yaml").project.raw_dir == raw.resolve()

    @needs_pydantic
    def test_existing_config_is_not_overwritten(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        again = invoke("init", str(tmp_path), "--name", "other", "--raw-dir", str(raw))
        assert again.exit_code != 0
        assert resolve_config(tmp_path / "project.yaml").project.name == "demo"

    @needs_pydantic
    def test_force_overwrites(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        invoke("init", str(tmp_path), "--name", "other", "--raw-dir", str(raw), "--force")
        assert resolve_config(tmp_path / "project.yaml").project.name == "other"

    def test_two_subjects_keep_the_default_roles(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw))
        assert "[parent, child]" in (tmp_path / "project.yaml").read_text(encoding="utf-8")

    def test_other_group_sizes_get_generated_roles(self, tmp_path):
        """parent/child is a two-person assumption; this is a general-purpose package."""
        raw = tmp_path / "raw"
        raw.mkdir()
        invoke("init", str(tmp_path), "--name", "demo", "--raw-dir", str(raw), "--n-subjects", "3")
        text = (tmp_path / "project.yaml").read_text(encoding="utf-8")
        assert "[subject_1, subject_2, subject_3]" in text
        assert "parent" not in text


@needs_pydantic
class TestRecordings:
    def test_lists_detected_ids(self, project):
        result = invoke("recordings", "-c", str(project.config))
        assert result.exit_code == 0
        assert result.output.split() == ["003", "004"]

    def test_writes_the_same_ids_to_a_file(self, project):
        out = project.root / "ids.txt"
        invoke("recordings", "-c", str(project.config), "--out", str(out))
        assert out.read_text(encoding="utf-8").split() == ["003", "004"]

    def test_prefix_filters(self, project):
        (project.raw / "dyad-9").mkdir()
        result = invoke("recordings", "-c", str(project.config), "--prefix", "dyad-")
        assert result.output.split() == ["dyad-9"]

    def test_exclusions_are_applied(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["project"]["excluded_recordings"] = ["003"]
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert invoke("recordings", "-c", str(project.config)).output.split() == ["004"]

    def test_stdout_stays_clean_under_quiet(self, project):
        """The ids are the result, not progress; a pipe must still receive them."""
        result = invoke("-q", "recordings", "-c", str(project.config))
        assert result.output.split() == ["003", "004"]


@needs_pydantic
class TestConfigShow:
    def test_prints_the_merged_result(self, project):
        result = invoke("config", "show", "-c", str(project.config))
        data = yaml.safe_load(result.output)
        assert data["project"]["name"] == "demo"
        assert "prosody" in data

    def test_one_section_only(self, project):
        data = yaml.safe_load(invoke("config", "show", "-c", str(project.config), "--section", "project").output)
        assert list(data) == ["project"]

    def test_an_unknown_section_is_rejected(self, project):
        assert invoke("config", "show", "-c", str(project.config), "--section", "nope").exit_code != 0

    def test_set_overrides_reach_the_output(self, project):
        result = invoke(
            "config", "show", "-c", str(project.config), "--section", "execution",
            "--set", "execution.n_workers=9",
        )
        assert yaml.safe_load(result.output)["execution"]["n_workers"] == 9


@needs_pydantic
class TestConfigValidate:
    def test_a_good_config_passes(self, project):
        result = invoke("config", "validate", "-c", str(project.config))
        assert result.exit_code == 0
        assert "ok" in result.output

    def test_an_unknown_key_fails(self, project):
        data = yaml.safe_load(project.config.read_text(encoding="utf-8"))
        data["prosody"] = {"no_such_key": 1}
        project.config.write_text(yaml.safe_dump(data), encoding="utf-8")
        assert invoke("config", "validate", "-c", str(project.config)).exit_code != 0

    def test_a_missing_file_fails(self, tmp_path):
        assert invoke("config", "validate", "-c", str(tmp_path / "nope.yaml")).exit_code != 0


class TestConfigDiff:
    def pair(self, tmp_path, left, right):
        a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
        a.write_text(yaml.safe_dump(left), encoding="utf-8")
        b.write_text(yaml.safe_dump(right), encoding="utf-8")
        return a, b

    def test_identical_configs(self, tmp_path):
        a, b = self.pair(tmp_path, {"prosody": {"f0_min_hz": 60}}, {"prosody": {"f0_min_hz": 60}})
        assert "identical" in invoke("config", "diff", str(a), str(b)).output

    def test_names_the_differing_section(self, tmp_path):
        a, b = self.pair(tmp_path, {"prosody": {"f0_min_hz": 60}}, {"prosody": {"f0_min_hz": 75}})
        assert "prosody" in invoke("config", "diff", str(a), str(b)).output

    def test_says_which_extractors_would_recompute(self, tmp_path):
        """A yaml diff alone does not answer the question the command is for."""
        a, b = self.pair(tmp_path, {"prosody": {"f0_min_hz": 60}}, {"prosody": {"f0_min_hz": 75}})
        output = invoke("config", "diff", str(a), str(b)).output
        assert "would recompute" in output
        assert "prosody.frame" in output

    def test_a_section_no_extractor_hashes_recomputes_nothing(self, tmp_path):
        a, b = self.pair(tmp_path, {"execution": {"n_workers": 2}}, {"execution": {"n_workers": 8}})
        assert "nothing would recompute" in invoke("config", "diff", str(a), str(b)).output

    def test_a_missing_file_fails(self, tmp_path):
        a, _ = self.pair(tmp_path, {"prosody": {}}, {"prosody": {}})
        assert invoke("config", "diff", str(a), str(tmp_path / "nope.yaml")).exit_code != 0


@needs_pydantic
class TestRunDryRun:
    def test_dependencies_are_pulled_in_and_ordered_first(self, project):
        """prosody.frame never appears without prep.media in front of it."""
        result = invoke("run", "-c", str(project.config), "-e", "prosody.frame", "--dry-run")
        assert result.exit_code == 0
        tasks = result.output.split()
        assert tasks == ["003:prep.media", "004:prep.media", "003:prosody.frame", "004:prosody.frame"]

    def test_workers_is_accepted_as_an_override(self, project):
        result = invoke("run", "-c", str(project.config), "-e", "prep.media", "-j", "2", "--dry-run")
        assert result.exit_code == 0

    def test_no_recordings_is_an_error_not_an_empty_run(self, project):
        for name in ("003", "004"):
            (project.raw / name).rmdir()
        assert invoke("run", "-c", str(project.config), "--dry-run").exit_code != 0

    def test_an_unknown_extractor_is_rejected(self, project):
        result = invoke("run", "-c", str(project.config), "-e", "no.such.thing", "--dry-run")
        assert result.exit_code != 0


@needs_pydantic
class TestStatus:
    def record(self, project, status, task_id="003:prep.media"):
        database = JobDB(make_layout(project.output, create=False).job_db_path)
        database.initialise()
        database.upsert(
            JobRecord(
                task_id=task_id,
                recording_id="003",
                extractor="prep.media",
                execution_id=EXECUTION,
                status=status,
                exit_code=1 if status == "failed" else 0,
                message="boom" if status == "failed" else None,
            )
        )

    def test_nothing_recorded_yet(self, project):
        result = invoke("status", "-c", str(project.config))
        assert result.exit_code == 0
        assert "no executions" in result.output

    def test_summarises_the_latest_execution(self, project):
        self.record(project, "done")
        result = invoke("status", "-c", str(project.config))
        assert EXECUTION in result.output
        assert "done 1" in result.output

    def test_failed_only_lists_the_tasks(self, project):
        self.record(project, "done")
        self.record(project, "failed", task_id="004:prep.media")
        result = invoke("status", "-c", str(project.config), "--failed")
        assert "004:prep.media" in result.output
        assert "003:prep.media" not in result.output


class TestListExtractors:
    def test_lists_the_implemented_ones(self, project):
        output = invoke("list-extractors").output
        for name in ("prep.media", "prosody.frame", "turns.structure", "coding.layer_a"):
            assert name in output

    def test_group_filters(self, project):
        output = invoke("list-extractors", "--group", "turns").output
        assert "turns.structure" in output
        assert "prep.media" not in output

    def test_backends_are_listed_on_request(self, project):
        assert "prosody" in invoke("list-extractors", "--backends").output

    @needs_pydantic
    def test_a_config_adds_an_availability_column(self, project):
        """Registered and installed are different questions; only a config answers the second."""
        plain = invoke("list-extractors").output.strip().splitlines()
        checked = invoke("list-extractors", "-c", str(project.config)).output.strip().splitlines()
        assert all(len(line.split("\t")) == 4 for line in plain)
        assert all(len(line.split("\t")) == 5 for line in checked)

    @needs_pydantic
    def test_pure_pandas_extractors_report_ok_without_any_environment(self, project):
        checked = invoke("list-extractors", "-c", str(project.config), "--group", "turns").output
        assert all(line.endswith("\tok") for line in checked.strip().splitlines())


@needs_pydantic
class TestClean:
    def test_an_empty_cache_removes_nothing(self, project):
        result = invoke("clean", "-c", str(project.config))
        assert result.exit_code == 0
        assert "removed 0" in result.output

    def test_all_asks_before_deleting_output(self, project):
        """Declining must leave the directory alone."""
        recording_dir = make_layout(project.output, create=False).recording_dir("003")
        recording_dir.mkdir(parents=True)
        result = runner.invoke(app, ["clean", "-c", str(project.config), "--all"], input="n\n")
        assert result.exit_code != 0
        assert recording_dir.is_dir()

    def test_all_deletes_once_confirmed(self, project):
        recording_dir = make_layout(project.output, create=False).recording_dir("003")
        recording_dir.mkdir(parents=True)
        runner.invoke(app, ["clean", "-c", str(project.config), "--all"], input="y\n")
        assert not recording_dir.exists()


class TestLogging:
    @pytest.mark.parametrize(
        "flags,level",
        [({}, "WARNING"), ({"verbose": True}, "INFO"), ({"quiet": True}, "ERROR")],
    )
    def test_levels(self, flags, level):
        assert logging.getLevelName(setup_logging(**flags).level) == level

    def test_verbose_wins_over_quiet(self):
        assert setup_logging(verbose=True, quiet=True).level == logging.INFO

    def test_handlers_do_not_stack(self):
        """Every command re-runs the callback; CliRunner would otherwise pile up output."""
        for _ in range(3):
            logger = setup_logging()
        assert len(logger.handlers) == 1

    def test_the_root_logger_is_left_alone(self):
        """Importing macaw as a library must not reconfigure someone else's logging."""
        assert setup_logging().propagate is False
        assert logging.getLogger(LOGGER_NAME).name == "macaw"


class TestEntrypoint:
    def test_a_macaw_error_prints_cleanly_and_exits_one(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(
            sys, "argv", ["macaw", "config", "validate", "-c", str(tmp_path / "nope.yaml")]
        )
        with pytest.raises(SystemExit) as exit_info:
            entrypoint()
        assert exit_info.value.code == 1
        captured = capsys.readouterr()
        assert "error:" in captured.err
        assert "Traceback" not in captured.err

    @needs_pydantic
    def test_success_still_exits_zero(self, monkeypatch, project):
        monkeypatch.setattr(sys, "argv", ["macaw", "config", "validate", "-c", str(project.config)])
        with pytest.raises(SystemExit) as exit_info:
            entrypoint()
        assert exit_info.value.code == 0


class TestBlockedCommandsStayHonest:
    """Commands whose layer is still a skeleton must fail, not pretend."""

    @pytest.mark.parametrize(
        "args",
        [
            ["qc", "-c", "project.yaml"],
            ["hpc", "generate", "-c", "project.yaml"],
        ],
    )
    def test_they_do_not_report_success(self, args, project, monkeypatch):
        monkeypatch.chdir(project.root)
        assert invoke(*args).exit_code != 0

    def test_eaf_export_says_there_is_nothing_to_export(self, project, monkeypatch):
        # it is implemented now, so it has to fail on the missing annotations and say so, rather
        # than on the format being unknown
        monkeypatch.chdir(project.root)
        result = invoke("export", "-c", "project.yaml", "-f", "eaf")
        assert result.exit_code != 0
        assert "annotations" in result.output.lower()


class TestExportArguments:
    def test_an_unknown_format_is_rejected(self, project):
        assert invoke("export", "-c", str(project.config), "-f", "xlsx").exit_code != 0

    def test_an_unknown_scale_is_rejected(self, project):
        assert invoke("export", "-c", str(project.config), "--scale", "hourly").exit_code != 0

    @needs_pydantic
    def test_exporting_before_running_says_so(self, project):
        result = invoke("export", "-c", str(project.config))
        assert result.exit_code != 0
        assert "macaw run" in str(result.exception)
