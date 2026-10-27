from __future__ import annotations

from types import SimpleNamespace

import pytest

from macaw.config.schema import PipelineSection
from macaw.execution.runner import plan, resolve_recordings, task_id_for
from macaw.extractors.base import Phase


@pytest.fixture
def registry(monkeypatch):
    import macaw.extractors as extractors

    def build(spec):
        table = {
            name: type(
                "FakeExtractor",
                (),
                {"name": name, "requires": tuple(requires), "phase": phase, "config_sections": (), "dependencies": classmethod(lambda cls, config: cls.requires)},
            )
            for name, (requires, phase) in spec.items()
        }
        monkeypatch.setattr(extractors, "_REGISTRY", table)
        monkeypatch.setattr(extractors, "_builtins_loaded", True)

    return build


def _config(excluded=()):
    return SimpleNamespace(
        project=SimpleNamespace(excluded_recordings=list(excluded)), pipeline=PipelineSection()
    )


LINE = {
    "prep.media": ((), Phase.PREP),
    "prosody.frame": (("prep.media",), Phase.EXTRACT),
}


class TestTaskId:
    def test_a_task_is_one_recording_and_one_extractor(self):
        assert task_id_for("sub-003", "prosody.frame") == "sub-003:prosody.frame"


class TestResolvingRecordings:
    def test_a_comma_list_is_accepted(self):
        assert resolve_recordings(_config(), "rec1,rec2") == ["rec1", "rec2"]

    def test_the_exclusion_list_is_applied(self):
        assert resolve_recordings(_config(excluded=["rec2"]), "rec1,rec2,rec3") == ["rec1", "rec3"]

    def test_a_file_of_ids_is_accepted(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("rec1\nrec2\n", encoding="utf-8")
        assert resolve_recordings(_config(), str(path)) == ["rec1", "rec2"]


class TestPlan:
    def test_a_dependency_is_pulled_in_and_ordered_first(self, registry):
        """Asking only for prosody still has to run prep first, and the plan has to say so."""
        registry(LINE)
        tasks = plan(_config(), ["rec1"], ["prosody.frame"])
        assert tasks == ["rec1:prep.media", "rec1:prosody.frame"]

    def test_every_recording_gets_every_task(self, registry):
        registry(LINE)
        assert len(plan(_config(), ["rec1", "rec2"], ["prosody.frame"])) == 4

    def test_an_extractor_finishes_across_recordings_before_the_next_starts(self, registry):
        """That grouping is what becomes one job array per extractor on the cluster."""
        registry(LINE)
        tasks = plan(_config(), ["rec1", "rec2"], ["prosody.frame"])
        assert tasks == [
            "rec1:prep.media",
            "rec2:prep.media",
            "rec1:prosody.frame",
            "rec2:prosody.frame",
        ]

    def test_no_request_plans_everything_registered(self, registry):
        registry(LINE)
        assert len(plan(_config(), ["rec1"], [])) == 2

    def test_no_recordings_plans_nothing(self, registry):
        registry(LINE)
        assert plan(_config(), [], ["prosody.frame"]) == []
