import sys
import types

import pytest

from macaw.coding.llm import backends as llm
from macaw.config.schema import LLMSection
from macaw.exceptions import BackendNotAvailableError
from macaw.extractors.hub import check_hub_model
from macaw.extractors.video.pose.backends import YOLOPoseBackend


class _Missing(Exception):
    pass


class _Gated(Exception):
    pass


class _NotFound(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    """A stand-in hub client: what is cached, and what the account may fetch."""
    state = {"cached": set(), "gated": set(), "unknown": set(), "offline_hub": False}
    module = types.ModuleType("huggingface_hub")
    errors = types.ModuleType("huggingface_hub.errors")
    errors.LocalEntryNotFoundError, errors.GatedRepoError, errors.RepositoryNotFoundError = _Missing, _Gated, _NotFound

    def snapshot_download(repo_id, local_files_only):
        if repo_id not in state["cached"]:
            raise _Missing(repo_id)

    def auth_check(repo_id):
        if state["offline_hub"]:
            raise ConnectionError("no route to host")
        if repo_id in state["gated"]:
            raise _Gated(repo_id)
        if repo_id in state["unknown"]:
            raise _NotFound(repo_id)

    module.snapshot_download, module.auth_check, module.errors = snapshot_download, auth_check, errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return state


class TestHubModel:
    def test_a_downloaded_model_passes_even_offline(self, hub, monkeypatch):
        hub["cached"].add("org/model")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        check_hub_model("org/model", "text.diarization_model")

    def test_a_model_not_downloaded_on_an_offline_machine_stops(self, hub, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(BackendNotAvailableError, match="not downloaded, and HF_HUB_OFFLINE"):
            check_hub_model("org/model", "text.diarization_model")

    def test_a_gated_model_without_access_names_where_to_accept_its_terms(self, hub):
        hub["gated"].add("org/model")
        with pytest.raises(BackendNotAvailableError, match=r"accept its terms at https://huggingface.co/org/model"):
            check_hub_model("org/model", "text.diarization_model")

    def test_a_misnamed_model_and_an_unreachable_hub_stop(self, hub):
        hub["unknown"].add("org/typo")
        with pytest.raises(BackendNotAvailableError, match="check the name"):
            check_hub_model("org/typo", "text.asr_model")
        hub["offline_hub"] = True
        with pytest.raises(BackendNotAvailableError, match="cannot be reached"):
            check_hub_model("org/model", "text.asr_model")

    def test_a_fetchable_model_and_a_local_folder_pass(self, hub, tmp_path):
        check_hub_model("org/model", "text.asr_model")
        hub["offline_hub"] = True
        check_hub_model(str(tmp_path), "text.asr_model")


class TestServedModel:
    def _backend(self, cls, monkeypatch, served, model="qwen3:14b"):
        monkeypatch.setattr(llm, "_get", lambda url, headers=None, timeout=5: {"data": [{"id": name} for name in served]})
        return cls(LLMSection(provider=cls.name, model=model))

    def test_ollama_without_the_model_says_to_pull_it(self, monkeypatch):
        backend = self._backend(llm.OllamaBackend, monkeypatch, ["qwen3:8b"])
        with pytest.raises(BackendNotAvailableError, match=r"no model qwen3:14b .*it has qwen3:8b\. Fetch it with `ollama pull`"):
            backend.check_available()

    def test_ollama_with_the_model_or_its_latest_tag_passes(self, monkeypatch):
        self._backend(llm.OllamaBackend, monkeypatch, ["qwen3:14b"]).check_available()
        self._backend(llm.OllamaBackend, monkeypatch, ["qwen3:latest"], model="qwen3").check_available()

    def test_ollama_with_nothing_pulled_stops(self, monkeypatch):
        with pytest.raises(BackendNotAvailableError, match="it has none"):
            self._backend(llm.OllamaBackend, monkeypatch, []).check_available()

    def test_llamacpp_serves_what_it_was_started_with(self, monkeypatch):
        self._backend(llm.LlamaCppBackend, monkeypatch, ["/models/x.gguf"]).check_available()


def test_yolo_weights_that_are_not_there_stop_before_the_run(tmp_path):
    with pytest.raises(BackendNotAvailableError, match="pose.model to point at a pose weights file"):
        YOLOPoseBackend().check_model(tmp_path / "yolo11x-pose.pt")
    weights = tmp_path / "yolo11x-pose.pt"
    weights.write_bytes(b"")
    YOLOPoseBackend().check_model(weights)
