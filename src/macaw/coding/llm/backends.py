"""Local and cloud are both first-class: some IRBs forbid uploading audio to third parties."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from macaw.config.schema import LLMSection
from macaw.exceptions import BackendNotAvailableError, LLMBackendError
from macaw.extractors import get_backend, register_backend
from macaw.extractors.base import Backend

BACKEND_GROUP = "llm"

_RETRIES = 3
_PROBE_TIMEOUT_S = 5


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMBackend(Backend):
    name: str
    is_local: bool
    default_base_url: str
    takes_seed: bool = True
    # printed in the run plan; None for a server on the user's own machine
    reporting: str | None = None

    def __init__(self, cfg: LLMSection):
        self.cfg = cfg
        self.base_url = (cfg.base_url or self.default_base_url).rstrip("/")
        self._seen_model: str | None = None

    @property
    def version(self) -> str:
        return self.model_version()

    def check_available(self) -> None:
        if not self.cfg.model:
            raise BackendNotAvailableError(
                f"coding.llm.model is not set. Name the model as {self.name} knows it "
                "(for example the tag `ollama list` shows)."
            )

    def complete(self, system: str, user: str, *, seed: int | None = None) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        return self._seen_model or str(self.cfg.model)

    def _post(self, url: str, body: dict, headers: dict[str, str]) -> dict:
        data = json.dumps(body).encode("utf-8")
        for attempt in range(_RETRIES + 1):
            request = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                if (error.code != 429 and error.code < 500) or attempt == _RETRIES:
                    raise LLMBackendError(f"{self.name} answered {error.code} at {url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == _RETRIES:
                    raise LLMBackendError(
                        f"{self.name} at {url} did not answer after {_RETRIES + 1} tries: {error}"
                    ) from None
            time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _get(url: str, headers: dict[str, str] | None = None, timeout: float = _PROBE_TIMEOUT_S) -> dict:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleBackend(LLMBackend):
    """The chat completions protocol that Ollama, vLLM, llama.cpp and OpenAI all speak."""

    api_key_variable: str | None = None

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.api_key_variable or "", "")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def check_available(self) -> None:
        super().check_available()
        if not self.is_local:
            if not os.environ.get(self.api_key_variable or ""):
                raise BackendNotAvailableError(
                    f"{self.name} needs an API key in the environment variable {self.api_key_variable}. "
                    "Keys are never read from project.yaml, which is archived with the outputs."
                )
            return
        try:
            _get(f"{self.base_url}/models", self._headers())
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as error:
            raise BackendNotAvailableError(
                f"No {self.name} server answering at {self.base_url} ({error}). Start it first, "
                "or set coding.llm.base_url to where it listens."
            ) from None

    def complete(self, system: str, user: str, *, seed: int | None = None) -> LLMResponse:
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if seed is not None:
            body["seed"] = seed
        answer = self._post(f"{self.base_url}/chat/completions", body, self._headers())
        try:
            text = answer["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise LLMBackendError(f"{self.name} returned no message: {str(answer)[:300]}") from None
        usage = answer.get("usage") or {}
        self._seen_model = answer.get("model") or self._seen_model
        return LLMResponse(
            text=text,
            model=self.model_version(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def model_version(self) -> str:
        if not self.is_local:
            return super().model_version()
        try:
            listed = _get(f"{self.base_url}/models", self._headers()).get("data") or []
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            return str(self.cfg.model)
        for entry in listed:
            if entry.get("id") == self.cfg.model:
                return str(entry.get("id"))
        return str(self.cfg.model)


@register_backend(BACKEND_GROUP)
class OllamaBackend(OpenAICompatibleBackend):
    name = "ollama"
    is_local = True
    default_base_url = "http://localhost:11434/v1"

    def model_version(self) -> str:
        # the tag can be pulled again with other weights; the digest cannot
        root = self.base_url.removesuffix("/v1")
        try:
            listed = _get(f"{root}/api/tags").get("models") or []
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            return str(self.cfg.model)
        for entry in listed:
            if entry.get("name") == self.cfg.model or entry.get("model") == self.cfg.model:
                return f"{self.cfg.model}@{entry.get('digest', '')[:12]}"
        return str(self.cfg.model)


@register_backend(BACKEND_GROUP)
class VLLMBackend(OpenAICompatibleBackend):
    name = "vllm"
    is_local = True
    default_base_url = "http://localhost:8000/v1"


@register_backend(BACKEND_GROUP)
class LlamaCppBackend(OpenAICompatibleBackend):
    name = "llamacpp"
    is_local = True
    default_base_url = "http://localhost:8080/v1"


@register_backend(BACKEND_GROUP)
class OpenAIBackend(OpenAICompatibleBackend):
    name = "openai"
    is_local = False
    default_base_url = "https://api.openai.com/v1"
    api_key_variable = "OPENAI_API_KEY"
    reporting = (
        "coding.llm.provider openai: the transcript text, speaker letters and times are sent to "
        "OpenAI's servers. Whether that is allowed is for your ethics approval to say."
    )


@register_backend(BACKEND_GROUP)
class AnthropicBackend(LLMBackend):
    name = "anthropic"
    is_local = False
    default_base_url = "https://api.anthropic.com/v1"
    takes_seed = False
    reporting = (
        "coding.llm.provider anthropic: the transcript text, speaker letters and times are sent to "
        "Anthropic's servers. Whether that is allowed is for your ethics approval to say."
    )

    def check_available(self) -> None:
        super().check_available()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise BackendNotAvailableError(
                "anthropic needs an API key in the environment variable ANTHROPIC_API_KEY. "
                "Keys are never read from project.yaml, which is archived with the outputs."
            )

    def complete(self, system: str, user: str, *, seed: int | None = None) -> LLMResponse:
        body = {
            "model": self.cfg.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        headers = {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""), "anthropic-version": "2023-06-01"}
        answer = self._post(f"{self.base_url}/messages", body, headers)
        parts = [part.get("text", "") for part in answer.get("content") or [] if part.get("type") == "text"]
        if not parts:
            raise LLMBackendError(f"anthropic returned no text: {str(answer)[:300]}")
        usage = answer.get("usage") or {}
        self._seen_model = answer.get("model") or self._seen_model
        return LLMResponse(
            text="".join(parts),
            model=self.model_version(),
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )


def resolve_llm_backend(cfg: LLMSection) -> LLMBackend:
    return get_backend(BACKEND_GROUP, cfg.provider)(cfg)
