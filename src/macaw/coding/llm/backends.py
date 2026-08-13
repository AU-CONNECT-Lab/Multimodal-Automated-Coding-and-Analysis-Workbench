"""Local and cloud are both first-class: some IRBs forbid uploading audio to third parties."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from macaw.config.schema import LLMSection


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMBackend(ABC):
    name: str
    is_local: bool

    def __init__(self, cfg: LLMSection):
        self.cfg = cfg

    @abstractmethod
    def check_available(self) -> None: ...

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse: ...

    @abstractmethod
    def model_version(self) -> str: ...


class OllamaBackend(LLMBackend):
    name = "ollama"
    is_local = True

    def check_available(self) -> None:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        raise NotImplementedError


class VLLMBackend(LLMBackend):
    name = "vllm"
    is_local = True

    def check_available(self) -> None:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    name = "llamacpp"
    is_local = True

    def check_available(self) -> None:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        raise NotImplementedError


class AnthropicBackend(LLMBackend):
    name = "anthropic"
    is_local = False

    def check_available(self) -> None:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    name = "openai"
    is_local = False

    def check_available(self) -> None:
        raise NotImplementedError

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def model_version(self) -> str:
        raise NotImplementedError


def resolve_llm_backend(cfg: LLMSection) -> LLMBackend:
    raise NotImplementedError


def warn_on_first_cloud_use(cfg: LLMSection) -> None:
    """Notify once that data leaves the machine. Do not block; that call is the user's IRB's."""
    raise NotImplementedError
