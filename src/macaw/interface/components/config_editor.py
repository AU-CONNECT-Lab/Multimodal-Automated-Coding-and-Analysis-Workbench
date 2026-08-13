from __future__ import annotations

from macaw.config.schema import MacawConfig


def layout(config: MacawConfig) -> object:
    raise NotImplementedError


def section_form(section_name: str, values: dict) -> object:
    raise NotImplementedError


def validation_banner(errors: list[str]) -> object:
    raise NotImplementedError
