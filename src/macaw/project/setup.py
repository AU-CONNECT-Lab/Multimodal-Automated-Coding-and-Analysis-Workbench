"""The setup file: which device folders a recording holds and whom each device records."""

from __future__ import annotations

from pathlib import Path

import yaml

from macaw.config.schema import MacawConfig, SetupSection
from macaw.exceptions import ConfigError


def setup_path(config: MacawConfig) -> Path:
    return Path(config.project.setups_dir) / f"{config.setup.name}.yaml"


def require_setup(config: MacawConfig) -> None:
    """A run reads files only through the setup, so it does not start before one is saved."""
    path = setup_path(config)
    if not path.is_file():
        raise ConfigError(
            f"the setup {config.setup.name} is not saved yet ({path}): describe the cameras and microphones on the "
            "Setup page of the workbench (macaw gui) and save it, or write that file by hand"
        )
    if not config.setup.devices():
        raise ConfigError(f"the setup {config.setup.name} names no camera or microphone, so no file can be read ({path})")


def setup_content(setup: SetupSection) -> dict:
    """What a setup file holds: everything but its name, which is the file's."""
    data = setup.model_dump(mode="json", exclude={"name"})
    data["cameras"] = {name: {key: value for key, value in camera.items() if value} for name, camera in data["cameras"].items()}
    return data


def write_setup(path: Path, setup: SetupSection) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(setup_content(setup), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def device_people(setup: SetupSection, device: str) -> tuple[list[str], list[str]]:
    """(people the device is read for, people whose voices its sound carries)."""
    if device in setup.cameras:
        camera = setup.cameras[device]
        return camera.people(), sorted(camera.hears)
    if device in setup.microphones:
        hears = sorted(setup.microphones[device].hears)
        return hears, hears
    raise ConfigError(f"setup {setup.name} has no device {device}")
