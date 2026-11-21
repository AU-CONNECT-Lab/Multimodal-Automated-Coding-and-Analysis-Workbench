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
    """What a setup file holds: everything but its name, which is the file's, and nothing the room leaves blank."""
    data = setup.model_dump(mode="json", exclude={"name", "room"})
    data["cameras"] = {name: {key: value for key, value in camera.items() if value} for name, camera in data["cameras"].items()}
    data["ignore"] = {name: [_plain(list(area)) for area in areas] for name, areas in data["ignore"].items() if areas}
    if not data["ignore"]:
        del data["ignore"]
    room = setup.room.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    room = {kind: _plain(placed) if kind == "size" else {name: _plain(place) for name, place in placed.items() if place} for kind, placed in room.items()}
    room = {kind: placed for kind, placed in room.items() if placed}
    if room:
        data["room"] = room
    return data


def _plain(value):
    """Whole numbers written without a decimal point, as a person would type them."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return int(value) if isinstance(value, float) and value.is_integer() else value


def write_setup(path: Path, setup: SetupSection) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from macaw.store._io import write_text_atomic

    content = setup_content(setup)
    if path.is_file():
        write_text_atomic(path, _kept_as_written(path.read_text(encoding="utf-8"), content))
        return path
    # short lists of names or numbers on one line
    write_text_atomic(path, yaml.safe_dump(content, sort_keys=False, allow_unicode=True, default_flow_style=None))
    return path


def _kept_as_written(text: str, content: dict) -> str:
    """The file's text with this content, a person's comments and layout kept wherever a value did not change."""
    import io

    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap

    rt = YAML()
    rt.preserve_quotes = True
    rt.width = 4096
    rt.indent(mapping=2, sequence=4, offset=2)
    data = rt.load(text)
    if not isinstance(data, CommentedMap):
        data = CommentedMap()
    _merge(data, content)
    out = io.StringIO()
    rt.dump(data, out)
    return out.getvalue()


def _merge(old, new: dict) -> None:
    for key in [key for key in old if key not in new]:
        del old[key]
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(old.get(key), dict):
            _merge(old[key], value)
        elif key not in old or old[key] != value:
            old[key] = _round_trip(value)


def _round_trip(value):
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    if isinstance(value, dict):
        return CommentedMap((key, _round_trip(item)) for key, item in value.items())
    if isinstance(value, list):
        seq = CommentedSeq(_round_trip(item) for item in value)
        if not any(isinstance(item, dict) for item in value):
            seq.fa.set_flow_style()
        return seq
    return value


def device_people(setup: SetupSection, device: str) -> tuple[list[str], list[str]]:
    """(people the device is read for, people whose voices its sound carries)."""
    if device in setup.cameras:
        camera = setup.cameras[device]
        return camera.people(), sorted(camera.hears)
    if device in setup.microphones:
        hears = sorted(setup.microphones[device].hears)
        return hears, hears
    raise ConfigError(f"setup {setup.name} has no device {device}")
