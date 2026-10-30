"""A project's own Python files in project.scripts_dir, imported whenever a project config is read."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

from macaw.exceptions import ConfigError

MODULE_PREFIX = "macaw_project_scripts"

_loaded: dict[Path, ModuleType] = {}
_failed: dict[Path, str] = {}


def load_project_scripts(directory: Path | None) -> None:
    # a file that fails is reported where it is used, so one broken script does not stop every command
    if directory is None or not directory.is_dir():
        return
    from macaw.extractors import load_builtin_extractors

    # built-ins first, so a script reusing a built-in's name is the one that fails
    load_builtin_extractors()
    for path in sorted(directory.glob("*.py")):
        path = path.resolve()
        if path in _loaded or path in _failed:
            continue
        name = f"{MODULE_PREFIX}.{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            _register_extractors(module)
        except Exception as error:
            del sys.modules[name]
            _failed[path] = f"{type(error).__name__}: {error}"
            logging.getLogger("macaw").warning("project script %s failed to load: %s", path, _failed[path])
            continue
        _loaded[path] = module


def _register_extractors(module: ModuleType) -> None:
    """A class made from the extractor template has no decorator: installed, its entry point registers it."""
    from macaw.extractors import Extractor, register_extractor

    for value in vars(module).values():
        if (
            isinstance(value, type) and issubclass(value, Extractor) and value.__module__ == module.__name__
            and not inspect.isabstract(value) and isinstance(getattr(value, "name", None), str)
        ):
            register_extractor(value)


def script_path(directory: Path, reference: str) -> Path:
    return directory / f"{reference.rpartition('.')[0]}.py"


def script_function(directory: Path, reference: str) -> Callable:
    """reference is file.function, the file being directory/file.py."""
    path = script_path(directory, reference)
    if not path.is_file():
        raise ConfigError(f"{reference}: there is no {path}")
    load_project_scripts(directory)
    path = path.resolve()
    if path in _failed:
        raise ConfigError(f"{reference}: {path} failed to load: {_failed[path]}")
    function = getattr(_loaded[path], reference.rpartition(".")[2], None)
    if not callable(function):
        raise ConfigError(f"{reference}: {path} has no function {reference.rpartition('.')[2]}")
    return function


def script_file_of(cls: type) -> Path | None:
    """The file an extractor defined in a project script comes from, so editing it reruns the step."""
    if not cls.__module__.startswith(f"{MODULE_PREFIX}."):
        return None
    module = sys.modules.get(cls.__module__)
    return Path(module.__file__) if module is not None and module.__file__ else None
