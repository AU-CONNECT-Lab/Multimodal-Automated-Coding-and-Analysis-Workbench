"""Model weights fetched before the steps run, so no step reaches for the network while it works."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from pathlib import Path

from macaw.config.hashing import hash_extractor_config, hash_value
from macaw.config.loader import resolve_config
from macaw.extractors import get_extractor
from macaw.extractors.base import Extractor, Phase
from macaw.project.layout import layout_for

# what the Hugging Face libraries read; the steps get it once their weights are on disk
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1"}
# loading on the CPU is all it takes to fetch
_ON_CPU = ("text.device=cpu", "face.device=cpu")
# where the libraries keep downloaded weights
_CACHE_PLACES = ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME")


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _ready_path(work_dir: Path) -> Path:
    return work_dir / "weights_ready.json"


def _machine() -> str:
    return hash_value([socket.gethostname(), {name: os.environ.get(name, "") for name in _CACHE_PLACES}])


def _read_ready(work_dir: Path) -> dict[str, str]:
    try:
        return json.loads(_ready_path(work_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_ready(work_dir: Path, ready: dict[str, str]) -> None:
    path = _ready_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ready, indent=1, sort_keys=True), encoding="utf-8")


def forget(work_dir: Path, names) -> None:
    """Has these steps' weights checked again on this machine, as after a task of theirs failed."""
    ready = _read_ready(work_dir)
    keys = [key for key in ready if key.split("|")[1] in set(names) and key.startswith(f"{_machine()}|")]
    if keys:
        _write_ready(work_dir, {key: value for key, value in ready.items() if key not in keys})


def fetching_steps(names) -> list[str]:
    """Those of these steps that fetch model weights."""
    steps = [get_extractor(name) for name in names]
    return [cls.name for cls in steps if cls.phase != Phase.INTERACTIVE and cls.prefetch is not Extractor.prefetch]


def fetch_models(
    config_path: Path, names, recording_ids, *, strict: bool, echo: Callable[[str], None]
) -> list[str]:
    """Fetches each step's weights once per distinct setting across the recordings and returns the steps
    that failed. Unless strict, a step whose backend is not installed is left to fail when it runs.
    Offline, a setting whose weights were found on this machine before is not loaded again."""
    failed = []
    configs = [resolve_config(config_path, cli_overrides=_ON_CPU)]
    configs += [resolve_config(config_path, recording_id=rid, cli_overrides=_ON_CPU) for rid in recording_ids]
    work_dir = layout_for(configs[0], create=False).work_dir
    ready = _read_ready(work_dir)
    for name in fetching_steps(names):
        cls = get_extractor(name)
        seen = set()
        for config in configs:
            setting = hash_extractor_config(config, name)
            if setting in seen:
                continue
            seen.add(setting)
            key = f"{_machine()}|{name}|{cls.version}|{setting}"
            if offline() and key in ready:
                echo(f"fetched {name}")
                continue
            step = cls(config)
            try:
                step.check_available()
            except Exception as error:
                if strict:
                    failed.append(name)
                    echo(f"failed {name}: {' '.join(str(error).split())}")
                else:
                    echo(f"skipped {name}: {' '.join(str(error).split())}")
                break
            try:
                step.prefetch()
            except Exception as error:
                failed.append(name)
                echo(f"failed {name}: {' '.join(str(error).split())}")
                break
            echo(f"fetched {name}")
            ready[key] = cls.name
            _write_ready(work_dir, ready)
    return failed
