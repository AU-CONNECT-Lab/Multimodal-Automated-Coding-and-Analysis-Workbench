"""Model weights fetched before the steps run, so no step reaches for the network while it works."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from macaw.config.hashing import hash_extractor_config
from macaw.config.loader import resolve_config
from macaw.extractors import get_extractor
from macaw.extractors.base import Extractor, Phase

# what the Hugging Face libraries read; the steps get it once their weights are on disk
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1"}
# loading on the CPU is all it takes to fetch
_ON_CPU = ("text.device=cpu", "face.device=cpu")


def offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def fetching_steps(names) -> list[str]:
    """Those of these steps that fetch model weights."""
    steps = [get_extractor(name) for name in names]
    return [cls.name for cls in steps if cls.phase != Phase.INTERACTIVE and cls.prefetch is not Extractor.prefetch]


def fetch_models(
    config_path: Path, names, recording_ids, *, strict: bool, echo: Callable[[str], None]
) -> list[str]:
    """Fetches each step's weights once per distinct setting across the recordings and returns the steps
    that failed. Unless strict, a step whose backend is not installed is left to fail when it runs."""
    failed = []
    configs = [resolve_config(config_path, cli_overrides=_ON_CPU)]
    configs += [resolve_config(config_path, recording_id=rid, cli_overrides=_ON_CPU) for rid in recording_ids]
    for name in fetching_steps(names):
        cls = get_extractor(name)
        seen = set()
        for config in configs:
            setting = hash_extractor_config(config, name)
            if setting in seen:
                continue
            seen.add(setting)
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
    return failed
