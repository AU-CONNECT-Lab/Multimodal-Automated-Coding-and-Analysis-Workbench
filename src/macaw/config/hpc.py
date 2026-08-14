"""Cluster settings, loaded on their own. Deliberately outside MacawConfig: where a job
runs must not change any hash, or a cache built on a laptop would miss on the cluster."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from macaw.exceptions import ConfigValidationError, InputNotFoundError

HPC_PATH = Path(__file__).parent / "hpc.yaml"
HPC_FILENAME = "hpc.yaml"
HPC_ENV_VAR = "MACAW_HPC_CONFIG"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ResourceProfile(_Section):
    partition: str | None = None
    account: str | None = None
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 1
    memory: str | None = None
    memory_per_cpu: str | None = None
    time: str | None = None
    gres: str | None = None
    array_limit: int | None = None
    additional_args: list[str] = Field(default_factory=list)


class HPCConfig(_Section):
    defaults: ResourceProfile = Field(default_factory=ResourceProfile)
    resource_profiles: dict[str, ResourceProfile] = Field(default_factory=dict)
    module_load: list[str] = Field(default_factory=list)
    env_activate: list[str] = Field(default_factory=list)
    hf_home: Path | None = None


def find_hpc_config(explicit: Path | None = None, project_yaml: Path | None = None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise InputNotFoundError(f"--hpc-config points at a missing file: {explicit}")
        return explicit

    from_env = os.environ.get(HPC_ENV_VAR)
    if from_env:
        path = Path(from_env)
        if not path.is_file():
            raise InputNotFoundError(f"${HPC_ENV_VAR} points at a missing file: {path}")
        return path

    if project_yaml is not None:
        beside = project_yaml.parent / HPC_FILENAME
        if beside.is_file():
            return beside

    return HPC_PATH


def load_hpc_config(explicit: Path | None = None, project_yaml: Path | None = None) -> HPCConfig:
    path = find_hpc_config(explicit, project_yaml)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return HPCConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigValidationError(f"{path}: {exc}") from exc


def resolve_profile(hpc: HPCConfig, name: str) -> ResourceProfile:
    """Defaults underneath, profile on top. Only keys the profile actually wrote win, so
    a profile that omits `partition` still inherits it instead of resetting it to None."""
    profile = hpc.resource_profiles.get(name)
    if profile is None:
        available = ", ".join(sorted(hpc.resource_profiles)) or "none defined"
        raise ConfigValidationError(
            f"Resource profile '{name}' is not defined in hpc.yaml. Available: {available}"
        )
    return hpc.defaults.model_copy(update=profile.model_dump(exclude_unset=True))
