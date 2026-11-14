"""Before a run starts: a model the steps fetch from the Hugging Face hub is already on this machine, or can be fetched."""

from __future__ import annotations

import os
from pathlib import Path

from macaw.exceptions import BackendNotAvailableError


def check_hub_model(repo_id: str, setting: str) -> None:
    """Stops when the model is neither downloaded nor fetchable here: offline, gated without access, or misnamed."""
    if Path(repo_id).expanduser().is_dir():
        return
    try:
        import huggingface_hub
        from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
    except ImportError:   # whatever loads the model brings the hub client with it and fails on its own
        return
    try:
        huggingface_hub.snapshot_download(repo_id, local_files_only=True)
        return
    except LocalEntryNotFoundError:
        pass
    except Exception:   # a cache this client cannot read is for the download to sort out
        return
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on"):
        raise BackendNotAvailableError(
            f"{setting} {repo_id} is not downloaded, and HF_HUB_OFFLINE keeps this machine from fetching it. "
            "Download it where the hub can be reached, into the cache HF_HOME names here."
        )
    # auth_check asks about the files, which a gated model's public card does not
    check = getattr(huggingface_hub, "auth_check", None) or huggingface_hub.model_info
    try:
        check(repo_id)
    except GatedRepoError:
        raise BackendNotAvailableError(
            f"{setting} {repo_id} is not downloaded, and the account logged in here may not fetch it: accept its terms "
            f"at https://huggingface.co/{repo_id} and log in with `hf auth login`."
        ) from None
    except RepositoryNotFoundError:
        raise BackendNotAvailableError(
            f"{setting} {repo_id} is not on the Hugging Face hub, or is private: check the name, or log in with `hf auth login`."
        ) from None
    except Exception as error:
        raise BackendNotAvailableError(
            f"{setting} {repo_id} is not downloaded, and the Hugging Face hub cannot be reached from here ({' '.join(str(error).split())[:200]})."
        ) from None
