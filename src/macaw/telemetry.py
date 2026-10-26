"""Off switches for usage reporting, self-updating and self-installing in the tools pymacaw calls."""

from __future__ import annotations

import os

# each is a dependency's documented switch; a value the user has set is left alone
QUIET_ENVIRONMENT: dict[str, str] = {
    "DO_NOT_TRACK": "1",                 # the shared convention, honoured by the model hub client
    "HF_HUB_DISABLE_TELEMETRY": "1",     # model downloads for speech recognition and diarisation
    "PYANNOTE_METRICS_ENABLED": "0",     # file durations and speaker counts from the diarisation pipeline
    "YOLO_OFFLINE": "1",                 # usage events, pip installs, update checks and weight downloads
}


def quiet_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    for key, value in QUIET_ENVIRONMENT.items():
        environment.setdefault(key, value)
    return environment


def quiet_this_process() -> None:
    """Has to run before the dependencies are imported: several read these once, at import."""
    for key, value in QUIET_ENVIRONMENT.items():
        os.environ.setdefault(key, value)


quiet_this_process()
