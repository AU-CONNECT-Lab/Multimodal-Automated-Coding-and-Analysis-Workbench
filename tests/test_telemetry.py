from __future__ import annotations

import os

from macaw.telemetry import QUIET_ENVIRONMENT, quiet_environment


def test_every_switch_is_on_by_default():
    assert quiet_environment({}) == QUIET_ENVIRONMENT


def test_a_choice_the_user_made_is_kept():
    assert quiet_environment({"PYANNOTE_METRICS_ENABLED": "1"})["PYANNOTE_METRICS_ENABLED"] == "1"


def test_importing_the_package_sets_them_for_this_process():
    import macaw  # noqa: F401

    assert all(key in os.environ for key in QUIET_ENVIRONMENT)
