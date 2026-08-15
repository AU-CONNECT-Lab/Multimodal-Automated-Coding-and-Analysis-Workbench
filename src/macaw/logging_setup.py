"""Progress goes to the log on stderr, results go to stdout. A caller piping recording ids
into another command must still get them under --quiet."""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "macaw"
_FORMAT = "%(levelname)s %(name)s: %(message)s"


def setup_logging(*, verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """--verbose wins over --quiet: asking for more is the more specific request."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO if verbose else logging.ERROR if quiet else logging.WARNING)
    # replace rather than add: this runs once per command, and a handler kept from an earlier
    # call still points at the stderr that was current back then
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    # the root logger belongs to whoever imports macaw, not to us
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
