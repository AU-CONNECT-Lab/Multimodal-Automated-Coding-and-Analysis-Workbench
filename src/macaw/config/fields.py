from __future__ import annotations

from typing import Any

from pydantic import Field

HASHED_WHEN_SET = "hashed_when_set"


def added(*args: Any, **kwargs: Any) -> Any:
    """A key added once results exist: it stays out of every cache key while it holds its default."""
    return Field(*args, json_schema_extra={HASHED_WHEN_SET: True}, **kwargs)
