"""Shared fixtures, plus a bridge that lets the numeric tests run before the full
environment exists. Both stubs disappear the moment the real package is importable."""

from __future__ import annotations

import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_MISSING = object()


class _FieldInfo:
    def __init__(self, default=_MISSING, default_factory=None):
        self.default = default
        self.default_factory = default_factory

    def resolve(self):
        if self.default_factory is not None:
            return self.default_factory()
        return None if self.default is _MISSING else self.default


class _BaseModel:
    """Enough of pydantic to read a section's defaults. These tests never validate config."""

    def __init__(self, **kwargs):
        for name, value in self._field_defaults().items():
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def _field_defaults(cls):
        out = {}
        for klass in reversed(cls.__mro__):
            for name in getattr(klass, "__annotations__", {}):
                if name.startswith("_") or name == "model_config":
                    continue
                value = getattr(klass, name, _MISSING)
                if isinstance(value, _FieldInfo):
                    out[name] = value.resolve()
                elif value is not _MISSING:
                    out[name] = value
                else:
                    out.setdefault(name, None)
        return out


def _install_stubs() -> None:
    try:
        import pydantic  # noqa: F401
    except ImportError:
        pydantic = types.ModuleType("pydantic")
        pydantic.BaseModel = _BaseModel
        pydantic.Field = lambda default=_MISSING, *, default_factory=None, **_: _FieldInfo(
            default, default_factory
        )
        pydantic.ConfigDict = lambda **kwargs: dict(kwargs)
        pydantic.ValidationError = type("ValidationError", (Exception,), {})
        sys.modules["pydantic"] = pydantic

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pa = types.ModuleType("pyarrow")
        pa.Table = type("Table", (), {})
        parquet = types.ModuleType("pyarrow.parquet")
        dataset = types.ModuleType("pyarrow.dataset")
        dataset.Expression = type("Expression", (), {})
        pa.parquet, pa.dataset = parquet, dataset
        sys.modules["pyarrow"] = pa
        sys.modules["pyarrow.parquet"] = parquet
        sys.modules["pyarrow.dataset"] = dataset

        from macaw.store import schema

        schema._PANDAS_DTYPES["string"] = "string"


_install_stubs()

SAMPLE_RATE = 16000


@pytest.fixture
def make_transcript():
    def build(rows):
        return pd.DataFrame(
            [
                {
                    "subject_id": speaker,
                    "t_start": float(start),
                    "t_end": float(end),
                    "text": text,
                    "n_words": len(text.split()),
                }
                for speaker, start, end, text in rows
            ]
        )

    return build


@pytest.fixture
def make_turns():
    def build(rows):
        return pd.DataFrame(
            [
                {
                    "subject_id": speaker,
                    "turn_index": index,
                    "t_start": float(start),
                    "t_end": float(end),
                    "n_utterances": 1,
                    "n_words": 1,
                    "text": "",
                }
                for index, (speaker, start, end) in enumerate(rows)
            ]
        )

    return build


@pytest.fixture
def make_recording():
    def build(subject_ids, entities=None):
        return SimpleNamespace(
            recording_id="sub-003_ses-12mo",
            subjects=[SimpleNamespace(subject_id=name, role=name) for name in subject_ids],
            entities=lambda: entities
            or {"group_id": "003", "session_id": "12mo", "task": None, "run": None},
        )

    return build


@pytest.fixture
def make_prep():
    def build(t_origin_s=0.0, duration_s=60.0):
        return SimpleNamespace(t_origin_s=t_origin_s, duration_s=duration_s)

    return build


@pytest.fixture
def write_wav():
    def build(path, samples):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = np.clip(samples, -1.0, 1.0)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(SAMPLE_RATE)
            handle.writeframes((data * 32767).astype("<i2").tobytes())
        return path

    return build
