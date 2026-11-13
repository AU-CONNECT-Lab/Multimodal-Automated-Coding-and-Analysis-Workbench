import importlib.metadata

import pytest

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors.audio.content.asr import WhisperXBackend
from macaw.extractors.audio.content.diarization import PyannoteBackend, refuse_older


def _installed(monkeypatch, versions: dict[str, str]) -> None:
    def version(name: str) -> str:
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", version)


def test_whisperx_before_pyannote_4_is_refused_by_name(monkeypatch):
    _installed(monkeypatch, {"whisperx": "3.7.2"})
    backend = WhisperXBackend.__new__(WhisperXBackend)
    with pytest.raises(BackendNotAvailableError, match=r"whisperx 3\.7\.2 is installed; MACAW needs 3\.8 or later"):
        backend.check_available()


def test_pyannote_3_is_refused_before_its_import_fails(monkeypatch):
    _installed(monkeypatch, {"pyannote.audio": "3.4.0"})
    with pytest.raises(BackendNotAvailableError, match=r"pyannote\.audio 3\.4\.0 is installed; MACAW needs 4\.0 or later"):
        PyannoteBackend().check_available()


@pytest.mark.parametrize("installed", ["3.8.4", "3.8.7rc1", "4.1"])
def test_a_release_at_or_past_the_floor_passes(monkeypatch, installed):
    _installed(monkeypatch, {"whisperx": installed})
    refuse_older("whisperx", (3, 8), "reason")


def test_a_missing_package_is_left_to_the_import(monkeypatch):
    _installed(monkeypatch, {})
    refuse_older("whisperx", (3, 8), "reason")
