from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from macaw.config.schema import MacawConfig
from macaw.extractors.prep import _window
from macaw.interface.pages.segments import analysed_text
from macaw.project.manifest import FeatureManifest, FeatureSpec
from macaw.project.recording import Recording
from macaw.qc.boilerplate import describe
from macaw.store.schema import Scale


def _config(**segment):
    return MacawConfig.model_validate({"project": {"name": "t", "raw_dir": "raw", "output_dir": "out"}, "segment": segment})


def _recording(keep_last=None, end=None, lengths=(400.0, 380.0)):
    media = [SimpleNamespace(duration_s=length, span_s=length, offset_s=0.0) for length in lengths]
    return Recording(recording_id="r", root=None, media=media, t_task_end=end, t_keep_last=keep_last)


def test_the_last_seconds_are_counted_back_from_the_latest_end():
    assert _window(_recording(keep_last=300)) == (100.0, 400.0)


def test_the_last_seconds_are_counted_back_from_the_task_end_when_set():
    assert _window(_recording(keep_last=300, end=350)) == (50.0, 350.0)


def test_a_recording_shorter_than_the_part_kept_is_kept_whole():
    assert _window(_recording(keep_last=600)) == (0.0, 400.0)


def test_a_start_and_a_length_from_the_end_cannot_both_be_set():
    with pytest.raises(ValidationError, match="set one"):
        _config(task_start_s=10, keep_last_s=300)
    with pytest.raises(ValidationError):
        _config(keep_last_s=0)


def test_the_methods_text_and_the_segments_page_say_what_was_kept():
    manifest = FeatureManifest()
    manifest.extend([FeatureSpec(name="n_media", description="", unit=None, scale=Scale.EVENT, group="prep", extractor="prep.media", extractor_version="0.1.0")])
    config = _config(keep_last_s=300)
    text = " ".join(" ".join(part) for part in describe(manifest, config, short=(2, 47)).values())
    assert "trimmed to the task window of the last 300 s before the end of the recording" in text
    assert "Two of the 47 recordings were shorter than the part to be kept and were analysed whole." in text
    assert analysed_text(config.segment) == "Analysed part: the last 300 s before the end."
    assert analysed_text(_config(keep_last_s=300, task_end_s=350).segment) == "Analysed part: the last 300 s before 350 s."
