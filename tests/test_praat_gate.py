import numpy as np
import pytest

pytest.importorskip("parselmouth")

from macaw.config.schema import MacawConfig
from macaw.extractors.audio.prosody.backends import ParselmouthBackend, loud_share
from macaw.project.manifest import FeatureManifest, FeatureSpec
from macaw.qc.boilerplate import describe
from macaw.store.schema import Scale

SR = 16000


def _voice(seconds=20.0, amplitude=0.02, f0=150.0):
    t = np.arange(int(SR * seconds)) / SR
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    # half a second of voice, half a second of pause
    return amplitude * wave / np.abs(wave).max() * (np.sin(2 * np.pi * 1.0 * t) > 0)


def _knocked(audio):
    out = audio.copy()
    start, width = out.size // 2, int(0.02 * SR)
    out[start:start + width] += np.hanning(width) * np.sign(np.sin(np.arange(width)))
    return out


def _coverage(audio):
    backend = ParselmouthBackend()
    f0, hnr = backend.f0(audio, SR, 75.0, 500.0, 0.01), backend.hnr(audio, SR, 0.01)
    voiced = np.isfinite(f0)
    return voiced.mean(), np.isfinite(hnr[voiced]).mean()


def test_one_loud_knock_leaves_pitch_and_hnr_of_the_rest_of_the_track_as_they_were():
    plain, knocked = _voice(), _knocked(_voice())
    assert loud_share(knocked, SR) < 0.1
    voiced, with_hnr = _coverage(plain)
    knocked_voiced, knocked_with_hnr = _coverage(knocked)
    assert voiced > 0.3 and with_hnr > 0.9
    assert knocked_voiced == pytest.approx(voiced, abs=0.02)
    assert knocked_with_hnr == pytest.approx(with_hnr, abs=0.05)


def test_the_loud_level_of_silence_or_of_a_too_short_track_leaves_praat_as_it_is():
    assert loud_share(np.zeros(SR), SR) == 1.0
    assert loud_share(np.ones(10), SR) == 1.0


def test_the_methods_text_says_how_the_silence_thresholds_were_taken():
    spec = FeatureSpec(name="f0_hz", description="", unit=None, scale=Scale.FRAME, group="prosody", extractor="prosody.frame",
                       extractor_version="0.3.0", backend="parselmouth", backend_version="0.4.7")
    manifest = FeatureManifest()
    manifest.extend([spec])
    config = MacawConfig.model_validate({"project": {"name": "t", "raw_dir": "raw", "output_dir": "out"}})
    text = " ".join(" ".join(part) for part in describe(manifest, config).values())
    assert "relative to the 99.5th percentile of the track's 10 ms peak amplitudes rather than to its maximum" in text
