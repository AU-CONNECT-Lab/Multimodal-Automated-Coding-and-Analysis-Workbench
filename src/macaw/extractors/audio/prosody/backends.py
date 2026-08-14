from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend

BACKEND_GROUP = "prosody"

# sound pressure reference; the absolute level of a decoded file is meaningless anyway, only
# differences within one track are, which is why intensity is not comparable across subjects
_PRESSURE_REF = 2e-5


def frame_times(n_samples: int, sr: float, step_s: float) -> np.ndarray:
    """The one grid every backend has to land on, so their columns line up row by row."""
    step = max(step_s, 1e-6)
    count = int(np.floor(n_samples / (step * sr))) + 1
    return np.arange(max(count, 1), dtype=float) * step


def _to_grid(values: np.ndarray, src_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Nearest neighbour, never interpolation: averaging across an unvoiced gap would invent
    a pitch that was never there."""
    values = np.asarray(values, dtype=float)
    src_times = np.asarray(src_times, dtype=float)
    shape = (target_times.size,) + values.shape[1:]
    if src_times.size == 0 or values.size == 0:
        return np.full(shape, np.nan)

    right = np.clip(np.searchsorted(src_times, target_times), 0, src_times.size - 1)
    left = np.maximum(right - 1, 0)
    pick = np.where(
        np.abs(src_times[right] - target_times) <= np.abs(src_times[left] - target_times),
        right,
        left,
    )
    return values[pick]


def load_audio(path: Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, float]:
    """Prepared tracks are plain 16-bit wav, so the standard library reads them and no
    external tool is needed for the audio feature path."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                width = handle.getsampwidth()
                channels = handle.getnchannels()
                rate = float(handle.getframerate())
                raw = handle.readframes(handle.getnframes())
            if width == 2 and (sample_rate is None or rate == sample_rate):
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
                if channels > 1:
                    samples = samples.reshape(-1, channels).mean(axis=1)
                return samples, rate
        except wave.Error:
            pass                       # a non-PCM wav still decodes through the external tool

    from macaw.media.ffmpeg import decode_audio

    rate = int(sample_rate or 16000)
    return decode_audio(path, sample_rate=rate), float(rate)


class ProsodyBackend(Backend):
    capabilities: frozenset[str] = frozenset({"f0", "intensity"})

    def f0(self, audio: np.ndarray, sr: float, f0_min: float, f0_max: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def intensity(self, audio: np.ndarray, sr: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def hnr(self, audio: np.ndarray, sr: float, step_s: float) -> np.ndarray:
        raise NotImplementedError

    def formants(self, audio: np.ndarray, sr: float, step_s: float, n_formants: int = 4) -> np.ndarray:
        raise NotImplementedError

    def supports(self, quantity: str) -> bool:
        return quantity in self.capabilities


@register_backend(BACKEND_GROUP)
class ParselmouthBackend(ProsodyBackend):
    name = "parselmouth"
    capabilities = frozenset({"f0", "intensity", "hnr", "formants"})

    @property
    def version(self) -> str:
        import parselmouth

        return getattr(parselmouth, "VERSION", "unknown")

    def check_available(self) -> None:
        try:
            import parselmouth  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "The 'parselmouth' backend needs praat-parselmouth: pip install praat-parselmouth"
            ) from error

    def _sound(self, audio: np.ndarray, sr: float):
        import parselmouth

        return parselmouth.Sound(np.asarray(audio, dtype=float), sampling_frequency=sr)

    def f0(self, audio, sr, f0_min, f0_max, step_s):
        pitch = self._sound(audio, sr).to_pitch_ac(
            time_step=step_s, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        values = np.asarray(pitch.selected_array["frequency"], dtype=float)
        values[values == 0.0] = np.nan          # the backend marks unvoiced frames as zero
        return _to_grid(values, np.asarray(pitch.xs()), frame_times(audio.size, sr, step_s))

    def intensity(self, audio, sr, step_s):
        # the analysis window is tied to the pitch floor, and 50 Hz is low enough to cover
        # any speaker without smearing the contour
        contour = self._sound(audio, sr).to_intensity(time_step=step_s, minimum_pitch=50.0)
        values = np.asarray(contour.values[0], dtype=float)
        return _to_grid(values, np.asarray(contour.xs()), frame_times(audio.size, sr, step_s))

    def hnr(self, audio, sr, step_s):
        harmonicity = self._sound(audio, sr).to_harmonicity_cc(time_step=step_s)
        values = np.asarray(harmonicity.values[0], dtype=float)
        values[values <= -200.0] = np.nan       # the backend's sentinel for undefined
        return _to_grid(values, np.asarray(harmonicity.xs()), frame_times(audio.size, sr, step_s))

    def formants(self, audio, sr, step_s, n_formants=4):
        formant = self._sound(audio, sr).to_formant_burg(
            time_step=step_s, max_number_of_formants=n_formants + 1
        )
        times = frame_times(audio.size, sr, step_s)
        out = np.full((times.size, n_formants), np.nan)
        for column in range(n_formants):
            for row, t in enumerate(times):
                out[row, column] = formant.get_value_at_time(column + 1, t)
        return out


@register_backend(BACKEND_GROUP)
class LibrosaBackend(ProsodyBackend):
    name = "librosa"
    capabilities = frozenset({"f0", "intensity"})

    @property
    def version(self) -> str:
        import librosa

        return librosa.__version__

    def check_available(self) -> None:
        try:
            import librosa  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "The 'librosa' backend needs librosa: pip install librosa"
            ) from error

    def f0(self, audio, sr, f0_min, f0_max, step_s):
        import librosa

        hop = max(int(round(step_s * sr)), 1)
        window = max(int(round(3.0 * sr / f0_min)), 2 * hop)
        values, _, _ = librosa.pyin(
            np.asarray(audio, dtype=float),
            fmin=f0_min,
            fmax=f0_max,
            sr=sr,
            frame_length=window,
            hop_length=hop,
        )
        times = np.arange(values.size) * hop / sr
        return _to_grid(values, times, frame_times(audio.size, sr, step_s))

    def intensity(self, audio, sr, step_s):
        import librosa

        hop = max(int(round(step_s * sr)), 1)
        window = max(4 * hop, 2)
        rms = librosa.feature.rms(
            y=np.asarray(audio, dtype=float), frame_length=window, hop_length=hop
        )[0]
        values = _db(np.asarray(rms, dtype=float))
        times = np.arange(values.size) * hop / sr
        return _to_grid(values, times, frame_times(audio.size, sr, step_s))


@register_backend(BACKEND_GROUP)
class BuiltinBackend(ProsodyBackend):
    """Autocorrelation f0. Last resort so a run never dies for lack of a dependency."""

    name = "builtin"
    version = "0.1.0"
    capabilities = frozenset({"f0", "intensity", "hnr"})

    # an autocorrelation peak this weak is noise, not a period
    voicing_threshold = 0.45
    # a peak within this much of the best one is preferred if it comes first; twice the true
    # period correlates just as well as the period itself, and taking the tallest peak is
    # what makes a tracker report half the pitch
    octave_tolerance = 0.85
    # frames are this many periods of the lowest pitch searched for, the usual compromise
    # between resolving a low voice and tracking a fast contour
    periods_per_window = 3.0
    _block = 2048

    def check_available(self) -> None:
        return None

    def _window_samples(self, sr: float, f0_min: float) -> int:
        return max(int(round(self.periods_per_window * sr / f0_min)), 8)

    def _blocks(self, audio: np.ndarray, sr: float, step_s: float, window: int):
        """Centred frames on the shared grid, yielded in blocks so a long recording never
        materialises the whole frame matrix at once."""
        times = frame_times(audio.size, sr, step_s)
        half = window // 2
        padded = np.pad(np.asarray(audio, dtype=float), (half, window), mode="constant")
        starts = np.round(times * sr).astype(int)
        for begin in range(0, times.size, self._block):
            chunk = starts[begin : begin + self._block]
            index = chunk[:, None] + np.arange(window)[None, :]
            yield begin, padded[index]

    def _autocorrelation(self, segments: np.ndarray, lag_max: int) -> np.ndarray:
        window = segments.shape[1]
        segments = segments - segments.mean(axis=1, keepdims=True)
        size = int(2 ** np.ceil(np.log2(2 * window)))
        spectrum = np.fft.rfft(segments, size, axis=1)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), size, axis=1)[:, : lag_max + 1]

        zero = correlation[:, :1].copy()
        zero[zero <= 0] = np.nan
        # overlap shrinks with lag, so a raw autocorrelation always slopes down and would
        # bias the peak towards short lags
        taper = window / np.maximum(window - np.arange(lag_max + 1), 1.0)
        return correlation / zero * taper

    def _first_strong_peak(self, band: np.ndarray) -> np.ndarray:
        """The earliest local maximum that comes close to the tallest one. Picking the
        tallest instead lands on a multiple of the period whenever the voice is very
        regular, which reads as the speaker dropping an octave."""
        filled = np.where(np.isnan(band), -np.inf, band)
        padded = np.pad(filled, ((0, 0), (1, 1)), constant_values=-np.inf)
        is_peak = (filled >= padded[:, :-2]) & (filled >= padded[:, 2:])

        best = filled.max(axis=1, keepdims=True)
        acceptable = is_peak & (filled >= best * self.octave_tolerance) & np.isfinite(filled)
        return np.where(acceptable.any(axis=1), acceptable.argmax(axis=1), filled.argmax(axis=1))

    def f0(self, audio, sr, f0_min, f0_max, step_s):
        audio = np.asarray(audio, dtype=float)
        times = frame_times(audio.size, sr, step_s)
        window = self._window_samples(sr, f0_min)
        lag_min = max(int(np.floor(sr / f0_max)), 1)
        lag_max = min(int(np.ceil(sr / f0_min)), window - 1)
        if lag_max <= lag_min:
            return np.full(times.size, np.nan)

        out = np.full(times.size, np.nan)
        for begin, segments in self._blocks(audio, sr, step_s, window):
            normalised = self._autocorrelation(segments, lag_max)
            band = normalised[:, lag_min : lag_max + 1]
            peak = self._first_strong_peak(band)
            rows = np.arange(band.shape[0])
            strength = band[rows, peak]
            lag = peak + lag_min

            # parabolic interpolation around the peak: at 16 kHz a period is a few hundred
            # samples, and rounding it to a whole one is a visible error at high pitch
            left = normalised[rows, np.maximum(lag - 1, 0)]
            right = normalised[rows, np.minimum(lag + 1, lag_max)]
            denominator = left - 2.0 * strength + right
            shift = np.where(np.abs(denominator) > 1e-12, 0.5 * (left - right) / denominator, 0.0)
            refined = lag + np.clip(shift, -0.5, 0.5)

            voiced = np.isfinite(strength) & (strength >= self.voicing_threshold) & (refined > 0)
            block = np.full(band.shape[0], np.nan)
            block[voiced] = sr / refined[voiced]
            out[begin : begin + block.size] = block
        return out

    def intensity(self, audio, sr, step_s):
        audio = np.asarray(audio, dtype=float)
        times = frame_times(audio.size, sr, step_s)
        window = self._window_samples(sr, 50.0)
        out = np.full(times.size, np.nan)
        for begin, segments in self._blocks(audio, sr, step_s, window):
            block = _db(np.sqrt(np.mean(np.square(segments), axis=1)))
            out[begin : begin + block.size] = block
        return out

    def hnr(self, audio, sr, step_s):
        """The autocorrelation peak is the harmonic fraction of the frame's energy, so the
        harmonics-to-noise ratio follows from it without a second analysis."""
        audio = np.asarray(audio, dtype=float)
        times = frame_times(audio.size, sr, step_s)
        window = self._window_samples(sr, 75.0)
        lag_min = max(int(np.floor(sr / 500.0)), 1)
        lag_max = min(int(np.ceil(sr / 75.0)), window - 1)
        if lag_max <= lag_min:
            return np.full(times.size, np.nan)

        out = np.full(times.size, np.nan)
        for begin, segments in self._blocks(audio, sr, step_s, window):
            band = self._autocorrelation(segments, lag_max)[:, lag_min : lag_max + 1]
            strength = np.nanmax(np.where(np.isnan(band), -np.inf, band), axis=1)
            usable = np.isfinite(strength)
            block = np.full(strength.shape, np.nan)
            bounded = np.clip(strength[usable], 1e-6, 1.0 - 1e-6)
            block[usable] = 10.0 * np.log10(bounded / (1.0 - bounded))
            out[begin : begin + block.size] = block
        return out


def _db(rms: np.ndarray) -> np.ndarray:
    rms = np.asarray(rms, dtype=float)
    return np.where(rms > 0, 20.0 * np.log10(np.maximum(rms, 1e-12) / _PRESSURE_REF), np.nan)


def resolve_backend(chain: list[str]) -> ProsodyBackend:
    """First available backend in the chain; logs which one was picked."""
    from macaw.extractors import get_backend

    tried: list[str] = []
    for name in chain:
        if name in tried:
            continue
        tried.append(name)
        backend = get_backend(BACKEND_GROUP, name)()
        try:
            backend.check_available()
        except BackendNotAvailableError:
            continue
        return backend
    raise BackendNotAvailableError(
        f"No prosody backend is installed. Tried: {', '.join(tried) or 'nothing'}"
    )
