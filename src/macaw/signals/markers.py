"""A file placed by the events its recording software marked, paired with marks a person put on the video under the same names."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from macaw.media.drift import MIN_SEGMENTS, estimate_drift

# s; a mark pairs with a file marker of its name this near where the best offset puts it
PAIR_S = 2.0
# s; a pair further than this from the offset the others give asks a person to look
RESIDUAL_S = 0.5
# s per s; two clocks drifting apart faster than this are taken as marks paired wrongly
MAX_DRIFT = 0.005


@dataclass
class MarkerMatch:
    offset_s: float                                      # recording time minus the file's own time, over all pairs
    pairs: list[tuple[str, float, float]]                # (name, recording time, the file's own time)
    unpaired: list[str] = field(default_factory=list)   # the file's names a person marked that found no partner
    # offsets of the other places the marks fit as well, as when one mark names an event the file has many times
    alternatives: list[float] = field(default_factory=list)
    drift_ppm: float | None = None
    drift_r2: float | None = None
    drift_intercept_s: float | None = None

    @property
    def span_s(self) -> float:
        """How much of the recording the pairs cover, over which a drift between them is believed."""
        times = [at for _, at, _ in self.pairs]
        return max(times) - min(times)

    def residuals(self, offset_at=None) -> list[float]:
        """Each pair's own offset less the one fitted, s."""
        fitted = offset_at or (lambda at: self.offset_s)
        return [at - own - fitted(at) for _, at, own in self.pairs]


def marker_key(name: str) -> str:
    """Names compared without case or spaces, and 1, 1.0 and 01 as the same number."""
    text = str(name).strip().lower()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else repr(number)


def match_markers(marks: list[tuple[str, float]], markers: list[tuple[str, float]], prefer: float | None = None) -> MarkerMatch | None:
    """The offset that pairs the most marks (name, recording time) with the file's markers (name, own time) of their
    names, of several that fit as well the one nearest prefer; None when no mark has a name the file carries."""
    own: dict[str, list[float]] = {}
    for name, onset in markers:
        times = own.setdefault(marker_key(name), [])
        # a file may list one event twice
        if not any(abs(onset - t) < 1e-6 for t in times):
            times.append(float(onset))
    named = [(marker_key(name), float(at)) for name, at in marks if marker_key(name) in own]
    if not named:
        return None
    fits: list[tuple[int, float, float, list[tuple[str, float, float]]]] = []
    for candidate in sorted({at - onset for key, at in named for onset in own[key]}):
        line = (candidate, 0.0)
        pairs = _paired(named, own, line)
        # a clock drifting apart walks later marks off a fixed offset; follow the line the pairs give
        for _ in range(3):
            fitted = _line(pairs) or line
            more = _paired(named, own, fitted)
            if len(more) <= len(pairs):
                break
            pairs, line = more, fitted
        intercept, slope = _line(pairs) or (float(np.mean([at - onset for _, at, onset in pairs])), 0.0)
        spread = float(np.sum([(at - onset - intercept - slope * at) ** 2 for _, at, onset in pairs]))
        fits.append((len(pairs), spread, float(np.mean([at - onset for _, at, onset in pairs])), pairs))
    most = max(count for count, *_ in fits)
    best = min((fit for fit in fits if fit[0] == most), key=lambda fit: fit[1])
    # places that pair as many marks as closely as the pairs are believed, each counted once
    alike = [fit for fit in fits if fit[0] == most and fit[1] <= max(best[1], most * RESIDUAL_S**2)]
    places: list[tuple[int, float, float, list]] = []
    for fit in sorted(alike, key=lambda fit: fit[1]):
        if all(abs(fit[2] - other[2]) > PAIR_S for other in places):
            places.append(fit)
    if prefer is not None:
        best = min(places, key=lambda fit: abs(fit[2] - prefer))
    pairs = best[3]
    paired = {(key, at) for key, at, _ in pairs}
    names = {marker_key(name): str(name).strip() for name, _ in marks}
    found = MarkerMatch(
        offset_s=float(np.mean([at - onset for _, at, onset in pairs])),
        pairs=[(names[key], at, onset) for key, at, onset in pairs],
        unpaired=[names[key] for key, at in named if (key, at) not in paired],
        alternatives=[fit[2] for fit in places if fit is not best],
    )
    if len(pairs) >= MIN_SEGMENTS:
        found.drift_ppm, found.drift_r2, found.drift_intercept_s = estimate_drift([(at, at - onset) for _, at, onset in pairs])
    return found


def _paired(named, own, line: tuple[float, float]) -> list[tuple[str, float, float]]:
    """(key, recording time, own time) for each mark whose name has an event near where the line (intercept, slope of
    the offset over recording time) expects it, each event taken once."""
    intercept, slope = line
    pairs, used = [], set()
    for key, at in named:
        expected = at - intercept - slope * at
        options = [onset for onset in own[key] if abs(onset - expected) <= PAIR_S and (key, onset) not in used]
        if options:
            onset = min(options, key=lambda t: abs(t - expected))
            used.add((key, onset))
            pairs.append((key, at, onset))
    return pairs


def _line(pairs) -> tuple[float, float] | None:
    """The offset over recording time fitted to the pairs, when they span enough to give a slope a clock could have."""
    times = np.array([at for _, at, _ in pairs])
    if times.size < 2 or np.ptp(times) <= 0:
        return None
    slope, intercept = np.polyfit(times, [at - onset for _, at, onset in pairs], 1)
    return (float(intercept), float(slope)) if abs(slope) <= MAX_DRIFT else None
