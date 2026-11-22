"""Static figures for review and for the report, one file each so a report links them rather
than embedding them."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.config.schema import AlignSection
from macaw.exceptions import AlignmentFailedError, BackendNotAvailableError
from macaw.media.align import (
    AGREEMENT_TOLERANCE_S,
    ENVELOPE_EXCLUSION_S,
    ENVELOPE_STEP_S,
    FINE_WINDOW_S,
    MAX_CORRELATION_S,
    PHAT_EXCLUSION_S,
    AlignResult,
    describe_agreement,
    describe_drift,
    drift_corrected,
    envelope,
    envelope_correlation,
    fine_pieces,
    phat_correlation,
    top_two_peaks,
)

DPI = 300
EXCERPT_S = 30.0
# a headset's bursts reach many times its 99th percentile and would flatten everything else
LOUDNESS_CLIP = 1.5
# a grey that reads on a light page and a dark one, for figures shown on a transparent ground
EITHER_INK = "#8c8c8c"
EITHER_THEME = {
    "text.color": EITHER_INK, "axes.labelcolor": EITHER_INK, "axes.edgecolor": EITHER_INK, "xtick.color": EITHER_INK,
    "ytick.color": EITHER_INK, "axes.facecolor": "none", "figure.facecolor": "none", "savefig.transparent": True,
}


def require_matplotlib():
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise BackendNotAvailableError('Review figures need matplotlib: pip install "pymacaw[report]"') from error
    return Figure


def loudest_shared_excerpt(
    reference_env: np.ndarray, target_env: np.ndarray, shift: int, width: int
) -> tuple[int, np.ndarray, np.ndarray] | None:
    """The stretch where both are loud, on the reference's frame grid: (first frame, reference
    piece, target piece). Target frame j sits at reference frame j + shift."""
    start = max(shift, 0)
    stop = min(reference_env.size, target_env.size + shift)
    if stop - start <= 0:
        return None
    reference_part = reference_env[start:stop]
    target_part = target_env[start - shift : stop - shift]
    width = min(width, reference_part.size)
    joint = np.convolve(np.minimum(reference_part, target_part), np.ones(width), mode="valid")
    at = int(np.argmax(joint))
    return start + at, reference_part[at : at + width], target_part[at : at + width]


def alignment_figure(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: float,
    result: AlignResult,
    cfg: AlignSection,
    path: Path,
) -> Path:
    """What a person or an agent needs to accept or reject one offset: each correlation peak
    against its nearest rival, both loudness envelopes on the recording clock, and whether the
    segments keep finding the same offset."""
    Figure = require_matplotlib()
    from matplotlib import rc_context

    with rc_context(EITHER_THEME):
        return _alignment_figure(Figure, reference, target, sample_rate, result, cfg, path)


def _alignment_figure(Figure, reference, target, sample_rate, result, cfg, path) -> Path:
    figure = Figure(figsize=(12, 11), layout="constrained")
    coarse, fine, envelopes, segments = figure.subplots(4, 1)
    figure.suptitle(f"{result.target} against {result.reference}: offset {result.offset_s:+.3f} s")

    lags, window = envelope_correlation(reference, target, sample_rate, cfg.search_window_s)
    _peaks(coarse, lags, window, int(ENVELOPE_EXCLUSION_S / ENVELOPE_STEP_S), "s")
    coarse.set(
        xlabel="lag (s)", ylabel="envelope correlation",
        title=f"coarse, whole files: confidence {_format(result.coarse_confidence)}",
    )

    shift = int(round((result.coarse_offset_s or 0.0) * sample_rate))
    try:
        pieces = fine_pieces(reference, target, shift, int(MAX_CORRELATION_S * sample_rate))
    except AlignmentFailedError as error:
        _empty(fine, f"fine correlation not drawn: {error}")
    else:
        lags, window = phat_correlation(*pieces, sample_rate, FINE_WINDOW_S)
        _peaks(fine, (lags + shift / sample_rate) * 1000, window, int(PHAT_EXCLUSION_S * sample_rate), "ms")
        fine.set(
            xlabel="lag (ms)", ylabel="PHAT correlation",
            title=f"fine, first minutes of overlap: confidence {_format(result.confidence)}",
        )

    reference_env = envelope(reference, sample_rate)
    target_env = envelope(target, sample_rate)
    reference_env /= max(float(np.percentile(reference_env, 99)), 1e-12)
    target_env /= max(float(np.percentile(target_env, 99)), 1e-12)
    excerpt = loudest_shared_excerpt(
        reference_env, target_env, int(round(result.offset_s / ENVELOPE_STEP_S)), int(EXCERPT_S / ENVELOPE_STEP_S)
    )
    if excerpt is None:
        _empty(envelopes, "the two files do not overlap at this offset")
    else:
        first, reference_part, target_part = excerpt
        t = (first + np.arange(reference_part.size)) * ENVELOPE_STEP_S
        reference_part = np.minimum(reference_part, LOUDNESS_CLIP)
        target_part = np.minimum(target_part, LOUDNESS_CLIP)
        envelopes.fill_between(t, 0, reference_part, color="C0", lw=0, label=result.reference)
        envelopes.fill_between(t, 0, -target_part, color="C1", lw=0, label=f"{result.target}, shifted")
        envelopes.axhline(0, color=EITHER_INK, lw=0.5)
        envelopes.set_ylim(-LOUDNESS_CLIP * 1.05, LOUDNESS_CLIP * 1.05)
        envelopes.set(xlabel="recording time (s)", ylabel="loudness", title="aligned, the two should rise and fall together")
        _legend(envelopes)

    if result.segment_offsets:
        _segments(segments, result, cfg)
    else:
        _empty(segments, "no sounding segments to cross-check")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    return path


def _peaks(axes, lags, window, exclusion: int, unit: str) -> None:
    best, rival = top_two_peaks(window, exclusion)
    # lines behind the curve and a dot on each peak, so a line never hides the peak it marks
    axes.axvline(lags[best], color="C3", lw=1, alpha=0.5, zorder=1)
    axes.axvline(lags[rival], color="C0", lw=1, ls="--", alpha=0.5, zorder=1)
    axes.plot(lags, window, lw=0.6, color=EITHER_INK, zorder=2)
    axes.plot(lags[best], window[best], "o", color="C3", ms=6, zorder=3, label=f"chosen {lags[best]:+.3f} {unit}")
    axes.plot(
        lags[rival], window[rival], "o", color="C0", mfc="none", ms=6, zorder=3,
        label=f"runner-up {lags[rival]:+.3f} {unit}",
    )
    _legend(axes)


def _segments(axes, result: AlignResult, cfg: AlignSection) -> None:
    t, offsets, confidence = np.array(result.segment_offsets, dtype=float).T
    difference = (offsets - result.offset_s) * 1000
    tolerance = AGREEMENT_TOLERANCE_S * 1000
    same = (confidence > 0) & (np.abs(difference) <= tolerance)
    axes.axhspan(-tolerance, tolerance, color="C2", alpha=0.15, lw=0, label=f"within {tolerance:.0f} ms")
    if same.any():
        axes.plot(t[same], difference[same], "o", ms=5, color="C0", label="same offset")
    if not same.all():
        axes.plot(t[~same], difference[~same], "o", ms=5, color="C3", mfc="none", label="another offset")
    title = describe_agreement(result)
    sharp = confidence >= cfg.min_confidence
    if result.drift_ppm is not None and sharp.sum() >= 2:
        if result.drift_intercept_s is not None:
            slope, intercept = result.drift_ppm * 1e-6, result.drift_intercept_s
        else:
            slope, intercept = np.polyfit(t[sharp], offsets[sharp], 1)
        label = "drift fit, corrected along it" if drift_corrected(result) else "drift fit"
        if result.drift_segment_offsets:
            again = np.array(result.drift_segment_offsets, dtype=float)
            kept = again[again[:, 2] >= cfg.min_confidence]
            axes.plot(kept[:, 0], (kept[:, 1] - result.offset_s) * 1000, "x", ms=5, color="C1", label="measured again at the fitted rate")
        axes.plot(t, (slope * t + intercept - result.offset_s) * 1000, color="C1", lw=1, label=label)
        title += f"; {describe_drift(result)}"
    axes.set(xlabel="recording time (s)", ylabel="segment offset minus offset (ms)", title=title)
    _legend(axes)


def imu_figure(head: tuple[np.ndarray, np.ndarray], imu: tuple[np.ndarray, np.ndarray], result: AlignResult, path: Path) -> Path:
    """What a person needs to keep or reject a signal file's offset: the correlation at every offset, the head's and the
    cap's turning speed where they move most at the one found, and whether each stretch finds the same."""
    Figure = require_matplotlib()
    from matplotlib import rc_context

    with rc_context(EITHER_THEME):
        return _imu_figure(Figure, head, imu, result, path)


def _imu_figure(Figure, head, imu, result: AlignResult, path) -> Path:
    from macaw.signals.imu import AGREEMENT_S, GRID_HZ, standardised, correlation_curve, on_grid

    figure = Figure(figsize=(10, 9), layout="constrained")
    curve, speeds, stretches = figure.subplots(3, 1)
    figure.suptitle(f"{result.target}: offset {result.offset_s:+.2f} s, confidence {_format(result.confidence)}")
    offsets, r = correlation_curve(*head, *imu)
    curve.plot(offsets, r, lw=0.5, color=EITHER_INK, zorder=2)
    curve.axvline(result.offset_s, color="C3", lw=1, alpha=0.6, zorder=1, label=f"found {result.offset_s:+.2f} s")
    if result.metadata_offset_s is not None:
        curve.axvline(result.metadata_offset_s, color="C0", lw=1, ls="--", alpha=0.6, zorder=1,
                      label=f"start times say {result.metadata_offset_s:+.1f} s")
    curve.set(xlabel="offset (s): recording time minus the file's own time", ylabel="correlation",
              title="how well the cap's turning follows the head's at each offset")
    _legend(curve)

    start = float(np.min(head[0]))
    count = int((float(np.max(head[0])) - start) * GRID_HZ) + 1
    video = standardised(on_grid(head[0], head[1], start, count))
    cap = standardised(on_grid(imu[0] + result.offset_s, imu[1], start, count))
    width = min(int(60 * GRID_HZ), count)
    scores = [np.nanvar(cap[i : i + width]) * np.isfinite(video[i : i + width]).mean() if np.isfinite(cap[i : i + width]).any() else -1
              for i in range(0, max(count - width, 1), int(GRID_HZ))]
    first = int(GRID_HZ) * int(np.argmax(scores))
    t = start + (first + np.arange(width)) / GRID_HZ
    speeds.plot(t, video[first : first + width], lw=0.8, color="C0", label="head on the video")
    speeds.plot(t, cap[first : first + width], lw=0.8, color="C1", label="cap gyroscope, at the offset found")
    speeds.set(xlabel="recording time (s)", ylabel="turning speed (z)", title="where they move most: the two should rise together")
    _legend(speeds)

    if result.segment_offsets:
        at, found, confidence = np.array(result.segment_offsets, dtype=float).T
        same = (confidence > 0) & (np.abs(found - result.offset_s) <= AGREEMENT_S)
        stretches.axhspan(-AGREEMENT_S, AGREEMENT_S, color="C2", alpha=0.15, lw=0, label=f"within {AGREEMENT_S:g} s")
        if same.any():
            stretches.plot(at[same], (found - result.offset_s)[same], "o", ms=5, color="C0", label="same offset")
        if not same.all():
            stretches.plot(at[~same], (found - result.offset_s)[~same], "o", ms=5, color="C3", mfc="none", label="another offset")
        stretches.set(xlabel="recording time (s)", ylabel="stretch offset less the found (s)", title="each stretch matched on its own")
        _legend(stretches)
    else:
        _empty(stretches, "too short for stretches to be matched on their own")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    return path


def _legend(axes) -> None:
    # outside the panel, so it never covers a peak or a point
    axes.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _empty(axes, message: str) -> None:
    axes.text(0.5, 0.5, message, ha="center", va="center", transform=axes.transAxes)
    axes.set_axis_off()


SEEN_COLOR = "#2a78d6"
VOICE_COLOR = "#eb6834"
LOOK_COLOR = "#fab219"
GRID_COLOR = "#e1e0d9"
MUTED_INK = "#898781"
# presence is drawn in steps this long, so a run of single dropped frames does not fray the bar
TIMELINE_STEP_S = 0.2


def present_spans(times: np.ndarray, step: float = TIMELINE_STEP_S) -> list[tuple[float, float]]:
    """(start, width) runs of the step-wide bins that hold at least one time."""
    if times.size == 0:
        return []
    bins = np.unique(np.floor(np.asarray(times, dtype=float) / step).astype(int))
    breaks = np.flatnonzero(np.diff(bins) > 1)
    starts = np.r_[bins[0], bins[breaks + 1]]
    stops = np.r_[bins[breaks], bins[-1]] + 1
    return [(float(a * step), float((b - a) * step)) for a, b in zip(starts, stops)]


def timeline_figure(
    subjects: list[str],
    seen: dict[str, list[tuple[float, float]]],
    heard: dict[str, list[tuple[float, float]]],
    heard_label: str,
    candidates,
    path: Path,
) -> Path:
    """Per person, (start, width) spans when a camera saw them and when they were heard, with the
    spans the checks flagged shaded behind; a person never seen, or seen but never heard, shows at once.
    A voice heard that is no one's, such as other, gets a lane of its own."""
    Figure = require_matplotlib()
    kinds = [kind for kind, spans in (("seen", seen), ("heard", heard)) if any(spans.values())]
    lanes = [(subject, kind) for subject in subjects for kind in kinds]
    lanes += [(voice, "heard") for voice, spans in heard.items() if voice not in subjects and spans]
    figure = Figure(figsize=(12, 0.9 + 0.32 * len(lanes)), layout="constrained")
    axes = figure.subplots()
    for row, (subject, kind) in enumerate(lanes):
        spans = (seen if kind == "seen" else heard).get(subject, [])
        color = SEEN_COLOR if kind == "seen" else VOICE_COLOR
        axes.broken_barh(spans, (row - 0.35, 0.7), color=color, lw=0)
    for span in candidates.itertuples():
        who = span.subject_id if isinstance(span.subject_id, str) else None
        rows = [row for row, (subject, _) in enumerate(lanes) if subject == who]
        low, high = (min(rows) - 0.5, max(rows) + 0.5) if rows else (-0.5, len(lanes) - 0.5)
        axes.fill_between([span.t_start, max(span.t_end, span.t_start + TIMELINE_STEP_S)], low, high,
                          color=LOOK_COLOR, alpha=0.35, lw=0, zorder=0)
    axes.set_yticks(range(len(lanes)), [f"{subject}  {kind}" for subject, kind in lanes], fontsize=8)
    axes.set_ylim(len(lanes) - 0.5, -0.5)
    axes.set_xlabel("recording time (s)")
    axes.grid(axis="x", color=GRID_COLOR, lw=0.5)
    axes.tick_params(colors=MUTED_INK, labelcolor="0.2")
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    from matplotlib.patches import Patch

    axes.legend(
        handles=[Patch(color=SEEN_COLOR, label="seen on camera")] * ("seen" in kinds)
        + [Patch(color=VOICE_COLOR, label=heard_label)] * ("heard" in kinds)
        + [Patch(color=LOOK_COLOR, alpha=0.35, label="flagged by a check")],
        loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8, frameon=False,
    )
    return _save(figure, path)


def pitch_figure(subjects: list[str], pitch: dict[str, tuple[np.ndarray, np.ndarray]], candidates, path: Path) -> Path:
    """Each person's f0 track, one panel each on a shared time axis, with the spans the octave
    check flagged shaded; a halving or doubling shows as a band of points an octave off."""
    return _panels(subjects, pitch, candidates, path, ylabel="f0 (Hz)", empty="no pitch", style=".")


def motion_figure(subjects: list[str], energy: dict[str, tuple[np.ndarray, np.ndarray]], candidates, path: Path) -> Path:
    """Each person's motion energy on a shared time axis, with the spans the motion check flagged shaded."""
    return _panels(subjects, energy, candidates, path, ylabel="share changed", empty="no region", style="-")


def _panels(subjects, series, candidates, path: Path, *, ylabel: str, empty: str, style: str) -> Path:
    """One panel per person; a span with no subject is shaded on every panel."""
    Figure = require_matplotlib()
    figure = Figure(figsize=(12, 0.6 + 1.6 * len(subjects)), layout="constrained")
    panels = np.atleast_1d(figure.subplots(len(subjects), 1, sharex=True, squeeze=False)[:, 0])
    for axes, subject in zip(panels, subjects):
        t, values = series.get(subject, (np.empty(0), np.empty(0)))
        mine = candidates["subject_id"].eq(subject).fillna(False).astype(bool) | candidates["subject_id"].isna()
        for span in candidates[mine].itertuples():
            axes.axvspan(span.t_start, max(span.t_end, span.t_start + TIMELINE_STEP_S), color=LOOK_COLOR, alpha=0.35, lw=0)
        if t.size:
            if style == ".":
                axes.plot(t, values, ".", ms=1.5, color=SEEN_COLOR, rasterized=True)
            else:
                axes.plot(t, values, "-", lw=0.6, color=SEEN_COLOR, rasterized=True)
        else:
            axes.text(0.5, 0.5, empty, ha="center", va="center", transform=axes.transAxes, color=MUTED_INK)
        axes.set_ylabel(ylabel, fontsize=8)
        axes.set_title(subject, loc="left", fontsize=9)
        axes.grid(color=GRID_COLOR, lw=0.5)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
    panels[-1].set_xlabel("recording time (s)")
    return _save(figure, path)


# one colour per person in a fixed order; yellow is left out, it marks what a check flagged
PERSON_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#008300", "#6250d6", "#e34948")
# seconds of recording per row, so a long recording stays readable at report width
OVERVIEW_ROW_S = 300.0
_TALK_ROW_IN = 0.22
_LANE_IN = 0.85
_CODE_ROW_IN = 0.18
_GAP_IN = 0.35


def overview_figure(
    people: list[str],
    talk: dict[str, list[tuple[float, float]]],
    lanes: list[tuple[str, list[tuple[str, str | None, np.ndarray, np.ndarray]], bool]],
    codes,
    candidates,
    span: tuple[float, float],
    path: Path,
) -> Path:
    """Talk spans, measure lanes and code tiers of every person on one clock, a row per OVERVIEW_ROW_S, flagged spans shaded."""
    Figure = require_matplotlib()
    from matplotlib.patches import Patch

    color = dict(zip(people, PERSON_COLORS))
    # a voice that is none of the people, such as other, talks in grey
    talkers = [person for person in people if talk.get(person)] + sorted(who for who in talk if who not in people and talk[who])
    tiers = sorted(codes["tier"].astype(str).unique()) if codes is not None and not codes.empty else []
    heights = [_TALK_ROW_IN * len(talkers) + 0.15] * bool(talkers) + [_LANE_IN] * len(lanes)
    heights += [_CODE_ROW_IN * len(tiers) + 0.15] * bool(tiers)
    starts = np.arange(span[0], span[1], OVERVIEW_ROW_S) if span[1] > span[0] else np.array([span[0]])
    ratios = []
    for index in range(len(starts)):
        ratios += heights + [_GAP_IN] * (index < len(starts) - 1)
    figure = Figure(figsize=(12, sum(ratios) + 0.5), layout="constrained")
    stack = list(np.atleast_1d(figure.subplots(len(ratios), 1, gridspec_kw={"height_ratios": ratios})))
    limits = [_lane_limits(series) for _, series, _ in lanes]

    for index, start in enumerate(starts):
        block = [stack.pop(0) for _ in heights]
        if index < len(starts) - 1:
            stack.pop(0).set_axis_off()
        end = start + OVERVIEW_ROW_S
        rows = iter(block)
        if talkers:
            axes = next(rows)
            for row, person in enumerate(talkers):
                axes.broken_barh(talk[person], (row - 0.35, 0.7), color=color.get(person, MUTED_INK), lw=0)
            axes.set_yticks(range(len(talkers)), talkers, fontsize=7)
            axes.set_ylim(len(talkers) - 0.5, -0.5)
        for (name, series, dots), limit in zip(lanes, limits):
            axes = next(rows)
            for label, person, t, y in series:
                inside = (t >= start) & (t <= end)
                style = dict(ms=1.5, ls="none", marker=".") if dots else dict(lw=0.7)
                axes.plot(t[inside], y[inside], color=color.get(person, MUTED_INK), label=label, rasterized=True, **style)
            if not series:
                axes.text(0.5, 0.5, "no values", ha="center", va="center", transform=axes.transAxes, color=MUTED_INK, fontsize=7)
            elif limit is not None:
                axes.set_ylim(*limit)
            if index == 0 and len(series) > 1:
                axes.legend(fontsize=6, loc="upper right", frameon=False, ncol=len(series))
            axes.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
            axes.tick_params(axis="y", labelsize=6)
        if tiers:
            axes = next(rows)
            for row, tier in enumerate(tiers):
                mine = codes[(codes["tier"].astype(str) == tier) & (codes["t_end"] >= start) & (codes["t_start"] <= end)]
                for subject_id, coded in mine.groupby(mine["subject_id"].astype(str), dropna=False):
                    widths = (coded["t_end"] - coded["t_start"]).clip(lower=0.4)
                    axes.broken_barh(list(zip(coded["t_start"], widths)), (row - 0.3, 0.6), color=color.get(subject_id, MUTED_INK), lw=0)
            axes.set_yticks(range(len(tiers)), tiers, fontsize=6)
            axes.set_ylim(len(tiers) - 0.5, -0.5)
        for axes in block:
            for flagged in candidates.itertuples():
                if flagged.t_end >= start and flagged.t_start <= end:
                    axes.axvspan(flagged.t_start, max(flagged.t_end, flagged.t_start + TIMELINE_STEP_S), color=LOOK_COLOR, alpha=0.35, lw=0, zorder=0)
            axes.set_xlim(start, end)
            axes.grid(axis="x", color=GRID_COLOR, lw=0.5)
            axes.tick_params(colors=MUTED_INK, labelcolor="0.2", labelbottom=axes is block[-1], labelsize=7)
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
    block[-1].set_xlabel("recording time (s)", fontsize=8)
    handles = [Patch(color=color[person], label=person) for person in people if person in color]
    handles.append(Patch(color=LOOK_COLOR, alpha=0.35, label="flagged by a check"))
    figure.legend(handles=handles, loc="outside upper left", ncol=len(handles), fontsize=8, frameon=False)
    return _save(figure, path)


def _lane_limits(series) -> tuple[float, float] | None:
    """One y range per lane across every row, so rows compare."""
    values = np.concatenate([y[np.isfinite(y)] for _, _, _, y in series]) if series else np.empty(0)
    if not values.size:
        return None
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * 0.05 or abs(high) * 0.05 or 1.0
    return low - pad, high + pad


IGNORED_COLOR = "#5f5e5a"
# points drawn per track at most; a seated person's middle stays in one small cloud
_TRACK_POINTS = 1500
# inches: the least a tracks figure is wide, so its legend fits beside a small picture, and the room around its panels
_TRACKS_MIN_IN = 6.0
_TRACKS_PAD_IN = 0.15
_TRACKS_TITLE_IN = 0.3
_TRACKS_LEGEND_ROW_IN = 0.18


def tracks_panel_width(views: list[dict], columns: int) -> float:
    """Inches a camera's panel is wide: a share of the page, and never more than its picture's own pixels give."""
    pictures = [view["picture"].shape[1] / DPI for view in views if view.get("picture") is not None]
    return min([12.0 / columns, *pictures])


def tracks_figure(views: list[dict], people: list[str], path: Path) -> Path:
    """Per camera, where the middle of each tracked body was through the analysed part, on a frame of that camera: the
    people named in their colours, bodies nobody named in grey, the areas the setup says to ignore hatched. A view holds
    camera, size, picture (an image array or None), areas and tracks, each track with who, ignored, label, x and y."""
    Figure = require_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    color = dict(zip(people, PERSON_COLORS))
    drawn = [track for view in views for track in view["tracks"]]
    handles = [
        Line2D([], [], marker="o", ls="", color=color.get(person, MUTED_INK), label=person)
        for person in people if any(track["who"] == person for track in drawn)
    ]
    if any(track["who"] is None and not track["ignored"] for track in drawn):
        handles.append(Line2D([], [], marker="o", ls="", color=MUTED_INK, label="nobody named"))
    if any(view["areas"] for view in views):
        handles.append(Patch(fill=False, hatch="///", ec=IGNORED_COLOR, ls="--", label="area to ignore"))
    if any(track["ignored"] for track in drawn):
        handles.append(Line2D([], [], marker="x", ls="", color=IGNORED_COLOR, label="left out: in an area to ignore"))

    columns = min(len(views), 2)
    rows = -(-len(views) // columns)
    wide = tracks_panel_width(views, columns)
    tall = [wide * view["size"][1] / view["size"][0] for view in views]
    row_tall = [max(tall[row * columns:(row + 1) * columns]) for row in range(rows)]
    legend_rows = -(-len(handles) // 3)
    legend_in = _TRACKS_LEGEND_ROW_IN * legend_rows + _TRACKS_PAD_IN / 2
    across = max(columns * wide + (columns + 1) * _TRACKS_PAD_IN, _TRACKS_MIN_IN)
    down = legend_in + sum(row_tall) + rows * (_TRACKS_TITLE_IN + _TRACKS_PAD_IN)
    figure = Figure(figsize=(across, down))
    for index, view in enumerate(views):
        row, column = divmod(index, columns)
        top = down - legend_in - sum(row_tall[:row]) - row * (_TRACKS_TITLE_IN + _TRACKS_PAD_IN) - _TRACKS_TITLE_IN
        left = _TRACKS_PAD_IN + column * (wide + _TRACKS_PAD_IN)
        # placed by hand: a layout that fills the page would draw the picture larger than its pixels
        axes = figure.add_axes([left / across, (top - tall[index]) / down, wide / across, tall[index] / down])
        pixels_wide, pixels_high = view["size"]
        if view.get("picture") is not None:
            axes.imshow(view["picture"], extent=(0, pixels_wide, pixels_high, 0), alpha=0.55, zorder=0, aspect="auto")
        for area_left, area_top, area_right, area_bottom in view["areas"]:
            axes.add_patch(Rectangle(
                (area_left, area_top), area_right - area_left, area_bottom - area_top, fill=False, hatch="///", ec=IGNORED_COLOR,
                lw=1.2, ls="--", zorder=1,
            ))
        for track in view["tracks"]:
            shade = IGNORED_COLOR if track["ignored"] else color.get(track["who"], MUTED_INK)
            step = max(len(track["x"]) // _TRACK_POINTS, 1)
            axes.plot(
                track["x"][::step], track["y"][::step], "x" if track["ignored"] else ".", ms=4, color=shade, alpha=0.4, mew=0.7, zorder=2,
            )
            if track["label"]:
                axes.annotate(
                    track["label"], (float(np.median(track["x"])), float(np.median(track["y"]))), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color="white", zorder=4, annotation_clip=False,
                    bbox={"boxstyle": "round,pad=0.25", "fc": shade, "ec": "white", "lw": 0.6},
                )
        axes.set_xlim(0, pixels_wide)
        axes.set_ylim(pixels_high, 0)
        axes.set_xticks([])
        axes.set_yticks([])
        for side in axes.spines.values():
            side.set_color(GRID_COLOR)
        axes.set_title(view["camera"], fontsize=9, loc="left", pad=4)
    if handles:
        figure.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(_TRACKS_PAD_IN / across, 1.0), ncol=3, fontsize=8, frameon=False,
            borderaxespad=0.3, columnspacing=1.2, handletextpad=0.4,
        )
    return _save(figure, path)


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)
    return path
