"""Results burned back into the video: skeletons and who they were assigned to, on the frames
worth looking at. A misassigned skeleton is obvious in one frame and invisible in a parquet file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors.video.pose.keypoints import select_videos
from macaw.extractors.video.pose.skeleton import SKELETON_EDGES, STANDARD_JOINTS
from macaw.project.recording import Recording
from macaw.store.schema import coerce_frame, empty_frame

BASELINE = "baseline"

REVIEW_SCHEMA: dict[str, str] = {
    "recording_id": "string",
    "subject_id": "string",
    "check": "string",        # null for the baseline
    "reason": "string",       # a candidate reason, or "baseline"
    "t": "float64",
    "score": "float64",
}

# BGR; subjects take colours in configured order, so a person keeps one colour across frames
_PALETTE = ((60, 180, 255), (255, 140, 40), (90, 220, 90), (200, 90, 220), (40, 220, 220), (220, 220, 60))
_UNASSIGNED = (170, 170, 170)


def review_sample(
    candidates: pd.DataFrame,
    t_range: tuple[float, float],
    *,
    recording_id: str,
    max_candidates: int,
    n_baseline: int,
    seed: int,
) -> pd.DataFrame:
    """Candidates serve precision and the uniform baseline serves recall. The baseline is drawn
    without looking at anything a check computed, because the failures that matter most are the
    confident ones no statistic noticed."""
    picked = candidates.assign(
        rank=candidates.groupby("reason", dropna=False)["score"].rank(method="first", ascending=False)
    )
    # round robin across reasons, so one noisy reason cannot crowd out the rest
    picked = picked.sort_values(["rank", "score"], ascending=[True, False]).head(max_candidates)
    picked = picked[list(REVIEW_SCHEMA)]

    start, end = t_range
    edges = np.linspace(start, end, n_baseline + 1)
    offsets = np.random.default_rng(seed).random(n_baseline)
    baseline = pd.DataFrame(
        {
            "recording_id": recording_id,
            "subject_id": None,
            "check": None,
            "reason": BASELINE,
            "t": edges[:-1] + offsets * np.diff(edges),
            "score": np.nan,
        }
    )
    parts = [frame for frame in (picked, baseline) if not frame.empty]
    if not parts:
        return empty_frame(REVIEW_SCHEMA)
    sample = pd.concat(parts, ignore_index=True).sort_values("t", kind="stable")
    return coerce_frame(sample.reset_index(drop=True), REVIEW_SCHEMA, "review sample")


def _cv2():
    try:
        import cv2
    except ImportError as error:
        raise BackendNotAvailableError('Rendering needs opencv: pip install "pymacaw[vision]"') from error
    return cv2


def _colour(subject_id, subjects: list[str]) -> tuple[int, int, int]:
    if subject_id is None or pd.isna(subject_id) or subject_id not in subjects:
        return _UNASSIGNED
    return _PALETTE[subjects.index(subject_id) % len(_PALETTE)]


def draw_overlay(image: np.ndarray, people: pd.DataFrame, subjects: list[str], header: list[str]) -> np.ndarray:
    """people is one frame's keypoint rows. Each person gets a box labelled with who they were
    assigned to and their mean confidence, so a reviewer can name the box that is wrong."""
    cv2 = _cv2()
    out = image.copy()
    height = out.shape[0]
    font_scale = max(0.4, height / 1200.0)
    thickness = max(1, int(round(height / 500.0)))

    standard = people[people["joint"].isin(STANDARD_JOINTS)]
    who = standard["subject_id"].astype(object).where(
        standard["subject_id"].notna(), "unassigned (track " + standard["track_id"].astype(str) + ")"
    )
    for name, person in standard.groupby(who, sort=False):
        colour = _colour(name, subjects)
        points = {
            row.joint: (int(round(row.x)), int(round(row.y)))
            for row in person.itertuples()
            if np.isfinite(row.x) and np.isfinite(row.y)
        }
        for a, b in SKELETON_EDGES:
            if a in points and b in points:
                cv2.line(out, points[a], points[b], colour, thickness, cv2.LINE_AA)
        for point in points.values():
            cv2.circle(out, point, thickness + 2, colour, -1, cv2.LINE_AA)
        if not points:
            continue

        xs, ys = zip(*points.values())
        pad = 4 * thickness
        top_left = (min(xs) - pad, min(ys) - pad)
        cv2.rectangle(out, top_left, (max(xs) + pad, max(ys) + pad), colour, thickness)
        label = f"{name}  conf {person['confidence'].mean():.2f}"
        _text(cv2, out, label, (top_left[0], max(top_left[1] - 6, 12)), font_scale, colour, thickness)

    y = int(24 * font_scale / 0.5)
    for line in header:
        _text(cv2, out, line, (8, y), font_scale, (255, 255, 255), thickness)
        y += int(22 * font_scale / 0.5)
    return out


def _text(cv2, image, text, origin, scale, colour, thickness) -> None:
    """A dark outline under the text keeps it readable on any background."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def _targets(row, videos: list) -> list:
    """The camera a moment should be seen from: the subject's own when they have one."""
    if row.subject_id is not None and not pd.isna(row.subject_id):
        own = [item for item in videos if item.subject_id == row.subject_id]
        if own:
            return own
    return videos


def render_review(
    recording: Recording,
    keypoints: pd.DataFrame,
    sample: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """One image per sampled moment and camera, plus an index.csv that says why each was drawn."""
    cv2 = _cv2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos, _ = select_videos(recording)
    subjects = [subject.subject_id for subject in recording.subjects]

    jobs = []
    for index, row in enumerate(sample.itertuples()):
        for item in _targets(row, videos):
            frame = int(round((row.t - item.offset_s) * (item.fps or 0.0)))
            jobs.append((item, frame, index, row))

    entries = []
    by_video: dict[Path, list] = {}
    for job in jobs:
        by_video.setdefault(job[0].path, []).append(job)
    for path, work in by_video.items():
        capture = cv2.VideoCapture(str(path))
        try:
            n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for item, frame, index, row in sorted(work, key=lambda job: job[1]):
                if frame < 0 or (n_frames > 0 and frame >= n_frames):
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = capture.read()
                if not ok:
                    continue
                people = keypoints[keypoints["frame"] == frame]
                if item.subject_id is not None:
                    people = people[(people["subject_id"] == item.subject_id).fillna(False)]
                header = [
                    f"{recording.recording_id}  t={row.t:.2f}s  frame {frame}  {Path(path).name}",
                    row.reason if row.reason == BASELINE else f"{row.check}: {row.reason}",
                ]
                name = f"{index:03d}_{row.reason}_{row.t:09.3f}s_{Path(path).stem}.jpg"
                cv2.imwrite(str(out_dir / name), draw_overlay(image, people, subjects, header))
                entries.append(
                    {
                        "file": name,
                        "recording_id": recording.recording_id,
                        "subject_id": row.subject_id,
                        "check": row.check,
                        "reason": row.reason,
                        "t": row.t,
                        "frame": frame,
                        "video": Path(path).name,
                        "score": row.score,
                    }
                )
        finally:
            capture.release()

    index = pd.DataFrame(
        entries, columns=["file", "recording_id", "subject_id", "check", "reason", "t", "frame", "video", "score"]
    ).sort_values("file", kind="stable")
    index.to_csv(out_dir / "index.csv", index=False)
    return index.reset_index(drop=True)
