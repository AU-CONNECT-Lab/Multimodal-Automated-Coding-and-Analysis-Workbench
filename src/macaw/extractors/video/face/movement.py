"""Mouth movement and blinks, read off the face landmarks face.au keeps; no model runs here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import apply_prep_timing
from macaw.extractors.video.face.au import LANDMARKS_NAME, FaceExtractor
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables, read_long_table
from macaw.store.schema import Scale

# indices in the 68-point layout every face backend writes
INNER_LIPS = ((61, 67), (62, 66), (63, 65))
EYES = ((36, 37, 38, 39, 40, 41), (42, 43, 44, 45, 46, 47))
NOSE_BASE = 33

# an eyelid this far below its own recent level is closed, and this close to it is open again
CLOSED_SHARE = 0.7
OPEN_SHARE = 0.85
BASELINE_S = 3.0
MAX_BLINK_S = 0.5
# frames either side of a closure that must show the eye open
REOPEN_FRAMES = 3
# beyond this the far eye is foreshortened and its landmarks guessed
MAX_YAW_DEG = 45.0
# a head lowered or raised this far from the person's usual pitch moves the lids by itself
MAX_PITCH_SHIFT_DEG = 15.0
# landmarks spanning less of the person's usual face width sit on an ear or the hair, not a face
MIN_FACE_WIDTH_SHARE = 0.7
_LOW_MEASURABLE = 0.5
# people blink many times a minute, so none in this much readable time means the landmarks never close
_NO_BLINK_S = 60.0

_FRAME_SPECS = {
    "mouth_opening": "Gap between the inner lips over the distance from the eyes to the base of the nose.",
    "mouth_movement": "Absolute change in mouth_opening from the previous frame, per second.",
    "eye_opening": "Eyelid gap over eye width, both eyes averaged; missing where the eyes are too small or the head turned too far to read.",
}


@register_extractor
class FaceMovementExtractor(Extractor):
    name = "face.movement"
    version = "0.2.0"
    scale = Scale.FRAME
    group = "face"
    config_sections = ("face.blink_min_eye_px", "segment")
    requires = (FaceExtractor.name,)

    def check_available(self) -> None:
        return None

    def _files(self, recording_id: str) -> tuple[Path, Path]:
        layout = layout_for(self.config, create=False)
        return (
            layout.artifacts_dir(recording_id, FaceExtractor.name) / LANDMARKS_NAME,
            layout.long_table_path(recording_id) / f"{FaceExtractor.name}.parquet",
        )

    def inputs_for(self, recording: Recording) -> list[Path]:
        return list(self._files(recording.recording_id))

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        landmarks_path, angles_path = self._files(recording.recording_id)
        if not landmarks_path.is_file():
            raise MacawError(f"{recording.recording_id}: face.movement needs the face landmarks; run face.au with face.landmarks true")
        landmarks = pd.read_parquet(landmarks_path)
        angles = head_angles(angles_path, recording.recording_id)
        prep = apply_prep_timing(self.config, recording)
        min_eye_px = self.config.face.blink_min_eye_px
        frames, warnings = [], []
        for subject_id, faces in landmarks.groupby("subject_id", sort=True):
            subject_id = str(subject_id)
            faces = faces.sort_values("frame").merge(angles.get(subject_id, _NO_ANGLES), on="t", how="left")
            opening = mouth_opening(faces)
            eyes, eye_width = eye_opening(faces)
            readable = (eye_width >= min_eye_px) & eyes_in_view(faces)
            eyes = eyes.where(readable)
            values = pd.DataFrame(
                {
                    "t_start": faces["t"].to_numpy(dtype=float),
                    "mouth_opening": opening.to_numpy(),
                    "mouth_movement": per_second_change(opening, faces["frame"], faces["t"]).to_numpy(),
                    "eye_opening": eyes.to_numpy(),
                }
            )
            common = dict(recording_id=recording.recording_id, source=self.name, entities=recording.entities(), subject_id=subject_id)
            frames.append(build_long_rows(values=values, scale=Scale.FRAME, **common))

            blinks = find_blinks(eyes, faces["frame"], faces["t"])
            if not blinks.empty:
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame(
                            {"t_start": blinks["t_start"], "t_end": blinks["t_end"], "blink_duration_s": blinks["t_end"] - blinks["t_start"]}
                        ),
                        scale=Scale.EVENT,
                        **common,
                    )
                )
            measurable_s = float(readable.sum()) * _frame_period(faces["t"])
            rate = len(blinks) / measurable_s * 60.0 if measurable_s > 0 else np.nan
            if blinks.empty and measurable_s >= _NO_BLINK_S:
                rate = np.nan
                warnings.append(
                    QCWarning(
                        check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity="warn",
                        message=(
                            f"not one blink in {measurable_s:.0f} s of readable eyes, so the eye landmarks do not follow the "
                            "lids (OpenFace's do not on small faces); blinks_per_min is left empty"
                        ),
                    )
                )
            frames.append(
                build_long_rows(
                    values=pd.DataFrame(
                        {"t_start": [prep.t_origin_s], "t_end": [prep.t_origin_s + prep.duration_s],
                         "blinks_per_min": [rate], "eyes_measurable_s": [measurable_s]}
                    ),
                    scale=Scale.RECORDING,
                    **common,
                )
            )
            warnings += self._eye_warnings(recording, subject_id, eye_width, readable, min_eye_px)
        rows = concat_long_tables(frames)
        return ExtractorResult(rows=rows, specs=self._specs(set(rows["feature"]) if len(rows) else set()), warnings=warnings)

    def _eye_warnings(self, recording, subject_id, eye_width: pd.Series, readable: pd.Series, min_eye_px: float) -> list[QCWarning]:
        if not len(readable):
            return []
        width = float(eye_width.median())
        if width < min_eye_px:
            message = f"eyes are about {width:.1f} px wide, under face.blink_min_eye_px ({min_eye_px:g}); blinks are not counted"
        elif readable.mean() < _LOW_MEASURABLE:
            message = f"the eyes could be read in {readable.mean():.0%} of the frames with a face; blinks_per_min rests on that share"
        else:
            return []
        return [QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=subject_id, severity="warn", message=message)]

    def _specs(self, emitted: set[str]) -> list[FeatureSpec]:
        params = {"blink_min_eye_px": self.config.face.blink_min_eye_px, "model": self.config.face.model or "v1"}
        described = {
            **{name: (text, None, Scale.FRAME) for name, text in _FRAME_SPECS.items()},
            "blink_duration_s": (
                f"A blink: eye_opening falls below {CLOSED_SHARE:.0%} of its median over {BASELINE_S:g} s and the eye "
                f"is seen open again within {MAX_BLINK_S:g} s.", "s", Scale.EVENT,
            ),
            "blinks_per_min": ("Blinks per minute of the time the eyes could be read.", "1/min", Scale.RECORDING),
            "eyes_measurable_s": (
                f"Time the eyes could be read: at least face.blink_min_eye_px wide, the head turned no more than {MAX_YAW_DEG:g} "
                f"degrees and tilted no more than {MAX_PITCH_SHIFT_DEG:g} degrees from the person's usual pitch.",
                "s", Scale.RECORDING,
            ),
        }
        return [
            FeatureSpec(
                name=name, description=description, unit=unit, scale=scale, group=self.group, extractor=self.name,
                extractor_version=self.version, params=params, interpolation="none",
                aggregation="sum" if name == "eyes_measurable_s" else "mean",
                # landmarks are placed differently by each face model, and face size and pose change what they can show
                comparable_across_backends=False, comparable_across_subjects=name == "eyes_measurable_s", validation="synthetic",
            )
            for name, (description, unit, scale) in described.items()
            if name in emitted
        ]


_NO_ANGLES = pd.DataFrame({"t": pd.Series(dtype=float), "yaw": pd.Series(dtype=float), "pitch": pd.Series(dtype=float)})


def head_angles(path: Path, recording_id: str) -> dict[str, pd.DataFrame]:
    """subject -> (t, yaw, pitch) from face.au's own rows; landmarks and angles share their frame times."""
    if not Path(path).exists():
        return {}
    rows = read_long_table(path, recording_id=recording_id, features=["face_yaw_deg", "face_pitch_deg"])
    if rows.empty:
        return {}
    wide = rows.pivot_table(index=["subject_id", "t_start"], columns="feature", values="value", observed=True).reset_index()
    wide = wide.rename(columns={"t_start": "t", "face_yaw_deg": "yaw", "face_pitch_deg": "pitch"})
    return {
        str(subject): part.reindex(columns=["t", "yaw", "pitch"]).reset_index(drop=True)
        for subject, part in wide.groupby("subject_id", observed=True)
    }


def eyes_in_view(faces: pd.DataFrame) -> pd.Series:
    """Frames whose eyes can be read: head not turned or tilted away from how this person usually holds it,
    and landmarks spread like this person's face; missing angles are not held against a frame."""
    xs = faces[[column for column in faces.columns if column.startswith("x_")]]
    width = xs.max(axis=1) - xs.min(axis=1)
    pitch = faces["pitch"]
    return (
        ~(faces["yaw"].abs() > MAX_YAW_DEG)
        & ~((pitch - pitch.median()).abs() > MAX_PITCH_SHIFT_DEG)
        & (width >= MIN_FACE_WIDTH_SHARE * width.median())
    )


def _point(faces: pd.DataFrame, index: int) -> np.ndarray:
    return faces[[f"x_{index}", f"y_{index}"]].to_numpy(dtype=float)


def _gap(faces: pd.DataFrame, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(_point(faces, a) - _point(faces, b), axis=1)


def mouth_opening(faces: pd.DataFrame) -> pd.Series:
    """Scaled by a stretch of face the jaw does not move, and vertical like the lip gap, so a nod shortens both."""
    lips = np.mean([_gap(faces, a, b) for a, b in INNER_LIPS], axis=0)
    eyes = np.mean([_point(faces, index) for eye in EYES for index in eye], axis=0)
    height = np.linalg.norm(_point(faces, NOSE_BASE) - eyes, axis=1)
    return pd.Series(np.divide(lips, height, out=np.full_like(lips, np.nan), where=height > 0), index=faces.index)


def eye_opening(faces: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(eyelid gap over eye width, eye width in pixels), both eyes averaged."""
    gaps, widths = [], []
    for corner_a, top_a, top_b, corner_b, bottom_b, bottom_a in EYES:
        gaps.append((_gap(faces, top_a, bottom_a) + _gap(faces, top_b, bottom_b)) / 2)
        widths.append(_gap(faces, corner_a, corner_b))
    gap, width = np.mean(gaps, axis=0), np.mean(widths, axis=0)
    ratio = np.divide(gap, width, out=np.full_like(gap, np.nan), where=width > 0)
    return pd.Series(ratio, index=faces.index), pd.Series(width, index=faces.index)


def per_second_change(values: pd.Series, frame: pd.Series, t: pd.Series) -> pd.Series:
    """Absolute change from the previous frame over the time between them; missing across a gap."""
    step = frame.diff() == 1
    return (values.diff().abs() / t.diff()).where(step)


def _frame_period(t: pd.Series) -> float:
    steps = t.diff().dropna()
    return float(steps.median()) if len(steps) else 0.0


def find_blinks(eyes: pd.Series, frame: pd.Series, t: pd.Series) -> pd.DataFrame:
    """Runs of closed eye no longer than MAX_BLINK_S, with the eye seen open on both sides; a slow
    lowering of the lids (looking down) never reopens in time and is not a blink."""
    eyes = pd.Series(eyes.to_numpy(dtype=float), index=frame.to_numpy())
    times = pd.Series(t.to_numpy(dtype=float), index=frame.to_numpy())
    full = pd.RangeIndex(int(frame.min()), int(frame.max()) + 1) if len(frame) else pd.RangeIndex(0)
    eyes = eyes.reindex(full)
    period = _frame_period(t)
    if period <= 0 or eyes.notna().sum() == 0:
        return pd.DataFrame({"t_start": pd.Series(dtype=float), "t_end": pd.Series(dtype=float)})
    window = max(int(round(BASELINE_S / period)), 3)
    baseline = eyes.rolling(window, center=True, min_periods=window // 3).median()
    closed = (eyes < CLOSED_SHARE * baseline).to_numpy()
    open_ = (eyes >= OPEN_SHARE * baseline).to_numpy()
    blinks, start = [], None
    for i in range(len(closed) + 1):
        if i < len(closed) and closed[i]:
            start = i if start is None else start
            continue
        if start is None:
            continue
        stop, start_at = i, start
        start = None
        if (stop - start_at) * period > MAX_BLINK_S:
            continue
        if not (open_[max(start_at - REOPEN_FRAMES, 0):start_at].any() and open_[stop:stop + REOPEN_FRAMES].any()):
            continue
        first, last = full[start_at], full[stop - 1]
        if first not in times.index or last not in times.index:
            continue
        blinks.append((times[first], times[last] + period))
    return pd.DataFrame(blinks, columns=["t_start", "t_end"]).astype(float)
