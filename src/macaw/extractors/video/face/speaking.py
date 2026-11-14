"""Whether each person's face is the one speaking, moment by moment, from their lips and the sound (Light-ASD)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError, MacawError
from macaw.extractors import register_extractor
from macaw.extractors.base import Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import MediaPrepExtractor, apply_prep_timing, load_prep
from macaw.extractors.video.face import light_asd
from macaw.extractors.video.face.au import FACES_NAME, LANDMARKS_NAME, FaceExtractor
from macaw.project.layout import layout_for
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.longtable import build_long_rows, concat_long_tables
from macaw.store.schema import Scale

SPEAKING = "speaking_score"
_DESCRIPTION = (
    "How much this person's face is the one speaking at this moment, from the movement of their lips together with the sound "
    "of the device their voice is read from (Light-ASD, AVA-ActiveSpeaker weights); above 0 reads as speaking. "
    f"{light_asd.FPS} values a second, at the frame of the person's face camera nearest each moment."
)
_MISSING = (
    "Empty where face.au found no face of this person for more than "
    f"{light_asd.MAX_GAP_FRAMES / light_asd.FPS:g} s, in runs of their face shorter than "
    f"{(light_asd.MIN_TRACK_FRAMES + 1) / light_asd.FPS:g} s, and for a person without a face camera or a sound to read; shorter gaps "
    "are bridged by placing the face between where it was seen."
)


@register_extractor
class FaceSpeakingExtractor(Extractor):
    name = "face.speaking"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "face"
    config_sections = ("face.device",)
    requires = (FaceExtractor.name, MediaPrepExtractor.name)
    resource_profile = "gpu_standard"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        from macaw.extractors.audio.content.diarization import SHARED_MODE

        # it is read to tell whose a line on a shared microphone is
        return config.text.diarization == SHARED_MODE

    def check_available(self) -> None:
        try:
            import python_speech_features  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "face.speaking needs PyTorch and python_speech_features: install PyTorch for your graphics card first, then "
                'pip install "pymacaw[vision]"'
            ) from error

    def prefetch(self) -> None:
        self._network("cpu")

    def _network(self, device: str):
        import torch

        try:
            state = torch.hub.load_state_dict_from_url(
                light_asd.WEIGHTS_URL, map_location="cpu", file_name=light_asd.WEIGHTS_FILE, weights_only=True, progress=False
            )
        except (OSError, RuntimeError) as error:
            raise ExternalToolError(
                f"could not fetch the Light-ASD weights ({error}); on a machine without internet, run macaw run on one with it "
                "first so they are in TORCH_HOME"
            ) from error
        return light_asd.network(state, device)

    def _files(self, recording_id: str) -> tuple[Path, Path]:
        directory = layout_for(self.config, create=False).artifacts_dir(recording_id, FaceExtractor.name)
        return directory / LANDMARKS_NAME, directory / FACES_NAME

    def inputs_for(self, recording: Recording) -> list[Path]:
        landmarks, faces = self._files(recording.recording_id)
        prep = load_prep(self.config, recording.recording_id)
        sounds = [prep.shared_track_path(name) for name in sorted(prep.shared_tracks)] or [prep.track_path(s) for s in sorted(prep.tracks)]
        return [landmarks, faces, *sounds]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        import torch

        rid = recording.recording_id
        landmarks_path, faces_path = self._files(rid)
        if not landmarks_path.is_file():
            raise MacawError(f"{rid}: face.speaking needs the face landmarks; run face.au with face.landmarks true")
        landmarks = pd.read_parquet(landmarks_path)
        views = json.loads(faces_path.read_text(encoding="utf-8"))["views"]
        prep = apply_prep_timing(self.config, recording)
        device = self.config.face.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self._network(device)
        by_name = {item.name: item for item in recording.media if item.kind == "video"}
        frames, warnings = [], []
        for video in sorted(set(views.values())):
            item = by_name[video]
            plans = {}
            for subject in sorted(subject for subject, view in views.items() if view == video):
                sound = self._sound(prep, subject)
                own = landmarks[landmarks["subject_id"].astype(str) == subject]
                if sound is None or own.empty:
                    warnings.append(self._warning(rid, subject, "no sound to read their voice from" if sound is None else "face.au found no face of theirs"))
                    continue
                xs = own[[f"x_{k}" for k in range(68)]].to_numpy(dtype=float)
                ys = own[[f"y_{k}" for k in range(68)]].to_numpy(dtype=float)
                x, y, s = light_asd.face_boxes(xs, ys)
                tracks = light_asd.plan_tracks(own["t"].to_numpy(dtype=float), x, y, s, fps=float(item.fps), offset_s=float(item.offset_s))
                plans[subject] = (tracks, sound)
            scored = score_video(item.frames_path, plans, net, device, prep.t_origin_s)
            for subject, (times, values) in scored.items():
                frames.append(
                    build_long_rows(
                        values=pd.DataFrame({"t_start": times, SPEAKING: values}), scale=Scale.FRAME, recording_id=rid,
                        source=self.name, entities=recording.entities(), subject_id=subject,
                    )
                )
        rows = concat_long_tables(frames)
        warnings.append(QCWarning(
            check=self.name, recording_id=rid, subject_id=None, severity="info",
            message="the speaking model learnt from adults in films; how well it reads young children's faces is not known",
        ))
        return ExtractorResult(rows=rows, specs=self._specs() if len(rows) else [], warnings=warnings)

    @staticmethod
    def _sound(prep, subject: str) -> Path | None:
        """The voices on a shared microphone, else the person's own: the sound whose speaker the face is asked about."""
        if len(prep.shared_tracks) == 1:
            return prep.shared_track_path(next(iter(prep.shared_tracks)))
        return prep.track_path(subject) if subject in prep.tracks else None

    def _warning(self, recording_id: str, subject_id: str, why: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording_id, subject_id=subject_id, severity="warn", message=f"not scored: {why}")

    def _specs(self) -> list[FeatureSpec]:
        return [
            FeatureSpec(
                name=SPEAKING, description=_DESCRIPTION, unit=None, scale=Scale.FRAME, group=self.group, extractor=self.name,
                extractor_version=self.version, backend="light_asd", backend_version="AVA-ActiveSpeaker",
                params={"weights": light_asd.WEIGHTS_URL.rsplit("/", 1)[-1]}, interpolation="none", aggregation="mean",
                missing=_MISSING, comparable_across_backends=False, validation="none",
            )
        ]


def score_video(video_path: Path, plans: dict, net, device: str, t_origin_s: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each person's times and scores on one camera, decoding it once and scoring each track in pieces as it fills,
    so the faces held at a time stay under a minute's worth per person."""
    import cv2

    from macaw.extractors.video.pose.backends import _seek

    wanted: dict[int, list[tuple[str, int, int]]] = {}
    for subject, (tracks, _) in plans.items():
        for number, track in enumerate(tracks):
            for index, frame in enumerate(track.frames):
                wanted.setdefault(int(frame), []).append((subject, number, index))
    found: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {subject: [] for subject in plans}
    if not wanted:
        return {}
    samples = {subject: _samples(sound) for subject, (_, sound) in plans.items()}
    pending: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExternalToolError(f"could not open video: {video_path}")
    try:
        first, last = min(wanted), max(wanted)
        frame = _seek(capture, cv2, max(first, 0), Path(video_path))
        image = None
        while frame <= last:
            ok, read = capture.read()
            if ok:
                image = read
            # past the file's end, the last picture stands in, as a face cannot be scored without one
            if image is None:
                break
            for subject, number, index in wanted.get(frame, ()):
                track = plans[subject][0][number]
                pending.setdefault((subject, number), {})[index] = light_asd.crop(image, track.x[index], track.y[index], track.s[index])
                _flush(found, pending, plans, samples, net, device, t_origin_s, subject, number, index, done=index == track.frames.size - 1)
            frame += 1
    finally:
        capture.release()
    # what a short file left unfilled is scored from the faces read
    for subject, number in list(pending):
        _flush(found, pending, plans, samples, net, device, t_origin_s, subject, number, None, done=True)
    out = {}
    for subject, pieces in found.items():
        if pieces:
            out[subject] = (np.concatenate([times for times, _ in pieces]), np.concatenate([values for _, values in pieces]))
    return out


def _flush(found, pending, plans, samples, net, device, t_origin_s, subject, number, index, *, done: bool) -> None:
    """Scores a track's piece once it holds a whole piece, or the track has ended."""
    held = pending.get((subject, number), {})
    if not held:
        return
    start = min(held)
    piece_end = start + light_asd.CHUNK_FRAMES - 1
    if not done and index != piece_end:
        return
    track = plans[subject][0][number]
    stop = min(piece_end, track.frames.size - 1)
    # moments whose frame was never read take the face read just before
    faces, last = [], None
    for position in range(start, stop + 1):
        last = held.get(position, last)
        if last is not None:
            faces.append(last)
    if faces:
        times = track.times[start: start + len(faces)]
        sound = light_asd.sound_for(samples[subject], float(times[0]) - t_origin_s, len(faces))
        values = light_asd.score(net, sound, np.stack(faces), device)
        found[subject].append((times[: values.size], values.astype("float32")))
    pending[(subject, number)] = {position: face for position, face in held.items() if position > stop}
    if not pending[(subject, number)]:
        del pending[(subject, number)]


def _samples(path: Path) -> np.ndarray:
    """The prepared track as 16-bit samples, as the model was trained on them."""
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="int16", always_2d=False)
    if rate != light_asd.SAMPLE_RATE:
        raise MacawError(f"{path}: face.speaking reads sound at {light_asd.SAMPLE_RATE} Hz, and this track is at {rate} Hz")
    return samples if samples.ndim == 1 else samples[:, 0]
