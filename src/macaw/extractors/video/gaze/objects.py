"""Things the project names in gaze.targets, found in each frame, so gaze.target can say which one a person looks at."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError, UnreadableMediaError
from macaw.extractors import register_backend, register_extractor
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning
from macaw.extractors.prep import apply_prep_timing
from macaw.extractors.video.pose.keypoints import PoseExtractor, PoseReviewGate
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.store import keypoints as keypoints_store
from macaw.store.schema import Scale

BACKEND_GROUP = "objects"
# one row per thing found: its name as gaze.targets gives it, its box in the video's pixels, and the detector's score
OBJECT_COLUMNS = ("video", "frame", "label", "x0", "y0", "x1", "y1", "score")
OBJECTS_NAME = "objects.parquet"
# the detector is slow, so things are looked for this often and gaze.target takes the nearest look
EVERY_S = 0.2
# a named thing found in fewer of the frames read than this is reported, as the name may not suit the detector
_RARELY_FOUND = 0.2


class ObjectBackend(Backend):
    def infer(self, video_path, *, frames: tuple[int, int], step: int, names: list[str], device: str, model: str | None,
              batch_size: int, min_score: float) -> pd.DataFrame:
        """OBJECT_COLUMNS but video, for every step-th frame from first to stop - 1 of the file, its own frame numbers."""
        raise NotImplementedError

    def prefetch(self, *, model: str | None) -> None:
        return None


@register_backend(BACKEND_GROUP)
class GroundingDinoBackend(ObjectBackend):
    """Grounding DINO through transformers; Apache-2.0 code and weights from the Hugging Face hub."""

    name = "grounding_dino"
    default_model = "IDEA-Research/grounding-dino-tiny"
    # the original authors' cut for keeping a word found in a box
    text_threshold = 0.25

    @property
    def version(self) -> str:
        import transformers

        return transformers.__version__

    def check_available(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "gaze.object_backend grounding_dino needs PyTorch and transformers: install PyTorch for your GPU first, "
                "then pip install transformers"
            ) from error

    def prefetch(self, *, model: str | None) -> None:
        self.network(model, "cpu")

    def network(self, model: str | None, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        name = model or self.default_model
        return AutoProcessor.from_pretrained(name), AutoModelForZeroShotObjectDetection.from_pretrained(name).eval().to(device)

    def infer(self, video_path, *, frames, step, names, device, model, batch_size, min_score):
        import cv2
        import torch
        from PIL import Image

        from macaw.extractors.video.pose.backends import _seek

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        processor, net = self.network(model, device)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")
        rows: list[dict] = []
        batch: list[tuple[int, Image.Image]] = []

        def flush():
            images = [image for _, image in batch]
            inputs = processor(images=images, text=[list(names)] * len(images), return_tensors="pt").to(device)
            with torch.inference_mode():
                found = net(**inputs)
            results = processor.post_process_grounded_object_detection(
                found, inputs.input_ids, threshold=min_score, text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1] for image in images],
            )
            for (frame, _), result in zip(batch, results):
                for box, score, label in zip(result["boxes"].tolist(), result["scores"].tolist(), result["text_labels"]):
                    if label in names:
                        rows.append({"frame": frame, "label": label, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "score": score})
            batch.clear()

        try:
            first, stop = frames
            index = _seek(capture, cv2, max(int(first), 0), Path(video_path))
            while index < stop:
                if (index - first) % step:
                    if not capture.grab():
                        break
                    index += 1
                    continue
                ok, image = capture.read()
                if not ok:
                    break
                batch.append((index, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
                index += 1
                if len(batch) >= batch_size:
                    flush()
            if batch:
                flush()
        finally:
            capture.release()
        return pd.DataFrame(rows, columns=[column for column in OBJECT_COLUMNS if column != "video"])


def resolve_backend(name: str) -> ObjectBackend:
    from macaw.extractors import get_backend

    backend = get_backend(BACKEND_GROUP, name)()
    backend.check_available()
    return backend


@register_extractor
class GazeObjectsExtractor(Extractor):
    name = "gaze.objects"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "gaze"
    config_sections = ("gaze.targets", "gaze.object_backend", "gaze.object_model", "gaze.object_score", "segment", "video", "setup.cameras")
    requires = (PoseReviewGate.name,)
    resource_profile = "gpu_standard"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.gaze.targets)

    def backend(self):
        return resolve_backend(self.config.gaze.object_backend)

    def backend_info(self) -> tuple[str | None, str | None]:
        backend = self.backend()
        return (backend.name, backend.version)

    def check_available(self) -> None:
        self.backend()

    def prefetch(self) -> None:
        self.backend().prefetch(model=self.config.gaze.object_model)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return [_keypoints_file(self.config, recording.recording_id)]

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.gaze
        backend = self.backend()
        prep = apply_prep_timing(self.config, recording)
        start, end = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        keypoints = keypoints_store.read_keypoints(
            _keypoints_file(self.config, recording.recording_id), recording_id=recording.recording_id
        )
        by_name = {item.name: item for item in recording.media if item.kind == "video"}
        found, warnings = [], []
        for video in sorted({item.name for item in recording.views("gaze").values()}):
            item = by_name[video]
            fps = item.fps or 0.0
            if fps <= 0:
                raise UnreadableMediaError(f"{item.path}: no frame rate to convert frames to time")
            first = max(math.floor((start - item.offset_s) * fps), 0)
            stop = math.ceil((end - item.offset_s) * fps) + 1
            step = max(round(EVERY_S * fps), 1)
            things = backend.infer(
                item.frames_path, frames=(first, stop), step=step, names=list(cfg.targets), device=cfg.device,
                model=cfg.object_model, batch_size=cfg.batch_size, min_score=cfg.object_score,
            )
            things.insert(0, "video", video)
            found.append(things)
            read = max(math.ceil((stop - first) / step), 1)
            for name in cfg.targets:
                share = things.loc[things["label"] == name, "frame"].nunique() / read
                if share < _RARELY_FOUND:
                    warnings.append(
                        QCWarning(
                            check=self.name, recording_id=recording.recording_id, subject_id=None, severity="warn",
                            message=(
                                f"{name} was found in {share:.0%} of the frames of {video}; if it is there more often, "
                                "try another word for it in gaze.targets"
                            ),
                        )
                    )
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        table = pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=list(OBJECT_COLUMNS))
        table.to_parquet(directory / OBJECTS_NAME, index=False)
        return ExtractorResult(rows=pd.DataFrame(), artifacts={OBJECTS_NAME: directory / OBJECTS_NAME}, warnings=warnings)


def objects_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).artifacts_dir(recording_id, GazeObjectsExtractor.name) / OBJECTS_NAME


def load_objects(config, recording_id: str) -> pd.DataFrame:
    path = objects_file(config, recording_id)
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=list(OBJECT_COLUMNS))


def _keypoints_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).keypoints_path(recording_id) / f"{PoseExtractor.name}.parquet"
