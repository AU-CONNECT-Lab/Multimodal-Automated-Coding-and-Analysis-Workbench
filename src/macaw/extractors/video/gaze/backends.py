from __future__ import annotations

import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError
from macaw.extractors import register_backend
from macaw.extractors.base import Backend

BACKEND_GROUP = "gaze"
# what infer returns per head and frame: the point looked at in the video's pixels, and how sure the model is it lies in the picture
GAZE_POINT_COLUMNS = ("frame", "subject_id", "gaze_x", "gaze_y", "in_frame")
HEAD_COLUMNS = ("frame", "subject_id", "x0", "y0", "x1", "y1")


class GazeBackend(Backend):
    # whether it finds the point looked at in the picture; one that does not reads head and eye angles instead
    finds_points: bool = True

    def infer(self, video_path, *, heads: pd.DataFrame, device: str, model: str | None, batch_size: int) -> pd.DataFrame:
        """heads holds HEAD_COLUMNS in the video's pixels and its own frame numbers; one row of GAZE_POINT_COLUMNS back per head."""
        raise NotImplementedError

    def prefetch(self, *, model: str | None) -> None:
        return None


@register_backend(BACKEND_GROUP)
class GazeLLEBackend(GazeBackend):
    """Gaze-LLE (Ryan et al. 2025): MIT code and decoder weights, on a DINOv2 backbone (Apache-2.0),
    both fetched through torch.hub on first use."""

    name = "gazelle"
    hub_repo = "fkryan/gazelle"
    weights_url = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/{model}.pt"
    # the in-or-out models also say whether the target is in the picture; the childplay ones were tuned on children playing with adults
    models = (
        "gazelle_dinov2_vitb14_inout", "gazelle_dinov2_vitl14_inout",
        "gazelle_dinov2_vitb14_inout_childplay", "gazelle_dinov2_vitl14_inout_childplay",
    )
    default_model = "gazelle_dinov2_vitl14_inout_childplay"
    input_size = 448
    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)

    @property
    def version(self) -> str:
        return "1.0.0"

    def check_available(self) -> None:
        try:
            import timm  # noqa: F401
            import torch  # noqa: F401
        except ImportError as error:
            raise BackendNotAvailableError(
                "gaze.backend gazelle needs PyTorch and timm: install PyTorch for your GPU first, then pip install timm. "
                "The model itself comes through torch.hub the first time it runs."
            ) from error

    def prefetch(self, *, model: str | None) -> None:
        self.network(model, "cpu")

    def network(self, model: str | None, device: str):
        import torch

        model = model or self.default_model
        if model not in self.models:
            raise BackendNotAvailableError(f"gaze.model for gazelle is one of {', '.join(self.models)}, not {model}")
        try:
            net, _ = torch.hub.load(self.hub_repo, model.removesuffix("_childplay"), pretrained=False, trust_repo=True, skip_validation=True)
            net.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(self.weights_url.format(model=model), map_location="cpu"))
        except (OSError, RuntimeError) as error:
            raise ExternalToolError(
                f"could not load Gaze-LLE {model} through torch.hub ({error}); on a machine without internet, "
                "run macaw run on one with it first so the model is in TORCH_HOME"
            ) from error
        return net.eval().to(device)

    def infer(self, video_path, *, heads, device, model, batch_size):
        import cv2
        import numpy as np
        import torch

        from macaw.extractors.video.pose.backends import _seek

        if heads.empty:
            return pd.DataFrame(columns=list(GAZE_POINT_COLUMNS))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self.network(model, device)
        mean = torch.tensor(self._mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self._std, device=device).view(1, 3, 1, 1)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExternalToolError(f"could not open video: {video_path}")
        width, height = capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        by_frame = {frame: group for frame, group in heads.groupby("frame", sort=True)}
        rows: list[dict] = []
        batch: list[tuple[int, np.ndarray]] = []

        def flush():
            images = torch.from_numpy(np.stack([image for _, image in batch])).to(device)
            images = (images.permute(0, 3, 1, 2).float() / 255.0 - mean) / std
            boxes = [
                [(g.x0 / width, g.y0 / height, g.x1 / width, g.y1 / height) for g in by_frame[frame].itertuples()]
                for frame, _ in batch
            ]
            with torch.inference_mode():
                found = net({"images": images, "bboxes": boxes})
            for (frame, _), maps, scores in zip(batch, found["heatmap"], found["inout"]):
                for head, heat, score in zip(by_frame[frame].itertuples(), maps, scores):
                    row, col = divmod(int(torch.argmax(heat)), heat.shape[1])
                    rows.append(
                        {
                            "frame": frame, "subject_id": head.subject_id,
                            "gaze_x": (col + 0.5) / heat.shape[1] * width, "gaze_y": (row + 0.5) / heat.shape[0] * height,
                            "in_frame": float(score),
                        }
                    )
            batch.clear()

        try:
            frames = list(by_frame)
            index = _seek(capture, cv2, max(int(frames[0]), 0))
            for frame in frames:
                while index < frame and capture.grab():
                    index += 1
                if index != frame:
                    break
                ok, image = capture.read()
                index += 1
                if not ok:
                    break
                small = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
                batch.append((frame, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                if len(batch) >= batch_size:
                    flush()
            if batch:
                flush()
        finally:
            capture.release()
        return pd.DataFrame(rows, columns=list(GAZE_POINT_COLUMNS))


@register_backend(BACKEND_GROUP)
class AnglesBackend(GazeBackend):
    """No model of its own: reads face.au's head and eye angles against gaze.seats and the frames a person labelled."""

    name = "angles"
    version = "1.0"
    finds_points = False

    def check_available(self) -> None:
        return None


def resolve_backend(name: str) -> GazeBackend:
    from macaw.extractors import get_backend

    backend = get_backend(BACKEND_GROUP, name)()
    backend.check_available()
    return backend
