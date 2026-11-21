"""The things a project names, each found by name or started by a person, then followed frame by frame as an outline."""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError, ExternalToolError, UnreadableMediaError
from macaw.extractors import register_backend, register_extractor
from macaw.extractors.base import Backend, Extractor, ExtractorResult, QCWarning, distribution_version
from macaw.extractors.hub import check_hub_model
from macaw.extractors.prep import MediaPrepExtractor, VideoPrepExtractor, apply_prep_timing
from macaw.extractors.video.objects.decisions import ABSENT, STARTS, read_decisions, tracking_key
from macaw.extractors.video.objects.outline import mask_outline
from macaw.project.layout import layout_for
from macaw.project.recording import MediaFile, Recording
from macaw.store.schema import Scale

FINDERS, TRACKERS = "object_finders", "object_trackers"
OBJECTS_NAME = "objects.parquet"
STARTS_NAME = "starts.json"
# one row per frame a thing is in view: its outline and box in the video's pixels, and how sure the tracker is it is there
OBJECT_COLUMNS = ("video", "frame", "label", "score", "in_view", "x0", "y0", "x1", "y1", "cx", "cy", "area", "outline")
FOUND_COLUMNS = ("frame", "label", "x0", "y0", "x1", "y1", "score")
# a thing is looked for by name this often, a stretch at a time, until it is found
FIND_EVERY_S = 0.2
FIND_STRETCH_S = 10.0
# the tracker's own cut for a thing being in view
IN_VIEW = 0.5
# a thing in view in fewer of the frames after its start than this is in doubt
_FOUND_SHARE = 0.5
# decoded frames held at once while following a thing back
_BACK_BYTES = 512e6
FINDER, PERSON, NOT_FOUND, NOT_HERE = "found by name", "a person", "not found", "not on this camera"


@dataclass(frozen=True)
class TrackStart:
    name: str
    frame: int
    box: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None


class ObjectFinder(Backend):
    default_model: str | None = None

    def infer(self, video_path, *, frames: tuple[int, int], step: int, names: list[str], device: str, model: str | None,
              min_score: float) -> pd.DataFrame:
        """frame, label, x0, y0, x1, y1, score for every step-th frame from first to stop - 1 of the file, its own frame numbers."""
        raise NotImplementedError

    def prefetch(self, *, model: str | None) -> None:
        return None


class ObjectTracker(Backend):
    default_model: str | None = None

    def track(self, video_path, *, frames: tuple[int, int], starts: list[TrackStart], device: str,
              model: str | None) -> Iterator[tuple[int, dict[str, tuple[np.ndarray, float]]]]:
        """(frame, {name: (boolean mask the frame's size, 0-1 that it is in view)}) for frames from first to stop - 1, each
        thing followed back from its first start to first and on from there to stop - 1, in any order of frames and with
        a frame at most once per thing; a later start of a thing puts it back where it says."""
        raise NotImplementedError

    def prefetch(self, *, model: str | None) -> None:
        return None


def _torch_and_transformers(setting: str, name: str) -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise BackendNotAvailableError(
            f"{setting} {name} needs PyTorch and transformers: install PyTorch for your GPU first, then pip install transformers"
        ) from error


def _device(device: str) -> str:
    import torch

    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


@register_backend(FINDERS)
class GroundingDinoFinder(ObjectFinder):
    """Grounding DINO through transformers; Apache-2.0 code and weights from the Hugging Face hub."""

    name = "grounding_dino"
    default_model = "IDEA-Research/grounding-dino-tiny"
    # the original authors' cut for keeping a word found in a box
    text_threshold = 0.25
    batch_size = 8

    @property
    def version(self) -> str:
        return distribution_version("transformers")

    def check_available(self) -> None:
        _torch_and_transformers("objects.finder", self.name)

    def prefetch(self, *, model: str | None) -> None:
        self.network(model, "cpu")

    def network(self, model: str | None, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        name = model or self.default_model
        loaded = self.__dict__.setdefault("_networks", {})
        if (name, device) not in loaded:
            loaded[name, device] = (AutoProcessor.from_pretrained(name), AutoModelForZeroShotObjectDetection.from_pretrained(name).eval().to(device))
        return loaded[name, device]

    def infer(self, video_path, *, frames, step, names, device, model, min_score):
        import torch

        device = _device(device)
        processor, net = self.network(model, device)
        rows: list[dict] = []
        for batch in _batches(_every_step(video_path, frames, step), self.batch_size):
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
        return pd.DataFrame(rows, columns=list(FOUND_COLUMNS))


@register_backend(FINDERS)
class Sam3Finder(ObjectFinder):
    """SAM 3 finding every instance of a name through transformers; weights under the SAM License, given on request on the
    Hugging Face hub."""

    name = "sam3"
    default_model = "facebook/sam3"
    batch_size = 4

    @property
    def version(self) -> str:
        return distribution_version("transformers")

    def check_available(self) -> None:
        _torch_and_transformers("objects.finder", self.name)
        import transformers

        if not hasattr(transformers, "Sam3Model"):
            raise BackendNotAvailableError(f"objects.finder {self.name} needs a transformers with Sam3Model: pip install -U transformers")

    def prefetch(self, *, model: str | None) -> None:
        self.network(model, "cpu")

    def network(self, model: str | None, device: str):
        import torch
        from transformers import Sam3Model, Sam3Processor

        name = model or self.default_model
        loaded = self.__dict__.setdefault("_networks", {})
        if (name, device) not in loaded:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            loaded[name, device] = (Sam3Processor.from_pretrained(name), Sam3Model.from_pretrained(name).to(device, dtype=dtype).eval(), dtype)
        return loaded[name, device]

    def infer(self, video_path, *, frames, step, names, device, model, min_score):
        import torch

        device = _device(device)
        processor, net, dtype = self.network(model, device)
        words = {name: processor(text=name, return_tensors="pt").to(device) for name in names}
        rows: list[dict] = []
        for batch in _batches(_every_step(video_path, frames, step), self.batch_size):
            images = [image for _, image in batch]
            sizes = [image.size[::-1] for image in images]
            pixels = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                # the picture is read once and asked for each name in turn
                seen = net.get_vision_features(pixel_values=pixels.pixel_values.to(dtype))
                for name, word in words.items():
                    found = net(
                        vision_embeds=seen, input_ids=word["input_ids"].expand(len(images), -1),
                        attention_mask=word["attention_mask"].expand(len(images), -1),
                    )
                    results = processor.post_process_object_detection(found, threshold=min_score, target_sizes=sizes)
                    for (frame, _), result in zip(batch, results):
                        for box, score in zip(result["boxes"].float().tolist(), result["scores"].float().tolist()):
                            rows.append({"frame": frame, "label": name, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "score": score})
        return pd.DataFrame(rows, columns=list(FOUND_COLUMNS))


def _every_step(video_path, frames: tuple[int, int], step: int):
    """(frame, RGB picture) for every step-th frame from first to stop - 1, the frames between only skipped over."""
    import cv2
    from PIL import Image

    from macaw.extractors.video.pose.backends import _seek

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExternalToolError(f"could not open video: {video_path}")
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
            yield index, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        capture.release()


def _batches(items, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class _SamTracker(ObjectTracker):
    """A SAM video tracker through transformers, fed one frame at a time and keeping only the memory it reads, so a long
    video takes no more GPU memory than a short one."""

    model_class: str
    processor_class: str
    setting_hint: str = ""

    @property
    def version(self) -> str:
        return distribution_version("transformers")

    def check_available(self) -> None:
        _torch_and_transformers("objects.tracker", self.name)
        import transformers

        if not hasattr(transformers, self.model_class):
            raise BackendNotAvailableError(
                f"objects.tracker {self.name} needs a transformers with {self.model_class}: pip install -U transformers"
            )

    def prefetch(self, *, model: str | None) -> None:
        self.network(model, "cpu")

    def network(self, model: str | None, device: str):
        import torch
        import transformers

        name = model or self.default_model
        loaded = self.__dict__.setdefault("_networks", {})
        if (name, device) not in loaded:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            processor = getattr(transformers, self.processor_class).from_pretrained(name)
            net = getattr(transformers, self.model_class).from_pretrained(name).to(device, dtype=dtype).eval()
            loaded[name, device] = (processor, net, dtype)
        return loaded[name, device]

    def track(self, video_path, *, frames, starts, device, model):
        if not starts:
            return
        device = _device(device)
        first, stop = frames
        earliest: dict[str, TrackStart] = {}
        for start in sorted(starts, key=lambda start: start.frame):
            earliest.setdefault(start.name, start)
        back = [start for start in earliest.values() if start.frame > first]
        if back:
            # a thing is followed back only over the frames before its own first start
            before = {start.name: start.frame for start in back}
            for index, seen in self._follow(_pictures_back(video_path, max(before.values()), first), back, device, model):
                mine = {name: found for name, found in seen.items() if index < before[name]}
                if mine:
                    yield index, mine
        on = min(start.frame for start in earliest.values())
        yield from self._follow(_pictures_on(video_path, on, stop), starts, device, model)

    def _follow(self, pictures, starts: list[TrackStart], device: str, model: str | None):
        """(frame, {name: (mask, 0-1 in view)}) for each (frame, picture) in the order given, a thing from its start on."""
        import cv2
        import torch

        processor, net, dtype = self.network(model, device)
        ids = {name: number for number, name in enumerate(sorted({start.name for start in starts}), start=1)}
        names = {number: name for name, number in ids.items()}
        by_frame: dict[int, list[TrackStart]] = {}
        for start in starts:
            by_frame.setdefault(start.frame, []).append(start)
        # memories of this many frames back are what the tracker reads: its own memory and the pointers to the thing
        keep = max(int(getattr(net.config, "num_maskmem", 7)), int(getattr(net.config, "max_object_pointers_in_encoder", 16))) + 1
        session = processor.init_video_session(inference_device=device, dtype=dtype)
        step = 0
        for index, image in pictures:
            started = by_frame.get(index, [])
            if step == 0 and not started:
                continue
            pixels = processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device=device, return_tensors="pt")
            for start in started:
                prompt = (
                    {"input_boxes": [[list(start.box)]]} if start.box is not None
                    else {"input_points": [[[list(start.point)]]], "input_labels": [[[1]]]}
                )
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=step, obj_ids=ids[start.name], original_size=pixels.original_sizes[0], **prompt,
                )
            if started:
                # each call names only its own thing as newly started, so several starting on one frame are named again
                session.obj_with_new_inputs = sorted({ids[start.name] for start in started})
            with torch.inference_mode():
                output = net(inference_session=session, frame=pixels.pixel_values[0].to(dtype), frame_idx=step)
                masks = processor.post_process_masks([output.pred_masks], original_sizes=pixels.original_sizes, binarize=True)[0]
                present = torch.sigmoid(output.object_score_logits.float().reshape(-1)).tolist()
            masks = masks[:, 0].cpu().numpy()
            yield index, {names[number]: (masks[position], present[position]) for position, number in enumerate(output.object_ids)}
            _forget(session, step, keep)
            step += 1


def _pictures_on(video_path, first: int, stop: int) -> Iterator[tuple[int, np.ndarray]]:
    import cv2

    from macaw.extractors.video.pose.backends import _seek

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExternalToolError(f"could not open video: {video_path}")
    try:
        index = _seek(capture, cv2, first, Path(video_path))
        while index < stop:
            ok, image = capture.read()
            if not ok:
                break
            yield index, image
            index += 1
    finally:
        capture.release()


def _pictures_back(video_path, last: int, first: int) -> Iterator[tuple[int, np.ndarray]]:
    """Frames from last down to first, read forward a stretch at a time and handed out backward."""
    import cv2

    from macaw.extractors.video.pose.backends import _seek

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ExternalToolError(f"could not open video: {video_path}")
    try:
        size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 3
        stretch = max(int(_BACK_BYTES // max(size, 1)), 8)
        end = last + 1
        while end > first:
            begin = max(end - stretch, first)
            index, held = _seek(capture, cv2, begin, Path(video_path)), []
            while index < end:
                ok, image = capture.read()
                if not ok:
                    break
                if index >= begin:
                    held.append((index, image))
                index += 1
            yield from reversed(held)
            end = begin
    finally:
        capture.release()


def _forget(session, step: int, keep: int) -> None:
    """Drops what the next frames no longer read: older frames' pictures and memories; the starts stay."""
    if session.processed_frames is not None:
        session.processed_frames.pop(step - 1, None)
    for outputs in session.output_dict_per_obj.values():
        outputs["non_cond_frame_outputs"].pop(step - keep, None)
    for tracked in session.frames_tracked_per_obj.values():
        tracked.pop(step - keep, None)


@register_backend(TRACKERS)
class Sam2Tracker(_SamTracker):
    """SAM 2.1 through transformers; Apache-2.0 code and weights from the Hugging Face hub."""

    name = "sam2"
    default_model = "facebook/sam2.1-hiera-base-plus"
    model_class, processor_class = "Sam2VideoModel", "Sam2VideoProcessor"


@register_backend(TRACKERS)
class Sam3Tracker(_SamTracker):
    """SAM 3's tracker through transformers; weights under the SAM License, given on request on the Hugging Face hub."""

    name = "sam3"
    default_model = "facebook/sam3"
    model_class, processor_class = "Sam3TrackerVideoModel", "Sam3TrackerVideoProcessor"


# the open models SAM 3 gives way to where this machine cannot have it
OPEN_FINDER, OPEN_TRACKER = "grounding_dino", "sam2"


@dataclass(frozen=True)
class ObjectModels:
    finder: str
    finder_model: str | None
    tracker: str
    tracker_model: str | None
    why: str = ""

    def note(self) -> str:
        """What a person is told when SAM 3 gave way."""
        return (
            f"SAM 3 cannot be used here, so things are found with {self.finder} and followed with {self.tracker} instead. "
            f"{self.why} Once it can, the next run follows them again with SAM 3; to keep these instead, set "
            f"objects.finder: {self.finder} and objects.tracker: {self.tracker}."
        )


_CHOSEN: dict[tuple, ObjectModels] = {}


def object_models(objects) -> ObjectModels:
    """The finder and tracker set, SAM 3 swapped for the open ones where this machine cannot have it; looked up once a process."""
    key = (objects.finder, objects.finder_model, objects.tracker, objects.tracker_model)
    if key not in _CHOSEN:
        finder, finder_model, tracker, tracker_model, why = *key, []
        if finder == "sam3" and (missing := _sam3_missing(finder_model or Sam3Finder.default_model)):
            finder, finder_model = OPEN_FINDER, None
            why.append(missing)
        if tracker == "sam3" and (missing := _sam3_missing(tracker_model or Sam3Tracker.default_model)):
            tracker, tracker_model = OPEN_TRACKER, None
            why.append(missing)
        _CHOSEN[key] = ObjectModels(finder, finder_model, tracker, tracker_model, " ".join(dict.fromkeys(why)))
    return _CHOSEN[key]


def _sam3_missing(model: str) -> str:
    """Why SAM 3 cannot be had here, or nothing; without transformers at all, the step's own check says what to install."""
    found = importlib.util.find_spec("transformers")
    if found is None or not found.origin:
        return ""
    if not all((Path(found.origin).parent / "models" / part).is_dir() for part in ("sam3", "sam3_tracker_video")):
        return "The transformers installed here has no SAM 3: pip install -U transformers."
    try:
        check_hub_model(model, "SAM 3")
    except BackendNotAvailableError as error:
        return str(error)
    return ""


def resolve_finder(name: str) -> ObjectFinder:
    from macaw.extractors import get_backend

    backend = get_backend(FINDERS, name)()
    backend.check_available()
    return backend


def resolve_tracker(name: str) -> ObjectTracker:
    from macaw.extractors import get_backend

    backend = get_backend(TRACKERS, name)()
    backend.check_available()
    return backend


def object_cameras(recording: Recording) -> list[MediaFile]:
    """The cameras someone's body or gaze is read from, where a thing's place in the picture is used."""
    used = {item.name for use in ("body", "gaze") for item in recording.views(use).values()}
    return [item for item in recording.cameras() if item.name in used]


def frame_span(item: MediaFile, start_s: float, end_s: float) -> tuple[int, int]:
    fps = item.fps or 0.0
    if fps <= 0:
        raise UnreadableMediaError(f"{item.path}: no frame rate to convert frames to time")
    return max(math.floor((start_s - item.offset_s) * fps), 0), math.ceil((end_s - item.offset_s) * fps) + 1


def proposals(found: pd.DataFrame, names: list[str]) -> dict[str, dict]:
    """name -> the start found by name in one stretch: its best box, and how many boxes in that frame could be it."""
    chosen = {}
    for name in names:
        mine = found[found["label"] == name]
        if mine.empty:
            continue
        best = mine.sort_values(["score", "frame"], ascending=[False, True]).iloc[0]
        others = int((mine["frame"] == best["frame"]).sum())
        chosen[name] = {
            "frame": int(best["frame"]), "box": [round(float(best[key]), 1) for key in ("x0", "y0", "x1", "y1")],
            "score": round(float(best["score"]), 3), "candidates": others,
        }
    return chosen


def outline_rows(video: str, frame: int, seen: dict[str, tuple[np.ndarray, float]]) -> list[dict]:
    """A row for each thing the tracker gave in the frame, its score always; in view where the score reaches IN_VIEW and the
    mask has an outline, which is kept below the cut as well, so that a person can lower it."""
    rows = []
    for name, (mask, present) in seen.items():
        row = {"video": video, "frame": int(frame), "label": name, "score": round(float(present), 4), "in_view": False, "outline": []}
        outline = mask_outline(mask) if mask.any() else []
        if outline:
            ys, xs = np.nonzero(mask)
            row.update(
                x0=float(xs.min()), y0=float(ys.min()), x1=float(xs.max() + 1), y1=float(ys.max() + 1),
                cx=round(float(xs.mean()) + 0.5, 2), cy=round(float(ys.mean()) + 0.5, 2), area=float(mask.sum()), outline=outline,
                in_view=bool(present >= IN_VIEW),
            )
        rows.append(row)
    return rows


def followed(frames_found: np.ndarray, first: int, stop: int, fps: float) -> dict:
    """How much of the stretch from first to stop a thing was in view, and its longest time out of view."""
    span = max(stop - first, 1)
    marks = np.zeros(span, dtype=bool)
    marks[np.clip(np.asarray(frames_found, dtype=int) - first, 0, span - 1)] = True
    gaps = np.diff(np.flatnonzero(np.r_[True, marks, True])) - 1
    return {"frames": int(span), "found": int(marks.sum()), "longest_gap_s": round(float(gaps.max(initial=0)) / fps, 2)}


def doubts(entry: dict) -> list[str]:
    """Why a person should look at this start before the outlines are used."""
    if entry["from"] == NOT_FOUND:
        return ["not found by name on this camera"]
    said = []
    if entry["from"] == FINDER and entry.get("candidates", 1) > 1:
        said.append(f"{entry['candidates']} things in the start frame could be it")
    if entry.get("frames") and entry["share_after_start"] < _FOUND_SHARE:
        said.append(f"in view in {entry['share_after_start']:.0%} of the frames after its start")
    return said


@register_extractor
class ObjectTrackExtractor(Extractor):
    name = "objects.track"
    version = "0.3.0"
    scale = Scale.FRAME
    group = "objects"
    config_sections = (
        "objects.names", "objects.finder", "objects.finder_model", "objects.finder_score", "objects.tracker", "objects.tracker_model",
        "segment", "video", "setup.cameras",
    )
    requires = (MediaPrepExtractor.name, VideoPrepExtractor.name)
    resource_profile = "gpu_standard"

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.objects.names)

    def models(self) -> ObjectModels:
        return object_models(self.config.objects)

    def finder(self) -> ObjectFinder:
        return resolve_finder(self.models().finder)

    def tracker(self) -> ObjectTracker:
        return resolve_tracker(self.models().tracker)

    def backend_info(self) -> tuple[str | None, str | None]:
        tracker = self.tracker()
        return (tracker.name, tracker.version)

    def backends(self) -> list[Backend]:
        from macaw.extractors import get_backend

        models = self.models()
        return [get_backend(FINDERS, models.finder)(), get_backend(TRACKERS, models.tracker)()]

    def check_available(self) -> None:
        models = self.models()
        for backend, model, setting in ((self.finder(), models.finder_model, "objects.finder_model"), (self.tracker(), models.tracker_model, "objects.tracker_model")):
            if model or backend.default_model:
                check_hub_model(model or backend.default_model, setting)

    def prefetch(self) -> None:
        models = self.models()
        self.finder().prefetch(model=models.finder_model)
        self.tracker().prefetch(model=models.tracker_model)

    def inputs_for(self, recording: Recording) -> list[Path]:
        paths = sorted(item.path for item in object_cameras(recording))
        offsets = layout_for(self.config, create=False).offset_table_path(recording.recording_id)
        return paths + ([offsets] if offsets.is_file() else [])

    def extra_key(self, recording: Recording) -> str:
        models = self.models()
        said = tracking_key(read_decisions(layout_for(self.config, create=False).objects_path(recording.recording_id)))
        # where SAM 3 gave way, the run once it can be had is a different one
        return said + (f"|{models.finder},{models.tracker}" if models.why else "")

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        cfg = self.config.objects
        prep = apply_prep_timing(self.config, recording)
        start_s, end_s = prep.t_origin_s, prep.t_origin_s + prep.duration_s
        said = read_decisions(layout_for(self.config, create=False).objects_path(recording.recording_id))
        models = self.models()
        finder, tracker = self.finder(), self.tracker()
        rows, summary, warnings = [], {}, []
        if models.why:
            warnings.append(self._warning(recording, "warn", models.note()))
        for item in object_cameras(recording):
            first, stop = frame_span(item, start_s, end_s)
            if stop <= first:
                continue
            fps = float(item.fps)
            given, absent = said[STARTS].get(item.name, {}), said[ABSENT].get(item.name, {})
            wanted = [name for name in cfg.names if name not in given and name not in absent]
            step = max(round(FIND_EVERY_S * fps), 1)
            found = self._find(finder, models.finder_model, item, (first, stop), step, wanted) if wanted else {}
            entries, starts = {}, []
            for name in cfg.names:
                if name in absent:
                    entries[name] = {"from": NOT_HERE}
                elif name in given:
                    points = [{**point, "frame": int(np.clip(round((point["t"] - item.offset_s) * fps), first, stop - 1))} for point in given[name]]
                    starts += [TrackStart(name, point["frame"], point=(point["x"], point["y"])) for point in points]
                    entries[name] = {"from": PERSON, "frame": points[0]["frame"], "points": points}
                elif name in found:
                    starts.append(TrackStart(name, found[name]["frame"], box=tuple(found[name]["box"])))
                    entries[name] = {"from": FINDER, **found[name]}
                else:
                    entries[name] = {"from": NOT_FOUND}
            mine: list[dict] = []
            for frame, seen in tracker.track(item.frames_path, frames=(first, stop), starts=starts, device=cfg.device, model=models.tracker_model):
                mine += outline_rows(item.name, frame, seen)
            shown = pd.DataFrame(mine, columns=list(OBJECT_COLUMNS))
            frames_by_name = shown[shown["in_view"].astype(bool)].groupby("label")["frame"]
            for name, entry in entries.items():
                if "frame" in entry:
                    entry["t"] = round(entry["frame"] / fps + item.offset_s, 3)
                    seen_at = frames_by_name.get_group(name).to_numpy() if name in frames_by_name.groups else np.array([], dtype=int)
                    entry.update(followed(seen_at, first, stop, fps))
                    after = followed(seen_at[seen_at >= entry["frame"]], entry["frame"], stop, fps)
                    entry["share_after_start"] = round(after["found"] / after["frames"], 3)
                entry["doubts"] = doubts(entry)
                entry.update(finder=models.finder, tracker=models.tracker)
                if entry["from"] == NOT_FOUND:
                    warnings.append(self._warning(
                        recording, "warn",
                        f"{name} was not found by name on {item.name}; start it by a click on the Review page, say it is not on this "
                        "camera, or try another word for it in objects.names",
                    ))
            rows += mine
            summary[item.name] = entries
        directory = workspace / self.name
        directory.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows, columns=list(OBJECT_COLUMNS)).sort_values(["video", "frame", "label"], kind="stable", ignore_index=True)
        table.to_parquet(directory / OBJECTS_NAME, index=False)
        (directory / STARTS_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return ExtractorResult(
            rows=pd.DataFrame(), artifacts={OBJECTS_NAME: directory / OBJECTS_NAME, STARTS_NAME: directory / STARTS_NAME}, warnings=warnings,
        )

    def _find(self, finder: ObjectFinder, model: str | None, item: MediaFile, span: tuple[int, int], step: int, names: list[str]) -> dict[str, dict]:
        """Looks a stretch at a time from the start, until each name is found or the video ends."""
        cfg, chosen = self.config.objects, {}
        stretch = max(round(FIND_STRETCH_S / FIND_EVERY_S), 1)
        first, stop = span
        while first < stop and len(chosen) < len(names):
            end = min(first + step * stretch, stop)
            looking = [name for name in names if name not in chosen]
            found = finder.infer(item.frames_path, frames=(first, end), step=step, names=looking, device=cfg.device, model=model, min_score=cfg.finder_score)
            chosen.update(proposals(found, looking))
            first = end
        return chosen

    def _warning(self, recording: Recording, severity: str, message: str) -> QCWarning:
        return QCWarning(check=self.name, recording_id=recording.recording_id, subject_id=None, severity=severity, message=message)


def objects_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).artifacts_dir(recording_id, ObjectTrackExtractor.name) / OBJECTS_NAME


def starts_file(config, recording_id: str) -> Path:
    return layout_for(config, create=False).artifacts_dir(recording_id, ObjectTrackExtractor.name) / STARTS_NAME


def load_objects(config, recording_id: str, *, everything: bool = False) -> pd.DataFrame:
    """The outlines of the things in view; with everything, also the frames the tracker scored below IN_VIEW."""
    path = objects_file(config, recording_id)
    table = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=list(OBJECT_COLUMNS))
    return table if everything else table[table["in_view"].astype(bool)].reset_index(drop=True)


def load_starts(config, recording_id: str) -> dict:
    path = starts_file(config, recording_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
