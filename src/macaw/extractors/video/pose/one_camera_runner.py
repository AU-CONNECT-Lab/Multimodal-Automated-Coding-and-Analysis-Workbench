"""Runs Multi-HMR with the Anny body model over one video, in the environment it is installed in, for pose.mesh.

Called as `python one_camera_runner.py job.json`; the job gives video, out, code (the Multi-HMR folder), checkpoint
(or null for the one under code/models/multiHMR), fov_deg (or null to let the model estimate it), every (a frame
step) and threshold (the detection score a person needs). It writes out as an .npz with, per person found in a frame:
frame, points (17 COCO keypoints in the camera's frame, metres, x right, y down, z ahead), pixels (the same keypoints
in the video's pixels), height (the standing height of the body the model fitted, metres) and focal (the focal length
used, in the video's pixels). A runner for another model writes the same arrays. It imports nothing from MACAW, which
need not be installed beside it."""

import json
import sys
from pathlib import Path

import numpy as np

COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
]
CHECKPOINT = "multiHMR_672_L_anny"
NMS_KERNEL = 3


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import cv2
    import torch
    from PIL import Image, ImageOps

    sys.path.insert(0, job["code"])
    from multi_hmr_anny.multi_hmr import Multi_HMR
    from utils import normalize_rgb

    try:
        from anny.keypoints import KeypointsRegressor
    except ImportError:
        sys.exit("pose.mesh needs anny 0.6 or newer, for its COCO keypoints: pip install -U anny")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(job.get("checkpoint") or Path(job["code"]) / "models" / "multiHMR" / f"{CHECKPOINT}.pt")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Multi_HMR(**vars(saved["args"])).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=False)
    model.eval()
    # newer anny no longer hands back blendshape_coeffs, which Multi-HMR copies but does not use at inference
    model.body_model.register_forward_hook(lambda module, args, output: output if "blendshape_coeffs" in output else {**output, "blendshape_coeffs": None})
    coco = KeypointsRegressor.coco(model.body_model, labels=COCO17).to(device)
    size = model.img_size

    video = cv2.VideoCapture(job["video"])
    every = max(1, int(job.get("every") or 1))
    frames, points, pixels, heights, focals = [], [], [], [], []
    index = -1
    while True:
        ok, frame = video.read()
        index += 1
        if not ok:
            break
        if index % every:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scale = min(size / image.width, size / image.height)
        contained = ImageOps.contain(image, (size, size))
        pad_x, pad_y = (size - contained.width) / 2, (size - contained.height) / 2
        x = torch.from_numpy(normalize_rgb(np.asarray(ImageOps.pad(contained, size=(size, size))))).unsqueeze(0).to(device)
        K = None
        if job.get("fov_deg"):
            focal = image.width / 2 / np.tan(np.radians(job["fov_deg"]) / 2) * scale
            K = torch.tensor([[[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]]], dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device, enabled=device == "cuda"):
            people = model(x, is_training=False, nms_kernel_size=NMS_KERNEL, det_thresh=float(job["threshold"]), K=K)
        # with nobody found the model hands back ({}, []) instead of a list of people
        if isinstance(people, tuple):
            people = people[1]
        for person in people:
            with torch.no_grad():
                keypoints = coco({"vertices": person["v3d"].float()[None]})[0]
                rest = model.body_model(pose_parameters=None, phenotype_kwargs=_phenotypes(model, person["shape"].float()[None]))
            used = person["K"].float()
            square = (keypoints @ used.T)[:, :2] / keypoints[:, 2:3]
            frames.append(index)
            points.append(keypoints.cpu().numpy())
            pixels.append(((square.cpu().numpy() - [pad_x, pad_y]) / scale))
            heights.append(_height(model, rest["vertices"][0], rest["bone_poses"][0]))
            focals.append(float(used[0, 0]) / scale)
    video.release()
    np.savez_compressed(
        job["out"], frame=np.asarray(frames, int), points=np.asarray(points, float).reshape(-1, 17, 3),
        pixels=np.asarray(pixels, float).reshape(-1, 17, 2), height=np.asarray(heights, float), focal=np.asarray(focals, float),
    )


def _phenotypes(model, shape):
    """The phenotype values Multi-HMR feeds the body model, as it does."""
    return {label: shape[:, index] for index, label in enumerate(model.body_model.phenotype_labels)
            if label in ("age", "gender", "weight", "height", "muscle", "proportions")}


def _height(model, vertices, bones) -> float:
    """The rest body's extent along its spine, from the root bone toward the head."""
    labels = list(model.body_model.bone_labels)
    up = bones[labels.index("head"), :3, 3] - bones[labels.index("root"), :3, 3]
    up = up / up.norm()
    along = vertices @ up
    return float(along.max() - along.min())


if __name__ == "__main__":
    main()
