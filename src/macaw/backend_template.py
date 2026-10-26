"""The package `macaw new-backend` writes: a working pose backend to edit, its entry point and a
test. A backend only says how to read people out of one video; time, identity, caching and QC
come with the framework."""

from __future__ import annotations

from pathlib import Path
from string import Template

from macaw.exceptions import MacawError
from macaw.plugin_template import names_for

PYPROJECT = Template('''\
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "$distribution"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pymacaw"]

# this line is what makes macaw find the backend once the package is installed
[project.entry-points."macaw.backends"]
"$name" = "$module.backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
''')

README = Template('''\
# $name

A MACAW pose backend. Install it into the environment macaw runs in:

    pip install -e .

and choose it in project.yaml:

    pose:
      backend: $name

The example reads keypoints that something else already produced: next to each video
`clip.mp4` it expects `clip.keypoints.csv` with the columns frame, track_id, joint, x, y,
confidence (frame counted from 0 in the video file, x and y in pixels, joint a COCO-17 name such
as left_wrist). Replace the body of `infer` in `src/$module/backend.py` with your own model and
keep what it returns: one row per frame, person and joint, in the columns of
BACKEND_KEYPOINT_SCHEMA, under the four rules in the docstring of `PoseBackend.infer`.

To change one of the backends that ship with macaw rather than start from nothing, subclass it
and override only what you need, for example:

    from macaw.extractors.video.pose.backends import YOLOPoseBackend

    class MyYolo(YOLOPoseBackend):
        name = "$name"

Keep the `register_backend` decorator and give the class its own name; a name already taken by
an installed backend is refused.
''')

BACKEND = Template('''\
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend
from macaw.extractors.video.pose.backends import (
    BACKEND_GROUP,
    BACKEND_KEYPOINT_SCHEMA,
    PoseBackend,
    PoseFrames,
)
from macaw.extractors.video.pose.skeleton import STANDARD_JOINTS, extra_joint
from macaw.media.probe import probe
from macaw.store.schema import UNITS_PIXEL, coerce_frame, empty_frame

NAME = "$name"
SUFFIX = ".keypoints.csv"


def keypoint_file(video_path: Path) -> Path:
    return Path(video_path).with_suffix(SUFFIX)


def read_keypoints(path: Path, *, min_confidence: float, frames: tuple[int, int] | None) -> pd.DataFrame:
    table = pd.read_csv(path)
    if frames is not None:
        table = table[(table["frame"] >= frames[0]) & (table["frame"] < frames[1])]
    if table.empty:
        return empty_frame(BACKEND_KEYPOINT_SCHEMA)
    unsure = table["confidence"] < min_confidence
    out = pd.DataFrame(
        {
            "frame": table["frame"].astype(int),
            # a point outside the 17 standard ones must not be read as one of them
            "joint": [joint if joint in STANDARD_JOINTS else extra_joint(NAME, joint) for joint in table["joint"]],
            "x": table["x"].where(~unsure, np.nan),
            "y": table["y"].where(~unsure, np.nan),
            "z": np.nan,
            "confidence": table["confidence"],
            "track_id": table["track_id"].astype("Int64"),
            "units": UNITS_PIXEL,
            "metric_scale_valid": None,
        }
    )
    return coerce_frame(out, BACKEND_KEYPOINT_SCHEMA, "backend keypoints")


@register_backend(BACKEND_GROUP)
class $cls(PoseBackend):
    name = NAME
    version = "0.1.0"

    def check_available(self) -> None:
        # raise BackendNotAvailableError here when a model or a package is missing
        return None

    def infer(self, video_path, *, device, min_confidence, model=None, max_people=1, frames=None, input_size=None):
        path = keypoint_file(video_path)
        if not path.is_file():
            raise BackendNotAvailableError(f"no keypoints file {path.name} next to {Path(video_path).name}")
        stream = next(iter(probe(video_path).video_streams()), None)
        size = (stream.width, stream.height) if stream is not None and stream.width else None
        return PoseFrames(
            keypoints=read_keypoints(path, min_confidence=min_confidence, frames=frames),
            frame_size=size,
        )
''')

TEST = Template('''\
import pandas as pd

from $module.backend import read_keypoints


def write(tmp_path, rows):
    path = tmp_path / "clip.keypoints.csv"
    pd.DataFrame(rows, columns=["frame", "track_id", "joint", "x", "y", "confidence"]).to_csv(path, index=False)
    return path


def test_a_point_below_the_threshold_is_missing(tmp_path):
    path = write(tmp_path, [(0, 1, "nose", 10.0, 20.0, 0.9), (0, 1, "left_wrist", 30.0, 40.0, 0.1)])
    table = read_keypoints(path, min_confidence=0.3, frames=None)
    assert table["x"].isna().tolist() == [False, True]


def test_only_the_frames_asked_for(tmp_path):
    path = write(tmp_path, [(frame, 1, "nose", 10.0, 20.0, 0.9) for frame in range(10)])
    assert read_keypoints(path, min_confidence=0.3, frames=(2, 5))["frame"].tolist() == [2, 3, 4]


def test_a_joint_of_its_own_is_kept_apart_from_the_standard_ones(tmp_path):
    path = write(tmp_path, [(0, 1, "chin", 10.0, 20.0, 0.9)])
    assert read_keypoints(path, min_confidence=0.3, frames=None)["joint"].tolist() != ["chin"]
''')


def scaffold_backend(name: str, directory: Path) -> Path:
    """Write the package into directory/<distribution>; refuses to overwrite."""
    values = names_for(name)
    values["cls"] = values["cls"].removesuffix("Extractor") + "Backend"
    root = Path(directory) / values["distribution"]
    if root.exists():
        raise MacawError(f"{root} already exists; choose another directory or remove it")

    files = {
        root / "pyproject.toml": PYPROJECT,
        root / "README.md": README,
        root / "src" / values["module"] / "__init__.py": Template(""),
        root / "src" / values["module"] / "backend.py": BACKEND,
        root / "tests" / "test_backend.py": TEST,
    }
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.substitute(values), encoding="utf-8")
    return root
