"""The package `macaw new-backend` writes: a working pose, ASR or prosody backend to edit, its
entry point and a test. A backend only says how to read one file; time, identity, caching and QC
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

POSE_README = Template('''\
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

POSE_BACKEND = Template('''\
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

POSE_TEST = Template('''\
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


ASR_README = Template('''\
# $name

A MACAW speech recogniser backend. Install it into the environment macaw runs in:

    pip install -e .

and choose it in project.yaml:

    text:
      asr_backend: $name
      asr_model: D:/transcripts      # the folder this example reads from

The example reads transcripts something else already produced, for instance by hand or by a
service: `<asr_model>/<recording_id>/<subject_id>.csv` with the columns t_start, t_end, text
and optionally confidence, times in seconds from the start of the prepared track (the start of
the task window). Replace the body of `transcribe` in `src/$module/backend.py` with your own
recogniser and keep what it returns: utterances with utterance_index, t_start, t_end, text,
confidence, language, and words (empty unless word timestamps are asked for). Word counts, the
long table rows and the quality warnings come from macaw.

To change one of the recognisers that ship with macaw, subclass it and give it its own name:

    from macaw.extractors.audio.content.asr import FasterWhisperBackend

    class MyWhisper(FasterWhisperBackend):
        name = "$name"
''')

ASR_BACKEND = Template('''\
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import BackendNotAvailableError
from macaw.extractors import register_backend
from macaw.extractors.audio.content.asr import BACKEND_GROUP, ASRBackend

NAME = "$name"


def transcript_file(folder: Path, audio_path: Path) -> Path:
    # a prepared track sits at recordings/<recording_id>/artifacts/prep.media/<subject_id>.wav
    recording_id = Path(audio_path).parents[2].name
    return Path(folder) / recording_id / f"{Path(audio_path).stem}.csv"


def read_transcript(path: Path, *, language: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(path).sort_values("t_start").reset_index(drop=True)
    utterances = pd.DataFrame(
        {
            "utterance_index": range(len(table)),
            "t_start": table["t_start"].astype(float),
            "t_end": table["t_end"].astype(float),
            "text": table["text"].fillna("").astype(str).str.strip(),
            "confidence": table["confidence"].astype(float) if "confidence" in table else np.nan,
            "language": language,
        }
    )
    words = pd.DataFrame(columns=["utterance_index", "t_start", "t_end", "word", "score"])
    return utterances, words


@register_backend(BACKEND_GROUP)
class $cls(ASRBackend):
    name = NAME
    version = "0.1.0"
    # mean confidence below which a transcript is suspect, on this backend's own scale; None
    # until you have measured one
    low_confidence = None

    def check_available(self) -> None:
        if not Path(self.model).is_dir():
            raise BackendNotAvailableError(f"text.asr_model should name the folder of transcripts, not {self.model}")

    def transcribe(self, audio_path, *, language, word_timestamps):
        path = transcript_file(Path(self.model), audio_path)
        if not path.is_file():
            raise BackendNotAvailableError(f"no transcript {path}")
        return read_transcript(path, language=language)
''')

ASR_TEST = Template('''\
from pathlib import Path

import pandas as pd

from $module.backend import read_transcript, transcript_file


def test_utterances_come_back_in_time_order(tmp_path):
    path = tmp_path / "parent.csv"
    pd.DataFrame({"t_start": [4.0, 1.0], "t_end": [5.0, 2.0], "text": ["later", " first "]}).to_csv(path, index=False)
    utterances, words = read_transcript(path, language="en")
    assert utterances["text"].tolist() == ["first", "later"]
    assert utterances["utterance_index"].tolist() == [0, 1] and words.empty


def test_the_file_is_found_by_recording_and_person(tmp_path):
    track = Path("out/recordings/003/artifacts/prep.media/parent.wav")
    assert transcript_file(tmp_path, track) == tmp_path / "003" / "parent.csv"
''')

PROSODY_README = Template('''\
# $name

A MACAW prosody backend. Install it into the environment macaw runs in:

    pip install -e .

and choose it in project.yaml:

    prosody:
      backend: $name

The example is a short autocorrelation pitch tracker with an RMS intensity, written to be
replaced. Each method gets one prepared track as samples and returns one value per frame on
the grid `frame_times(len(audio), sr, step_s)`, NaN where there is nothing to measure (an
unvoiced frame has no pitch). Say which quantities it gives in `capabilities`; prosody.frame
asks only for those.

To change one of the backends that ship with macaw, subclass it and give it its own name:

    from macaw.extractors.audio.prosody.backends import ParselmouthBackend

    class MyPraat(ParselmouthBackend):
        name = "$name"
''')

PROSODY_BACKEND = Template('''\
from __future__ import annotations

import numpy as np

from macaw.extractors import register_backend
from macaw.extractors.audio.prosody.backends import BACKEND_GROUP, ProsodyBackend, frame_times

NAME = "$name"
# an autocorrelation peak weaker than this is noise, not a period
VOICING = 0.5


def frames(audio: np.ndarray, sr: float, step_s: float, width: int) -> np.ndarray:
    # one row per frame on the shared grid, each centred on its frame time
    times = frame_times(audio.size, sr, step_s)
    padded = np.pad(np.asarray(audio, dtype=float), width // 2)
    starts = np.round(times * sr).astype(int)
    return np.stack([padded[start : start + width] for start in starts])


def pitch(audio: np.ndarray, sr: float, f0_min: float, f0_max: float, step_s: float) -> np.ndarray:
    width = int(3 * sr / f0_min)
    out = []
    for frame in frames(audio, sr, step_s, width):
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0:
            out.append(np.nan)
            continue
        correlation = np.correlate(frame, frame, mode="full")[width - 1 :] / energy
        low, high = int(sr / f0_max), min(int(sr / f0_min), width - 1)
        lag = low + int(np.argmax(correlation[low:high]))
        out.append(sr / lag if correlation[lag] >= VOICING else np.nan)
    return np.array(out)


@register_backend(BACKEND_GROUP)
class $cls(ProsodyBackend):
    name = NAME
    version = "0.1.0"
    capabilities = frozenset({"f0", "intensity"})

    def check_available(self) -> None:
        return None

    def f0(self, audio, sr, f0_min, f0_max, step_s):
        return pitch(audio, sr, f0_min, f0_max, step_s)

    def intensity(self, audio, sr, step_s):
        rms = np.sqrt(np.mean(np.square(frames(audio, sr, step_s, int(0.04 * sr))), axis=1))
        return 20 * np.log10(np.maximum(rms, 1e-10) / 2e-5)
''')

PROSODY_TEST = Template('''\
import numpy as np

from macaw.extractors.audio.prosody.backends import frame_times
from $module.backend import pitch


def test_a_steady_tone_reads_as_its_own_pitch():
    sr = 16000
    t = np.arange(sr) / sr
    f0 = pitch(np.sin(2 * np.pi * 200 * t), sr, 75, 500, 0.01)
    assert abs(np.nanmedian(f0) - 200) < 5


def test_silence_has_no_pitch_and_the_grid_is_the_shared_one():
    sr = 16000
    f0 = pitch(np.zeros(sr), sr, 75, 500, 0.01)
    assert np.isnan(f0).all() and f0.size == frame_times(sr, sr, 0.01).size
''')

KINDS = {
    "pose": (POSE_README, POSE_BACKEND, POSE_TEST),
    "asr": (ASR_README, ASR_BACKEND, ASR_TEST),
    "prosody": (PROSODY_README, PROSODY_BACKEND, PROSODY_TEST),
}


def scaffold_backend(name: str, directory: Path, *, kind: str = "pose") -> Path:
    """Write the package into directory/<distribution>; refuses to overwrite."""
    if kind not in KINDS:
        raise MacawError(f"no backend template for '{kind}'; choose one of {', '.join(KINDS)}")
    readme, backend, test = KINDS[kind]
    values = names_for(name)
    values["cls"] = values["cls"].removesuffix("Extractor") + "Backend"
    root = Path(directory) / values["distribution"]
    if root.exists():
        raise MacawError(f"{root} already exists; choose another directory or remove it")

    files = {
        root / "pyproject.toml": PYPROJECT,
        root / "README.md": readme,
        root / "src" / values["module"] / "__init__.py": Template(""),
        root / "src" / values["module"] / "backend.py": backend,
        root / "tests" / "test_backend.py": test,
    }
    for path, template in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.substitute(values), encoding="utf-8")
    return root
