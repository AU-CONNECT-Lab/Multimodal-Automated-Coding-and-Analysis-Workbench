"""Picture files read and written through OpenCV's codecs, with Python doing the file access."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from macaw.exceptions import BackendNotAvailableError, ExternalToolError

# OpenCV's own imread and imwrite fail without a word on Windows paths with letters such as 项目


def _cv2():
    try:
        import cv2
    except ImportError:
        raise BackendNotAvailableError('reading and writing pictures needs OpenCV: pip install "pymacaw[vision]"') from None
    return cv2


def read_image(path: Path, flags: int | None = None) -> np.ndarray | None:
    """The picture in the file as OpenCV reads it (flags as for cv2.imread); None when it cannot be decoded."""
    cv2 = _cv2()
    raw = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR if flags is None else flags) if raw.size else None


def write_image(path: Path, image: np.ndarray, params: list[int] | None = None) -> None:
    ok, encoded = _cv2().imencode(Path(path).suffix, image, params or [])
    if not ok:
        raise ExternalToolError(f"could not encode the picture for {Path(path).name}")
    Path(path).write_bytes(encoded.tobytes())
