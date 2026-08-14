from __future__ import annotations

from pathlib import Path

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class PoseExtractor(Extractor):
    name = "pose.keypoints"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Writes keypoints to the keypoint table; only track-quality metrics go to rows."""
        raise NotImplementedError
