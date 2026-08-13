from __future__ import annotations

from pathlib import Path

from macaw.extractors.base import Extractor, ExtractorResult
from macaw.project.session import Session
from macaw.store.schema import Scale


class PoseExtractor(Extractor):
    name = "pose.keypoints"
    version = "0.1.0"
    scale = Scale.FRAME
    group = "pose"
    config_sections = ("pose",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, session: Session) -> list[Path]:
        raise NotImplementedError

    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        """Writes keypoints to the keypoint table; only track-quality metrics go to rows."""
        raise NotImplementedError
