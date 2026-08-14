"""Everything downstream depends on this: probe, trim to the task window, split audio tracks."""

from __future__ import annotations

from pathlib import Path

from macaw.extractors.base import Extractor, ExtractorResult, Phase
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class MediaPrepExtractor(Extractor):
    name = "prep.media"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.PREP
    config_sections = ("segment", "align")

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Reads the offset table produced by `macaw align`; it does not estimate offsets."""
        raise NotImplementedError


class AlignReviewGate(Extractor):
    """A human confirms low-confidence offsets. Present so the DAG knows the edge exists."""

    name = "align.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.INTERACTIVE
    config_sections = ("align",)

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        """Fails with a clear message when run non-interactively and confirmation is missing."""
        raise NotImplementedError


class IdentityConfirmGate(Extractor):
    """Three-source vote is automatic; accepting it is not."""

    name = "identity.confirm"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "prep"
    phase = Phase.INTERACTIVE
    config_sections = ("text",)
    requires = ("text.asr", "prosody.frame")

    def check_available(self) -> None:
        raise NotImplementedError

    def inputs_for(self, recording: Recording) -> list[Path]:
        raise NotImplementedError

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        raise NotImplementedError
