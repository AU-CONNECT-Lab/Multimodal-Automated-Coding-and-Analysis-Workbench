"""The one abstraction that matters: everything that produces features is an Extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from macaw.config.schema import MacawConfig
from macaw.exceptions import GateNotClearedError
from macaw.project.manifest import FeatureSpec
from macaw.project.recording import Recording
from macaw.store.schema import Scale


class Phase(str, Enum):
    PREP = "prep"                # media probing, segmentation, track extraction
    INTERACTIVE = "interactive"  # needs a human; never goes into a job array
    EXTRACT = "extract"          # the expensive per-recording work
    DERIVE = "derive"            # cheap quantities computed from extract output
    QC = "qc"
    AGGREGATE = "aggregate"      # cross-recording; one job after every recording finishes


@dataclass
class QCWarning:
    check: str
    recording_id: str
    subject_id: str | None
    severity: str                      # info | warn | fail
    message: str
    t_start: float | None = None
    t_end: float | None = None
    details: dict = field(default_factory=dict)


@dataclass
class ExtractorResult:
    rows: pd.DataFrame                              # LONG_TABLE_SCHEMA
    specs: list[FeatureSpec] = field(default_factory=list)
    keypoints: pd.DataFrame | None = None           # KEYPOINT_SCHEMA, pose backends only
    body_model: pd.DataFrame | None = None          # BODY_MODEL_SCHEMA, 3D backends only
    annotations: pd.DataFrame | None = None         # ANNOTATION_SCHEMA, coding only
    artifacts: dict[str, Path] = field(default_factory=dict)
    warnings: list[QCWarning] = field(default_factory=list)


class Extractor(ABC):
    name: str
    version: str
    scale: Scale
    group: str
    phase: Phase = Phase.EXTRACT
    config_sections: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()                  # other extractor names -> DAG edges
    runs_after: tuple[str, ...] = ()                # ordered after these when they run too, never pulled in by it
    resource_profile: str = "standard"              # tier name defined in hpc.yaml
    cached: bool = True                             # whether the result cache keeps a copy to replay

    def __init__(self, config: MacawConfig):
        self.config = config

    @classmethod
    def dependencies(cls, config: MacawConfig) -> tuple[str, ...]:
        """requires, plus any step this project's settings put in front of this one."""
        return cls.requires

    @classmethod
    def wanted_by_default(cls, config: MacawConfig) -> bool:
        """Whether a run that names no steps includes this one."""
        return True

    @classmethod
    def resources(cls, config: MacawConfig) -> str:
        """The hpc.yaml tier a cluster job running this step asks for."""
        return cls.resource_profile

    def prefetch(self) -> None:
        """Download whatever model this step would otherwise fetch while it runs."""
        return None

    @abstractmethod
    def check_available(self) -> None:
        """Raise BackendNotAvailableError with install instructions if deps are missing."""

    @abstractmethod
    def inputs_for(self, recording: Recording) -> list[Path]:
        """Files whose hashes go into the cache key."""

    @abstractmethod
    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        ...

    def backend_info(self) -> tuple[str | None, str | None]:
        """(backend name, backend version) recorded in the manifest for reproducibility."""
        return (None, None)


class Gate(Extractor):
    """A step a person clears by a recorded decision. The runner checks it in-process and never schedules it."""

    phase = Phase.INTERACTIVE

    @abstractmethod
    def pending(self, recording: Recording) -> str | None:
        """None once cleared; otherwise what the person has to do."""

    def notes(self, recording: Recording) -> list[QCWarning]:
        """Warnings a cleared gate leaves behind, e.g. that a person was skipped."""
        return []

    def clear_by_itself(self, recording: Recording) -> None:
        """Record the software's own decision where the configuration lets it stand. Called by a run before pending."""
        return None

    def check_available(self) -> None:
        return None

    def run(self, recording: Recording, workspace: Path) -> ExtractorResult:
        reason = self.pending(recording)
        if reason is not None:
            raise GateNotClearedError(f"{recording.recording_id}: {reason}")
        return ExtractorResult(rows=pd.DataFrame(), warnings=self.notes(recording))


class Backend(ABC):
    """A swappable implementation behind one Extractor."""

    name: str
    version: str

    @abstractmethod
    def check_available(self) -> None: ...
