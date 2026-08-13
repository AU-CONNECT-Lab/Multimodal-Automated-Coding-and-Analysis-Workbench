"""The one abstraction that matters: everything that produces features is an Extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from macaw.config.schema import MacawConfig
from macaw.project.manifest import FeatureSpec
from macaw.project.session import Session
from macaw.store.schema import Scale


class Phase(str, Enum):
    PREP = "prep"                # media probing, segmentation, track extraction
    INTERACTIVE = "interactive"  # needs a human; never goes into a job array
    EXTRACT = "extract"          # the expensive per-session work
    DERIVE = "derive"            # cheap quantities computed from extract output
    QC = "qc"
    AGGREGATE = "aggregate"      # cross-session; one job after every session finishes


@dataclass
class QCWarning:
    check: str
    session_id: str
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

    def __init__(self, config: MacawConfig):
        self.config = config

    @abstractmethod
    def check_available(self) -> None:
        """Raise BackendNotAvailableError with install instructions if deps are missing."""

    @abstractmethod
    def inputs_for(self, session: Session) -> list[Path]:
        """Files whose hashes go into the cache key."""

    @abstractmethod
    def run(self, session: Session, workspace: Path) -> ExtractorResult:
        ...

    def backend_info(self) -> tuple[str | None, str | None]:
        """(backend name, backend version) recorded in the manifest for reproducibility."""
        return (None, None)


class Backend(ABC):
    """A swappable implementation behind one Extractor."""

    name: str
    version: str

    @abstractmethod
    def check_available(self) -> None: ...
