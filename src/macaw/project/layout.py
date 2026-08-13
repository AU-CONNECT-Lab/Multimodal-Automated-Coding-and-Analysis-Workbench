"""Output directory conventions. Every path in the package comes from here."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    output_dir: Path

    @property
    def cache_dir(self) -> Path: ...

    @property
    def logs_dir(self) -> Path: ...

    @property
    def reports_dir(self) -> Path: ...

    @property
    def manifest_path(self) -> Path: ...

    @property
    def resolved_config_path(self) -> Path: ...

    @property
    def job_db_path(self) -> Path: ...

    def session_dir(self, session_id: str) -> Path: ...

    def long_table_path(self, session_id: str) -> Path: ...

    def keypoints_path(self, session_id: str) -> Path: ...

    def annotations_path(self, session_id: str) -> Path: ...

    def artifacts_dir(self, session_id: str, extractor: str) -> Path: ...

    def qc_report_path(self, session_id: str | None = None) -> Path: ...

    def review_queue_path(self) -> Path: ...


def make_layout(output_dir: Path, *, create: bool = True) -> Layout:
    raise NotImplementedError


def new_run_id() -> str:
    raise NotImplementedError
