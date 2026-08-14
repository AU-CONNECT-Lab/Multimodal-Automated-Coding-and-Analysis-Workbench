"""Output directory conventions. Every path in the package comes from here."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RESOLVED_FILENAME = "resolved_config.yaml"


@dataclass(frozen=True)
class Layout:
    output_dir: Path

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / ".cache"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.output_dir / RESOLVED_FILENAME

    @property
    def job_db_path(self) -> Path:
        return self.output_dir / "jobs.sqlite"

    def session_dir(self, session_id: str) -> Path:
        return self.output_dir / "sessions" / session_id

    def long_table_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "features.parquet"

    def keypoints_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "keypoints.parquet"

    def annotations_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "annotations.parquet"

    def artifacts_dir(self, session_id: str, extractor: str) -> Path:
        return self.session_dir(session_id) / "artifacts" / extractor

    def qc_report_path(self, session_id: str | None = None) -> Path:
        if session_id is None:
            return self.reports_dir / "qc_project.html"
        return self.reports_dir / f"qc_{session_id}.html"

    def review_queue_path(self) -> Path:
        return self.output_dir / "review_queue.parquet"


def make_layout(output_dir: Path, *, create: bool = True) -> Layout:
    layout = Layout(Path(output_dir))
    if create:
        for directory in (layout.output_dir, layout.logs_dir, layout.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
    return layout


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
