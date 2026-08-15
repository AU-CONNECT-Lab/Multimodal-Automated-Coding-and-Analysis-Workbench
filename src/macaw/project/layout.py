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
        # a directory of per-task files, not a database file; see execution/job_db.py
        return self.output_dir / "jobs"

    @property
    def array_map_path(self) -> Path:
        return self.output_dir / "array_map.tsv"

    # manifest entries and QC warnings arrive from concurrent extractors just like feature
    # rows do, so they are written as fragments and merged by whoever reads them
    def manifest_fragment_path(self, recording_id: str, extractor: str) -> Path:
        return self.recording_dir(recording_id) / "manifest" / f"{extractor}.json"

    def warnings_path(self, recording_id: str, extractor: str) -> Path:
        return self.recording_dir(recording_id) / "warnings" / f"{extractor}.json"

    @property
    def recordings_dir(self) -> Path:
        return self.output_dir / "recordings"

    def recording_dir(self, recording_id: str) -> Path:
        return self.recordings_dir / recording_id

    def offset_table_path(self, recording_id: str) -> Path:
        return self.recording_dir(recording_id) / "offsets.json"

    # the three tables are directories of parquet files, one per writer; store/_io.py
    # explains why a single file cannot survive concurrent extractors
    def long_table_path(self, recording_id: str) -> Path:
        return self.recording_dir(recording_id) / "features"

    def keypoints_path(self, recording_id: str) -> Path:
        return self.recording_dir(recording_id) / "keypoints"

    def body_model_path(self, recording_id: str) -> Path:
        return self.recording_dir(recording_id) / "body_model"

    def annotations_path(self, recording_id: str) -> Path:
        return self.recording_dir(recording_id) / "annotations"

    def artifacts_dir(self, recording_id: str, extractor: str) -> Path:
        return self.recording_dir(recording_id) / "artifacts" / extractor

    def qc_report_path(self, recording_id: str | None = None) -> Path:
        if recording_id is None:
            return self.reports_dir / "qc_project.html"
        return self.reports_dir / f"qc_{recording_id}.html"

    def review_queue_path(self) -> Path:
        return self.output_dir / "review_queue.parquet"


def make_layout(output_dir: Path, *, create: bool = True) -> Layout:
    layout = Layout(Path(output_dir))
    if create:
        for directory in (layout.output_dir, layout.logs_dir, layout.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
    return layout


def new_execution_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
