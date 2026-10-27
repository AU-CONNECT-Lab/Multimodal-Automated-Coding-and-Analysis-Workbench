"""Pictures, clips and report pages from the project folders, served to the page by their own path
so the links inside a QC report still resolve."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote, unquote

from macaw.config.schema import MacawConfig
from macaw.project.layout import layout_for

PREFIX = "/files/"


def allowed_roots(config: MacawConfig) -> list[Path]:
    layout = layout_for(config, create=False)
    return [root.resolve() for root in (layout.output_dir, layout.work_dir, Path(config.project.raw_dir))]


def file_url(path: Path) -> str:
    return PREFIX + quote(Path(path).resolve().as_posix().lstrip("/"))


def path_from_url(rest: str, config: MacawConfig) -> Path | None:
    """The file a /files/ address names, or None when it lies outside the project's folders."""
    text = unquote(rest)
    path = Path(text if re.match(r"^[A-Za-z]:/", text) else "/" + text).resolve()
    if not any(path == root or root in path.parents for root in allowed_roots(config)):
        return None
    return path if path.is_file() else None
