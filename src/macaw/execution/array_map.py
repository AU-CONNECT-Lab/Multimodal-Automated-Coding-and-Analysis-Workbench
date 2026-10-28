"""Stable recording to array-index mapping.

aftercorr pairs element N of one array with element N of the next, so the mapping must be
identical across every array in a run. Cached recordings still occupy their index and run a
task that exits immediately; indices are never compacted."""

from __future__ import annotations

from pathlib import Path

from macaw.exceptions import MacawError

HEADER = ("index", "recording_id")


def build_map(recording_ids: list[str]) -> dict[str, int]:
    ordered = sorted(dict.fromkeys(recording_ids))
    return {recording_id: index for index, recording_id in enumerate(ordered)}


def write_map(mapping: dict[str, int], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(HEADER)]
    lines += [
        f"{index}\t{recording_id}"
        for recording_id, index in sorted(mapping.items(), key=lambda item: item[1])
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_map(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith(HEADER[0]):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise MacawError(f"{path}:{number}: expected 'index<TAB>recording_id', got {line!r}")
        mapping[parts[1].strip()] = int(parts[0])
    return mapping


def recording_for_index(mapping: dict[str, int], index: int) -> str:
    for recording_id, value in mapping.items():
        if value == index:
            return recording_id
    raise MacawError(f"Array index {index} is not in the map; it was built for a different run")


def array_spec(mapping: dict[str, int], *, max_concurrent: int | None = None) -> str:
    """Renders 0-30 or 0-30%4 for the --array argument."""
    if not mapping:
        raise MacawError("Cannot build an array spec for zero recordings")
    spec = f"0-{max(mapping.values())}"
    return f"{spec}%{max_concurrent}" if max_concurrent else spec
