"""Expected outputs declared in YAML: glob, minimum size, minimum count."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from macaw.exceptions import ConfigValidationError

# what the store layer writes for any extractor, so most extractors need no rule at all
DEFAULT_PATTERN = "features/{extractor}.parquet"


@dataclass
class OutputRule:
    extractor: str
    pattern: str
    min_size_kb: float = 0.0
    min_count: int = 1
    optional: bool = False


@dataclass
class CheckResult:
    task_id: str
    complete: bool
    missing: list[str]
    undersized: list[str]


def load_rules(path: Path) -> list[OutputRule]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigValidationError(f"{path}: expected a mapping of extractor to rule list")

    rules: list[OutputRule] = []
    for extractor, entries in payload.items():
        for entry in entries or []:
            if "pattern" not in entry:
                raise ConfigValidationError(f"{path}: a rule for {extractor} has no pattern")
            unknown = set(entry) - {"pattern", "min_size_kb", "min_count", "optional"}
            if unknown:
                raise ConfigValidationError(
                    f"{path}: rule for {extractor} has unknown keys: {', '.join(sorted(unknown))}"
                )
            rules.append(OutputRule(extractor=extractor, **entry))
    return rules


def rules_for(extractor: str, rules: list[OutputRule]) -> list[OutputRule]:
    declared = [rule for rule in rules if rule.extractor == extractor]
    if declared:
        return declared
    return [OutputRule(extractor=extractor, pattern=DEFAULT_PATTERN.format(extractor=extractor))]


def check_task(
    recording_dir: Path, extractor: str, rules: list[OutputRule], task_id: str
) -> CheckResult:
    missing: list[str] = []
    undersized: list[str] = []

    for rule in rules_for(extractor, rules):
        matches = sorted(Path(recording_dir).glob(rule.pattern))
        # a directory of parquet parts counts by its files, not as one entry
        files = [path for match in matches for path in _files_under(match)]
        if len(files) < rule.min_count:
            if not rule.optional:
                missing.append(rule.pattern)
            continue
        if rule.min_size_kb:
            small = [str(path) for path in files if path.stat().st_size < rule.min_size_kb * 1024]
            undersized.extend(small)

    return CheckResult(
        task_id=task_id,
        complete=not missing and not undersized,
        missing=missing,
        undersized=undersized,
    )


def _files_under(path: Path) -> list[Path]:
    if path.is_dir():
        return [item for item in sorted(path.rglob("*")) if item.is_file()]
    return [path] if path.is_file() else []


def completed_tasks(output_dir: Path, rules: list[OutputRule]) -> set[str]:
    """Drives resume: a task whose outputs already validate is skipped."""
    recordings_root = Path(output_dir) / "recordings"
    if not recordings_root.is_dir():
        return set()

    extractors = sorted({rule.extractor for rule in rules})
    done: set[str] = set()
    for recording_dir in sorted(recordings_root.iterdir()):
        if not recording_dir.is_dir():
            continue
        # without declared rules the written parquet files are the only evidence available
        candidates = extractors or [
            path.stem for path in sorted((recording_dir / "features").glob("*.parquet"))
        ]
        for extractor in candidates:
            task_id = f"{recording_dir.name}:{extractor}"
            if check_task(recording_dir, extractor, rules, task_id).complete:
                done.add(task_id)
    return done
