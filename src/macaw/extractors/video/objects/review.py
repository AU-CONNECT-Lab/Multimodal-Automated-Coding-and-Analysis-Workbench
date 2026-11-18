"""The check a person makes before the outlines are used: that each thing found by name is the thing meant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from macaw.extractors import register_extractor
from macaw.extractors.base import PERSON as BY_PERSON
from macaw.extractors.base import SOFTWARE, Gate, QCWarning
from macaw.extractors.video.objects.decisions import ABSENT, KEPT, STARTS, read_decisions
from macaw.extractors.video.objects.track import FINDER, NOT_FOUND, NOT_HERE, PERSON, ObjectTrackExtractor, load_starts
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.store.schema import Scale


@dataclass(frozen=True)
class Waiting:
    camera: str
    name: str
    why: str


def start_of(entry: dict) -> dict:
    """What a person keeps when they keep a start found by name: where and on which frame."""
    return {"frame": entry.get("frame"), "box": entry.get("box")}


def waiting(config, recording_id: str, review: str | None = None) -> list[Waiting]:
    """The starts a person still has to look at; with review always every one found by name, with flagged those in doubt."""
    review = review or config.objects.review
    if review == "skip":
        return []
    said = read_decisions(layout_for(config, create=False).objects_path(recording_id))
    found = []
    for camera, entries in load_starts(config, recording_id).items():
        for name, entry in entries.items():
            given, absent = name in said[STARTS].get(camera, {}), name in said[ABSENT].get(camera, {})
            if (given and entry["from"] != PERSON) or (absent and entry["from"] != NOT_HERE):
                found.append(Waiting(camera, name, "followed again from what a person said on the next run"))
            elif entry["from"] == NOT_FOUND:
                found.append(Waiting(camera, name, "not found by name: click it on a frame, or say it is not on this camera"))
            elif entry["from"] == FINDER:
                kept = said[KEPT].get(camera, {}).get(name)
                if kept and kept.get("start") == start_of(entry):
                    continue
                if review == "always" or entry.get("doubts"):
                    found.append(Waiting(camera, name, "; ".join(entry.get("doubts") or ["found by name, not looked at yet"])))
    return found


def unlooked(config, recording_id: str) -> list[tuple[str, str, dict]]:
    """Starts found by name that stand without a person having looked: (camera, name, entry)."""
    said = read_decisions(layout_for(config, create=False).objects_path(recording_id))
    return [
        (camera, name, entry)
        for camera, entries in load_starts(config, recording_id).items()
        for name, entry in entries.items()
        if entry["from"] in (FINDER, NOT_FOUND) and said[KEPT].get(camera, {}).get(name, {}).get("start") != start_of(entry)
    ]


@register_extractor
class ObjectReviewGate(Gate):
    """Held until a person has looked at where each thing found by name starts (review always), or at those in doubt (flagged)."""

    name = "objects.review"
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "objects"
    config_sections = ("objects.names", "objects.review")
    requires = (ObjectTrackExtractor.name,)

    @classmethod
    def wanted_by_default(cls, config) -> bool:
        return bool(config.objects.names)

    def inputs_for(self, recording: Recording) -> list[Path]:
        return []

    def pending(self, recording: Recording) -> str | None:
        found = waiting(self.config, recording.recording_id)
        if not found:
            return None
        rid = recording.recording_id
        listed = "; ".join(f"{item.name} on {item.camera}: {item.why}" for item in found)
        return (
            f"where each thing starts waits for a person (objects.review is {self.config.objects.review}): {listed}. Look on the "
            f"Review page, or list them with `macaw objects -r {rid}`; `--keep` keeps the starts found by name, `--start NAME` "
            "with --camera, --at and --point gives one, `--not-here NAME` says a thing is not on a camera"
        )

    def decided_by(self, recording: Recording) -> str | None:
        if not self.config.objects.names:
            return None
        said = read_decisions(layout_for(self.config, create=False).objects_path(recording.recording_id))
        if any(said[key] for key in (STARTS, ABSENT, KEPT)):
            return BY_PERSON
        return SOFTWARE

    def notes(self, recording: Recording) -> list[QCWarning]:
        left = [(camera, name, entry) for camera, name, entry in unlooked(self.config, recording.recording_id) if entry.get("doubts")]
        if not left:
            return []
        lost = [f"{name} on {camera}" for camera, name, entry in left if entry["from"] == NOT_FOUND]
        doubted = [f"{name} on {camera} ({', '.join(entry['doubts'])})" for camera, name, entry in left if entry["from"] == FINDER]
        said = ([f"things in doubt were followed from where their names found them, without a person looking: {'; '.join(doubted)}"] if doubted else [])
        said += [f"not found by name and so not followed: {', '.join(lost)}"] if lost else []
        return [
            QCWarning(
                check="objects_unreviewed", recording_id=recording.recording_id, subject_id=None, severity="warn",
                message=f"objects.review is {self.config.objects.review}; " + "; ".join(said),
            )
        ]
