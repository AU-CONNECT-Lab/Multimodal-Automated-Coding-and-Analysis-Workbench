"""HTML pages to read a run top to bottom, one per recording and one for the project. Pictures
stay files the page links to, so a long recording does not make a heavy page."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.config.schema import MacawConfig
from macaw.extractors.base import Gate, QCWarning
from macaw.media.align import describe_drift, drift_flagged, read_offset_table, review_reason
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.exceptions import BackendNotAvailableError
from macaw.qc.checks import RecordingResults, read_candidates
from macaw.extractors.audio.content.asr import DROPPED_NAME, DROPPED_SCHEMA, TRANSCRIPT_NAME, read_transcript
from macaw.project.identity import read_confirmation
from macaw.store import _io
from macaw.qc.downsample import min_max_downsample
from macaw.qc.figures import (
    OVERVIEW_ROW_S,
    motion_figure,
    overview_figure,
    pitch_figure,
    present_spans,
    require_matplotlib,
    timeline_figure,
)
from macaw.store import annotations as annotations_store
from macaw.store.schema import Scale

TEMPLATES = Path(__file__).parent / "templates"
SEVERITY_ORDER = {"fail": 0, "warn": 1, "info": 2}


def _environment():
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    return Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html", "j2"]))


def _href(target: Path, page: Path) -> str:
    """Relative where it can be, so the folder can move; a file URI across Windows drives."""
    try:
        return Path(os.path.relpath(target, page.parent)).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def _number(value, digits: int = 3):
    if value is None or pd.isna(value):
        return None
    return f"{float(value):.{digits}g}" if abs(float(value)) < 1e4 else f"{float(value):.0f}"


def _waiting(config: MacawConfig, recording) -> list[tuple[str, str]]:
    from macaw.extractors import get_extractor, list_extractors

    waiting = []
    for name in list_extractors():
        cls = get_extractor(name)
        if not (isinstance(cls, type) and issubclass(cls, Gate)):
            continue
        try:
            message = cls(config).pending(recording)
        except Exception as error:   # a gate that cannot read its inputs still says so on the page
            message = f"could not be checked: {error}"
        if message:
            waiting.append((name, message))
    return waiting


def _media(config: MacawConfig, recording) -> list[dict]:
    from macaw.extractors.prep import apply_prep_timing

    try:
        apply_prep_timing(config, recording)
    except Exception:
        pass   # before prep.media has run the files are still worth listing
    rows = []
    for item in sorted(recording.media, key=lambda media: media.path.name):
        detail = []
        if item.kind == "video" and item.fps:
            detail.append(f"{item.fps:g} fps")
        if item.duration_s:
            detail.append(f"{item.duration_s:.0f} s")
        if item.offset_s:
            detail.append(f"starts at {item.offset_s:+.3f} s")
        rows.append(
            {"name": item.path.name, "owner": item.subject_id or f"shared {item.kind}", "detail": ", ".join(detail)}
        )
    return rows


def _alignment(config: MacawConfig, layout, recording_id: str, page: Path) -> tuple[list[dict], list[dict]]:
    path = layout.offset_table_path(recording_id)
    if not path.is_file():
        return [], []
    rows = []
    for result in read_offset_table(path):
        if result.target == result.reference:
            continue
        rows.append(
            {
                "target": result.target,
                "reference": result.reference,
                "offset": f"{result.offset_s:+.3f}" if result.offset_s is not None else "-",
                "confidence": _number(result.confidence, 2) or "-",
                "agreement": (
                    f"{round(result.agreement * len(result.segment_offsets or ()))} of {len(result.segment_offsets or ())}"
                    if result.agreement is not None
                    else "-"
                ),
                "drift": describe_drift(result) if result.drift_ppm is not None and drift_flagged(result) else "",
                "method": result.method,
                "flag": review_reason(result, config.align) if not result.accepted else None,
                "accepted": result.accepted,
            }
        )
    figures = sorted((layout.renders_dir(recording_id) / "align").glob("*.png"))
    return rows, [{"name": figure.name, "href": _href(figure, page)} for figure in figures]


def _cameras(layout, results: RecordingResults, recording_id: str) -> list[dict]:
    from macaw.extractors.video.pose.keypoints import PoseExtractor, read_owner_file

    owners_path = results.artifact(PoseExtractor.name, "owners")
    if owners_path is None:
        return []
    owners = read_owner_file(owners_path)
    accepted = read_owner_file(layout.camera_owners_accepted_path(recording_id))
    cameras = []
    for file_name, owner in sorted(owners.items()):
        kept = accepted.get(file_name, {})
        cameras.append(
            {
                "file": file_name,
                "people": [
                    {
                        "subject": subject,
                        "tracks": ", ".join(str(track) for track in tracks[:10])
                        + (f" and {len(tracks) - 10} more" if len(tracks) > 10 else "")
                        or "none",
                        "all_tracks": ", ".join(str(track) for track in tracks),
                        "chosen_by": owner["chosen_by"].get(subject, ""),
                    }
                    for subject, tracks in owner["people"].items()
                ],
                "doubt": owner["doubt"],
                "accepted": kept.get("by") if kept.get("people") == owner["people"] else None,
            }
        )
    return cameras


def _voices(config: MacawConfig, layout, recording, page: Path) -> list[dict]:
    """Each voice found on a shared microphone, what it says and its pitch, whom the suggestion and
    the person gave it to, and its clips to listen to."""
    from macaw.extractors.audio.content.diarization import NO_ONE, OTHER, DiarizationExtractor, others_take_turns, read_voice_owners, read_voices
    from macaw.extractors.audio.content.speaker_id import read_voice_material, suggest_voices, suggested_people, telling_lines

    recording_id = recording.recording_id
    voices = read_voices(config, recording_id)
    if not voices:
        return []
    named = read_voice_owners(layout.voice_owners_path(recording_id))
    lines, pitch = read_voice_material(config, recording_id)
    try:
        suggestion = suggest_voices(config, recording)
    except Exception:   # the page still shows the voices when a suggestion cannot be made
        suggestion = None
    suggested = suggested_people(suggestion) if suggestion else {}
    directory = layout.artifacts_dir(recording_id, DiarizationExtractor.name)
    left_out = f"{OTHER} (left out of the people's measures)" if others_take_turns(config) else "left out"
    rows = []
    for label, voice in sorted(voices.items()):
        who = named.get("people", {}).get(label)
        own = lines[lines["subject_id"].astype(str) == label].sort_values("t_start")
        median = pitch.get(label, {}).get("median_hz")
        rows.append(
            {
                "label": label,
                "speech": f"{voice.get('speech_s', 0.0):.0f} s",
                "pitch": "" if median is None else f"{median:.0f} Hz",
                "says": telling_lines(lines, label),
                "lines": [{"at": f"{row.t_start:.1f}", "text": row.text} for row in own.itertuples()],
                "suggested": suggested.get(label, ""),
                "person": None if who is None else (left_out if who == NO_ONE else who),
                "clips": [
                    {"href": _href(directory / clip["file"], page), "at": f"{clip['t_start']:.0f} s"}
                    for clip in voice.get("clips", [])
                ],
            }
        )
    return rows


def _dropped(results: RecordingResults) -> list[dict]:
    path = results.artifact("text.asr", DROPPED_NAME)
    if path is None:
        return []
    dropped = _io.read_file(path, DROPPED_SCHEMA, "dropped utterances").sort_values("t_start")
    return [
        {
            "who": row.subject_id, "when": f"{row.t_start:.1f}", "length": f"{row.t_end - row.t_start:.1f}",
            "text": row.text if isinstance(row.text, str) else "", "reason": row.reason,
            "detail": row.detail if isinstance(row.detail, str) else "",
        }
        for row in dropped.itertuples(index=False)
    ]


def _who_and_when(stamp: str | None) -> str:
    """'ann 2026-10-26T10:00:00' as 'ann, 2026-10-26'."""
    if not stamp:
        return ""
    who, _, when = stamp.partition(" ")
    return f"{who}, {when[:10]}" if when else who


def _decisions(
    config: MacawConfig, layout, results: RecordingResults, recording_id: str,
    alignment: list[dict], cameras: list[dict], voices: list[dict], dropped: list[dict], frames: list[dict],
) -> list[dict]:
    """What the software decided by itself for this recording, beside what a person decided, with where to check it."""
    from macaw.extractors.audio.content.diarization import AUTOMATIC, read_voice_owners

    rows = []
    confirmation = read_confirmation(layout.identity_path)
    if config.identity.confirm == "skip":
        rows.append({"what": "Which file is whose", "decided": "taken from the file names, not confirmed (identity.confirm: skip)", "by": "software", "href": None})
    elif confirmation is not None:
        stamp = f"{confirmation.get('confirmed_by', '')} {confirmation.get('confirmed_at', '')}".strip()
        rows.append({"what": "Which file is whose", "decided": "confirmed for the project", "by": _who_and_when(stamp), "href": None})
    else:
        rows.append({"what": "Which file is whose", "decided": "not confirmed yet", "by": "waiting", "href": None})

    if alignment:
        kept = sum(1 for row in alignment if row["accepted"])
        manual = sum(1 for row in alignment if row["method"] == "manual")
        automatic = len(alignment) - kept - manual
        rows.append({
            "what": "Offsets between files",
            "decided": f"{len(alignment)} file(s): {kept} kept by a person, {manual} set by hand, {automatic} automatic",
            "by": "person" if not automatic else "software", "href": "#Alignment",
        })
    video = _prepared_video(config, recording_id)
    if video is not None:
        rows.append(video)
    if cameras:
        kept = sum(1 for camera in cameras if camera["accepted"])
        doubt = sum(1 for camera in cameras if camera["doubt"] and not camera["accepted"])
        rows.append({
            "what": "Who is on each camera",
            "decided": f"{len(cameras)} camera(s): {kept} kept by a person, {len(cameras) - kept} automatic" + (f", {doubt} in doubt" if doubt else ""),
            "by": "person" if kept == len(cameras) else "software", "href": "#Camera",
        })
    swaps = results.artifact("pose.keypoints", "swaps.json")
    if swaps is not None:
        count = sum(len(events) for events in json.loads(swaps.read_text(encoding="utf-8")).values())
        rows.append({
            "what": "Track swaps repaired",
            "decided": f"{count} swap(s) between two people's tracks undone",
            "by": "software", "href": "#Frames" if frames else None,
        })
    if voices:
        by = read_voice_owners(layout.voice_owners_path(recording_id)).get("by")
        rows.append({
            "what": "Whose voice is which",
            "decided": f"{len(voices)} voice(s) on the shared microphone" + ("" if by else ", not named yet"),
            "by": "waiting" if not by else "software" if by == AUTOMATIC else _who_and_when(by), "href": "#Voices",
        })
    stamp_path = layout.corrected_transcript_stamp_path(recording_id)
    if stamp_path.is_file():
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        rows.append({
            "what": "Transcript", "decided": f"corrected by hand from {stamp.get('from', 'an ELAN file')}",
            "by": _who_and_when(f"{stamp.get('by', '')} {stamp.get('at', '')}".strip()), "href": None,
        })
    elif results.artifact("text.asr", TRANSCRIPT_NAME) is not None:
        rows.append({
            "what": "Transcript", "decided": f"as recognised; to correct it, macaw transcript export -r {recording_id}",
            "by": "software", "href": None,
        })
    if dropped:
        reasons = Counter(row["reason"] for row in dropped)
        rows.append({
            "what": "Speech left out",
            "decided": ", ".join(f"{count} as {reason}" for reason, count in reasons.items()),
            "by": "software", "href": "#Dropped",
        })
    coded = _model_coding(layout, recording_id)
    if coded is not None:
        rows.append(coded)
    return rows


def _prepared_video(config: MacawConfig, recording_id: str) -> dict | None:
    from macaw.extractors.prep import load_video_prep

    videos = list(load_video_prep(config, recording_id).values())
    if not videos:
        return None
    made = [video for video in videos if video["file"]]
    if not made:
        return {"what": "Video", "decided": f"{len(videos)} file(s) read as recorded (video.normalise: false)", "by": "software", "href": "#Warnings"}
    rates = ", ".join(sorted({f"{video['fps']:g}" for video in made}, key=float))
    variable = sum(video["variable_rate"] for video in made)
    repeated = sum(video["frames_repeated"] or 0 for video in made)
    dropped = sum(video["frames_dropped"] or 0 for video in made)
    toned = sum(video.get("tone_mapped", False) for video in made)
    return {
        "what": "Video",
        "decided": f"{len(made)} file(s) re-encoded at a constant {rates} fps, upright, with square pixels and frame 0 at their first sound"
        + (f"; {variable} had a variable frame rate" if variable else "")
        + (f"; {toned} tone-mapped from high dynamic range" if toned else "")
        + f"; {repeated} frame(s) repeated and {dropped} left out",
        "by": "software", "href": "#Warnings",
    }


def _frame_rates(config: MacawConfig, recording_ids) -> list[dict]:
    """Each frame rate the project's videos are read at, with how many videos and recordings; empty when there is only one."""
    from macaw.extractors.prep import load_video_prep

    videos: Counter = Counter()
    recordings: dict[float, set[str]] = {}
    for recording_id in recording_ids:
        for video in load_video_prep(config, recording_id).values():
            rate = round(float(video["fps"]), 3)
            videos[rate] += 1
            recordings.setdefault(rate, set()).add(recording_id)
    if len(videos) < 2:
        return []
    return [{"fps": f"{rate:g}", "videos": videos[rate], "recordings": len(recordings[rate])} for rate in sorted(videos)]


def _common_rate(rates: list[float]) -> float | None:
    """The lowest rate when every other one is a whole multiple of it, so video.fps can bring them all to it."""
    lowest = min(rates)
    whole = all(abs(rate / lowest - round(rate / lowest)) <= 1e-3 * round(rate / lowest) for rate in rates)
    return lowest if whole else None


def _model_coding(layout, recording_id: str) -> dict | None:
    from macaw.coding.llm.judge import LAYER
    from macaw.coding.review_queue import REVIEWED_NAME
    from macaw.store.schema import ANNOTATION_SCHEMA

    directory = layout.annotations_path(recording_id)
    machine = directory / f"{LAYER}.parquet"
    if not machine.is_file():
        return None
    units = _io.read_file(machine, ANNOTATION_SCHEMA, "annotations")
    reviewed = directory / REVIEWED_NAME
    n_reviewed = len(_io.read_file(reviewed, ANNOTATION_SCHEMA, "annotations")) if reviewed.is_file() else 0
    flagged = int(units["needs_review"].fillna(False).sum())
    return {
        "what": "Speech acts",
        "decided": f"{len(units)} unit(s) coded by a model, {flagged} sent for review, {n_reviewed} decided by a person; "
        f"macaw review --export queue.csv -r {recording_id}",
        "by": "person" if flagged and n_reviewed >= flagged else "software",
        "href": None,
    }


def _feature_notes(layout, recording_id: str) -> tuple[dict[str, dict], list[dict]]:
    """Descriptions and units by feature name, and one version row per step, from the manifest."""
    notes: dict[str, dict] = {}
    versions: dict[str, dict] = {}
    for path in sorted(layout.manifest_fragment_path(recording_id, "x").parent.glob("*.json")):
        for spec in json.loads(path.read_text(encoding="utf-8")).get("features", []):
            notes[spec["name"]] = spec
            versions[spec["extractor"]] = {
                key: spec.get(key) for key in ("extractor", "extractor_version", "backend", "backend_version")
            }
    return notes, list(versions.values())


def _measures(layout, results: RecordingResults, recording_id: str, notes: dict) -> list[dict]:
    tables = []
    for path in sorted(layout.long_table_path(recording_id).glob("*.parquet")):
        rows = results.long_table(path.stem)
        rows = rows[rows["scale"].astype(str) == Scale.RECORDING.value]
        if rows.empty:
            continue
        who = rows["subject_id"].astype(object).where(rows["subject_id"].notna(), "everyone")
        pair = rows["subject_id_b"].notna()
        who[pair] = rows.loc[pair, "subject_id"].astype(str) + " & " + rows.loc[pair, "subject_id_b"].astype(str)
        values = rows.assign(who=who).pivot_table(index="feature", columns="who", values="value", aggfunc="first")
        columns = [column for column in values.columns if column != "everyone"]
        columns += ["everyone"] if "everyone" in values.columns else []
        tables.append(
            {
                "source": path.stem,
                "columns": columns,
                "features": [
                    {
                        "name": feature,
                        "description": notes.get(feature, {}).get("description", ""),
                        "unit": notes.get(feature, {}).get("unit"),
                        "cells": [_number(values.at[feature, column]) for column in columns],
                    }
                    for feature in values.index
                ],
            }
        )
    return tables


def _frames(layout, recording_id: str, page: Path) -> list[dict]:
    directory = layout.renders_dir(recording_id)
    index = directory / "index.csv"
    if not index.is_file():
        return []
    table = pd.read_csv(index)
    groups = []
    for reason, rows in table.groupby("reason", sort=False):
        items = []
        for row in rows.itertuples():
            who = row.subject_id if isinstance(row.subject_id, str) else "anyone"
            items.append(
                {
                    "href": _href(directory / row.file, page),
                    "caption": f"{row.t:.1f} s, {who}, {row.video}",
                }
            )
        groups.append({"reason": str(reason).replace("_", " "), "shots": items})
    # the baseline is the sample to compare against, so it reads last
    return sorted(groups, key=lambda group: group["reason"] == "baseline")


def _over_time(config, layout, results: RecordingResults, recording_id: str, subjects: list[str], page: Path) -> list[dict]:
    """Drawn when the page is written, from what the steps left; skipped without matplotlib."""
    try:
        require_matplotlib()
    except BackendNotAvailableError:
        return []
    keypoints = results.keypoints()
    keypoints = keypoints[keypoints["subject_id"].notna() & keypoints["x"].notna()]
    seen = {str(who): present_spans(group["t"].to_numpy(dtype=float)) for who, group in keypoints.groupby("subject_id")}
    frames = results.long_table("prosody.frame", features=["f0_hz"])
    frames = frames[frames["subject_id"].notna()]
    pitch_rows = frames[(frames["feature"] == "f0_hz") & frames["value"].notna()]
    pitch = {
        str(who): (group["t_start"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float))
        for who, group in pitch_rows.groupby("subject_id")
    }
    # what the recogniser kept is speech with crosstalk taken out; pitch alone also catches a neighbour
    transcript = results.artifact("text.asr", TRANSCRIPT_NAME)
    if transcript is not None:
        utterances = read_transcript(transcript)
        heard = {
            str(who): list(zip(group["t_start"].astype(float), (group["t_end"] - group["t_start"]).astype(float)))
            for who, group in utterances.groupby("subject_id")
        }
        heard_label = "speaking (transcript)"
    else:
        heard = {who: present_spans(t) for who, (t, _) in pitch.items()}
        heard_label = "pitch found on their mic"
    candidates_path = layout.qc_dir(recording_id) / "candidates.parquet"
    candidates = read_candidates(candidates_path) if candidates_path.is_file() else pd.DataFrame(
        columns=["subject_id", "t_start", "t_end", "reason"]
    )
    directory = layout.renders_dir(recording_id) / "report"
    figures = []
    path = _overview(config, layout, results, recording_id, subjects, candidates, directory / "overview.png")
    if path is not None:
        figures.append({"name": "Overview", "hint": "each person's talk, the measures in qc.overview_features (by default movement and distance) and the codes, on one clock; shaded where a check flagged", "href": _href(path, page)})
    if seen or heard:
        path = timeline_figure(subjects, seen, heard, heard_label, candidates, directory / "timeline.png")
        figures.append({"name": "Seen and heard", "hint": f"when a camera saw each person and when they were heard ({heard_label}); shaded spans were flagged by a check", "href": _href(path, page)})
    if pitch:
        octave = candidates[candidates["reason"].astype(str).str.contains("octave")]
        path = pitch_figure(subjects, pitch, octave, directory / "pitch.png")
        figures.append({"name": "Pitch", "hint": "each person's f0; a band of points an octave above or below the rest is a tracking error, shaded where the octave check found one", "href": _href(path, page)})
    motion = results.long_table("motion.mea", features=["motion_energy"])
    motion = motion[motion["subject_id"].notna()]
    if not motion.empty:
        energy = {
            str(who): (group["t_start"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float))
            for who, group in motion.sort_values("t_start").groupby("subject_id")
        }
        flagged = candidates[candidates["check"].astype(str) == "motion_energy"] if "check" in candidates else candidates.iloc[:0]
        path = motion_figure(subjects, energy, flagged, directory / "motion.png")
        figures.append({"name": "Motion energy", "hint": "share of each person's region that changed from one frame to the next; shaded where the whole picture changed or froze, or nothing moved for a long time", "href": _href(path, page)})
    return figures


def _overview(config, layout, results: RecordingResults, recording_id: str, subjects: list[str], candidates, path: Path) -> Path | None:
    transcript = results.artifact("text.asr", TRANSCRIPT_NAME)
    talk = {}
    if transcript is not None:
        utterances = read_transcript(transcript)
        talk = {
            str(who): list(zip(group["t_start"].astype(float), (group["t_end"] - group["t_start"]).astype(float)))
            for who, group in utterances.groupby("subject_id")
        }
    sources = sorted(path.stem for path in layout.long_table_path(recording_id).glob("*.parquet"))
    lanes = [_lane(config, results, sources, name) for name in _overview_features(config, sources)]
    codes = None
    if layout.annotations_path(recording_id).exists():
        codes = annotations_store.read_annotations(layout.annotations_path(recording_id), recording_id=recording_id)
        codes = codes[~codes["abstained"].astype(bool)]
    times = [edge for spans in talk.values() for start, width in spans for edge in (start, start + width)]
    times += [edge for _, series, _ in lanes for _, _, t, _ in series if t.size for edge in (t.min(), t.max())]
    if codes is not None and not codes.empty:
        times += [codes["t_start"].min(), codes["t_end"].max()]
    if not times:
        return None
    return overview_figure(subjects, talk, lanes, codes, candidates, (float(min(times)), float(max(times))), path)


def _overview_features(config, sources: list[str]) -> list[str]:
    if config.qc.overview_features:
        return list(config.qc.overview_features)
    names = ["motion_energy"] if "motion.mea" in sources else []
    if "pose.kinematics" in sources:
        names += ["left_wrist_speed", "right_wrist_speed"] * (not names)
        names += ["interpersonal_distance", *(measure.name for measure in config.kinematics.measures)]
    return names


def _lane(config, results: RecordingResults, sources: list[str], name: str):
    """(name, [(label, person, t, y)], dots): one series per person or pair, the first table holding the feature."""
    rows = next((found for source in sources if not (found := results.long_table(source, features=[name])).empty), None)
    if rows is None:
        return name, [], False
    rows = rows.sort_values("t_start")
    who, whom = rows["subject_id"].fillna("").astype(str), rows["subject_id_b"].fillna("").astype(str)
    target = config.qc.report_downsample_px * max(1, int(np.ceil(np.ptp(rows["t_start"]) / OVERVIEW_ROW_S)))
    series = []
    for (a, b), group in rows.groupby([who, whom], sort=True):
        t, y = min_max_downsample(group["t_start"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float), target)
        series.append((f"{a} to {b}" if b else a, a or None, t, y))
    return name, series, not (rows["scale"].astype(str) == Scale.FRAME.value).all()


def _warnings(layout, results: RecordingResults, recording_id: str) -> list[QCWarning]:
    found = results.extractor_warnings()
    qc_file = layout.qc_dir(recording_id) / "warnings.json"
    if qc_file.is_file():
        found += [QCWarning(**item) for item in json.loads(qc_file.read_text(encoding="utf-8"))]
    unique = {(w.check, w.subject_id, w.severity, w.message): w for w in found}
    return sorted(unique.values(), key=lambda w: (SEVERITY_ORDER.get(w.severity, 3), w.check, w.subject_id or ""))


def _base(title: str, heading: str, chip: str, sections: list[tuple[str, str]]) -> dict:
    from macaw import __version__

    return {
        "title": title,
        "heading": heading,
        "chip": chip,
        "sections": sections,
        "version": __version__,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "css": (TEMPLATES / "report.css").read_text(encoding="utf-8"),
    }


def write_recording_report(config: MacawConfig, recording_id: str) -> tuple[Path, dict]:
    """The page, and the counts and waiting gates the project page lists for it."""
    layout = layout_for(config, create=False)
    page = layout.qc_report_path(recording_id)
    recording = build_recording(recording_id, config)
    results = RecordingResults(layout, recording_id)

    notes, versions = _feature_notes(layout, recording_id)
    alignment, alignment_figures = _alignment(config, layout, recording_id, page)
    cameras = _cameras(layout, results, recording_id)
    voices = _voices(config, layout, recording, page)
    dropped = _dropped(results)
    frames = _frames(layout, recording_id, page)
    warnings = _warnings(layout, results, recording_id)
    counts = Counter(warning.severity for warning in warnings)
    waiting = _waiting(config, recording)
    context = {
        "recording_id": recording_id,
        "subjects": [subject.subject_id for subject in recording.subjects],
        "steps": [row["extractor"] for row in versions],
        "media": _media(config, recording),
        "waiting": waiting,
        "counts": counts,
        "alignment": alignment,
        "alignment_figures": alignment_figures,
        "decisions": _decisions(config, layout, results, recording_id, alignment, cameras, voices, dropped, frames),
        "cameras": cameras,
        "voices": voices,
        "dropped": dropped,
        "over_time": _over_time(config, layout, results, recording_id, [subject.subject_id for subject in recording.subjects], page),
        "measures": _measures(layout, results, recording_id, notes),
        "frames": frames,
        "warnings": [
            {**asdict(warning), "when": f"{warning.t_start:.1f}" if warning.t_start is not None else ""}
            for warning in warnings
        ],
        "versions": versions,
    }
    sections = [("Summary", "Summary"), ("Decisions", "Decisions")]
    sections += [("Alignment", "Alignment")] if alignment else []
    sections += [("Camera", "On camera")] if cameras else []
    sections += [("Voices", "Voices")] if voices else []
    sections += [("Dropped", "Speech left out")] if dropped else []
    sections += [("Time", "Over time")] if context["over_time"] else []
    sections += [("Measures", "Measures")] if context["measures"] else []
    sections += [("Frames", "Review frames")] if context["frames"] else []
    sections += [("Warnings", "Warnings"), ("Versions", "Versions")]
    context.update(_base(f"QC {recording_id}", f"Quality review: {recording_id}", config.project.name, sections))

    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_environment().get_template("recording.html.j2").render(**context), encoding="utf-8")
    return page, {"counts": counts, "waiting": [gate for gate, _ in waiting]}


def write_project_report(config: MacawConfig, pages: dict[str, tuple[Path, dict]]) -> Path:
    layout = layout_for(config, create=False)
    page = layout.qc_report_path()
    rows = [
        {"recording_id": recording_id, "href": _href(path, page), **summary}
        for recording_id, (path, summary) in sorted(pages.items())
    ]
    rates = _frame_rates(config, sorted(pages))
    context = _base(
        f"QC {config.project.name}", f"Quality review: {config.project.name}", f"{len(rows)} recording(s)",
        [("Recordings", "Recordings")] + ([("Rates", "Frame rates")] if rates else []),
    )
    context["recordings"] = rows
    context["frame_rates"] = rates
    context["common_rate"] = _common_rate([float(row["fps"]) for row in rates]) if rates else None
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(_environment().get_template("project.html.j2").render(**context), encoding="utf-8")
    return page
