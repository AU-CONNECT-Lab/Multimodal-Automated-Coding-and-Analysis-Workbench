"""Who speaks each line on a shared microphone whose voices a person found mixed: what mouth movement says, and what was kept."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from macaw.exceptions import MacawError
from macaw.extractors import register_extractor
from macaw.extractors.audio.content.asr import SHARED_ASR, read_shared_transcript, split_by_voice
from macaw.extractors.audio.content.diarization import (
    CHANGED,
    LINES_GATE,
    NO_ONE,
    SHARED_MODE,
    SpeakerReviewGate,
    diarization_path,
    line_changes,
    read_diarization,
    read_line_speakers,
    voice_names,
    voices_mixed,
    with_changes,
)
from macaw.extractors.base import PERSON, SOFTWARE, Gate, QCWarning
from macaw.extractors.prep import load_prep
from macaw.project.layout import layout_for
from macaw.project.recording import Recording
from macaw.store.cache import hash_file
from macaw.store.schema import Scale

# who made a change that no person looked at
SUGGESTION = "suggestion"
SUGGESTED, BY_HAND = "suggested", "by hand"

_BIN_S = 0.2
# a line whose face shows in fewer bins than this says nothing
_MIN_BINS = 2
# with both mouths read: the line is changed without a person over SURE, and shown as doubtful over LIKELY
_SURE_Z = 2.0
_LIKELY_Z = 1.5
# a shorter line holds too few frames for two mouths to settle it
_SURE_MIN_S = 2.0
# one mouth alone is wrong too often to change anything; over this it is shown as doubtful
_ONE_FACE_Z = 2.5


def spoken_lines(config, recording: Recording) -> pd.DataFrame:
    """Each line as it stands, on the recording clock: the voice it falls in, whose it is by that voice (base) and whose now (person)."""
    rid = recording.recording_id
    prep = load_prep(config, rid)
    utterances, words = read_shared_transcript(config, rid)
    changes = line_changes(config, rid)
    turns = read_diarization(config, rid)
    turns = with_changes(turns[turns["exclusive"].astype(bool)], changes)
    turns = turns.assign(t_start=turns["t_start"] - prep.t_origin_s, t_end=turns["t_end"] - prep.t_origin_s)
    labels = {str(label): str(label) for label in turns["label"].unique()}
    lines, placed, _ = split_by_voice(utterances, words, turns, labels)
    if lines.empty:
        return pd.DataFrame(columns=["t_start", "t_end", "voice", "base", "person", "by", "why", "text", "words"])
    names = voice_names(config, rid)
    rows = []
    for line in lines.sort_values("t_start").itertuples(index=False):
        start, end = float(line.t_start) + prep.t_origin_s, float(line.t_end) + prep.t_origin_s
        label = str(line.subject_id)
        own = placed[(placed["subject_id"] == label) & (placed["utterance_index"] == line.utterance_index)]
        row = {
            "t_start": start, "t_end": end, "voice": label, "base": names.get(label) or NO_ONE, "by": None, "why": "",
            "text": line.text, "words": [(float(a) + prep.t_origin_s, float(b) + prep.t_origin_s) for a, b in zip(own["t_start"], own["t_end"])],
        }
        row["person"] = row["base"]
        if label.startswith(CHANGED):
            middle = (start + end) / 2
            made = next((change for change in changes if change["t_start"] <= middle <= change["t_end"]), {})
            row.update(
                voice=made.get("voice", label), base=made.get("was", NO_ONE), person=label[len(CHANGED):],
                by=made.get("by"), why=made.get("why", ""),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def mouths(config, recording_id: str) -> dict[str, pd.Series]:
    """Per person whose face was read, the rank of their mouth movement among their own, in short bins."""
    from macaw.qc.checks import MOUTH_SOURCE, RecordingResults

    table = RecordingResults(layout_for(config, create=False), recording_id).long_table(MOUTH_SOURCE, features=["mouth_movement"])
    out = {}
    for subject_id, rows in table.groupby("subject_id", dropna=True, observed=True):
        bins = (rows["t_start"].to_numpy(dtype=float) // _BIN_S).astype(int)
        moving = pd.Series(rows["value"].to_numpy(dtype=float)).groupby(bins).mean().dropna()
        if moving.size >= 10:
            out[str(subject_id)] = moving.rank(pct=True)
    return out


def with_evidence(lines: pd.DataFrame, ranks: dict[str, pd.Series]) -> pd.DataFrame:
    """Adds whom mouth movement gives each line to (suggested, None where it only speaks against the voice's person),
    whether that is sure enough to keep without listening, and why. A mouth counts as moving against the level halfway
    between the lines of that person's voice and the other lines, so no line needs a known speaker."""
    out = lines.copy()
    out["suggested"], out["sure"], out["doubt"] = None, False, ""
    if out.empty or not ranks:
        return out
    moving = {}
    for who, rank in ranks.items():
        seen = [rank.reindex(range(int(np.ceil(a / _BIN_S)), int(b // _BIN_S) + 1)).dropna() for a, b in zip(out["t_start"], out["t_end"])]
        n = np.array([bins.size for bins in seen])
        level = np.array([bins.mean() if bins.size else np.nan for bins in seen])
        usable = n >= _MIN_BINS
        own = (out["base"] == who).to_numpy()
        sides = [np.median(level[usable & side]) for side in (own, ~own) if (usable & side).any()]
        middle = float(np.mean(sides)) if len(sides) == 2 else 0.5
        moving[who] = np.where(usable, (level - middle) * np.sqrt(12 * n), np.nan)
    people = sorted({*out["base"], *ranks} - {NO_ONE})
    suggested, sure, doubt = [], [], []
    for index, line in enumerate(out.itertuples(index=False)):
        own = moving.get(line.base, np.full(len(out), np.nan))[index]
        others = {who: z[index] for who, z in moving.items() if who != line.base and np.isfinite(z[index])}
        rival = max(others, key=others.get) if others else None
        found = (None, False, "")
        if rival is not None and np.isfinite(own):
            against = others[rival] - own
            if against > _LIKELY_Z:
                found = (rival, against > _SURE_Z and line.t_end - line.t_start >= _SURE_MIN_S, f"the mouth of {rival} moves and that of {line.base} is still")
        elif rival is not None and others[rival] > _ONE_FACE_Z:
            found = (rival, False, f"the mouth of {rival} moves")
        elif rival is None and np.isfinite(own) and -own > _ONE_FACE_Z:
            rest = [who for who in people if who != line.base]
            found = (rest[0] if len(rest) == 1 else None, False, f"the mouth of {line.base} is still")
        suggested.append(found[0])
        sure.append(found[1])
        doubt.append(found[2])
    # as objects, so a line without a suggestion keeps None
    out["suggested"] = pd.Series(suggested, index=out.index, dtype=object)
    out["sure"], out["doubt"] = sure, doubt
    return out


def suggested_lines(config, recording: Recording) -> pd.DataFrame:
    return with_evidence(spoken_lines(config, recording), mouths(config, recording.recording_id))


def keep_suggested(lines: pd.DataFrame) -> pd.DataFrame:
    """The lines with every sure suggestion taken."""
    out = lines.copy()
    take = out["sure"] & out["suggested"].notna() & (out["suggested"] != out["person"])
    out.loc[take, "person"] = out.loc[take, "suggested"]
    out.loc[take, "by"] = SUGGESTION
    out.loc[take, "why"] = out.loc[take, "doubt"]
    return out


def with_picks(lines: pd.DataFrame, picks: dict[float, str], by: str) -> pd.DataFrame:
    """The lines with a person's choices, keyed by each line's start to the millisecond."""
    out = lines.copy()
    keys = out["t_start"].round(3)
    unknown = sorted(set(picks) - set(keys))
    if unknown:
        raise MacawError(f"no line starts at {', '.join(f'{t:g}' for t in unknown)} s")
    for index, key in keys.items():
        pick = picks.get(key)
        if pick is not None and pick != out.at[index, "person"]:
            out.at[index, "person"], out.at[index, "by"], out.at[index, "why"] = pick, by, ""
    return out


def split_at(lines: pd.DataFrame, at_s: float, person: str, by: str) -> pd.DataFrame:
    """From the word being said at at_s, or the next one, to the end of its line, person speaks."""
    inside = lines[(lines["t_start"] <= at_s) & (lines["t_end"] > at_s)]
    if inside.empty:
        raise MacawError(f"no line is being said at {at_s:g} s")
    index = inside.index[0]
    line = lines.loc[index]
    words = sorted(line["words"])
    cut = next((start for start, end in words if end > at_s), None)
    if cut is None or cut <= line["t_start"]:
        return with_picks(lines, {round(float(line["t_start"]), 3): person}, by)
    first, second = line.copy(), line.copy()
    first["words"], second["words"] = [word for word in words if word[0] < cut], [word for word in words if word[0] >= cut]
    first["t_end"], second["t_start"] = first["words"][-1][1], cut
    if person != second["person"]:
        second["person"], second["by"], second["why"] = person, by, ""
    rest = lines.drop(index=index)
    return pd.concat([rest, pd.DataFrame([first, second])], ignore_index=True).sort_values("t_start", ignore_index=True)


def changes_of(lines: pd.DataFrame) -> list[dict]:
    return [
        {
            "t_start": float(line.t_start), "t_end": float(line.t_end), "person": line.person, "was": line.base,
            "voice": line.voice, "by": line.by, "why": line.why,
        }
        for line in lines.itertuples(index=False)
        if line.person != line.base
    ]


def fingerprint(config, recording_id: str) -> str:
    return hash_file(diarization_path(config, recording_id))


def write_line_speakers(config, recording_id: str, lines: pd.DataFrame, stamp: str, route: str) -> Path:
    path = layout_for(config, create=False).line_speakers_path(recording_id)
    left = lines["suggested"].notna() & (lines["suggested"] != lines["person"]) if "suggested" in lines else pd.Series(False, index=lines.index)
    payload = {
        "by": stamp, "route": route, "diarization": fingerprint(config, recording_id), "n_lines": int(len(lines)),
        "n_doubted_left": int(left.sum()), "changes": changes_of(lines),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def describe_kept(kept: dict) -> str:
    changes = kept.get("changes", [])
    unheard = sum(change.get("by") == SUGGESTION for change in changes)
    parts = [f"{len(changes)} of {kept.get('n_lines', 0)} lines were given to another person than their voice's"]
    if changes:
        parts.append(f"{unheard} by mouth movement without a person listening, {len(changes) - unheard} by a person")
    if kept.get("n_doubted_left"):
        parts.append(f"{kept['n_doubted_left']} line(s) mouth movement speaks against were left as they were")
    return "; ".join(parts)


@register_extractor
class LineSpeakersGate(Gate):
    """Held where a person found the voices mixed, until someone has said who speaks each line or kept the suggestions."""

    name = LINES_GATE
    version = "0.1.0"
    scale = Scale.RECORDING
    group = "text"
    config_sections = ("text.diarization",)
    requires = (SpeakerReviewGate.name, SHARED_ASR)
    runs_after = ("face.movement",)

    @classmethod
    def brings_along(cls, config) -> tuple[str, ...]:
        # mouth movement, read off the faces the run reads anyway, is what the suggestions go on
        return ("face.movement",) if config.text.diarization == SHARED_MODE else ()

    def inputs_for(self, recording: Recording) -> list[Path]:
        return []

    def _mixed(self, recording: Recording) -> bool:
        return self.config.text.diarization == SHARED_MODE and voices_mixed(self.config, recording.recording_id)

    def pending(self, recording: Recording) -> str | None:
        if not self._mixed(recording):
            return None
        rid = recording.recording_id
        kept = read_line_speakers(self.config, rid)
        if kept and kept.get("diarization") == fingerprint(self.config, rid):
            return None
        again = "the voices were separated again since the lines were checked; " if kept else ""
        return (
            f"{again}the voices were marked as mixed, so someone has to say who speaks each line: on the Review page, or list them with "
            f"`macaw line-speakers -r {rid}` and keep the sure suggestions with `--suggested`, your own with `--apply FILE`, "
            "or all as they are with `--as-they-are`"
        )

    def decided_by(self, recording: Recording) -> str | None:
        if not self._mixed(recording):
            return None
        kept = read_line_speakers(self.config, recording.recording_id)
        if not kept:
            return None
        return SOFTWARE if kept.get("route") == SUGGESTED else PERSON

    def notes(self, recording: Recording) -> list[QCWarning]:
        if not self._mixed(recording):
            return []
        kept = read_line_speakers(self.config, recording.recording_id)
        unheard = any(change.get("by") == SUGGESTION for change in kept.get("changes", []))
        return [
            QCWarning(
                check="line_speakers", recording_id=recording.recording_id, subject_id=None,
                severity="warn" if unheard or kept.get("n_doubted_left") else "info",
                message=f"a person found the voices mixed: {describe_kept(kept)}",
            )
        ]
