"""Methods text and BibTeX describing only the steps a project actually ran."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import yaml

from macaw import __version__
from macaw.extractors.audio.content.diarization import others_take_turns
from macaw.extractors.prep import PREP_SAMPLE_RATE
from macaw.project.manifest import FeatureManifest, FeatureSpec
from macaw.project.scale import pixel_power
from macaw.project.segments import read_segments

METHODS_NAME = "methods.md"
BIB_NAME = "references.bib"

# extractors in the order a reader meets them; anything else (plugins) follows
ORDER = (
    "prep.media", "prep.video", "prosody.frame", "prosody.voice_report", "prosody.standard_sets", "text.diarize", "text.asr", "text.features", "text.syntax",
    "turns.structure", "turns.transitions", "turns.entropy", "coding.layer_a", "coding.judge", "pose.detect", "pose.keypoints",
    "pose.kinematics", "pose.gestures", "pose.kinematics3d", "pose.gestures3d", "motion.mea", "face.au", "face.au_check", "face.movement",
    "gaze.target",
)
SECTIONS = {
    "prep": "Media", "prosody": "Voice", "text": "Speech", "turns": "Turn-taking", "coding": "Turn-taking",
    "pose": "Body", "motion": "Body", "face": "Face", "gaze": "Gaze",
}

# writers cite as [key] or [key; key]; render turns that into author-year and collects the keys
_CITATION = re.compile(r"\[([a-z0-9_]+(?:; [a-z0-9_]+)*)\]")


def recordings_scaled(config_path: Path, config) -> tuple[int, int]:
    """(recordings with a scale, recordings) in a project."""
    from macaw.config.loader import resolve_config
    from macaw.execution.runner import resolve_recordings
    from macaw.project.layout import layout_for
    from macaw.project.scale import recording_scale

    layout = layout_for(config, create=False)
    recording_ids = resolve_recordings(config, None)
    found = sum(recording_scale(resolve_config(config_path, recording_id=rid), layout, rid) is not None for rid in recording_ids)
    return found, len(recording_ids)


def load_references() -> dict[str, dict]:
    return yaml.safe_load(files("macaw.qc").joinpath("references.yaml").read_text(encoding="utf-8"))


def write_methods(
    manifest: FeatureManifest, config, out_dir: Path, *, scaled: tuple[int, int] | None = None, looked: dict | None = None,
) -> tuple[Path, Path]:
    references = load_references()
    text, used = render(describe(manifest, config, scaled=scaled, looked=looked), references)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = out_dir / METHODS_NAME
    methods.write_text(text, encoding="utf-8")
    bib = out_dir / BIB_NAME
    bib.write_text("\n".join(to_bibtex(key, references[key]) for key in used), encoding="utf-8")
    return methods, bib


def describe(
    manifest: FeatureManifest, config, *, scaled: tuple[int, int] | None = None, looked: dict | None = None,
) -> dict[str, list[str]]:
    """scaled is (recordings with a scale, recordings), when known; looked is what qc.look.project_summary gives."""
    by_extractor: dict[str, list[FeatureSpec]] = {}
    for spec in manifest.specs.values():
        by_extractor.setdefault(spec.extractor, []).append(spec)
    names = [name for name in ORDER if name in by_extractor] + sorted(set(by_extractor) - set(ORDER))
    sections: dict[str, list[str]] = {}
    for name in names:
        specs = by_extractor[name]
        writer = _WRITERS.get(name) or (_judge if name.startswith("coding.judge.") else _plugin)
        text = writer(config, specs[0], {spec.name for spec in specs})
        if not text:
            continue
        sections.setdefault(SECTIONS.get(name.split(".")[0], "Other features"), []).append(text)
    in_pixels = any(pixel_power(spec.unit) for spec in manifest.specs.values())
    if scaled and scaled[0] and in_pixels:
        sections.setdefault("Body", []).append(_centimetres(*scaled))
    found = read_segments(config)
    if found and sections:
        sections.setdefault("Segments", []).append(_segments(found))
    if looked:
        sections.setdefault("Quality review", []).append(_looked(looked))
    return sections


def _looked(found: dict) -> str:
    where = "each recording" if found["recordings"] == 1 else f"{found['recordings']} recordings"
    text = (
        f"In {where}, the frames drawn for review were also shown to the vision language model {_listed(found['models'])} "
        f"({_listed(found['providers'])}). For each frame it was asked how many people were in view and, for each person the "
        "tracker had drawn, cut out with only that person's skeleton, whether it was a real person"
        + (f" and, given a description of the {_listed(found['described'])}, which of them it was" if found.get("described") else "")
        + "; a frame where its answers disagreed with the tracker was flagged for a person to check. "
        f"It flagged {found['flagged']} of {found['frames']} frames"
    )
    text += f", and a person confirmed {found['flagged_right']} of the {found['flagged_checked']} checked." if found["flagged_checked"] else "."
    if found["drawn_checked"]:
        text += (
            f" Of the frames it passed, {found['drawn_checked']} drawn at random were checked by a person, who found "
            f"{found['drawn_wrong']} of them wrong."
        )
    return text


def _segments(found: dict) -> str:
    names = sorted({segment.name for segments in found.values() for segment in segments})
    named = f" ({_listed(names)})" if len(names) <= 6 else ""
    where = "a recording" if len(found) == 1 else f"{len(found)} recordings"
    return (
        f"Within {where}, named segments{named} were set by hand, and every recording-level measure was computed again "
        "within each from the output of the whole recording, as if the recording had been cut at the segment's edges: an "
        "utterance, turn, code, look or blink that crossed an edge counted only with its part inside, a word where its middle "
        "fell, and turns, pauses and floor transfers were formed again from what remained. Speech recognition, the "
        "separation of voices, the tracking of people and the decisions about who was who were made once over the whole "
        "recording and held in every segment."
    )


def _centimetres(with_scale: int, recordings: int) -> str:
    which = "each recording" if with_scale == recordings else f"{with_scale} of the {recordings} recordings"
    return (
        f"For {which}, lengths in pixels were also converted to centimetres. The scale was the length of an object of "
        "known size divided by the distance between its two ends as marked by a person on a frame of the recording, or "
        "of a calibration clip filmed with the camera unmoved (the mean where it was marked more than once), or a number "
        "set by hand; where the object was filmed at each participant's place, each participant had their own. A scale "
        "holds only at the distance from the camera at which it was taken, and measures between two participants were "
        "converted only when both had the same scale."
    )


def render(sections: dict[str, list[str]], references: dict[str, dict]) -> tuple[str, list[str]]:
    used: list[str] = []

    def cite(match: re.Match) -> str:
        keys = match.group(1).split("; ")
        used.extend(key for key in keys if key not in used)
        return "(" + "; ".join(label(references[key]) for key in keys) + ")"

    lines = [
        "# Methods",
        "",
        f"Recordings were processed with pymacaw {__version__}. This text was generated from the feature manifest "
        "and the configuration of the run and describes only the steps that were run; it may be copied and edited freely.",
    ]
    for section, paragraphs in sections.items():
        lines += ["", f"## {section}", "", _CITATION.sub(cite, " ".join(paragraphs))]
    lines += ["", "## References"]
    for key in sorted(used, key=lambda key: label(references[key])):
        lines += ["", formatted(references[key])]
    return "\n".join(lines) + "\n", used


def _authors(entry: dict) -> list[str]:
    return [name.strip() for name in str(entry["author"]).split(" and ")]


def _initials(name: str) -> str:
    surname, _, given = name.partition(", ")
    parts = [".-".join(piece[0] for piece in word.split("-")) + "." for word in given.replace(".", " ").split()]
    return f"{surname}, {' '.join(parts)}" if parts else surname


def label(entry: dict) -> str:
    surnames = [name.split(",")[0] for name in _authors(entry)]
    if len(surnames) == 1:
        who = surnames[0]
    elif len(surnames) == 2:
        who = f"{surnames[0]} & {surnames[1]}"
    else:
        who = f"{surnames[0]} et al."
    return f"{who}, {entry.get('year', 'n.d.')}"


def formatted(entry: dict) -> str:
    names = [_initials(name) for name in _authors(entry)]
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", & " + names[-1]
    parts = [f"{who} ({entry.get('year', 'n.d.')}).", str(entry["title"]).rstrip(".") + "."]
    venue = entry.get("journal") or entry.get("booktitle") or entry.get("howpublished") or entry.get("publisher")
    if venue:
        detail = str(venue)
        if "volume" in entry:
            detail += f", {entry['volume']}" + (f"({entry['number']})" if "number" in entry else "")
        if "pages" in entry:
            detail += f", {str(entry['pages']).replace('--', '-')}"
        parts.append(detail + ".")
    return " ".join(parts)


def to_bibtex(key: str, entry: dict) -> str:
    fields = [f"  {name} = {{{value}}}," for name, value in entry.items() if name != "type"]
    return f"@{entry['type']}{{{key},\n" + "\n".join(fields) + "\n}\n"


def _seconds(value: float) -> str:
    return f"{value:g} s"


def _count(value: int) -> str:
    return ("two", "three", "four", "five", "six", "seven", "eight", "nine")[value - 2] if 2 <= value <= 9 else str(value)


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _prep(config, spec, names) -> str:
    segment, align = config.segment, config.align
    window = ""
    if segment.task_start_s is not None or segment.task_end_s is not None:
        window = f" from {_seconds(segment.task_start_s or 0.0)}"
        if segment.task_end_s is not None:
            window += f" to {_seconds(segment.task_end_s)}"
    text = (
        f"Media were probed and trimmed to the task window{window}, and each participant's audio was written to a "
        f"mono {PREP_SAMPLE_RATE / 1000:g} kHz track."
    )
    if align.method == "gcc_phat":
        text += (
            " Devices were aligned by cross-correlating loudness envelopes over a search range of "
            f"±{align.search_window_s:g} s and refining the offset with the generalised cross-correlation with phase "
            "transform [knapp1976] within ±1 s"
        )
        text += {
            "always": "; every offset was checked by a person before use.",
            "flagged": "; every offset flagged as uncertain was checked by a person before use.",
        }.get(align.review, ".")
        if align.drift_check:
            text += f" Clock drift was estimated over {align.drift_segment_s:g} s segments."
    elif align.method == "manual":
        text += " Offsets between devices were set by hand."
    else:
        text += " Devices were treated as already synchronised."
    return text


def _video(config, spec, names) -> str:
    video = config.video
    if not video.normalise:
        return "Videos were read as recorded, at their average frame rate."
    rate = f"a constant {video.fps:g} frames per second" if video.fps else "a constant frame rate, the one it was recorded at"
    return (
        f"Before any video measure, each video was re-encoded once with FFmpeg as H.264 ({video.encoder}, quality "
        f"{video.quality}) at {rate}, upright, with square pixels and deinterlaced where needed, its first frame placed "
        "at the first audio sample of its file; video recorded in high dynamic range was tone-mapped to BT.709 with "
        "FFmpeg's tonemap filter (Möbius curve)."
    )


def _backend(phrases: dict[str, str], spec: FeatureSpec, **extra) -> str:
    phrase = phrases.get(spec.backend, "the {name} {version} backend")
    return phrase.format(name=spec.backend, version=spec.backend_version, **extra)


_PITCH = {
    "parselmouth": "Praat through Parselmouth {version} [boersma_praat; jadoul2018]",
    "librosa": "librosa {version} [mcfee2015]",
    "builtin": "pymacaw's built-in autocorrelation pitch tracker",
}


def _f0_range(prosody) -> str:
    text = f"between {prosody.f0_min_hz:g} and {prosody.f0_max_hz:g} Hz"
    if prosody.f0_range_by_role:
        ranges = "; ".join(f"{role} {low:g} to {high:g} Hz" for role, (low, high) in prosody.f0_range_by_role.items())
        text += f" ({ranges})"
    return text


def _frame(config, spec, names) -> str:
    measured = ["pitch", "intensity"] + (["harmonics-to-noise ratio"] if "hnr_db" in names else [])
    if "f1_hz" in names:
        measured.append("formants F1 to F4")
    return (
        f"On each participant's own track, {_listed(measured)} were measured every "
        f"{config.prosody.frame_step_s * 1000:g} ms with {_backend(_PITCH, spec)}, searching pitch "
        f"{_f0_range(config.prosody)}. Intensity depends on the distance to the microphone and was only compared "
        "within a speaker."
    )


def _voice_report(config, spec, names) -> str:
    praat = spec.params.get("praat_version")
    return (
        f"For every turn, a voice report was computed with Praat{f' {praat}' if praat else ''} through Parselmouth "
        f"{spec.backend_version} [boersma_praat; jadoul2018]: pitch mean, median, standard deviation, minimum, maximum, "
        "interquartile range, range in semitones and least-squares slope in semitones per second; the voiced fraction; "
        "five jitter measures (local, local absolute, RAP, PPQ5, DDP) and six shimmer measures (local, local dB, APQ3, "
        "APQ5, APQ11, DDA) with Praat's standard settings (periods 0.1 to 20 ms, period factor 1.3, amplitude factor "
        "1.6); the mean harmonics-to-noise ratio; and smoothed cepstral peak prominence [hillenbrand1996] with a "
        "least-squares straight trend line."
    )


def _standard_sets(config, spec, names) -> str:
    known = {
        "egemaps": "the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS v02) [eyben2016]",
        "compare": "the ComParE 2016 feature set [schuller2016]",
    }
    chosen = [known[prefix] for prefix in sorted({name.split("_", 1)[0] for name in names}) if prefix in known]
    return (
        f"For every turn, {_listed(chosen)} {'was' if len(chosen) == 1 else 'were'} extracted from the speaker's own "
        f"track with openSMILE {spec.backend_version} [eyben2010]."
    )


_ASR = {
    "whisperx": "WhisperX {version} [bain2023] with the Whisper {model} model [radford2023]",
    "faster_whisper": "faster-whisper {version}, a reimplementation of Whisper [radford2023], with the {model} model",
}


def _diarize(config, spec, names) -> str:
    if "n_voices_found" not in names:
        return ""
    extra = spec.params.get("extra_voices", 0)
    speakers = (
        f"allowing up to {extra} voice{'s' if extra > 1 else ''} besides the participants'. Speech in any other voice "
        "was transcribed as one further speaker, who took turns and was a state of its own in the turn-taking "
        "measures, but was left out of the participants' language and voice measures. "
        if extra else "with the number of speakers set to the number of participants. "
    )
    return (
        f"The shared microphone was diarized with pyannote.audio {spec.backend_version} [bredin2023] using the "
        f"{spec.params.get('model')} pipeline, " + speakers + _VOICE_NAMING[config.text.voice_review] + _voice_hints(config)
    )


def _voice_hints(config) -> str:
    from macaw.extractors.audio.content.speaker_id import PITCH_GAP_HZ

    identity = config.identity
    used = ["phrases set for this study as marking a speaker's role"] * bool(identity.voice_hints)
    used += [f"the expected order of the voices' median pitch, where they differed by at least {PITCH_GAP_HZ:g} Hz"] * bool(identity.pitch_order)
    return f" Suggested names also drew on {' and on '.join(used)}." if used else ""


_VOICE_NAMING = {
    "always": "A researcher listened to clips of each voice found and named the participant it belonged to.",
    "flagged": (
        "Each voice was named from the loudness of the participants' own microphones or their mouth movement where "
        "that decided every voice, and otherwise by a researcher who listened to clips of each voice."
    ),
    "skip": (
        "Each voice was named from the loudness of the participants' own microphones or their mouth movement; "
        "voices this could not decide were left out."
    ),
}


def _asr(config, spec, names) -> str:
    text_cfg = config.text
    language = f" in {text_cfg.language}" if text_cfg.language else ""
    if text_cfg.diarization == "pyannote":
        return (
            f"The shared microphone was transcribed{language} with {_backend(_ASR, spec, model=text_cfg.asr_model)}. "
            "Each word was given to the voice whose diarized turn it overlapped most, and consecutive words of one "
            "participant within a recogniser segment formed an utterance. A participant with a microphone of their own "
            "was transcribed from it instead, and an utterance on it lying mostly in the voice of a participant heard "
            "only on the shared microphone was removed as bleed."
        )
    return (
        f"Speech on each participant's own track was transcribed{language} with "
        f"{_backend(_ASR, spec, model=text_cfg.asr_model)}. After each microphone was normalised to its own level, an "
        f"utterance more than {text_cfg.crosstalk_margin_db:g} dB louder on another participant's microphone was "
        "removed as cross-talk."
    )


def _text_features(config, spec, names) -> str:
    params = spec.params
    counted = "words"
    if params.get("first_use_words"):
        counted = "words from a researcher-supplied list"
    exclude = params.get("first_use_exclude", "none")
    if exclude == "english_stopwords":
        counted += ", excluding English function words, fillers and backchannels,"
    elif exclude != "none":
        counted += ", excluding a researcher-supplied list of words,"
    text = (
        "Lexical measures were computed for each speaker on lower-cased English word tokens: mean length of utterance "
        f"in words, type-token ratio, unigram entropy, and, of the {counted} used by at least "
        f"{_count(params.get('first_use_min_speakers', 2))} speakers, the share the speaker said first."
    )
    model = params.get("embedding_model")
    if model:
        text += (
            f" Each utterance was embedded with the {model} sentence-transformers model [reimers2019]. For each pair of "
            "speakers, the mean cosine similarity of one speaker's utterances to the utterance just before each by the other "
            "was taken as how closely the first followed on from the second."
        )
    return text


def _turns(config, spec, names) -> str:
    turns = config.turns
    return (
        "Consecutive utterances of one speaker were merged into a turn when no one else spoke in between and the pause "
        f"was at most {_seconds(turns.max_gap_s)}. Stretches of at least {_seconds(turns.silence_min_s)} in which "
        "nobody spoke were counted as joint silences."
    )


def _transitions(config, spec, names) -> str:
    turns = config.turns
    return (
        "For each pair of adjacent turns by different speakers, the floor transfer offset was the time from the end of "
        "one turn to the start of the next, positive for gaps and negative for overlaps [heldner2010]. Transfers within "
        f"{_seconds(turns.smooth_transfer_gap_s)} were counted as smooth; taking the floor with an overlap longer than "
        f"{_seconds(turns.interruption_overlap_s)} was counted as an interruption, and a shorter overlap after which "
        "the floor returned to the first speaker as a backchannel."
    )


def _entropy(config, spec, names) -> str:
    turns = config.turns
    text = (
        "Turn-taking entropies were the entropy of the split of floor time between speakers, the entropy of which "
        "speaker followed which and the entropy of the next speaker given the current one, the latter two with the "
        "Miller-Madow bias correction [miller1955], and the permutation entropy of the floor transfer offsets of order "
        f"{turns.permutation_entropy_order} [bandt2002]."
    )
    if turns.window_s:
        hop = f" every {_seconds(turns.window_hop_s or turns.window_s / 2.0)}"
        text += f" They were computed over the whole recording and in windows of {_seconds(turns.window_s)}{hop}."
    return text


def _layer_a(config, spec, names) -> str:
    coding = config.coding
    text = (
        f"Silences of at least {_seconds(coding.min_turn_internal_pause_s)} within a speaker's own turn were coded as "
        "turn-internal pauses."
    )
    if coding.backchannel_lexicon:
        text += (
            f" Turns shorter than {_seconds(coding.backchannel_max_s)} made only of words from a list of "
            f"{len(coding.backchannel_lexicon)} backchannel tokens, after which the floor returned, were coded as "
            "backchannels."
        )
    return text


def _shown(show) -> str:
    """What the model was given for each line, as a Methods phrase."""
    parts = ["the speaker letter", "the words"]
    if "timing" in show:
        parts += ["its start and length", "the pause or overlap before it"]
    voice = [term for item, term in (("pitch", "pitch"), ("loudness", "loudness"), ("rate", "speech rate")) if item in show]
    if voice:
        parts.append(f"its {_listed(voice)} relative to the median of that speaker's lines")
    shown = "the speaker letter and the words only" if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1]
    if "frames" in show:
        shown += "; each unit to code came with three video frames near its start, middle and end, one unit per request"
    return shown


def _prompt_extras(params: dict) -> str:
    """What else the prompt held: the researchers' note, examples people coded, reasons asked for, reasoning switched."""
    text = ""
    if params.get("notes"):
        note = params["notes"]
        text += f" The prompt also held this note from the researchers: \"{note}\"" + ("" if note[-1] in ".!?" else ".")
    if params.get("n_examples"):
        text += (
            f" It showed {_count(params['n_examples'])} units people had coded in other recordings as examples, "
            "none from the recording being coded."
        )
    if params.get("reasons"):
        text += " The model was asked to name in a few words what each code rested on."
    if params.get("think") is not None:
        text += f" The model's reasoning before answering was turned {'on' if params['think'] else 'off'}."
    return text


def _judge(config, spec, names) -> str:
    params = spec.params
    if params.get("judge"):
        return _judge_version(config, params, spec)
    model = spec.backend_version or params.get("model")
    given = "only the names of the codes" if params.get("names_only") else "the codebook's definitions and examples"
    text = (
        f"Each {params.get('codebook_unit', 'unit')} was coded with the codebook \"{params.get('codebook')}\" by the language model "
        f"{model} ({params.get('provider')}), given {given} and the "
        f"{params.get('context_units')} preceding lines as context. For each line the model was given "
        f"{_shown(params.get('show', ['timing']))}."
    ) + _prompt_extras(params)
    coded = params.get("coded_roles") or []
    if coded:
        text += f" Only the units of the {_listed(coded)} were coded; the others' lines were shown in place as context."
    if others_take_turns(config):
        text += " Lines in other voices were shown in place as context and not coded."
    samples = params.get("n_samples", 1)
    if samples > 1:
        text += (
            f" Each unit was coded {_count(samples)} times at temperature {params.get('temperature')}; the label is the "
            f"majority answer and its share the confidence. Units below {params.get('review_threshold')}, ties and units "
            "the model could not code were reviewed by a person."
        )
    drawn = config.coding.llm.review_random
    if drawn:
        text += (
            f" Besides those, {_count(drawn)} units per recording that the model was sure of were drawn at random for a person "
            "to check, to estimate how often its sure answers were wrong."
        )
    return text


_VERSION_TERMS = {
    "temperature": "temperature", "n_samples": "answers per unit", "context_units": "earlier lines as context",
    "batch_size": "units per request", "allow_abstain": "may answer that it cannot tell", "coded_roles": "people coded",
}


def _judge_version(config, params: dict, spec) -> str:
    model = spec.backend_version or params.get("model")
    names = " (code names only)" if params.get("names_only") else ""
    text = (
        f"For comparison, the same units were also coded in a version named {params['judge']}, with the codebook "
        f"\"{params.get('codebook')}\"{names} and the language model {model} ({params.get('provider')}), given for each line "
        f"{_shown(params.get('show', ['timing']))}"
    )
    base = config.coding.llm
    # with frames one unit per request is already said
    framed = "frames" in params.get("show", [])
    others = [
        f"{term} {_setting(params.get(key))}"
        for key, term in _VERSION_TERMS.items()
        if key in params and params[key] != getattr(base, key) and not (framed and key == "batch_size")
    ]
    own = config.coding.judges.get(params["judge"])
    changed = {
        "notes": params.get("notes") != base.notes.strip(),
        "n_examples": own is not None and own.examples is not None,
        "reasons": params.get("reasons") != base.reasons,
        "think": params.get("think") != base.think,
    }
    extras = _prompt_extras({key: value for key, value in params.items() if changed.get(key)})
    return text + (f" ({'; '.join(others)})." if others else ".") + extras


def _setting(value) -> str:
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "everyone"
    return str(value)


_POSE = {
    "mediapipe": "MediaPipe Pose Landmarker {version}{model} [bazarevsky2020; lugaresi2019]",
    "yolo": "Ultralytics YOLO{model} [jocher_ultralytics] with ByteTrack tracking [zhang2022]",
}


def _detect(config, spec, names) -> str:
    pose = config.pose
    model = f" ({Path(pose.model).name})" if pose.model else ""
    return (
        f"Body keypoints were detected in every video frame of the task window with {_backend(_POSE, spec, model=model)} "
        f"at an input size of {pose.input_size} px and mapped to the 17 COCO joints; joints below a confidence of "
        f"{pose.min_confidence:g} and coordinates outside the frame were set to missing."
    )


def _keypoints(config, spec, names) -> str:
    pose = config.pose
    text = "Detections were linked into tracks and each track was assigned to a participant automatically"
    text += {
        "always": "; every assignment was checked by a person.",
        "flagged": "; assignments the software could not make with confidence were confirmed by a person.",
    }.get(pose.review, ".")
    if pose.track_repair:
        text += f" Identity swaps between tracks across gaps of up to {_seconds(pose.track_repair_max_gap_s)} were repaired."
    return text


def _kinematics(config, spec, names) -> str:
    return (
        f"Wrist speed, acceleration and jerk were estimated with a {_seconds(config.kinematics.smoothing_s)} "
        "Savitzky-Golay filter of polynomial order 3 [savitzky1964] within each unbroken run of detections. The distance "
        "between the wrists, head yaw and roll from the eyes and nose, postural openness, the distance between two "
        "people's shoulder midpoints and the overlap of their body outlines were also derived. All are measured in the "
        "image plane, in pixels or relative to body size, not in metric units."
    ) + _measures(config.kinematics.measures)


def _gestures(config, spec, names) -> str:
    gestures = config.kinematics.gestures
    unit = spec.unit.split("/")[0] if spec.unit and "/" in spec.unit else "px"
    still = (
        f"where both wrists and both elbows moved slower than {gestures.still_speed:g} {unit}/s, counted once per gesture"
        if gestures.holds == "both_hands"
        else f"where wrist and elbow both moved slower than {gestures.still_speed:g} {unit}/s"
    )
    return (
        f"The kinematics of each gesture coded on the {gestures.tier} tier were measured from each wrist following "
        "[trujillo2019]: peak speed; submovements, the speed peaks rising at least "
        f"{gestures.peak_prominence:g} {unit}/s above their surroundings; holds between the first and last submovement, "
        f"{still}; the area the wrist's path covered; the highest level it reached relative to the body; the outermost "
        "and most used gesture space [mcneill1992]; and how many of the eight cells of the periphery around the centre "
        "space it entered. Rhythm was the standard deviation of the intervals between submovements [pouw2021]. Speeds "
        f"were estimated with a {_seconds(config.kinematics.smoothing_s)} Savitzky-Golay filter [savitzky1964] and "
        "are in image units. Gesture spaces were drawn in the image plane, which follows the body only while a person "
        "faces the camera or has their back to it."
    )


def _gestures_3d(config, spec, names) -> str:
    gestures = config.kinematics.gestures
    still = "both wrists and both elbows" if gestures.holds == "both_hands" else "wrist and elbow"
    return (
        f"The same measures of each coded gesture were also taken from the wrists placed in 3D, with speeds in m/s, "
        f"submovements rising at least {gestures.peak_prominence_3d:g} m/s above their surroundings and holds where {still} "
        f"moved slower than {gestures.still_speed_3d:g} m/s. The area, the levels and the gesture spaces were taken in each "
        "person's own frontal plane, set by their hips and shoulders, so that they do not depend on where the cameras stood."
    )


def _kinematics_3d(config, spec, names) -> str:
    from macaw.extractors.video.pose.cameras import read_setup
    from macaw.extractors.video.pose.triangulate import calibration_path

    pose3d = config.pose3d
    try:
        setup = read_setup(calibration_path(config), pose3d.units)
        cameras, made_here = f"{len(setup.cameras)} ", setup.metadata.get("made_by") == "macaw calibrate-cameras"
    except Exception:   # the calibration file moved since the run; the text stays true without the count
        cameras, made_here = "", False
    board = pose3d.board
    calibrated = (
        f"The cameras were calibrated from a ChArUco board [garrido2014] of {board.square_mm:g} mm squares filmed by all of "
        "them: each lens first from its own views [zhang2000], then every camera's place and lens together by bundle "
        "adjustment [triggs2000]."
        if made_here else "The cameras' calibration was supplied as a file."
    )
    return (
        f"Each person's keypoints were placed in 3D from the {cameras}calibrated cameras that saw them by weighted linear "
        f"triangulation of the undistorted points [hartley2003], leaving a camera out of a point while its view missed the "
        f"others' by more than {pose3d.max_reprojection_px:g} px; no point was placed where the remaining views still "
        f"missed each other by more, or where it fell behind a camera. {calibrated} In 3D, wrist speed, acceleration and jerk "
        f"were estimated with a {_seconds(config.kinematics.smoothing_s)} Savitzky-Golay filter [savitzky1964], together "
        "with the distance between the wrists, between two people's shoulder midpoints and between the midpoints of their "
        "ears, and the angle between where a person's head points, from the midpoint of the ears to the nose, and the "
        "direction to the other person's head."
    ) + _measures_3d(config)


def _measures_3d(config) -> str:
    from macaw.extractors.video.pose.triangulate import measure_spec_3d, measures_3d

    measures = measures_3d(config)
    if not measures:
        return ""
    described = [f"{measure.name}_3d: {measure_spec_3d(measure)[0].rstrip('.')}" for measure in measures]
    return " Measures defined for this study were also derived in 3D (" + "; ".join(described) + ")."


def _measures(measures) -> str:
    from macaw.extractors.video.pose.kinematics import measure_spec

    if not measures:
        return ""
    described = [f"{measure.name}: {measure_spec(measure)[0].rstrip('.')}" for measure in measures]
    return " Measures defined for this study were also derived (" + "; ".join(described) + ")."


def _mea(config, spec, names) -> str:
    motion = config.motion
    smoothing = f", smoothed over {_seconds(motion.smoothing_s)}" if motion.smoothing_s else ""
    return (
        "Motion energy was measured by frame differencing in a region drawn by hand around each participant "
        f"[ramseyer2011]: the share of the region's pixels whose grey level changed by more than {motion.threshold} "
        f"between consecutive frames{smoothing}."
    )


_FACE = {
    "pyfeat": "Py-Feat {version} [cheong2023] using its {model} models",
    "openface": "OpenFace {version} [baltrusaitis2018]",
    "mediapipe": "the MediaPipe Face Landmarker {version} [kartynnik2019; lugaresi2019]",
}
_FACE_NAMES = {"pyfeat": "Py-Feat", "openface": "OpenFace", "mediapipe": "MediaPipe"}
_FACE_CITED = {"pyfeat": "Py-Feat [cheong2023]", "openface": "OpenFace [baltrusaitis2018]", "mediapipe": "MediaPipe [kartynnik2019]"}


def _face(config, spec, names) -> str:
    model = spec.params.get("model") or config.face.model or "v1"
    measured = []
    if any(name.startswith("au") for name in names):
        measured.append("facial action unit intensities")
    if any(name.startswith("face_yaw") for name in names):
        measured.append("head pose")
    if any(name.startswith("gaze_") for name in names):
        measured.append("gaze direction")
    text = (
        f"Faces were detected and {_listed(measured or ['landmarks'])} estimated with {_backend(_FACE, spec, model=model)}; "
        "each face was assigned to the person whose nose or eyes fell inside it."
    )
    if spec.backend == "openface" and "facial action unit intensities" in measured:
        text += " OpenFace's intensities (0 to 5) were divided by 5."
    if spec.backend == "pyfeat" and "gaze direction" in measured and model == "v1":
        text += " Gaze came from L2CS-Net [abdelrahman2023]."
    return text


def _au_check(config, spec, names) -> str:
    params = spec.params
    first = _FACE_NAMES.get(params.get("backend"), params.get("backend"))
    second = _FACE_NAMES.get(params.get("check_backend"), params.get("check_backend"))
    if params.get("check_backend") == params.get("backend"):
        checked = f"{second}'s {params.get('check_model')} models were run on the same faces"
        against = f"the {params.get('model')} models"
    else:
        checked = f"{_FACE_CITED.get(params.get('check_backend'), second)} was run on the same faces"
        against = first
    return (
        f"As a check on the action units, {checked} and each action unit's agreement with {against} was measured as "
        "the Spearman correlation of their 1 s means."
    )


def _movement(config, spec, names) -> str:
    return (
        "From the face landmarks, mouth opening (the gap between the inner lips over the distance from the eyes to the "
        "base of the nose) and its change per second, eye opening and blinks (eye opening below 70% of its median over "
        f"3 s) were derived; eyes narrower than {config.face.blink_min_eye_px:g} px or in a head turned more than 45° "
        "were not read."
    )


def _gaze(config, spec, names) -> str:
    from macaw.extractors.video.gaze.heads import HEAD_SCALE

    settle = config.gaze.min_look_s
    people = (
        "mutual gaze was coded when two people looked at each other in the same frame. "
        + (
            f"A run of frames with one target lasting less than {settle:g} s took the target of the run before it (the first "
            "run, that of the run after), and each remaining run was kept as one coded interval"
            if settle > 0 else "Each run of frames with the same target was kept as one coded interval"
        )
    )
    if spec.backend == "angles":
        cfg = config.gaze
        signal = "gaze direction where the face model gave one and head pose elsewhere" if config.face.gaze else "head pose"
        where = (
            "the seats measured for the session (each participant's position and eye height relative to the camera), "
            "corrected by each participant's median offset on frames labelled by hand"
        ) if cfg.seats else "frames labelled by hand"
        return (
            f"Gaze targets were inferred from {signal}: each participant was coded as looking at the target whose direction, "
            f"taken from {where}, lay nearest their own and within {cfg.tolerance_deg:g}°; a target with three or more "
            f"labelled frames took its direction from them; {people}."
        )
    model = spec.params.get("model") or ""
    text = (
        f"For each participant in each frame, the point they looked at was estimated with Gaze-LLE [ryan2025] ({model}"
        f"{', fine-tuned on ChildPlay [tafasca2023]' if model.endswith('childplay') else ''}) from the whole frame and a head "
        f"box drawn from the participant's pose keypoints ({HEAD_SCALE:g} times the spread of the face keypoints, or the "
        "shoulder width if larger). A participant was coded as looking at another while that point lay inside the other's "
        f"head box and the model's in-frame score was at least {spec.params.get('in_frame_cut', 0.5):g}; {people}."
    )
    if config.gaze.targets:
        text += (
            f" Objects named {', '.join(config.gaze.targets)} were detected five times per second with Grounding DINO "
            f"[liu2024] (box score at least {config.gaze.object_score:g}); a gaze point on no head but inside an object's box "
            "in the nearest such frame was assigned to that object, the smallest box first."
        )
    return text


def _syntax(config, spec, names) -> str:
    model = spec.params.get("syntax_model") or config.text.syntax_model
    version = spec.params.get("syntax_model_version")
    return (
        f"Each utterance was parsed on its own with spaCy {spec.backend_version} [honnibal2020] and its {model} model"
        f"{f' {version}' if version else ''}. For each speaker, the mean dependency distance [liu2008], the share of "
        "dependencies between adjacent words, subordinate clauses per utterance, lexical density and the share of each "
        "part of speech were computed."
    )


def _plugin(config, spec, names) -> str:
    backend = f", backend {spec.backend} {spec.backend_version}" if spec.backend else ""
    return f"Features from the {spec.extractor} {spec.extractor_version} extractor{backend} were included."


_WRITERS = {
    "prep.media": _prep,
    "prep.video": _video,
    "prosody.frame": _frame,
    "prosody.voice_report": _voice_report,
    "prosody.standard_sets": _standard_sets,
    "text.diarize": _diarize,
    "text.asr": _asr,
    "text.features": _text_features,
    "text.syntax": _syntax,
    "turns.structure": _turns,
    "turns.transitions": _transitions,
    "turns.entropy": _entropy,
    "coding.layer_a": _layer_a,
    "coding.judge": _judge,
    "pose.detect": _detect,
    "pose.keypoints": _keypoints,
    "pose.kinematics": _kinematics,
    "pose.gestures": _gestures,
    "pose.kinematics3d": _kinematics_3d,
    "pose.gestures3d": _gestures_3d,
    "motion.mea": _mea,
    "face.au": _face,
    "face.au_check": _au_check,
    "face.movement": _movement,
    "gaze.target": _gaze,
}
