"""Methods text and BibTeX describing only the steps a project actually ran."""

from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path

import yaml

from macaw import __version__
from macaw.extractors.prep import PREP_SAMPLE_RATE
from macaw.project.manifest import FeatureManifest, FeatureSpec

METHODS_NAME = "methods.md"
BIB_NAME = "references.bib"

# extractors in the order a reader meets them; anything else (plugins) follows
ORDER = (
    "prep.media", "prep.video", "prosody.frame", "prosody.voice_report", "prosody.standard_sets", "text.diarize", "text.asr", "text.features", "text.syntax",
    "turns.structure", "turns.transitions", "turns.entropy", "coding.layer_a", "coding.judge", "pose.detect", "pose.keypoints",
    "pose.kinematics", "motion.mea", "face.au", "face.au_check", "face.movement",
)
SECTIONS = {
    "prep": "Media", "prosody": "Voice", "text": "Speech", "turns": "Turn-taking", "coding": "Turn-taking",
    "pose": "Body", "motion": "Body", "face": "Face",
}

# writers cite as [key] or [key; key]; render turns that into author-year and collects the keys
_CITATION = re.compile(r"\[([a-z0-9_]+(?:; [a-z0-9_]+)*)\]")


def load_references() -> dict[str, dict]:
    return yaml.safe_load(files("macaw.qc").joinpath("references.yaml").read_text(encoding="utf-8"))


def write_methods(manifest: FeatureManifest, config, out_dir: Path) -> tuple[Path, Path]:
    references = load_references()
    text, used = render(describe(manifest, config), references)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = out_dir / METHODS_NAME
    methods.write_text(text, encoding="utf-8")
    bib = out_dir / BIB_NAME
    bib.write_text("\n".join(to_bibtex(key, references[key]) for key in used), encoding="utf-8")
    return methods, bib


def describe(manifest: FeatureManifest, config) -> dict[str, list[str]]:
    by_extractor: dict[str, list[FeatureSpec]] = {}
    for spec in manifest.specs.values():
        by_extractor.setdefault(spec.extractor, []).append(spec)
    names = [name for name in ORDER if name in by_extractor] + sorted(set(by_extractor) - set(ORDER))
    sections: dict[str, list[str]] = {}
    for name in names:
        specs = by_extractor[name]
        text = _WRITERS.get(name, _plugin)(config, specs[0], {spec.name for spec in specs})
        if not text:
            continue
        sections.setdefault(SECTIONS.get(name.split(".")[0], "Other features"), []).append(text)
    return sections


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
    venue = entry.get("journal") or entry.get("booktitle") or entry.get("howpublished")
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
        "at the first audio sample of its file."
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
    return (
        f"The shared microphone was diarized with pyannote.audio {spec.backend_version} [bredin2023] using the "
        f"{spec.params.get('model')} pipeline, with the number of speakers set to the number of participants. "
        + _VOICE_NAMING[config.text.voice_review]
    )


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
        text += f" Each utterance was embedded with the {model} sentence-transformers model [reimers2019]."
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


def _judge(config, spec, names) -> str:
    params = spec.params
    model = spec.backend_version or params.get("model")
    text = (
        f"Each {params.get('codebook_unit', 'unit')} was coded with the codebook \"{params.get('codebook')}\" by the language model "
        f"{model} ({params.get('provider')}), given the codebook's definitions and examples, the speaker letters and timing, "
        f"and the {params.get('context_units')} preceding units as context."
    )
    samples = params.get("n_samples", 1)
    if samples > 1:
        text += (
            f" Each unit was coded {_count(samples)} times at temperature {params.get('temperature')}; the label is the "
            f"majority answer and its share the confidence. Units below {params.get('review_threshold')}, ties and units "
            "the model could not code were reviewed by a person."
        )
    return text


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
    )


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
    "motion.mea": _mea,
    "face.au": _face,
    "face.au_check": _au_check,
    "face.movement": _movement,
}
