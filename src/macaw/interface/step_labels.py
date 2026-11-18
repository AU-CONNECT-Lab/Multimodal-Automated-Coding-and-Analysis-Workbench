"""Plain names for the steps, as the workbench shows them; the step's own name stays in commands and on hover."""

from __future__ import annotations

LABELS = {
    "prep.video": "Video copies",
    "prep.media": "Sound and timing",
    "prep.depth": "Depth copies",
    "align.review": "Check: file offsets",
    "pose.detect": "Find people",
    "pose.keypoints": "Body points",
    "pose.review": "Check: who is who",
    "pose.kinematics": "Body movement",
    "pose.kinematics3d": "Body movement, 3D",
    "pose.listening": "Movement while listening",
    "pose.moving": "Hands moving",
    "pose.gestures": "Gestures",
    "pose.gestures3d": "Gestures, 3D",
    "pose.triangulate": "Body in 3D",
    "pose.depth": "Body depth",
    "pose.mesh": "Body model",
    "pose.one_camera": "Body in 3D, one camera",
    "face.au": "Facial expressions",
    "face.au_check": "Expressions, second model",
    "face.movement": "Face movement",
    "face.shares": "Expression shares",
    "face.speaking": "Speaking faces",
    "gaze.target": "Where people look",
    "objects.track": "Things followed",
    "motion.mea": "Motion energy",
    "text.diarize": "Separate voices",
    "text.review": "Check: whose voice",
    "text.lines": "Check: who speaks each line",
    "text.shared_asr": "Speech, shared sound",
    "text.asr": "Transcript",
    "text.features": "Word measures",
    "text.syntax": "Sentence structure",
    "turns.structure": "Turns",
    "turns.transitions": "Gaps between turns",
    "turns.entropy": "Turn balance",
    "prosody.frame": "Pitch and loudness",
    "prosody.voice_report": "Voice quality",
    "prosody.standard_sets": "Standard voice sets",
    "prosody.turn_pairs": "Voice across turns",
    "coding.layer_a": "Talk coding",
    "coding.judge": "Speech acts by the model",
    "segments.summary": "Segment summaries",
}
# a version under coding.judges or coding.ratings is named after its version
PREFIXES = {"coding.judge.": "Speech acts, ", "coding.rate.": "Ratings, "}
# shorter still, under the group a column stands in on the Overview grid
SHORT = {
    "prep.video": "Video", "prep.media": "Sound", "prep.depth": "Depth", "align.review": "Offsets",
    "pose.detect": "Find people", "pose.keypoints": "Points", "pose.review": "Who is who", "pose.kinematics": "Movement",
    "pose.kinematics3d": "Movement 3D", "pose.listening": "Listening", "pose.moving": "Hands moving", "pose.gestures": "Gestures", "pose.gestures3d": "Gestures 3D",
    "pose.triangulate": "3D", "pose.depth": "Depth", "pose.mesh": "Model", "pose.one_camera": "3D, one camera",
    "face.au": "Expressions", "face.au_check": "2nd model", "face.movement": "Movement", "face.shares": "Shares", "face.speaking": "Speaking",
    "gaze.target": "Who looks where", "objects.track": "Things",
    "motion.mea": "Motion energy",
    "text.diarize": "Voices", "text.review": "Whose voice", "text.lines": "Line speakers", "text.shared_asr": "Shared sound",
    "text.asr": "Transcript", "text.features": "Words", "text.syntax": "Sentences",
    "turns.structure": "Turns", "turns.transitions": "Gaps", "turns.entropy": "Balance",
    "prosody.frame": "Pitch", "prosody.voice_report": "Quality", "prosody.standard_sets": "Standard sets", "prosody.turn_pairs": "Across turns",
    "coding.layer_a": "Talk coding", "coding.judge": "Speech acts", "segments.summary": "Summaries",
}
SHORT_PREFIXES = {"coding.judge.": "Acts, ", "coding.rate.": "Ratings, "}


def label(step: str) -> str:
    if step in LABELS:
        return LABELS[step]
    for prefix, name in PREFIXES.items():
        if step.startswith(prefix):
            return name + step[len(prefix):]
    return step


def short(step: str) -> str:
    if step in SHORT:
        return SHORT[step]
    for prefix, name in SHORT_PREFIXES.items():
        if step.startswith(prefix):
            return name + step[len(prefix):]
    return label(step)
