from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from macaw.coding.llm.judge import JudgementExtractor
from macaw.config.loader import resolve_config, scaffold_project_config
from macaw.extractors.audio.content.asr import TRANSCRIPT_NAME, ASRExtractor, write_transcript
from macaw.extractors.audio.content.diarization import DIARIZATION_NAME, DIARIZATION_SCHEMA, DiarizationExtractor
from macaw.extractors.audio.content.features import TextFeatureExtractor
from macaw.extractors.audio.prosody.frame import FrameProsodyExtractor
from macaw.extractors.prep import PREP_INFO_NAME, MediaPrepExtractor, PrepInfo
from macaw.extractors.video.face.au import AU_CHECK_NAME, FaceAUCheckExtractor, FaceExtractor, au_agreement
from macaw.extractors.video.face.movement import FaceMovementExtractor
from macaw.extractors.video.gaze.target import POINTS_NAME, GazeTargetExtractor
from macaw.extractors.video.pose import keypoints as keypoints_module
from macaw.extractors.video.pose.keypoints import DETECTIONS_NAME, SWAPS_NAME, PoseDetectExtractor, PoseExtractor
from macaw.extractors.video.pose.track_repair import STANDARD_JOINTS
from macaw.project.layout import layout_for
from macaw.project.recording import build_recording
from macaw.project.segments import Segment
from macaw.store import _io
from macaw.store.annotations import build_annotations, write_annotations
from macaw.store.keypoints import write_keypoints
from macaw.store.longtable import build_long_rows, concat_long_tables, write_long_table
from macaw.store.schema import KEYPOINT_SCHEMA, Scale, coerce_frame
from setup_helpers import mics, put, recording_of, video_file, write_setup

from conftest import SAMPLE_RATE

RID = "003"


@pytest.fixture
def project(tmp_path):
    for who in ("parent", "child"):
        put(tmp_path / "raw", RID, f"{who}_mic", "a.wav")
    path = tmp_path / "project.yaml"
    scaffold_project_config(path, name="demo", raw_dir=tmp_path / "raw", output_dir=tmp_path / "out")
    write_setup(path, microphones=mics())

    def config(*overrides):
        resolved = resolve_config(path, cli_overrides=list(overrides))
        return SimpleNamespace(config=resolved, layout=layout_for(resolved), recording=build_recording(RID, resolved))

    return config


def values(rows):
    rows = rows[rows["scale"] == Scale.RECORDING.value]
    return {
        (row.feature, row.subject_id if isinstance(row.subject_id, str) else None,
         row.subject_id_b if isinstance(row.subject_id_b, str) else None): row.value
        for row in rows.itertuples()
    }


def same(a, b):
    assert a and a.keys() == b.keys()
    for key, value in a.items():
        assert b[key] == pytest.approx(value, nan_ok=True), key


def frame_rows(feature, subject, times, numbers, *, source, scale=Scale.FRAME, ends=None):
    data = {"t_start": np.asarray(times, dtype=float), feature: np.asarray(numbers, dtype=float)}
    if ends is not None:
        data["t_end"] = np.asarray(ends, dtype=float)
    return build_long_rows(values=pd.DataFrame(data), scale=scale, recording_id=RID, source=source, entities={}, subject_id=subject)


def test_the_models_units_are_counted_inside_and_peoples_decisions_left_out(project):
    p = project()
    step = JudgementExtractor(p.config)
    units = build_annotations(
        recording_id=RID, tier="act", layer=step.layer(), annotator="llm:stub/m", subject_id="parent", entities={},
        intervals=pd.DataFrame({
            "t_start": [1.0, 5.0, 9.0], "t_end": [3.0, 7.0, 11.0], "label": ["ask", "tell", None],
            "confidence": [0.9, 0.5, np.nan], "abstained": [False, False, True], "needs_review": [False, True, True],
        }),
    )
    write_annotations(units, p.layout.annotations_path(RID))
    decided = units.iloc[[1]].assign(annotator="a person", needs_review=False)
    _io.write_file(decided, p.layout.reviewed_codes_path(RID), *_annotation_schema())

    same(values(step._rows(p.recording, units)), values(step.summarise(p.recording, 0.0, 20.0)))
    part = values(step.summarise(p.recording, 4.0, 20.0))
    assert part[("n_units_judged", "parent", None)] == 2
    assert part[("share_abstained", "parent", None)] == 0.5
    assert part[("share_needs_review", "parent", None)] == 1.0


def _annotation_schema():
    from macaw.store.schema import ANNOTATION_SCHEMA

    return ANNOTATION_SCHEMA, "annotations"


def test_voices_and_their_overlap_are_counted_inside(project):
    p = project("text.diarization=pyannote")
    turns = pd.DataFrame({
        "track": "shared_room.wav", "label": ["S0", "S1", "S0", "S1"],
        "t_start": [0.0, 5.0, 0.0, 3.5], "t_end": [3.5, 9.0, 4.0, 9.0], "exclusive": [True, True, False, False],
    })
    folder = p.layout.artifacts_dir(RID, DiarizationExtractor.name)
    _io.write_file(turns, folder / DIARIZATION_NAME, DIARIZATION_SCHEMA, "diarization")
    step = DiarizationExtractor(p.config)
    whole = values(step.summarise(p.recording, 0.0, 20.0))
    assert whole[("n_voices_found", None, None)] == 2
    assert whole[("overlapped_speech_fraction", None, None)] == pytest.approx(0.5 / 9.0)
    part = values(step.summarise(p.recording, 5.0, 20.0))
    assert part[("n_voices_found", None, None)] == 1 and part[("overlapped_speech_fraction", None, None)] == 0.0


def test_pitch_is_summarised_over_the_frames_inside(project, write_wav):
    p = project()
    folder = p.layout.artifacts_dir(RID, MediaPrepExtractor.name)
    prep = PrepInfo(recording_id=RID, t_origin_s=10.0, duration_s=2.0, sample_rate=SAMPLE_RATE,
                    tracks={"parent": "parent.wav", "child": "child.wav"})
    folder.mkdir(parents=True)
    (folder / PREP_INFO_NAME).write_text(json.dumps(prep.to_dict()), encoding="utf-8")
    for who in ("parent", "child"):
        write_wav(folder / f"{who}.wav", np.zeros(2 * SAMPLE_RATE))
    voiced = np.arange(100)
    pitch = frame_rows("f0_hz", "parent", 10.0 + voiced * 0.01, 200.0 + voiced, source=FrameProsodyExtractor.name)
    write_long_table(pitch, p.layout.long_table_path(RID))
    step = FrameProsodyExtractor(p.config)

    every = np.full(201, np.nan)
    every[:100] = 200.0 + voiced
    expected = values(step._summary(p.recording, "parent", every, every.size, (10.0, 12.0)))
    whole = values(step.summarise(p.recording, 10.0, 12.0))
    same(expected, {key: value for key, value in whole.items() if key[1] == "parent"})
    assert whole[("voiced_fraction", "child", None)] == 0.0 and ("f0_median_hz", "child", None) not in whole
    part = values(step.summarise(p.recording, 10.505, 12.0))
    assert part[("voiced_fraction", "parent", None)] == pytest.approx(49 / 150)
    assert part[("f0_median_hz", "parent", None)] == pytest.approx(275.0)

    # under the same floor as the whole recording: the parent is voiced in the first second only
    late = step.segment_warnings(p.recording, Segment("late", 11.2, 12.0))
    assert [(w.check, w.subject_id) for w in late] == [("voiced_fraction", "parent"), ("voiced_fraction", "child")]
    assert late[0].message.startswith("Only 0.0% of frames are voiced in segment late, below the expected 10.0%")
    assert (late[0].t_start, late[0].t_end) == (11.2, 12.0)
    assert [w.subject_id for w in step.segment_warnings(p.recording, Segment("early", 10.0, 11.0))] == ["child"]


def _keypoints(frames, seen):
    rows = [
        {"recording_id": RID, "subject_id": "parent", "source": PoseExtractor.name, "video": "room.mp4", "frame": frame,
         "t": frame / 10.0, "joint": joint, "x": 1.0 if frame in seen else np.nan, "y": 1.0, "z": np.nan,
         "confidence": 0.8, "track_id": 1, "units": "pixel", "metric_scale_valid": None}
        for frame in frames for joint in STANDARD_JOINTS[:3]
    ]
    return coerce_frame(pd.DataFrame(rows), KEYPOINT_SCHEMA, "keypoints")


def test_tracking_is_summarised_over_the_frames_inside(project):
    p = project()
    keypoints = _keypoints(range(100), set(range(40)) | set(range(60, 100)))
    write_keypoints(keypoints, p.layout.keypoints_path(RID))
    folder = p.layout.artifacts_dir(RID, PoseExtractor.name)
    folder.mkdir(parents=True)
    events = [{"t": 2.0, "frame": 20, "track_a": 1, "track_b": 2, "evidence": "", "score": 1.0},
              {"t": 7.0, "frame": 70, "track_a": 1, "track_b": 2, "evidence": "", "score": 1.0}]
    (folder / SWAPS_NAME).write_text(json.dumps({"room.mp4": events}), encoding="utf-8")
    step = PoseExtractor(p.config)

    swaps = {"room.mp4": events}
    same(values(step._rows(p.recording, keypoints, swaps, [])), values(step.summarise(p.recording, 0.0, 10.0)))
    part = values(step.summarise(p.recording, 0.0, 5.0))
    assert part[("n_track_swaps", None, None)] == 1
    assert part[("detection_rate", "parent", None)] == pytest.approx(40 / 50)


def test_a_person_lost_within_a_segment_is_warned_about(project):
    p = project()
    keypoints = _keypoints(range(100), set(range(40)) | set(range(60, 100)))
    write_keypoints(keypoints, p.layout.keypoints_path(RID))
    recording = recording_of([video_file("room.mp4")], recording_id=RID)
    step = PoseExtractor(p.config)
    lost = step.segment_warnings(recording, Segment("gap", 4.0, 6.0))
    said = {(w.subject_id, w.message) for w in lost}
    # the same floor as for the whole recording; the child's body was never found on this camera
    assert ("parent", "detected in only 0% of frames in segment gap") in said
    assert ("child", "not detected at all in segment gap") in said
    assert all((w.check, w.t_start, w.t_end) == (PoseExtractor.name, 4.0, 6.0) for w in lost)
    assert [w.subject_id for w in step.segment_warnings(recording, Segment("seen", 0.0, 4.0))] == ["child"]


def test_detection_is_counted_over_the_frames_inside(project, monkeypatch):
    p = project()
    folder = p.layout.artifacts_dir(RID, PoseDetectExtractor.name)
    folder.mkdir(parents=True)
    frames = np.repeat(np.arange(60), 2)
    pd.DataFrame({"frame": frames, "video": "room.mp4", "t": frames / 10.0}).to_parquet(folder / DETECTIONS_NAME, index=False)
    (folder / "backend.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(keypoints_module, "apply_prep_timing", lambda config, recording: None)
    recording = recording_of([video_file("room.mp4", fps=10.0, offset_s=0.0)], recording_id=RID)
    step = PoseDetectExtractor(p.config)
    assert values(step.summarise(recording, 0.0, 9.9))[("frames_with_anyone", None, None)] == pytest.approx(0.6)
    assert values(step.summarise(recording, 5.0, 9.9))[("frames_with_anyone", None, None)] == pytest.approx(0.2)


def test_blinks_are_counted_over_the_readable_time_inside(project):
    p = project()
    times = np.arange(100) / 10.0
    eyes = np.where(np.arange(100) < 20, np.nan, 0.3)
    rows = concat_long_tables([
        frame_rows("eye_opening", "parent", times, eyes, source=FaceMovementExtractor.name),
        frame_rows("blink_duration_s", "parent", [3.0, 7.0], [0.2, 0.2], ends=[3.2, 7.2], scale=Scale.EVENT,
                   source=FaceMovementExtractor.name),
    ])
    write_long_table(rows, p.layout.long_table_path(RID))
    step = FaceMovementExtractor(p.config)
    same(values(step._summary(p.recording, "parent", 2, 8.0, (0.0, 10.0))), values(step.summarise(p.recording, 0.0, 10.0)))
    part = values(step.summarise(p.recording, 0.0, 5.0))
    assert part[("eyes_measurable_s", "parent", None)] == pytest.approx(3.0)
    assert part[("blinks_per_min", "parent", None)] == pytest.approx(20.0)


def test_two_face_models_are_compared_over_the_bins_inside(project):
    p = project()
    rng = np.random.default_rng(0)
    times = np.arange(300) / 10.0
    signal = np.sin(times)
    write_long_table(frame_rows("au12", "parent", times, signal, source=FaceExtractor.name), p.layout.long_table_path(RID))
    check = pd.DataFrame({"t": times, "au12": signal + rng.normal(0, 0.3, times.size), "subject_id": "parent"})
    folder = p.layout.artifacts_dir(RID, FaceAUCheckExtractor.name)
    folder.mkdir(parents=True)
    check.to_parquet(folder / AU_CHECK_NAME, index=False)
    step = FaceAUCheckExtractor(p.config)

    expected = au_agreement(pd.DataFrame({"t": times, "au12": signal}), check.drop(columns="subject_id"))["au12"]
    whole = values(step.summarise(p.recording, 0.0, 30.0))
    assert whole[("au12_agreement", "parent", None)] == pytest.approx(expected)
    # under ten one-second bins there is too little to correlate
    assert np.isnan(values(step.summarise(p.recording, 0.0, 8.0))[("au12_agreement", "parent", None)])


def test_looks_are_summarised_over_the_frames_inside(project):
    p = project()
    frames = np.arange(100)
    points = pd.concat([
        pd.DataFrame({"frame": frames, "subject_id": "parent", "t": frames / 10.0,
                      "target": np.where(frames < 50, "child", "elsewhere"), "to_child": 0.1}),
        pd.DataFrame({"frame": frames, "subject_id": "child", "t": frames / 10.0,
                      "target": np.where((frames >= 25) & (frames < 75), "parent", "elsewhere"), "to_parent": 0.1}),
    ], ignore_index=True)
    points["target_read"] = points["target"]
    folder = p.layout.artifacts_dir(RID, GazeTargetExtractor.name)
    folder.mkdir(parents=True)
    points.to_parquet(folder / POINTS_NAME, index=False)
    step = GazeTargetExtractor(p.config)

    same(values(step._rows(p.recording, points, ["parent", "child"], [])), values(step.summarise(p.recording, 0.0, 10.0)))
    part = values(step.summarise(p.recording, 0.0, 5.0))
    assert part[("looks_at_share", "parent", "child")] == 1.0
    assert part[("mutual_gaze_share", "child", "parent")] == pytest.approx(0.5)


def test_who_brings_a_word_in_is_decided_inside_the_segment(project):
    p = project("text.exclude_words=none")
    transcript = pd.DataFrame([
        {"recording_id": RID, "subject_id": who, "utterance_index": index, "t_start": start, "t_end": start + 1.0,
         "text": text, "n_words": len(text.split()), "confidence": 0.9, "avg_logprob": np.nan, "no_speech_prob": np.nan,
         "language": "en"}
        for who, index, start, text in [("parent", 0, 1.0, "a tower"), ("child", 0, 5.0, "my tower"), ("parent", 1, 8.0, "the tower")]
    ])
    folder = p.layout.artifacts_dir(RID, ASRExtractor.name)
    folder.mkdir(parents=True)
    write_transcript(transcript, folder / TRANSCRIPT_NAME)
    step = TextFeatureExtractor(p.config)
    whole = values(step.summarise(p.recording, 0.0, 20.0))
    part = values(step.summarise(p.recording, 4.0, 20.0))
    assert whole[("first_use_share", "parent", None)] == 1.0
    assert part[("first_use_share", "child", None)] == 1.0 and part[("n_tokens", "parent", None)] == 2
    assert whole[("mean_lexical_overlap", "child", "parent")] == 0.5 and ("mean_lexical_overlap", "child", "parent") not in part
    assert part[("mean_lexical_overlap", "parent", "child")] == 0.5


def test_syntax_is_parsed_again_from_the_utterances_inside(project):
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_sm"):
        pytest.skip("needs the en_core_web_sm parser")
    from macaw.extractors.audio.content.syntax import SyntaxExtractor

    p = project("text.syntax_model=en_core_web_sm")
    transcript = pd.DataFrame([
        {"recording_id": RID, "subject_id": "parent", "utterance_index": index, "t_start": start, "t_end": start + 1.0,
         "text": text, "n_words": len(text.split()), "confidence": 0.9, "avg_logprob": np.nan, "no_speech_prob": np.nan,
         "language": "en"}
        for index, (start, text) in enumerate([(1.0, "I think that we should go"), (5.0, "the dog runs"), (8.0, "she ate an apple")])
    ])
    folder = p.layout.artifacts_dir(RID, ASRExtractor.name)
    folder.mkdir(parents=True)
    write_transcript(transcript, folder / TRANSCRIPT_NAME)
    step = SyntaxExtractor(p.config)
    assert values(step.summarise(p.recording, 0.0, 20.0))[("n_utterances_parsed", "parent", None)] == 3
    part = values(step.summarise(p.recording, 4.0, 20.0))
    assert part[("n_utterances_parsed", "parent", None)] == 2
    assert part[("subordinate_clauses_per_utterance", "parent", None)] == 0.0
