from __future__ import annotations

import re

import pytest

from macaw.config.schema import MacawConfig
from macaw.project.manifest import FeatureManifest, FeatureSpec
from macaw.qc import boilerplate
from macaw.qc.boilerplate import describe, load_references, render, write_methods
from macaw.store.schema import Scale


def spec(extractor, name, backend=None, version=None, **params):
    return FeatureSpec(
        name=name, description="", unit=None, scale=Scale.EVENT, group=extractor.split(".")[0], extractor=extractor,
        extractor_version="0.1.0", backend=backend, backend_version=version, params=params,
    )


def config(**sections):
    base = {"project": {"name": "t", "raw_dir": "raw", "output_dir": "out"}}
    return MacawConfig.model_validate({**base, **sections})


def manifest(*specs):
    out = FeatureManifest()
    out.extend(list(specs))
    return out


def face_check(spec_, backend):
    spec_.params["backend"] = backend
    return spec_


EVERYTHING = manifest(
    spec("prep.media", "n_media"), spec("prep.video", "n_videos"),
    spec("prosody.frame", "f0_hz", "parselmouth", "0.4.7"), spec("prosody.frame", "hnr_db", "parselmouth", "0.4.7"),
    spec("prosody.frame", "f1_hz", "parselmouth", "0.4.7"),
    spec("prosody.voice_report", "jitter_local", "parselmouth", "0.4.7", praat_version="6.1.38"),
    spec("prosody.standard_sets", "egemaps_x", "opensmile", "2.6.0"), spec("prosody.standard_sets", "compare_x", "opensmile", "2.6.0"),
    spec("text.asr", "n_words", "whisperx", "3.4"),
    spec("text.features", "mlu_words", embedding_model="all-MiniLM-L6-v2"),
    spec("turns.structure", "n_turns"), spec("turns.transitions", "fto_s"), spec("turns.entropy", "speaker_entropy_bits"),
    spec("coding.layer_a", "n_turn_internal_pauses"),
    spec(
        "coding.judge", "n_units_judged", "ollama", "qwen3:14b@abc", codebook="acts", codebook_unit="utterance",
        provider="ollama", n_samples=5, temperature=0.3, review_threshold=0.7, context_units=5,
    ),
    spec("pose.detect", "n_videos_read", "yolo", "8.4"), spec("pose.keypoints", "detection_rate", "yolo", "8.4"),
    spec("pose.kinematics", "hand_distance"), spec("motion.mea", "motion_energy"),
    spec("face.au", "au01", "pyfeat", "2.1.5", model="v1"), spec("face.au", "face_yaw_deg", "pyfeat", "2.1.5", model="v1"),
    spec("face.au", "gaze_yaw_deg", "pyfeat", "2.1.5", model="v1"),
    face_check(spec("face.au_check", "au12_agreement", "pyfeat", "2.1.5", model="v1", check_backend="pyfeat", check_model="v2"), "pyfeat"),
    spec("face.movement", "mouth_opening"),
    spec("my.plugin", "thing", "mine", "1.0"),
)
ALL_ON = config(
    segment={"task_start_s": 30, "task_end_s": 630},
    prosody={"f0_range_by_role": {"child": [150, 600]}},
    text={"language": "en"},
    turns={"window_s": 60, "window_hop_s": 30},
    coding={"backchannel_lexicon": ["mhm", "yeah"]},
    pose={"backend": "yolo", "model": "models/yolo11x-pose.pt"},
)


def test_every_step_renders_and_every_citation_is_in_the_reference_list():
    text, used = render(describe(EVERYTHING, ALL_ON), load_references())
    assert "[" not in re.sub(r"\(.*?\)", "", text.split("## References")[0])
    assert set(used) <= set(load_references())
    assert {"bain2023", "radford2023", "eyben2016", "schuller2016", "reimers2019", "abdelrahman2023", "zhang2022"} <= set(used)
    assert "from 30 s to 630 s" in text and "child 150 to 600 Hz" in text and "yolo11x-pose.pt" in text
    assert "my.plugin 0.1.0 extractor, backend mine 1.0" in text
    assert "v2 models were run on the same faces" in text
    assert "Each utterance was coded with the codebook \"acts\" by the language model qwen3:14b@abc" in text
    assert "coded five times at temperature 0.3" in text


def test_only_what_ran_is_described():
    text, used = render(describe(manifest(spec("turns.structure", "n_turns")), config()), load_references())
    assert "## Turn-taking" in text and "## Voice" not in text and "## Body" not in text
    assert used == []


def test_neighbouring_citations_share_one_pair_of_brackets():
    text, _ = render(describe(manifest(spec("prosody.voice_report", "x", "parselmouth", "0.4.7")), config()), load_references())
    assert "(Boersma & Weenink, n.d.; Jadoul et al., 2018)" in text


def test_references_are_in_apa_order_and_shape():
    references = load_references()
    assert boilerplate.formatted(references["heldner2010"]) == (
        "Heldner, M., & Edlund, J. (2010). Pauses, gaps and overlaps in conversations. Journal of Phonetics, 38(4), 555-568."
    )
    assert boilerplate.formatted(references["lugaresi2019"]).startswith("Lugaresi, C., Tang, J.,")
    assert boilerplate._initials("Chang, Chuo-Ling") == "Chang, C.-L."


@pytest.mark.parametrize("embedding, expected", [(None, False), ("all-MiniLM-L6-v2", True)])
def test_embeddings_are_mentioned_only_when_made(embedding, expected):
    text, _ = render(describe(manifest(spec("text.features", "mlu_words", embedding_model=embedding)), config()), load_references())
    assert ("sentence-transformers" in text) is expected


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, "of the words used by at least two speakers"),
        ({"first_use_exclude": "english_stopwords"}, "excluding English function words, fillers and backchannels, used"),
        ({"first_use_words": "topic.txt", "first_use_min_speakers": 3}, "researcher-supplied list used by at least three"),
    ],
)
def test_the_first_use_rule_that_ran_is_described(params, expected):
    text, _ = render(describe(manifest(spec("text.features", "first_use_share", **params)), config()), load_references())
    assert expected in text


def test_write_methods_writes_the_text_and_only_the_cited_entries(tmp_path):
    methods, bib = write_methods(manifest(spec("turns.transitions", "fto_s")), config(), tmp_path)
    assert "Heldner & Edlund, 2010" in methods.read_text(encoding="utf-8")
    assert bib.read_text(encoding="utf-8").count("@") == 1


def test_a_shared_microphone_is_described_only_when_one_was_diarized():
    own = render(describe(manifest(spec("text.diarize", "n_shared_tracks_diarized"), spec("text.asr", "n_words", "whisperx", "3.8")), config()), load_references())[0]
    assert "diarized" not in own and "own track" in own
    shared = manifest(
        spec("text.diarize", "n_voices_found", "pyannote", "4.0.7", model="pyannote/speaker-diarization-community-1"),
        spec("text.asr", "n_words", "whisperx", "3.8"),
    )
    text, used = render(describe(shared, config(text={"diarization": "pyannote"})), load_references())
    assert "pyannote.audio 4.0.7 (Bredin, 2023)" in text and "community-1 pipeline" in text
    assert "The shared microphone was transcribed" in text and "cross-talk" not in text
    assert "bredin2023" in used
    assert "set to the number of participants" in text


def test_voices_besides_the_participants_are_described_as_one_further_speaker():
    shared = manifest(spec("text.diarize", "n_voices_found", "pyannote", "4.0.7", model="m", extra_voices=1))
    text, _ = render(describe(shared, config(text={"diarization": "pyannote"})), load_references())
    assert "allowing up to 1 voice besides the participants'" in text and "one further speaker" in text
    assert "set to the number of participants" not in text


def test_the_face_backend_that_ran_is_the_one_named():
    specs = manifest(
        spec("face.au", "au12", "openface", "2.2.0"), spec("face.au", "face_yaw_deg", "openface", "2.2.0"),
        face_check(spec("face.au_check", "au12_agreement", "pyfeat", "2.1.5", model=None, check_backend="pyfeat", check_model="v1"), "openface"),
    )
    text, used = render(describe(specs, config(face={"backend": "openface", "check_backend": "pyfeat"})), load_references())
    assert "OpenFace 2.2.0" in text and "Py-Feat" in text.split("As a check")[1]
    assert "Py-Feat (Cheong et al., 2023) was run on the same faces" in text and "agreement with OpenFace" in text
    assert {"baltrusaitis2018", "cheong2023"} <= set(used)


@pytest.mark.parametrize(
    ("video", "expected"),
    [
        ({}, "constant frame rate, the one it was recorded at"),
        ({"fps": 15}, "constant 15 frames per second"),
        ({"normalise": False}, "read as recorded"),
    ],
)
def test_how_the_videos_were_prepared_is_described(video, expected):
    text, _ = render(describe(manifest(spec("prep.video", "n_videos")), config(video=video)), load_references())
    assert expected in text


def test_coding_only_some_people_and_other_voices_as_context_is_said():
    judged = manifest(spec(
        "coding.judge", "n_units_judged", "ollama", "qwen3:14b@abc", codebook="acts", codebook_unit="utterance",
        provider="ollama", n_samples=1, context_units=5, coded_roles=["child"],
    ))
    shared = config(text={"diarization": "pyannote", "extra_voices": 1})
    text, _ = render(describe(judged, shared), load_references())
    assert "Only the units of the child were coded" in text
    assert "Lines in other voices were shown in place as context" in text
    text, _ = render(describe(judged, config()), load_references())
    assert "other voices" not in text


def test_what_the_model_was_shown_and_each_version_are_said():
    common = dict(codebook="acts", codebook_unit="utterance", provider="ollama", n_samples=5, context_units=5, temperature=0.3)
    judged = manifest(
        spec("coding.judge", "n_units_judged", "ollama", "qwen3:8b@abc", show=["timing"], judge=None, **common),
        spec("coding.judge.words", "n_units_judged_words", "ollama", "qwen3:8b@abc", show=[], judge="words", **common),
        spec(
            "coding.judge.voice", "n_units_judged_voice", "ollama", "qwen3:8b@abc", show=["timing", "pitch", "rate"],
            judge="voice", **{**common, "n_samples": 3},
        ),
    )
    text, _ = render(describe(judged, config()), load_references())
    assert "the speaker letter, the words, its start and length, and the pause or overlap before it." in text
    assert "a version named words" in text and "the speaker letter and the words only." in text
    assert "a version named voice" in text and "before it, and its pitch and speech rate relative to the median of that speaker's lines" in text
    assert "lines (answers per unit 3)." in text
    assert text.index("Each utterance was coded") < text.index("a version named voice") < text.index("a version named words")


def test_names_only_notes_examples_reasons_reasoning_and_frames_are_said_and_a_version_says_only_its_own():
    common = dict(codebook="acts", codebook_unit="utterance", provider="ollama", n_samples=1, context_units=5, temperature=0.3)
    told = dict(notes="A is the mother.", n_examples=6, reasons=True, think=False)
    judged = manifest(
        spec("coding.judge", "n_units_judged", "ollama", "qwen3:8b@abc", show=["timing"], judge=None, names_only=True, **told, **common),
        spec(
            "coding.judge.seen", "n_units_judged_seen", "ollama", "qwen3-vl:8b-instruct@abc", show=["timing", "frames"],
            judge="seen", batch_size=1, **told, **common,
        ),
    )
    shared = config(coding={"llm": {"notes": "A is the mother.", "reasons": True, "think": False}, "judges": {"seen": {"show": ["timing", "frames"]}}})
    text, _ = render(describe(judged, shared), load_references())
    assert 'given only the names of the codes and the 5 preceding lines' in text
    assert 'The prompt also held this note from the researchers: "A is the mother."' in text
    assert "It showed six units people had coded in other recordings as examples, none from the recording being coded." in text
    assert "asked to name in a few words what each code rested on" in text and "reasoning before answering was turned off" in text
    assert "each unit to code came with three video frames near its start, middle and end, one unit per request" in text
    # the version shares the note, reasons and reasoning with the main coding, so they are said once
    assert text.count("A is the mother.") == 1 and text.count("turned off") == 1
    assert "five units per recording that the model was sure of were drawn at random for a person to check" in text
    unchecked = config(coding={"llm": {"review_random": 0}})
    assert "drawn at random" not in render(describe(judged, unchecked), load_references())[0]


def test_measures_defined_for_the_study_are_listed_only_when_there_are_some():
    ran = manifest(spec("pose.kinematics", "hand_distance"))
    defined = config(kinematics={"measures": [
        {"name": "hand_to_head", "person": "parent", "joints": ["left_wrist"], "to_person": "child", "to_joints": ["nose"]},
    ]})
    text, _ = render(describe(ran, defined), load_references())
    assert "hand_to_head: Distance from parent's left_wrist to child's nose" in text
    text, _ = render(describe(ran, config()), load_references())
    assert "defined for this study" not in text


def test_gesture_kinematics_name_the_tier_and_both_thresholds_with_their_sources():
    ran = manifest(FeatureSpec(
        name="gesture_left_peak_speed", description="", unit="px/s", scale=Scale.EVENT, group="pose",
        extractor="pose.gestures", extractor_version="0.1.0",
    ))
    gestures = config(kinematics={"gestures": {"tier": "gesture", "still_speed": 12, "peak_prominence": 40}})
    text, used = render(describe(ran, gestures), load_references())
    assert "coded on the gesture tier" in text and "at least 40 px/s" in text and "slower than 12 px/s" in text
    assert "eight cells of the periphery" in text and "both wrists" not in text
    both = config(kinematics={"gestures": {"tier": "gesture", "still_speed": 12, "peak_prominence": 40, "holds": "both_hands"}})
    text, _ = render(describe(ran, both), load_references())
    assert "both wrists and both elbows moved slower than 12 px/s, counted once per gesture" in text
    assert {"trujillo2019", "mcneill1992", "pouw2021", "savitzky1964"} <= set(used)
    assert "University of Chicago Press." in text


def test_voice_hints_are_said_to_feed_the_suggestion_only_when_set():
    shared = manifest(spec("text.diarize", "n_voices_found", "pyannote", "4.0.7", model="m"))
    hinted = config(
        project={"name": "t", "raw_dir": "raw", "output_dir": "out", "roles": ["mother", "child"]},
        text={"diarization": "pyannote"},
        identity={"voice_hints": [{"says": "mommy,", "role": "child"}], "pitch_order": ["mother", "child"]},
    )
    text, _ = render(describe(shared, hinted), load_references())
    assert "phrases set for this study as marking a speaker's role and on the expected order" in text
    assert "at least 40 Hz" in text
    text, _ = render(describe(shared, config(text={"diarization": "pyannote"})), load_references())
    assert "phrases set for this study" not in text


def test_gaze_names_the_model_its_tuning_and_the_things_looked_for_or_the_seats():
    point = spec("gaze.target", "looks_at", "gazelle", "1.0.0", model="gazelle_dinov2_vitl14_inout_childplay", in_frame_cut=0.5)
    text, used = render(describe(manifest(point), config(gaze={"targets": ["bottle", "book"]})), load_references())
    assert {"ryan2025", "tafasca2023", "liu2024"} <= set(used)
    assert "## Gaze" in text and "bottle, book" in text and "at least 0.5" in text and "less than 0.2 s" in text
    every_frame, _ = render(describe(manifest(point), config(gaze={"min_look_s": 0})), load_references())
    assert "less than" not in every_frame and "Each run of frames with the same target" in every_frame
    plain, used = render(describe(manifest(spec("gaze.target", "looks_at", "gazelle", "1.0.0", model="gazelle_dinov2_vitb14_inout", in_frame_cut=0.5)), config()), load_references())
    assert "tafasca2023" not in used and "liu2024" not in used
    seats = {"camera": [0, 0, 100], "parent": [-50, 100, 110]}
    text, used = render(describe(manifest(spec("gaze.target", "looks_at", "angles", "1.0")), config(project={"name": "t", "raw_dir": "raw", "output_dir": "out", "roles": ["parent", "child"]}, gaze={"backend": "angles", "seats": seats, "tolerance_deg": 20})), load_references())
    assert "head pose" in text and "within 20°" in text and "seats" in text and not used


def test_name_suggestions_are_described_only_when_set_and_the_picture_model_only_when_described():
    ran = manifest(spec("pose.keypoints", "detection_rate", "yolo", "8.0"))
    text, _ = render(describe(ran, config()), load_references())
    assert "a researcher marked each participant" in text and "most in view" not in text
    suggested = config(pose={"suggest_names": True})
    text, _ = render(describe(ran, suggested), load_references())
    assert "head motion" in text and "the person most in view and largest" in text and "vision-language model" not in text
    looked = config(pose={"suggest_names": True}, qc={"look": {"model": "qwen3-vl:8b-instruct", "people": {"parent": "an adult"}}})
    text, _ = render(describe(ran, looked), load_references())
    assert "vision-language model (qwen3-vl:8b-instruct)" in text and "up to 5 frames" in text
