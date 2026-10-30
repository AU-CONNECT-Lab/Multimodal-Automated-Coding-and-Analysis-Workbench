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


def test_measures_defined_for_the_study_are_listed_only_when_there_are_some():
    ran = manifest(spec("pose.kinematics", "hand_distance"))
    defined = config(kinematics={"measures": [
        {"name": "hand_to_head", "person": "parent", "joints": ["left_wrist"], "to_person": "child", "to_joints": ["nose"]},
    ]})
    text, _ = render(describe(ran, defined), load_references())
    assert "hand_to_head: Distance from parent's left_wrist to child's nose" in text
    text, _ = render(describe(ran, config()), load_references())
    assert "defined for this study" not in text
