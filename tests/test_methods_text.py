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


EVERYTHING = manifest(
    spec("prep.media", "n_media"),
    spec("prosody.frame", "f0_hz", "parselmouth", "0.4.7"), spec("prosody.frame", "hnr_db", "parselmouth", "0.4.7"),
    spec("prosody.frame", "f1_hz", "parselmouth", "0.4.7"),
    spec("prosody.voice_report", "jitter_local", "parselmouth", "0.4.7", praat_version="6.1.38"),
    spec("prosody.standard_sets", "egemaps_x", "opensmile", "2.6.0"), spec("prosody.standard_sets", "compare_x", "opensmile", "2.6.0"),
    spec("text.asr", "n_words", "whisperx", "3.4"),
    spec("text.features", "mlu_words", embedding_model="all-MiniLM-L6-v2"),
    spec("turns.structure", "n_turns"), spec("turns.transitions", "fto_s"), spec("turns.entropy", "speaker_entropy_bits"),
    spec("coding.layer_a", "n_turn_internal_pauses"),
    spec("pose.detect", "n_videos_read", "yolo", "8.4"), spec("pose.keypoints", "detection_rate", "yolo", "8.4"),
    spec("pose.kinematics", "hand_distance"), spec("motion.mea", "motion_energy"),
    spec("face.au", "au01", "pyfeat", "2.1.5", model="v1"), spec("face.au", "face_yaw_deg", "pyfeat", "2.1.5", model="v1"),
    spec("face.au", "gaze_yaw_deg", "pyfeat", "2.1.5", model="v1"),
    spec("face.au_check", "au12_agreement", "pyfeat", "2.1.5", model="v1", check_model="v2"),
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


def test_write_methods_writes_the_text_and_only_the_cited_entries(tmp_path):
    methods, bib = write_methods(manifest(spec("turns.transitions", "fto_s")), config(), tmp_path)
    assert "Heldner & Edlund, 2010" in methods.read_text(encoding="utf-8")
    assert bib.read_text(encoding="utf-8").count("@") == 1
