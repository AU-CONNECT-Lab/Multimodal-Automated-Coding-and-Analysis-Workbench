from macaw.config.schema import MacawConfig
from macaw.execution.dag import build_dag


def _config(diarization="pyannote", voice_review="flagged"):
    return MacawConfig.model_validate({
        "project": {"name": "t", "raw_dir": "raw", "output_dir": "out", "n_subjects": 2, "roles": ["parent", "child"]},
        "text": {"diarization": diarization, "voice_review": voice_review},
    })


def test_mouth_movement_joins_a_run_that_reads_faces_and_names_voices_on_shared_sound():
    steps = set(build_dag(_config(), ["text.asr", "face.au"]).nodes)
    assert {"text.review", "face.au", "face.movement"} <= steps
    order = build_dag(_config(), ["text.asr", "face.au"]).topological_order()
    assert order.index("face.movement") < order.index("text.review")


def test_it_does_not_bring_the_faces_themselves_nor_join_without_shared_voices():
    assert "face.movement" not in build_dag(_config(), ["text.asr"]).nodes
    assert "face.au" not in build_dag(_config(), ["text.asr"]).nodes
    assert "face.movement" not in build_dag(_config("per_track"), ["text.asr", "face.au"]).nodes
    # the suggestion a person confirms by default reads the mouths too
    assert "face.movement" in build_dag(_config(voice_review="always"), ["text.asr", "face.au"]).nodes
