from __future__ import annotations

import sys

import pytest

from macaw.execution.backends.base import JobSpec
from macaw.execution.backends.local import LocalBackend
from macaw.execution.job_db import JobDB, JobRecord

EXECUTION = "exec-1"


def _command(marker, *, code=0):
    body = (
        "import pathlib, time, sys; "
        f"pathlib.Path(r'{marker}').write_text(repr(time.time())); "
        f"sys.exit({code})"
    )
    return [sys.executable, "-c", body]


@pytest.fixture
def database(tmp_path):
    db = JobDB(tmp_path / "jobs")
    db.initialise()
    return db


@pytest.fixture
def make_spec(tmp_path, database):
    def build(task_id, *, depends_on=(), code=0):
        marker = tmp_path / "markers" / f"{task_id.replace(':', '_')}.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        recording_id, extractor = task_id.split(":", 1)
        database.upsert(
            JobRecord(
                task_id=task_id,
                recording_id=recording_id,
                extractor=extractor,
                execution_id=EXECUTION,
                status="pending",
            )
        )
        return JobSpec(
            task_id=task_id,
            recording_id=recording_id,
            extractor=extractor,
            command=_command(marker, code=code),
            resources=None,
            log_path=tmp_path / "logs" / f"{task_id.replace(':', '_')}.log",
            depends_on=list(depends_on),
        ), marker

    return build


def test_a_task_runs_with_the_environment_its_spec_adds(tmp_path, database, make_spec):
    spec, marker = make_spec("r1:text.asr")
    spec.command = [sys.executable, "-c", f"import os, pathlib; pathlib.Path(r'{marker}').write_text(os.environ.get('HF_HUB_OFFLINE', 'unset'))"]
    spec.env = {"HF_HUB_OFFLINE": "1"}
    _run(database, [spec])
    assert marker.read_text() == "1"


def _run(database, specs, *, n_workers=4):
    backend = LocalBackend(n_workers=n_workers, job_db=database)
    backend.wait(backend.submit(specs))
    return backend


def _finished_at(marker):
    return float(marker.read_text(encoding="utf-8"))


def test_a_task_that_found_its_result_in_the_cache_is_counted_as_cached(tmp_path, database, make_spec):
    spec, _ = make_spec("r1:prep.media")
    spec.command = [
        sys.executable, "-c",
        f"from macaw.execution.job_db import JobDB; JobDB(r'{tmp_path / 'jobs'}').set_status('r1:prep.media', 'cached')",
    ]
    _run(database, [spec])
    assert database.get("r1:prep.media").status == "cached"


class TestRunning:
    def test_a_task_runs_and_is_marked_done(self, database, make_spec):
        spec, marker = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert marker.is_file()
        assert database.get("rec1:prep.media").status == "done"

    def test_the_exit_code_is_recorded(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media", code=3)
        _run(database, [spec])
        record = database.get("rec1:prep.media")
        assert (record.status, record.exit_code) == ("failed", 3)

    def test_a_log_file_records_the_command(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert "-c" in spec.log_path.read_text(encoding="utf-8")

    def test_a_duration_is_recorded(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media")
        _run(database, [spec])
        assert database.get("rec1:prep.media").duration_s >= 0.0

    def test_independent_tasks_all_run(self, database, make_spec):
        specs = [make_spec(f"rec{index}:prep.media") for index in range(4)]
        _run(database, [spec for spec, _ in specs])
        assert all(marker.is_file() for _, marker in specs)


class TestDependencies:
    def test_a_dependency_runs_first(self, database, make_spec):
        first, first_marker = make_spec("rec1:prep.media")
        second, second_marker = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        _run(database, [second, first])
        assert _finished_at(first_marker) < _finished_at(second_marker)

    def test_a_chain_does_not_deadlock_with_one_worker(self, database, make_spec):
        """Each worker waiting on its own predecessors is what would deadlock here; the
        dispatch loop exists to keep that from happening."""
        a, a_marker = make_spec("rec1:a")
        b, b_marker = make_spec("rec1:b", depends_on=["rec1:a"])
        c, c_marker = make_spec("rec1:c", depends_on=["rec1:b"])
        _run(database, [c, b, a], n_workers=1)
        assert _finished_at(a_marker) < _finished_at(b_marker) < _finished_at(c_marker)

    def test_a_chain_finishes_with_fewer_workers_than_tasks(self, database, make_spec):
        specs = [make_spec("rec1:a")]
        specs.append(make_spec("rec1:b", depends_on=["rec1:a"]))
        specs.append(make_spec("rec1:c", depends_on=["rec1:a"]))
        _run(database, [spec for spec, _ in specs], n_workers=2)
        assert all(database.get(spec.task_id).status == "done" for spec, _ in specs)


class TestFailure:
    def test_a_downstream_task_is_cancelled_not_left_hanging(self, database, make_spec):
        """Without this the dispatch loop would spin forever waiting on a predecessor that
        is never going to arrive."""
        failing, _ = make_spec("rec1:prep.media", code=1)
        downstream, downstream_marker = make_spec(
            "rec1:prosody.frame", depends_on=["rec1:prep.media"]
        )
        _run(database, [failing, downstream])
        assert database.get("rec1:prep.media").status == "failed"
        assert database.get("rec1:prosody.frame").status == "cancelled"
        assert not downstream_marker.is_file()

    def test_a_task_whose_bookkeeping_breaks_counts_as_failed_and_the_rest_still_run(self, database, make_spec, monkeypatch):
        broken, _ = make_spec("rec1:a")
        downstream, downstream_marker = make_spec("rec1:b", depends_on=["rec1:a"])
        other, other_marker = make_spec("rec2:a")
        backend = LocalBackend(n_workers=1, job_db=database)
        run_one = backend._run_one

        def breaks(spec):
            if spec.task_id == "rec1:a":
                raise RuntimeError("the record could not be written")
            return run_one(spec)

        monkeypatch.setattr(backend, "_run_one", breaks)
        backend.wait(backend.submit([broken, downstream, other]))
        assert database.get("rec1:a").status == "failed" and "could not be written" in database.get("rec1:a").message
        assert database.get("rec1:b").status == "cancelled" and not downstream_marker.is_file()
        assert database.get("rec2:a").status == "done" and other_marker.is_file()

    def test_a_cancelled_task_says_why(self, database, make_spec):
        failing, _ = make_spec("rec1:prep.media", code=1)
        downstream, _ = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        _run(database, [failing, downstream])
        assert "upstream" in database.get("rec1:prosody.frame").message

    def test_another_recording_is_not_dragged_down(self, database, make_spec):
        """Recordings are independent, so one failure must not stop the other 30."""
        failing, _ = make_spec("rec1:prep.media", code=1)
        blocked, _ = make_spec("rec1:prosody.frame", depends_on=["rec1:prep.media"])
        healthy, healthy_marker = make_spec("rec2:prep.media")
        _run(database, [failing, blocked, healthy])
        assert healthy_marker.is_file()
        assert database.get("rec2:prep.media").status == "done"

    def test_the_failure_message_points_at_the_log(self, database, make_spec):
        spec, _ = make_spec("rec1:prep.media", code=1)
        _run(database, [spec])
        assert str(spec.log_path) in database.get("rec1:prep.media").message


class TestStatus:
    def test_status_reports_each_task(self, database, make_spec):
        ok, _ = make_spec("rec1:a")
        bad, _ = make_spec("rec2:a", code=1)
        backend = _run(database, [ok, bad])
        assert backend.status(["rec1:a", "rec2:a"]) == {"rec1:a": "done", "rec2:a": "failed"}

    def test_an_unknown_task_reads_as_pending(self, database, make_spec):
        spec, _ = make_spec("rec1:a")
        backend = _run(database, [spec])
        assert backend.status(["rec9:a"]) == {"rec9:a": "pending"}


def _timed(spec, marker, seconds=0.6):
    """The task writes when it started and ended and which card it was shown."""
    spec.command = [sys.executable, "-c", (
        "import os, pathlib, time; start = time.time(); time.sleep(" + str(seconds) + "); "
        f"pathlib.Path(r'{marker}').write_text(f\"{{start}} {{time.time()}} {{os.environ.get('CUDA_VISIBLE_DEVICES', '-')}}\")"
    )]
    return spec


def _spans(markers):
    found = []
    for marker in markers:
        start, end, card = marker.read_text().split()
        found.append((float(start), float(end), card))
    return found


def _most_at_once(spans):
    return max(sum(start <= moment < end for start, end, _ in spans) for moment, _, _ in spans)


def test_steps_that_use_a_card_run_one_per_card_each_on_its_own(database, make_spec, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    built = [make_spec(f"r{i}:pose.detect") for i in range(4)]
    specs = [_timed(spec, marker) for spec, marker in built]
    for spec in specs:
        spec.gpu = True
    backend = LocalBackend(n_workers=4, job_db=database, cards=["0", "1"])
    backend.wait(backend.submit(specs))
    spans = _spans(marker for _, marker in built)
    assert _most_at_once(spans) == 2
    assert {card for _, _, card in spans} == {"0", "1"}
    for start, end, card in spans:
        overlapping = [other for s, e, other in spans if (s, e, other) != (start, end, card) and s < end and start < e]
        assert card not in overlapping


def test_with_one_card_other_steps_run_beside_the_one_that_uses_it(database, make_spec, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu = [make_spec(f"r{i}:face.au") for i in range(2)]
    cpu = [make_spec(f"r{i}:motion.mea") for i in range(2)]
    specs = [_timed(spec, marker) for spec, marker in gpu + cpu]
    for spec in specs[:2]:
        spec.gpu = True
    backend = LocalBackend(n_workers=4, job_db=database, cards=["0"])
    backend.wait(backend.submit(specs))
    on_card = _spans(marker for _, marker in gpu)
    assert _most_at_once(on_card) == 1 and {card for _, _, card in on_card} == {"0"}
    assert _most_at_once(_spans(marker for _, marker in gpu + cpu)) == 3
    assert {card for _, _, card in _spans(marker for _, marker in cpu)} == {"-"}


def test_without_a_card_or_a_number_set_nothing_is_held_back(database, make_spec, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    built = [make_spec(f"r{i}:pose.detect") for i in range(3)]
    specs = [_timed(spec, marker) for spec, marker in built]
    for spec in specs:
        spec.gpu = True
    backend = LocalBackend(n_workers=4, job_db=database, cards=[])
    backend.wait(backend.submit(specs))
    spans = _spans(marker for _, marker in built)
    assert _most_at_once(spans) == 3 and {card for _, _, card in spans} == {"-"}


def test_the_cards_are_those_named_or_those_found(monkeypatch):
    from types import SimpleNamespace

    from macaw.execution.backends import local

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1, 3")
    assert local.visible_cards() == ["1", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert local.visible_cards() == []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    listing = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-a)\nGPU 1: NVIDIA A100 (UUID: GPU-b)\n"
    monkeypatch.setattr(local.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=listing))
    assert local.visible_cards() == ["0", "1"]

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(local.subprocess, "run", missing)
    assert local.visible_cards() == []
    assert local.gpu_limit(None, ["0", "1"]) == 2 and local.gpu_limit(None, []) is None and local.gpu_limit(3, []) == 3


def test_the_plan_says_how_many_steps_share_the_cards():
    from macaw.execution.backends.local import gpu_notice

    assert gpu_notice(["pose.detect", "face.au"], 4, None, ["0", "1"]) == (
        "pose.detect, face.au use a graphics card: 2 at a time (execution.gpu_workers), on cards 0, 1; CUDA_VISIBLE_DEVICES picks the cards"
    )
    assert "1 at a time" in gpu_notice(["pose.detect"], 4, None, ["0"]) and "on card 0;" in gpu_notice(["pose.detect"], 4, None, ["0"])
    assert gpu_notice(["pose.detect"], 4, None, []) == "no NVIDIA card was found, so pose.detect run side by side like any other step"
    assert gpu_notice(["pose.detect"], 1, None, ["0"]) is None and gpu_notice([], 4, None, ["0"]) is None
