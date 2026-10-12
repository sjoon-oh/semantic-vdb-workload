"""Tests for semantic_vdb.resume and for continuing a scan of the universe after an interruption.

How a whole build is interrupted and continued is tested in test_end_to_end.py.
"""

import numpy as np
import pytest

from reference import reference_knn
from semantic_vdb import universe
from semantic_vdb.groundtruth import KnnAccumulator
from semantic_vdb.resume import ResumeError, ResumeState, array_digest, canonical
from test_universe import cpu_plan, write_universe


# ---------------------------------------------------------------------------
# the state directory
# ---------------------------------------------------------------------------


def test_a_value_is_stored_once_and_must_stay_equal(tmp_path):
    state = ResumeState(tmp_path / "state")
    assert state.check_or_set("config", {"sizes": (1, 2), "path": tmp_path}) is False
    # Tuples and paths come back as the JSON data they were stored as.
    assert state.check_or_set("config", {"sizes": [1, 2], "path": str(tmp_path)}) is True
    with pytest.raises(ResumeError, match=r"config differs.*sizes"):
        state.check_or_set("config", {"sizes": (1, 3), "path": tmp_path})


def test_the_difference_names_every_changed_setting(tmp_path):
    state = ResumeState(tmp_path)
    state.check_or_set("fingerprint", {"stream": {"seed": 5, "total": 100}, "name": "a"})
    with pytest.raises(ResumeError) as error:
        state.check_or_set("fingerprint", {"stream": {"seed": 6, "total": 100}, "name": "b", "extra": 1})
    text = str(error.value)
    assert "stream.seed: was 5, now 6" in text and "name: was" in text and "extra" in text
    assert "total" not in text  # unchanged settings are not mentioned


def test_stages_are_marked_and_listed(tmp_path):
    state = ResumeState(tmp_path)
    assert not state.any_progress() and state.completed() == []
    (tmp_path / "data.bin").write_bytes(b"x" * 10)
    state.mark("pool_g200", {"alpha": 0.125}, files=[tmp_path / "data.bin"])
    state.mark("common")
    assert state.done("pool_g200") and state.done("common") and not state.done("mix_u0")
    assert state.info("pool_g200") == {"alpha": 0.125} and state.info("common") == {}
    assert state.completed() == ["common", "pool_g200"] and state.any_progress()
    # A second object on the same directory sees the same state (this is the resumed process).
    assert ResumeState(tmp_path).done("common")


def test_a_directory_is_a_state_only_when_it_has_a_fingerprint(tmp_path):
    assert not ResumeState.exists(tmp_path)
    state = ResumeState(tmp_path)
    assert not ResumeState.exists(tmp_path)  # markers alone do not make it a resumable build
    state.check_or_set("fingerprint", {"a": 1})
    assert ResumeState.exists(tmp_path)


@pytest.mark.parametrize("name", ["", "a/b", "../x", "a b", "é"])
def test_stage_and_value_names_cannot_leave_the_directory(tmp_path, name):
    state = ResumeState(tmp_path)
    with pytest.raises(ValueError):
        state.mark(name)
    with pytest.raises(ValueError):
        state.check_or_set(name, 1)


def test_the_digest_sees_dtype_shape_and_content():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) != array_digest(a.reshape(4, 3))
    assert array_digest(a) != array_digest(a.astype(np.int8))
    changed = a.copy()
    changed[2, 3] += 1
    assert array_digest(a) != array_digest(changed)
    assert array_digest(None) != array_digest(np.empty((0, 4), np.uint8))
    assert array_digest(a, None) != array_digest(a)


def test_canonical_turns_python_values_into_json_data(tmp_path):
    assert canonical({"t": (1, 2), "p": tmp_path, "f": np.float32(0.5), "i": np.int64(3)}) == {
        "t": [1, 2], "p": str(tmp_path), "f": 0.5, "i": 3,
    }


# ---------------------------------------------------------------------------
# continuing a scan of the universe
# ---------------------------------------------------------------------------


def scan(tmp_path, queries, workers=1, group_rows=4, resume=None, k=9, scratch="scratch"):
    return universe.scan_lists(
        queries, k, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", 30, "l2",
        cpu_plan(workers), tmp_path / scratch, "q", query_group_rows=group_rows, resume=resume,
    )


def reference(base, inserts, queries, k=9):
    everything = np.vstack([base, inserts])
    return reference_knn(everything, np.arange(len(everything)), queries, k)


def test_a_scan_interrupted_between_query_groups_computes_only_the_missing_groups(tmp_path, monkeypatch):
    base, inserts = write_universe(tmp_path)
    queries = np.random.default_rng(2).integers(0, 40, size=(11, 6)).astype(np.uint8)  # groups 4 + 4 + 3
    state = ResumeState(tmp_path / "state")

    finished = []
    original_finish = KnnAccumulator.finish

    def dying_third_group(self):
        if len(finished) == 2:
            raise KeyboardInterrupt
        finished.append(1)
        return original_finish(self)

    monkeypatch.setattr(KnnAccumulator, "finish", dying_third_group)
    with pytest.raises(KeyboardInterrupt):
        scan(tmp_path, queries, resume=state)
    markers = sorted(path.name for path in (tmp_path / "scratch" / "q_done").iterdir())
    assert markers == ["0_4.done", "4_8.done"]

    calls = []

    def counting_finish(self):
        calls.append(self.num_queries)
        return original_finish(self)

    monkeypatch.setattr(KnnAccumulator, "finish", counting_finish)
    lists = scan(tmp_path, queries, resume=ResumeState(tmp_path / "state"))
    assert calls == [3]  # only the last group of 3 queries was scanned again

    expected_ids, expected_dists = reference(base, inserts, queries)
    assert np.array_equal(lists.ids.astype(np.int64), expected_ids)
    assert np.array_equal(lists.dists.astype(np.float64), expected_dists)
    # A third call finds every group finished and scans nothing.
    calls.clear()
    again = scan(tmp_path, queries, resume=ResumeState(tmp_path / "state"))
    assert calls == [] and np.array_equal(again.ids, lists.ids)


def test_a_finished_scan_with_several_workers_is_not_repeated(tmp_path):
    base, inserts = write_universe(tmp_path)
    queries = np.random.default_rng(3).integers(0, 40, size=(11, 6)).astype(np.uint8)
    first = scan(tmp_path, queries, workers=2, resume=ResumeState(tmp_path / "state"))
    ids_file = tmp_path / "scratch" / "q_ids.npy"
    written = ids_file.stat().st_mtime_ns
    markers = sorted(path.name for path in (tmp_path / "scratch" / "q_done").iterdir())
    assert len(markers) >= 3  # 11 queries in groups of 4, split over two workers

    second = scan(tmp_path, queries, workers=2, resume=ResumeState(tmp_path / "state"))
    assert ids_file.stat().st_mtime_ns == written  # the list files were not touched again
    assert np.array_equal(first.ids, second.ids) and np.array_equal(first.dists, second.dists)
    expected_ids, _ = reference(base, inserts, queries)
    assert np.array_equal(second.ids.astype(np.int64), expected_ids)


def test_a_scan_of_other_queries_than_the_interrupted_one_is_refused(tmp_path):
    write_universe(tmp_path)
    rng = np.random.default_rng(4)
    queries = rng.integers(0, 40, size=(8, 6)).astype(np.uint8)
    scan(tmp_path, queries, resume=ResumeState(tmp_path / "state"))
    other = queries.copy()
    other[0, 0] ^= 1
    with pytest.raises(ResumeError, match="scan_q"):
        scan(tmp_path, other, resume=ResumeState(tmp_path / "state"))
    with pytest.raises(ResumeError, match="list_length"):
        scan(tmp_path, queries, k=7, resume=ResumeState(tmp_path / "state"))


def test_markers_without_their_rows_are_not_trusted(tmp_path):
    base, inserts = write_universe(tmp_path)
    queries = np.random.default_rng(5).integers(0, 40, size=(9, 6)).astype(np.uint8)
    scan(tmp_path, queries, resume=ResumeState(tmp_path / "state"))
    (tmp_path / "scratch" / "q_ids.npy").unlink()  # the rows are gone, the markers are not

    lists = scan(tmp_path, queries, resume=ResumeState(tmp_path / "state"))
    expected_ids, _ = reference(base, inserts, queries)
    assert np.array_equal(lists.ids.astype(np.int64), expected_ids)


def test_without_a_resume_state_a_scan_leaves_no_markers(tmp_path):
    write_universe(tmp_path)
    queries = np.random.default_rng(6).integers(0, 40, size=(5, 6)).astype(np.uint8)
    scan(tmp_path, queries)
    assert not (tmp_path / "scratch" / "q_done").exists()
