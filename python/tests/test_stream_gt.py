"""The follow-up stage that answers the stream queries (``stream_gt.py``).

What is checked, each on the files that the stage writes:

* the answers made afterwards are byte for byte the answers that one build with the stage
  makes (and the golden hashes of ``test_golden.py`` pin those to the previous algorithm);
* a pair that has its answers is skipped, new pools are added without redoing the old ones,
  and an interrupted stage continues without changing a byte of the result;
* an answer made for another ``stream_k`` is refused and nothing is touched;
* families made before the stage existed are opened as they are.
"""

import copy
import dataclasses
import shutil
from collections import namedtuple

import numpy as np
import pytest

from semantic_vdb import binio, manifest, stream_gt, workload
from test_end_to_end import (
    build_base,
    dataset,  # noqa: F401  (pytest fixture)
    eval_only_groundtruth,
    named,
    read_all,
    workload_manifest,
)
from test_extend import make_config, same_family, tree_state, with_pools
from test_groundtruth import needs_gpu


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def stage_on(cfg, **changes):
    """The configuration with the stream answering stage switched on."""
    groundtruth = dataclasses.replace(cfg.groundtruth, stream_groundtruth=True, **changes)
    return dataclasses.replace(cfg, groundtruth=groundtruth)


def later_config(base_set, query_path, **more):
    """A configuration whose build leaves the stream queries without answers."""
    return make_config(base_set, query_path, groundtruth=eval_only_groundtruth(), **more)


def extend_with_stage(cfg, **changes):
    """Run the same command again: ``existing = extend`` and the stage switched on."""
    return stage_on(dataclasses.replace(cfg, existing="extend"), **changes)


def answer_files(family):
    """{relative path: bytes} of every stream answer file of a family."""
    return {
        path.relative_to(family).as_posix(): path.read_bytes()
        for path in sorted(family.glob("*/*/*/groundtruth/stream_*/step_*.bin"))
    }


def every_record_is_true(family):
    """Each stream a workload.json calls complete has all its answer files, as recorded."""
    for manifest_path in sorted(family.rglob("workload.json")):
        set_dir = manifest_path.parent
        info = manifest.read_json(manifest_path)
        for name, record in info["groundtruth"].get("stream", {}).items():
            if not record["complete"]:
                continue
            files = sorted((set_dir / "groundtruth" / name).glob("step_*.bin"))
            assert len(files) == record["steps"], (set_dir, name)
            for path in files:
                recorded = info["files"][path.relative_to(set_dir).as_posix()]
                assert recorded["sha256"] == manifest.sha256_file(path), path
    return True


def workload_hashes_are_in_the_index(family):
    index = manifest.read_json(family / "family.json")
    return all(
        entry["workload_json_sha256"] == manifest.sha256_file(family / entry["path"] / "workload.json")
        for entry in index["sets"]
    )


class ScanCounter:
    """Counts the universe scans of the stage and the vectors they were asked for."""

    def __init__(self, monkeypatch):
        self.calls = 0
        self.vectors = 0
        original = stream_gt.scan_lists

        def counting(queries, *args, **kwargs):
            self.calls += 1
            self.vectors += len(queries)
            return original(queries, *args, **kwargs)

        monkeypatch.setattr(stream_gt, "scan_lists", counting)


# ---------------------------------------------------------------------------
# the answers are the answers of a build
# ---------------------------------------------------------------------------


def test_answers_made_afterwards_equal_the_answers_of_one_build_with_the_stage(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = later_config(base_set, query_path, name="later")
    family = workload.build_workload(start)
    assert answer_files(family) == {}  # the build alone leaves the stream queries unanswered
    assert not (family / "shared" / "stream_lists").exists()

    later = workload.build_workload(extend_with_stage(start))
    once = workload.build_workload(named(make_config(base_set, query_path), "once"))

    assert answer_files(later), "the stage wrote no answers"
    assert answer_files(later) == answer_files(once)
    same_family(later, once)
    assert workload_hashes_are_in_the_index(later)


def test_every_set_records_what_the_stage_did(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(make_config(base_set, dataset["query_path"]))
    for set_dir in sorted(family.glob("o70/g3/u*")):
        info = workload_manifest(set_dir)
        record = info["groundtruth"]["stream"]
        assert sorted(record) == sorted(info["streams"])
        assert info["groundtruth"]["stream_groundtruth"] is True
        assert info["groundtruth"]["stream_k"] == 5
        for name, entry in record.items():
            assert entry["complete"] is True and entry["stream_k"] == 5
            assert entry["steps"] == len(info["schedule"]["inserts_per_step"])
            for path in (set_dir / "groundtruth" / name).glob("step_*.bin"):
                relative = path.relative_to(set_dir).as_posix()
                assert info["files"][relative]["sha256"] == manifest.sha256_file(path)


def test_the_stage_writes_a_report(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(make_config(base_set, dataset["query_path"]))
    report = manifest.read_json(family / "build_report_stream_groundtruth_1.json")
    assert report["stream_k"] == 5 and report["pairs_answered"] > 0
    assert report["vectors_scanned"] > 0 and report["seconds"] >= 0


# ---------------------------------------------------------------------------
# skipping and adding
# ---------------------------------------------------------------------------


def test_a_family_whose_pairs_are_all_answered_is_left_untouched(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"])
    family = workload.build_workload(cfg)
    before = tree_state(family)

    counter = ScanCounter(monkeypatch)
    workload.build_workload(dataclasses.replace(cfg, existing="extend"))
    assert counter.calls == 0
    assert tree_state(family) == before


def test_new_pools_get_answers_and_the_old_ones_are_not_computed_again(dataset, monkeypatch):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    cfg = make_config(base_set, query_path, name="grown")
    family = workload.build_workload(cfg)
    old_answers = answer_files(family)

    counter = ScanCounter(monkeypatch)
    grown = workload.build_workload(with_pools(cfg, sizes=(3, 6)))
    # The old pool keeps its answer files; the lists of the new pool are scanned, but not the
    # vectors that the pool of group size 3 already has (they are the first members).
    assert {key: grown.joinpath(key).read_bytes() for key in old_answers} == old_answers
    assert counter.calls == 1 and counter.vectors > 0

    once = workload.build_workload(named(make_config(base_set, query_path, sizes=(3, 6)), "once"))
    same_family(grown, once)
    assert answer_files(grown) == answer_files(once)


def test_naming_pools_answers_only_those_and_the_rest_follows_later(dataset, monkeypatch):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = later_config(base_set, query_path, sizes=(3, 6), name="narrow")
    family = workload.build_workload(start)

    first = stage_on(dataclasses.replace(start, existing="extend"), stream_pools=("o70/g3",))
    workload.build_workload(first)
    assert answer_files(family) and all("/g3/" in key for key in answer_files(family))
    assert not list(family.glob("o70/g6/*/groundtruth/stream_*"))
    first_lists = sum(
        manifest.read_json(path / "meta.json")["count"]
        for path in (family / "shared" / "stream_lists").glob("segment_*")
    )

    counter = ScanCounter(monkeypatch)
    workload.build_workload(extend_with_stage(start))  # all pools now
    assert answer_files(family)
    assert any("/g6/" in key for key in answer_files(family))
    assert counter.vectors > 0  # the g6 vectors that g3 did not have

    once = workload.build_workload(named(make_config(base_set, query_path, sizes=(3, 6)), "once"))
    assert answer_files(family) == answer_files(once)
    assert first_lists > 0


def test_a_glob_names_several_pools_and_a_wrong_name_lists_the_pools(dataset):
    base_set = build_base(dataset)
    start = later_config(base_set, dataset["query_path"], sizes=(3, 6), name="globs")
    family = workload.build_workload(start)
    wrong = stage_on(dataclasses.replace(start, existing="extend"), stream_pools=("o99/*",))
    with pytest.raises(stream_gt.StreamGtError, match=r"matches no pool.*o70/g3, o70/g6"):
        workload.build_workload(wrong)
    assert answer_files(family) == {}

    workload.build_workload(
        stage_on(dataclasses.replace(start, existing="extend"), stream_pools=("o70/*",))
    )
    assert any("/g3/" in key for key in answer_files(family))
    assert any("/g6/" in key for key in answer_files(family))


def test_a_glob_may_stand_anywhere_in_a_pool_name(dataset):
    base_set = build_base(dataset)
    start = later_config(base_set, dataset["query_path"], sizes=(3, 6), name="inner_glob")
    family = workload.build_workload(start)
    workload.build_workload(
        stage_on(dataclasses.replace(start, existing="extend"), stream_pools=("o*/g6",))
    )
    keys = list(answer_files(family))
    assert keys and all("/g6/" in key for key in keys)  # g3 was not named


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_answers_made_for_another_stream_k_are_refused_and_nothing_changes(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"])
    family = workload.build_workload(cfg)
    before = tree_state(family)

    counter = ScanCounter(monkeypatch)
    with pytest.raises(stream_gt.StreamGtError, match=r"another stream_k than the configured 7"):
        workload.build_workload(stage_on(dataclasses.replace(cfg, existing="extend"), stream_k=7))
    assert counter.calls == 0
    assert tree_state(family) == before


def test_lists_made_for_another_universe_are_refused(dataset):
    base_set = build_base(dataset)
    start = later_config(base_set, dataset["query_path"], name="mixed")
    family = workload.build_workload(start)
    workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))

    # Pretend the stored lists were made for another list length.
    segment = sorted((family / "shared" / "stream_lists").glob("segment_*"))[0]
    meta = manifest.read_json(segment / "meta.json")
    meta["universe"]["list_length"] += 1
    manifest.write_json(segment / "meta.json", meta)
    for stream_dir in family.glob("o70/*/*/groundtruth/stream_*"):
        shutil.rmtree(stream_dir)  # force the stage to look at the lists again
    for manifest_path in family.glob("o70/*/*/workload.json"):
        info = manifest.read_json(manifest_path)
        info["groundtruth"]["stream"] = {}
        info["groundtruth"]["stream_groundtruth"] = False
        manifest.write_json(manifest_path, info)
    with pytest.raises(stream_gt.StreamGtError, match="list_length: was"):
        workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))


def test_a_damaged_list_file_is_found_by_its_hash(dataset):
    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"])
    family = workload.build_workload(cfg)
    segment = sorted((family / "shared" / "stream_lists").glob("segment_*"))[0]
    path = segment / "ids.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1  # one wrong bit, same size
    path.write_bytes(bytes(data))

    store = stream_gt.ListStore(segment.parent)
    universe = manifest.read_json(segment / "meta.json")["universe"]
    with pytest.raises(stream_gt.StreamGtError, match="does not match its recorded sha256"):
        store.open(universe, workers=1)
    path.unlink()
    with pytest.raises(stream_gt.StreamGtError, match="missing or has another size"):
        stream_gt.ListStore(segment.parent).open(universe, workers=1)


def test_too_little_disk_space_stops_the_stage_before_any_scan(dataset, monkeypatch):
    base_set = build_base(dataset)
    start = later_config(base_set, dataset["query_path"], name="small_disk")
    family = workload.build_workload(start)
    before = tree_state(family)

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(stream_gt.shutil, "disk_usage", lambda path: Usage(100, 99, 1))
    counter = ScanCounter(monkeypatch)
    with pytest.raises(stream_gt.StreamGtError, match="not enough disk space"):
        workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))
    assert counter.calls == 0
    leftover = {key for key in tree_state(family) if key not in before}
    assert not leftover


# ---------------------------------------------------------------------------
# interruption
# ---------------------------------------------------------------------------


class Boom(Exception):
    """An error injected at one point of the stage."""


def inject(monkeypatch, module, name, after_calls):
    """Make ``module.name`` raise ``Boom`` on call number ``after_calls + 1``."""
    original = getattr(module, name)
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) > after_calls:
            raise Boom(name)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, failing)
    return original


FAULTS = {
    # (name in stream_gt that is made to fail, calls allowed before it fails)
    "while a segment of lists is scanned": ("scan_lists", 1),
    "just before a segment is named": ("write_text_durable", 1),
    "while the answers of a pair are written": ("write_groundtruth", 7),
    "between two pairs": ("_answer_pair", 2),
    "after the sets are recorded, before the family index is updated": ("_sync_family_hashes", 1),
}


@pytest.mark.parametrize("where", sorted(FAULTS))
def test_an_interrupted_stage_continues_and_gives_the_same_files(dataset, monkeypatch, where):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    # Small segments, so that several of them are written and an interruption can fall
    # between two of them.
    monkeypatch.setattr(stream_gt, "SEGMENT_VECTORS", 40)
    start = later_config(base_set, query_path, name="interrupted", percents=(0.0, 25.0))
    family = workload.build_workload(start)
    clean = workload.build_workload(
        named(make_config(base_set, query_path, percents=(0.0, 25.0)), "clean")
    )

    name, allowed = FAULTS[where]
    original = inject(monkeypatch, stream_gt, name, allowed)
    with pytest.raises(Boom):
        workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))
    monkeypatch.setattr(stream_gt, name, original)

    # The family is still usable and what its workload.json files call complete is complete.
    assert (family / "family.json").is_file()
    assert every_record_is_true(family)
    workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))

    assert answer_files(family) == answer_files(clean)
    same_family(family, clean)
    assert every_record_is_true(family) and workload_hashes_are_in_the_index(family)
    assert not list(family.glob("**/.tmp_*"))
    assert not list(family.parent.glob("*.extending.lock"))


def test_a_finished_segment_is_not_scanned_again_after_an_interruption(dataset, monkeypatch):
    base_set = build_base(dataset)
    monkeypatch.setattr(stream_gt, "SEGMENT_VECTORS", 40)
    start = later_config(base_set, dataset["query_path"], name="segments", percents=(0.0,))
    family = workload.build_workload(start)

    allowed = 2
    original = inject(monkeypatch, stream_gt, "scan_lists", allowed)
    with pytest.raises(Boom):
        workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))
    monkeypatch.setattr(stream_gt, "scan_lists", original)
    finished = sorted((family / "shared" / "stream_lists").glob("segment_*"))
    assert len(finished) == allowed

    counter = ScanCounter(monkeypatch)
    workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))
    total = sum(
        manifest.read_json(path / "meta.json")["count"]
        for path in (family / "shared" / "stream_lists").glob("segment_*")
    )
    assert counter.vectors == total - sum(
        manifest.read_json(path / "meta.json")["count"] for path in finished
    )


def test_a_leftover_of_an_interrupted_stage_is_not_taken_for_an_answer(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = later_config(base_set, query_path, name="leftover", percents=(0.0,))
    family = workload.build_workload(start)
    set_dir = family / "o70" / "g3" / "u0"
    # What a stage that died between the rename and the record leaves: an answer folder (here
    # with garbage) that no workload.json mentions, and a half-filled .tmp_ folder.
    (set_dir / "groundtruth" / "stream_zipf_0.5").mkdir(parents=True)
    (set_dir / "groundtruth" / "stream_zipf_0.5" / "step_00001.bin").write_bytes(b"garbage")
    (set_dir / "groundtruth" / (".tmp_stream_zipf_0.99")).mkdir()
    (set_dir / "groundtruth" / ".tmp_stream_zipf_0.99" / "step_00001.bin").write_bytes(b"half")

    workload.build_workload(stage_on(dataclasses.replace(start, existing="extend")))
    clean = workload.build_workload(
        named(make_config(base_set, query_path, percents=(0.0,)), "clean")
    )
    assert answer_files(family) == answer_files(clean)
    assert not list(family.glob("**/.tmp_*"))


# ---------------------------------------------------------------------------
# families made before the stage existed
# ---------------------------------------------------------------------------


def age_like_the_previous_version(family, stream_k=5):
    """Rewrite a finished family the way the version before round 00077 stored it.

    The stage settings sit in ``common`` and in the configuration recorded by every set, and
    a set has no per-stream record (only the flag and the list length).
    """
    index = manifest.read_json(family / "family.json")
    index["common"]["config"]["groundtruth"].update(stream_groundtruth=False, stream_k=stream_k)
    for entry in index["sets"]:
        path = family / entry["path"] / "workload.json"
        info = manifest.read_json(path)
        info["config"]["groundtruth"].update(stream_groundtruth=False, stream_k=stream_k)
        info["groundtruth"].pop("stream", None)
        info["groundtruth"].pop("stream_k", None)
        info["groundtruth"].update(
            stream_groundtruth=False, stream_list_length=None, distinct_query_vectors=None,
            pool_rows_with_own_list=None, pool_rows_sharing_a_list=None,
            stream_requests_answered_by_scan=None,
        )
        manifest.write_json(path, info)
        entry["workload_json_sha256"] = manifest.sha256_file(path)
    manifest.write_json(family / "family.json", index)


def test_a_family_of_the_previous_version_without_stream_answers_gets_them(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = later_config(base_set, query_path, name="old")
    family = workload.build_workload(start)
    age_like_the_previous_version(family)

    later = workload.build_workload(extend_with_stage(start))
    once = workload.build_workload(named(make_config(base_set, query_path), "once"))
    assert answer_files(later) == answer_files(once)
    # It also accepts new pools with the old common fingerprint.
    assert workload_hashes_are_in_the_index(later)


def test_a_set_of_the_previous_version_that_has_answers_counts_as_answered(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"], name="old_with_answers")
    family = workload.build_workload(cfg)
    answers = answer_files(family)
    age_like_the_previous_version(family, stream_k=5)
    for entry in manifest.read_json(family / "family.json")["sets"]:
        path = family / entry["path"] / "workload.json"
        info = manifest.read_json(path)
        info["groundtruth"]["stream_groundtruth"] = True  # the old flag, no per-stream record
        manifest.write_json(path, info)
    stream_gt._sync_family_hashes(family)

    counter = ScanCounter(monkeypatch)
    workload.build_workload(dataclasses.replace(cfg, existing="extend"))
    assert counter.calls == 0 and answer_files(family) == answers
    with pytest.raises(stream_gt.StreamGtError, match="another stream_k"):
        workload.build_workload(stage_on(dataclasses.replace(cfg, existing="extend"), stream_k=6))


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------


def test_digests_depend_on_the_bytes_the_dtype_and_the_length():
    rows = np.array([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]], dtype=np.uint8)
    digests = stream_gt.digests_as_array(stream_gt.row_digests(rows))
    assert digests.shape == (3, 2)
    assert (digests[0] == digests[2]).all() and not (digests[0] == digests[1]).all()
    # The same numbers as another dtype, or cut to another length, are another vector.
    as_int8 = stream_gt.digests_as_array(stream_gt.row_digests(rows.astype(np.int8)))
    shorter = stream_gt.digests_as_array(stream_gt.row_digests(rows[:, :3]))
    assert not (as_int8[0] == digests[0]).all() and not (shorter[0] == digests[0]).all()


def test_the_digest_index_finds_positions_and_reports_missing_digests():
    digests = np.array([[5, 1], [3, 9], [7, 7], [3, 4]], dtype=np.uint64)
    index = stream_gt.DigestIndex(digests)
    queries = np.array([[7, 7], [3, 4], [3, 9], [3, 5], [1, 1]], dtype=np.uint64)
    assert index.lookup(queries).tolist() == [2, 3, 1, -1, -1]
    assert stream_gt.DigestIndex(np.empty((0, 2), dtype=np.uint64)).lookup(queries).tolist() == [-1] * 5


def test_the_digest_index_separates_vectors_whose_first_halves_are_equal():
    # Two different vectors whose digests agree in the first 64 bits are possible (2^-64 per
    # pair): the second half has to decide, also when there are more than two of them.
    digests = np.array([[4, 1], [4, 2], [4, 3], [9, 9]], dtype=np.uint64)
    index = stream_gt.DigestIndex(digests)
    queries = np.array([[4, 3], [4, 1], [4, 2], [4, 7]], dtype=np.uint64)
    assert index.lookup(queries).tolist() == [2, 0, 1, -1]


def test_a_missing_set_a_complete_set_and_a_set_of_another_k_are_told_apart():
    def info(stream_flag, records, recorded_k=5):
        groundtruth = {"stream_groundtruth": stream_flag}
        if records is not None:
            groundtruth["stream"] = records
        return {"groundtruth": groundtruth, "config": {"groundtruth": {"stream_k": recorded_k}}}

    done = {"s": {"complete": True, "stream_k": 5}}
    assert stream_gt.stream_state(info(True, done), "s", 5) == "complete"
    assert stream_gt.stream_state(info(True, done), "s", 6) == "mismatch"
    assert stream_gt.stream_state(info(False, {}), "s", 5) == "missing"
    assert stream_gt.stream_state(info(False, {"s": {"complete": False, "stream_k": 5}}), "s", 5) == "missing"
    assert stream_gt.stream_state(info(True, None), "s", 5) == "complete"  # previous version
    assert stream_gt.stream_state(info(True, None, recorded_k=3), "s", 5) == "mismatch"
    assert stream_gt.stream_state(info(False, None), "s", 5) == "missing"


def test_a_list_store_keeps_its_segments_and_gathers_rows_across_them(tmp_path):
    universe = {"list_length": 3, "universe_size": 10}
    store = stream_gt.ListStore(tmp_path / "lists")
    (tmp_path / "lists").mkdir()
    store.open(universe, workers=1)
    assert len(store) == 0 and store.find(np.array([[1, 2]], dtype=np.uint64)).tolist() == [-1]

    def add(first_id, keys):
        scratch = store.directory / (stream_gt.TMP_PREFIX + store.next_segment_name())
        scratch.mkdir()
        count = len(keys)
        ids = np.arange(first_id, first_id + count * 3, dtype=np.int32).reshape(count, 3)
        np.save(scratch / "list_ids.npy", ids)
        np.save(scratch / "list_dists.npy", ids.astype(np.float32) / 2)
        np.save(scratch / "list_queries.npy", np.zeros((count, 2), dtype=np.uint8))
        store.add_segment(np.array(keys, dtype=np.uint64), scratch, universe, None)

    add(0, [[1, 1], [2, 2]])
    add(100, [[3, 3]])
    assert len(store) == 3
    rows = store.find(np.array([[3, 3], [1, 1], [9, 9], [2, 2]], dtype=np.uint64))
    assert rows.tolist() == [2, 0, -1, 1]
    ids, dists = store.gather(rows[[0, 1, 3]])
    assert ids.tolist() == [[100, 101, 102], [0, 1, 2], [3, 4, 5]]
    assert dists.tolist() == [[50.0, 50.5, 51.0], [0.0, 0.5, 1.0], [1.5, 2.0, 2.5]]

    # A new object finds the same segments, and removes what an interrupted stage left.
    (store.directory / (stream_gt.TMP_PREFIX + "segment_000009")).mkdir()
    again = stream_gt.ListStore(store.directory)
    again.open(universe, workers=2)
    assert len(again) == 3 and not list(store.directory.glob(stream_gt.TMP_PREFIX + "*"))
    assert again.find(np.array([[2, 2]], dtype=np.uint64)).tolist() == [1]


# ---------------------------------------------------------------------------
# progress lines
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_durations_are_written_in_the_largest_two_units():
    assert [stream_gt.format_duration(x) for x in (0, 41, 725, 3600, 11_000, 88_000)] == [
        "0s", "41s", "12m 05s", "1h 00m", "3h 03m", "24h 26m",
    ]


def test_progress_prints_at_the_interval_with_speed_and_time_left():
    lines, clock = [], FakeClock()
    progress = stream_gt.Progress(lines.append, interval=60, clock=clock)
    progress.start("lists", 1000, "vectors")
    clock.advance(30)
    progress.update(100)
    assert lines == []  # the interval has not passed
    clock.advance(40)  # 70 s since the start
    progress.update(350)
    assert len(lines) == 1
    # 350 vectors in 70 s = 5 per second; 650 left = 130 s.
    assert "lists 350/1,000 (35.0%)" in lines[0] and "5.0 vectors/s recent" in lines[0]
    assert "eta 2m 10s" in lines[0] and "elapsed 1m 10s" in lines[0]
    clock.advance(10)
    progress.update(400)
    assert len(lines) == 1  # 10 s after the last line: quiet again
    progress.finish("done")
    assert len(lines) == 2 and lines[1].endswith("done")


def test_progress_says_unknown_while_nothing_has_advanced_and_zero_when_done():
    lines, clock = [], FakeClock()
    progress = stream_gt.Progress(lines.append, interval=10, clock=clock)
    progress.start("answers", 4, "pairs")
    clock.advance(20)
    progress.report_if_due("waiting")
    assert "eta unknown" in lines[0] and "0.0 pairs/s recent" in lines[0]
    clock.advance(20)
    progress.update(4)
    assert "(100.0%)" in lines[1] and "eta 0s" in lines[1]


def test_progress_speed_follows_the_recent_window_not_the_whole_run():
    lines, clock = [], FakeClock()
    progress = stream_gt.Progress(lines.append, interval=1, clock=clock)
    progress.start("lists", 100_000, "vectors")
    for _ in range(10):  # fast at first: 100 per second for 10 minutes
        clock.advance(60)
        progress.update(int((clock() - 1000) * 100))
    for _ in range(10):  # then slow: 10 per second for 10 minutes
        clock.advance(60)
        done = 60_000 + int((clock() - 1600) * 10)
        progress.update(done)
    assert "10.0 vectors/s recent" in lines[-1]  # not the 55 per second of the whole run
    assert "55.0 overall" in lines[-1]


def test_the_stage_reports_its_phases(dataset):
    base_set = build_base(dataset)
    lines = []
    cfg = make_config(base_set, dataset["query_path"], name="logged")
    workload.build_workload(cfg, log=lines.append)
    text = "\n".join(lines)
    assert "[stream gt] scope:" in text and "0 already answered" in text
    assert "[stream gt] reading query files" in text
    assert "[stream gt] lists" in text and "[stream gt] answers" in text
    assert "[stream gt] distinct query vectors without a list" in text
    assert "(100.0%)" in text and "done in" in text


# ---------------------------------------------------------------------------
# the GPU memory that the build leaves behind
# ---------------------------------------------------------------------------


def test_the_stage_gives_back_the_gpu_memory_cached_by_the_build(monkeypatch):
    freed = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def free_all_blocks(self):
            freed.append(self.name)

    class FakeCupy:
        @staticmethod
        def get_default_memory_pool():
            return Pool("device")

        @staticmethod
        def get_default_pinned_memory_pool():
            return Pool("pinned")

    monkeypatch.setitem(stream_gt.sys.modules, "cupy", FakeCupy)
    stream_gt.release_device_memory()
    assert freed == ["device", "pinned"]

    monkeypatch.delitem(stream_gt.sys.modules, "cupy")
    stream_gt.release_device_memory()  # no cupy loaded: nothing to give back, no error
    assert freed == ["device", "pinned"]


@needs_gpu
def test_a_gpu_stage_runs_after_a_build_that_left_gigabytes_in_the_memory_pool(dataset, monkeypatch):
    """The real case: SIFT100M's build left 10 GB cached, and GPU 1 then looked busy."""
    import cupy

    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"], name="gpu_after_build")
    cfg = dataclasses.replace(
        cfg,
        groundtruth=dataclasses.replace(cfg.groundtruth, device="gpu", stream_groundtruth=False),
        parallel=dataclasses.replace(cfg.parallel, gpus=(1,), cpu_workers=1, numa_node="none"),
    )
    family = workload.build_workload(cfg)  # the build alone; the stage comes next

    with cupy.cuda.Device(1):
        cached = cupy.zeros(2 * 1024**3, dtype=cupy.uint8)  # 2 GiB, more than the 1 GiB limit
        del cached  # the pool keeps the block: nvidia-smi still shows the GPU as used
        later = stage_on(dataclasses.replace(cfg, existing="extend"))
        # Without giving the memory back the GPU would count as busy and the stage would stop.
        monkeypatch.setattr(stream_gt, "release_device_memory", lambda: None)
        with pytest.raises(Exception, match="GPU 1 is in use"):
            workload.build_workload(later)
        monkeypatch.undo()
        workload.build_workload(later)
    assert answer_files(family)
