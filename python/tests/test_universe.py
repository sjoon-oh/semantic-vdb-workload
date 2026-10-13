"""Tests for semantic_vdb.universe: list length, answers from lists, scans, shared lists."""

from math import comb

import numpy as np
import pytest

from reference import reference_knn
from semantic_vdb import binio, universe
from semantic_vdb.parallel import ParallelPlan


def cpu_plan(workers=1, base_rows=7, chunk_rows=5):
    """A hand-made plan: CPU only, small blocks so that blocking is exercised."""
    return ParallelPlan(
        device="cpu", gpus=(), workers_per_gpu=1, cpu_workers=workers, physical_cores=workers,
        numa_node=None, node_cpus=None, gpu_cpus=(), base_batch_rows=base_rows,
        query_chunk_rows=chunk_rows,
    )


# ---------------------------------------------------------------------------
# how long a list must be
# ---------------------------------------------------------------------------


def exact_tail(trials, needed, p):
    """P[Binomial(trials, p) < needed] from exact binomial coefficients."""
    return sum(comb(trials, j) * p**j * (1 - p) ** (trials - j) for j in range(needed))


@pytest.mark.parametrize("needed,p", [(1, 0.5), (3, 0.5), (3, 0.9), (5, 0.667), (10, 0.8)])
def test_list_length_is_the_shortest_one_with_a_small_enough_tail(needed, p):
    eps = 1e-6
    length = universe.choose_list_length(needed, p, 10_000, failure_probability=eps)
    assert exact_tail(length, needed, p) <= eps
    # One entry less must NOT be enough (otherwise the list is longer than necessary).
    assert length == needed or exact_tail(length - 1, needed, p) > eps


def test_list_length_grows_when_fewer_entries_are_alive():
    lengths = [universe.choose_list_length(10, p, 10**6) for p in (1.0, 0.9, 0.7, 0.5, 0.3)]
    assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)
    assert lengths[0] == 10  # everything alive: the list is the answer


def test_list_length_is_capped_by_the_universe():
    assert universe.choose_list_length(10, 0.5, 12) == 12
    assert universe.choose_list_length(10, 0.001, 500) == 500
    assert universe.choose_list_length(10, 0.5, 10) == 10


def test_list_length_matches_the_planned_values():
    # The values quoted in the plan of round 00067 (alive fractions 0.8, 2/3, 0.526).
    assert universe.choose_list_length(10, 1_000_000 / 1_250_000, 1_250_000) == 32
    assert universe.choose_list_length(10, 1_000_000 / 1_500_000, 1_500_000) == 44
    assert universe.choose_list_length(100, 1_000_000 / 1_900_000, 1_900_000) == 285


@pytest.mark.parametrize("bad", [(0, 0.5, 10), (3, 0.0, 10), (3, 1.5, 10), (3, 0.5, 0)])
def test_list_length_rejects_impossible_arguments(bad):
    with pytest.raises(ValueError):
        universe.choose_list_length(*bad)


# ---------------------------------------------------------------------------
# reading an answer out of a list
# ---------------------------------------------------------------------------


def slow_alive_prefix(ids_row, dists_row, limit, deleted, k):
    kept = [(i, d) for i, d in zip(ids_row, dists_row) if i < limit and not deleted[i]]
    return kept[:k], len(kept) >= k


def test_select_alive_prefix_matches_a_plain_loop():
    rng = np.random.default_rng(0)
    universe_size, length, rows, k = 60, 14, 80, 4
    list_ids = np.array([rng.permutation(universe_size)[:length] for _ in range(rows)], dtype=np.int32)
    list_dists = np.sort(rng.random((rows, length)), axis=1).astype(np.float32)
    deleted = rng.random(universe_size) < 0.5
    limit = 40  # ids >= 40 are not inserted yet

    ids, dists, enough = universe.select_alive_prefix(list_ids, list_dists, limit, deleted, k)
    for row in range(rows):
        expected, expected_enough = slow_alive_prefix(list_ids[row], list_dists[row], limit, deleted, k)
        assert bool(enough[row]) == expected_enough
        if expected_enough:
            assert ids[row].tolist() == [i for i, _ in expected]
            assert dists[row].tolist() == [d for _, d in expected]
    assert 0 < enough.sum() < rows  # the test covers both outcomes


def test_select_alive_prefix_keeps_the_order_in_long_lists():
    # A long list: an unstable sort would reorder equal "not alive" keys once rows are long.
    rng = np.random.default_rng(9)
    length = 300
    list_ids = np.array([rng.permutation(1000)[:length] for _ in range(20)], dtype=np.int32)
    list_dists = np.sort(rng.random((20, length)), axis=1).astype(np.float32)
    deleted = rng.random(1000) < 0.5
    ids, dists, enough = universe.select_alive_prefix(list_ids, list_dists, 800, deleted, 40)
    for row in range(20):
        expected, expected_enough = slow_alive_prefix(list_ids[row], list_dists[row], 800, deleted, 40)
        assert expected_enough and enough[row]
        assert ids[row].tolist() == [i for i, _ in expected]


def test_select_alive_prefix_keeps_the_nearest_first_order():
    ids = np.array([[5, 3, 9, 1, 7]], dtype=np.int32)
    dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32)
    deleted = np.zeros(10, dtype=bool)
    deleted[3] = True
    out_ids, out_dists, enough = universe.select_alive_prefix(ids, dists, 8, deleted, 3)
    # 3 is deleted and 9 is not inserted yet (>= 8): the nearest three alive are 5, 1, 7.
    assert out_ids.tolist() == [[5, 1, 7]] and out_dists.tolist() == [[pytest.approx(0.1), pytest.approx(0.4), pytest.approx(0.5)]]
    assert enough.tolist() == [True]


def test_select_alive_prefix_flags_rows_with_too_few_alive_entries():
    ids = np.array([[0, 1, 2, 3]], dtype=np.int32)
    dists = np.zeros((1, 4), dtype=np.float32)
    deleted = np.array([True, True, False, False])
    _, _, enough = universe.select_alive_prefix(ids, dists, 4, deleted, 3)
    assert enough.tolist() == [False]  # only 2 and 3 are alive
    with pytest.raises(ValueError):
        universe.select_alive_prefix(ids, dists, 4, deleted, 5)  # k longer than the lists


# ---------------------------------------------------------------------------
# scanning the universe
# ---------------------------------------------------------------------------


def write_universe(tmp_path, num_base=70, num_inserts=30, dim=6, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 40, size=(num_base, dim)).astype(np.uint8)  # many ties
    inserts = rng.integers(0, 40, size=(num_inserts, dim)).astype(np.uint8)
    binio.write_array(tmp_path / "base.u8bin", base)
    binio.write_array(tmp_path / "insert_pool.u8bin", inserts)
    return base, inserts


@pytest.mark.parametrize("workers", [1, 2, 3])
@pytest.mark.parametrize("group_rows", [4, 1000])
def test_scan_lists_equals_the_reference_top_k_of_the_whole_universe(tmp_path, workers, group_rows):
    base, inserts = write_universe(tmp_path)
    queries = np.random.default_rng(2).integers(0, 40, size=(11, 6)).astype(np.uint8)
    everything = np.vstack([base, inserts])

    lists = universe.scan_lists(
        queries, 9, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", len(inserts), "l2",
        cpu_plan(workers), tmp_path / "scratch", "q", query_group_rows=group_rows,
    )
    expected_ids, expected_dists = reference_knn(everything, np.arange(len(everything)), queries, 9)
    assert np.array_equal(lists.ids.astype(np.int64), expected_ids)
    assert np.array_equal(lists.dists.astype(np.float64), expected_dists)


def test_scan_lists_can_use_only_a_prefix_of_the_insertions(tmp_path):
    base, inserts = write_universe(tmp_path)
    queries = np.random.default_rng(3).integers(0, 40, size=(5, 6)).astype(np.uint8)
    used = 12
    universe_vectors = np.vstack([base, inserts[:used]])
    lists = universe.scan_lists(
        queries, 6, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", used, "l2",
        cpu_plan(), tmp_path / "scratch", "q",
    )
    expected, _ = reference_knn(universe_vectors, np.arange(len(universe_vectors)), queries, 6)
    assert np.array_equal(lists.ids.astype(np.int64), expected)


def test_scan_lists_without_insertions_scans_the_base_only(tmp_path):
    base, _ = write_universe(tmp_path)
    queries = np.random.default_rng(4).integers(0, 40, size=(5, 6)).astype(np.uint8)
    lists = universe.scan_lists(
        queries, 6, tmp_path / "base.u8bin", None, 0, "l2", cpu_plan(), tmp_path / "scratch", "q"
    )
    expected, _ = reference_knn(base, np.arange(len(base)), queries, 6)
    assert np.array_equal(lists.ids.astype(np.int64), expected)


def test_a_list_answers_every_step_like_a_scan_of_the_active_set(tmp_path):
    """The central claim: filtering the universe list gives the exact top-k of any active set."""
    base, inserts = write_universe(tmp_path, num_base=80, num_inserts=40)
    everything = np.vstack([base, inserts])
    queries = np.random.default_rng(5).integers(0, 40, size=(15, 6)).astype(np.uint8)
    k = 4
    length = universe.choose_list_length(k, 80 / 120, 120, failure_probability=1e-12)
    lists = universe.scan_lists(
        queries, length, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", 40, "l2",
        cpu_plan(), tmp_path / "scratch", "q",
    )
    rng = np.random.default_rng(6)
    for inserted in (0, 10, 40):
        # An active set of constant size: delete as many alive vectors as were inserted.
        deleted = np.zeros(120, dtype=bool)
        deleted[rng.choice(80, size=inserted, replace=False)] = True
        alive_ids = np.flatnonzero(~deleted[:80 + inserted] & (np.arange(80 + inserted) < 80 + inserted))
        ids, dists, enough = universe.select_alive_prefix(
            np.array(lists.ids), np.array(lists.dists), 80 + inserted, deleted, k
        )
        assert enough.all()
        expected_ids, expected_dists = reference_knn(everything[alive_ids], alive_ids, queries, k)
        assert np.array_equal(ids.astype(np.int64), expected_ids)
        assert np.array_equal(dists.astype(np.float64), expected_dists)


def test_the_fallback_scanner_answers_a_moment_of_a_set(tmp_path):
    base, inserts = write_universe(tmp_path)
    everything = np.vstack([base, inserts])
    queries = np.random.default_rng(7).integers(0, 40, size=(8, 6)).astype(np.uint8)
    deleted = np.zeros(100, dtype=bool)
    deleted[[3, 17, 70, 75]] = True  # two base vectors and two inserted ones
    scanner = universe.ActiveSetScanner(
        tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", "l2", cpu_plan()
    )
    try:
        result = scanner.top(queries, 5, 20, deleted)  # 20 insertions have happened
    finally:
        scanner.close()
    alive = np.array([i for i in range(90) if not deleted[i]])
    expected, _ = reference_knn(everything[alive], alive, queries, 5)
    assert np.array_equal(result.neighbour_ids.astype(np.int64), expected)


def test_split_ranges_cover_everything_without_overlap():
    from semantic_vdb.parallel import split_ranges

    assert split_ranges(10, 3) == [(0, 3), (3, 6), (6, 10)]
    assert split_ranges(2, 5) == [(0, 1), (1, 2)]
    assert split_ranges(0, 4) == []
    covered = [i for a, b in split_ranges(101, 7) for i in range(a, b)]
    assert covered == list(range(101))


# ---------------------------------------------------------------------------
# on a real GPU (skipped when no free GPU is available)
# ---------------------------------------------------------------------------


def cpu_reference_lists(tmp_path, queries, length):
    return universe.scan_lists(
        queries, length, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", 30, "l2",
        cpu_plan(), tmp_path / "scratch_cpu", "q",
    )


@pytest.mark.parametrize("workers_per_gpu", [1, 2])
def test_gpu_lists_equal_the_cpu_lists(tmp_path, workers_per_gpu):
    from gpu_helpers import gpu_plan_or_skip

    plan = gpu_plan_or_skip((1,), workers_per_gpu)
    write_universe(tmp_path)
    queries = np.random.default_rng(8).integers(0, 40, size=(23, 6)).astype(np.uint8)
    expected = cpu_reference_lists(tmp_path, queries, 11)

    # workers_per_gpu = 2 starts two spawned processes that share the GPU.
    lists = universe.scan_lists(
        queries, 11, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", 30, "l2", plan,
        tmp_path / "scratch_gpu", "q", query_group_rows=5,
    )
    assert np.array_equal(lists.ids, expected.ids) and np.array_equal(lists.dists, expected.dists)


def test_two_gpus_share_a_scan_when_both_are_free(tmp_path):
    from gpu_helpers import gpu_plan_or_skip

    plan = gpu_plan_or_skip((0, 1), 1)
    write_universe(tmp_path)
    queries = np.random.default_rng(9).integers(0, 40, size=(23, 6)).astype(np.uint8)
    expected = cpu_reference_lists(tmp_path, queries, 11)
    lists = universe.scan_lists(
        queries, 11, tmp_path / "base.u8bin", tmp_path / "insert_pool.u8bin", 30, "l2", plan,
        tmp_path / "scratch_gpu", "q",
    )
    assert plan.scan_workers == 2
    assert np.array_equal(lists.ids, expected.ids) and np.array_equal(lists.dists, expected.dists)
