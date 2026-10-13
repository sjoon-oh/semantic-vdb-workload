"""The filtered merge of ``KnnAccumulator`` against the dense merge it replaced.

Round 00077 made the scan encode and merge only the entries that can still reach the top-k of
their query. These tests check that the winners are byte-identical to the previous dense
algorithm (``dense_reference.py``) and to the independent float64 reference, in the situations
where a filter can go wrong: equal distances, ids that do not increase from block to block,
blocks smaller than k, queries that have fewer than k candidates for a long time.
"""

import numpy as np
import pytest

from semantic_vdb import groundtruth
from dense_reference import DenseKnnAccumulator
from reference import reference_knn
from test_groundtruth import needs_gpu, tie_heavy_uint8


def scan(accumulator_class, vectors, ids, queries, k, metric, device, block_rows, chunk_rows,
         **options):
    """Feed ``vectors`` (with the given global ids) block by block; return (ids, dists, acc)."""
    accumulator = accumulator_class(queries, k, metric, device, chunk_rows, **options)
    for start in range(0, len(vectors), block_rows):
        stop = start + block_rows
        accumulator.add_block(ids[start:stop], vectors[start:stop])
    result = accumulator.finish()
    return result.neighbour_ids, result.neighbour_dists, accumulator


def make_case(name):
    """(vectors, ids, queries, k, block_rows, chunk_rows) of a named situation."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32) if False else sum(map(ord, name)))
    ids = np.arange(900, dtype=np.int64)
    if name == "tie_heavy_uint8":
        vectors = tie_heavy_uint8(900, 12, seed=1)
        queries = rng.integers(0, 256, size=(25, 12)).astype(np.uint8)
        return vectors, ids, queries, 10, 97, 7
    if name == "shuffled_ids":
        # Later blocks hold SMALLER ids than earlier ones, so a tie at the threshold distance
        # must be won by the later entry: dropping equal distances would be wrong.
        vectors = tie_heavy_uint8(900, 12, seed=2)
        queries = rng.integers(0, 256, size=(25, 12)).astype(np.uint8)
        return vectors, rng.permutation(900).astype(np.int64), queries, 10, 97, 7
    if name == "int8_negative":
        vectors = rng.integers(-128, 128, size=(900, 10)).astype(np.int8)
        queries = rng.integers(-128, 128, size=(20, 10)).astype(np.int8)
        return vectors, ids, queries, 8, 64, 6
    if name == "float32_gaussian":
        vectors = rng.normal(size=(900, 16)).astype(np.float32)
        queries = rng.normal(size=(20, 16)).astype(np.float32)
        return vectors, ids, queries, 9, 70, 5
    if name == "all_equal":
        # Every distance is equal: all entries survive the filter (dense fallback), and the
        # k smallest ids must win.
        vectors = np.tile(np.array([[3, 4, 5, 6]], dtype=np.uint8), (900, 1))
        queries = np.array([[1, 2, 3, 4], [9, 9, 9, 9]], dtype=np.uint8)
        return vectors, rng.permutation(900).astype(np.int64), queries, 7, 50, 2
    if name == "block_smaller_than_k":
        vectors = tie_heavy_uint8(300, 6, seed=3)
        queries = rng.integers(0, 256, size=(12, 6)).astype(np.uint8)
        return vectors, np.arange(300, dtype=np.int64), queries, 25, 4, 5
    if name == "k_equals_everything":
        vectors = tie_heavy_uint8(40, 6, seed=4)
        queries = rng.integers(0, 256, size=(5, 6)).astype(np.uint8)
        return vectors, np.arange(40, dtype=np.int64), queries, 40, 7, 2
    raise ValueError(name)


CASES = [
    "tie_heavy_uint8",
    "shuffled_ids",
    "int8_negative",
    "float32_gaussian",
    "all_equal",
    "block_smaller_than_k",
    "k_equals_everything",
]


@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("name", CASES)
def test_filtered_merge_is_byte_identical_to_the_dense_merge(name, metric):
    vectors, ids, queries, k, block_rows, chunk_rows = make_case(name)
    new = scan(groundtruth.KnnAccumulator, vectors, ids, queries, k, metric, "cpu",
               block_rows, chunk_rows)
    old = scan(DenseKnnAccumulator, vectors, ids, queries, k, metric, "cpu",
               block_rows, chunk_rows)
    assert np.array_equal(new[0], old[0])
    assert np.array_equal(new[1], old[1])


@pytest.mark.parametrize("name", [case for case in CASES if case != "float32_gaussian"])
def test_filtered_merge_equals_the_independent_reference(name):
    vectors, ids, queries, k, block_rows, chunk_rows = make_case(name)
    got_ids, got_dists, _ = scan(groundtruth.KnnAccumulator, vectors, ids, queries, k, "l2",
                                 "cpu", block_rows, chunk_rows)
    ref_ids, ref_dists = reference_knn(vectors, ids, queries, k)
    assert np.array_equal(got_ids, ref_ids)
    assert np.array_equal(got_dists.astype(np.float64), ref_dists)


def test_a_later_block_with_smaller_ids_wins_the_tie_at_the_threshold():
    # 300 copies of one vector tie for the nearest places; the first blocks carry the large
    # ids, so by the time the small ids arrive every query has k entries at exactly that
    # distance, and only an inclusive threshold lets the small ids in.
    copy = np.array([[10, 20, 30, 40]], dtype=np.uint8)
    vectors = np.vstack([np.tile(copy, (300, 1)), np.full((60, 4), 250, dtype=np.uint8)])
    ids = np.concatenate([np.arange(300, 0, -1) + 1000, np.arange(60) + 5000]).astype(np.int64)
    ids[:300] = np.sort(ids[:300])[::-1]  # strictly decreasing: later blocks, smaller ids
    queries = np.array([[11, 20, 30, 40], [0, 0, 0, 0]], dtype=np.uint8)
    got_ids, _, _ = scan(groundtruth.KnnAccumulator, vectors, ids, queries, 5, "l2", "cpu", 20, 2)
    ref_ids, _ = reference_knn(vectors, ids, queries, 5)
    assert np.array_equal(got_ids, ref_ids)
    assert got_ids[0].tolist() == sorted(ids[:300].tolist())[:5]


def test_the_filter_runs_and_lets_few_entries_through():
    rng = np.random.default_rng(9)
    vectors = rng.integers(0, 256, size=(4000, 16)).astype(np.uint8)
    queries = rng.integers(0, 256, size=(20, 16)).astype(np.uint8)
    _, _, accumulator = scan(groundtruth.KnnAccumulator, vectors, np.arange(4000), queries, 10,
                             "l2", "cpu", 200, 7)
    counts = accumulator.merge_counts
    assert counts["filtered"] > 0  # the fast path really ran
    assert counts["dense"] > 0  # and the first blocks used the dense one
    assert counts["entries_kept"] < 0.5 * counts["entries_seen"]


def test_a_chunk_uses_the_dense_merge_until_every_query_holds_k_entries():
    vectors = tie_heavy_uint8(400, 8, seed=5)
    queries = vectors[:6].copy()
    accumulator = groundtruth.KnnAccumulator(queries, 30, "l2", "cpu", 6)
    ids = np.arange(400, dtype=np.int64)
    accumulator.add_block(ids[:20], vectors[:20])  # 20 < k = 30: nothing to threshold yet
    assert accumulator.merge_counts == {"filtered": 0, "kernel": 0, "dense": 1,
                                        "entries_seen": 120, "entries_kept": 120}
    assert accumulator._chunk_is_full == [False]
    accumulator.add_block(ids[20:60], vectors[20:60])  # still no threshold for THIS merge
    assert accumulator.merge_counts["filtered"] == 0
    assert accumulator._chunk_is_full == [True]  # 60 >= k entries were seen

    # With k = 5 the threshold becomes tight quickly, so a big block lets few entries
    # through and the filtered path runs.
    tight = groundtruth.KnnAccumulator(queries, 5, "l2", "cpu", 6)
    tight.add_block(ids[:20], vectors[:20])
    tight.add_block(ids[20:60], vectors[20:60])
    tight.add_block(ids[60:360], vectors[60:360])
    assert tight.merge_counts["filtered"] == 1
    assert tight.merge_counts["entries_kept"] < tight.merge_counts["entries_seen"]


def test_the_threshold_describes_the_current_best_list():
    vectors = tie_heavy_uint8(300, 8, seed=6)
    queries = vectors[:5].copy() + np.uint8(1)
    k = 8
    accumulator = groundtruth.KnnAccumulator(queries, k, "l2", "cpu", 5)
    for start in range(0, 300, 50):
        accumulator.add_block(np.arange(start, start + 50, dtype=np.int64), vectors[start:start + 50])
        seen = start + 50
        _, ref_dists = reference_knn(vectors[:seen], np.arange(seen), queries, min(k, seen))
        if seen >= k:
            # The stored threshold is the distance of the k-th nearest vector seen so far.
            assert np.array_equal(accumulator._threshold[0].astype(np.float64), ref_dists[:, -1])


@needs_gpu
def test_gpu_filtered_merge_equals_the_dense_merge_on_the_cpu():
    vectors = tie_heavy_uint8(3000, 128, seed=20)
    ids = np.random.default_rng(1).permutation(3000).astype(np.int64)
    queries = np.random.default_rng(21).integers(0, 256, size=(60, 128)).astype(np.uint8)
    gpu = scan(groundtruth.KnnAccumulator, vectors, ids, queries, 10, "l2", "gpu", 400, 32)
    cpu = scan(DenseKnnAccumulator, vectors, ids, queries, 10, "l2", "cpu", 400, 32)
    assert np.array_equal(gpu[0], cpu[0]) and np.array_equal(gpu[1], cpu[1])
    assert gpu[2].merge_counts["filtered"] > 0


# ---------------------------------------------------------------------------
# the GPU kernel path
# ---------------------------------------------------------------------------


@needs_gpu
@pytest.mark.parametrize("metric", ["l2", "ip"])
@pytest.mark.parametrize("name", CASES)
def test_gpu_kernel_is_byte_identical_to_the_dense_merge(name, metric):
    vectors, ids, queries, k, block_rows, chunk_rows = make_case(name)
    kernel = scan(groundtruth.KnnAccumulator, vectors, ids, queries, k, metric, "gpu",
                  block_rows, chunk_rows)
    portable = scan(groundtruth.KnnAccumulator, vectors, ids, queries, k, metric, "gpu",
                    block_rows, chunk_rows, use_device_kernel=False)
    dense = scan(DenseKnnAccumulator, vectors, ids, queries, k, metric, "gpu",
                 block_rows, chunk_rows)
    assert np.array_equal(kernel[0], dense[0]) and np.array_equal(kernel[1], dense[1])
    assert np.array_equal(portable[0], dense[0]) and np.array_equal(portable[1], dense[1])
    assert kernel[2].merge_counts["kernel"] == 0 or kernel[2].merge_counts["filtered"] > 0


@needs_gpu
def test_the_kernel_path_runs_and_the_portable_path_does_not_use_it():
    rng = np.random.default_rng(9)
    vectors = rng.integers(0, 256, size=(4000, 16)).astype(np.uint8)
    queries = rng.integers(0, 256, size=(20, 16)).astype(np.uint8)
    ids = np.arange(4000)
    _, _, with_kernel = scan(groundtruth.KnnAccumulator, vectors, ids, queries, 10, "l2", "gpu",
                             200, 7)
    _, _, without = scan(groundtruth.KnnAccumulator, vectors, ids, queries, 10, "l2", "gpu",
                         200, 7, use_device_kernel=False)
    assert with_kernel.merge_counts["kernel"] > 0
    assert with_kernel.merge_counts["kernel"] == with_kernel.merge_counts["filtered"]
    assert without.merge_counts["kernel"] == 0 and without.merge_counts["filtered"] > 0
    assert with_kernel.merge_counts["entries_kept"] == without.merge_counts["entries_kept"]


@needs_gpu
def test_a_query_with_more_survivors_than_slots_sends_the_block_to_the_dense_merge():
    # 200 identical vectors per block: every entry ties with the threshold, so all of them
    # survive; 200 > the 64 slots of a query with k = 5, and the answer must still be right.
    copy = np.array([[7, 7, 7, 7]], dtype=np.uint8)
    vectors = np.tile(copy, (1000, 1))
    ids = np.random.default_rng(3).permutation(1000).astype(np.int64)
    queries = np.array([[0, 0, 0, 0]], dtype=np.uint8)
    got_ids, _, accumulator = scan(groundtruth.KnnAccumulator, vectors, ids, queries, 5, "l2",
                                   "gpu", 200, 1)
    assert got_ids[0].tolist() == sorted(ids.tolist())[:5]
    assert accumulator.merge_counts["kernel"] == 0  # every block overflowed the slots


@needs_gpu
def test_a_chunk_too_tall_for_the_kernel_grid_uses_the_portable_filter():
    queries = np.zeros((3, 4), dtype=np.uint8)
    accumulator = groundtruth.KnnAccumulator(queries, 2, "l2", "gpu", 70000)
    assert accumulator._device_kernel is False
