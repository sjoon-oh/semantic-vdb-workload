"""Tests for semantic_vdb.groundtruth against an independent float64 reference."""

import numpy as np
import pytest

from semantic_vdb import binio, groundtruth
from reference import reference_knn

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def gpu_available():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - any failure means "no usable GPU"
        return False


needs_gpu = pytest.mark.skipif(not gpu_available(), reason="no usable cupy/GPU")


def tie_heavy_uint8(num_vectors, dim, seed):
    """Random uint8 vectors where a third of the rows are exact copies of earlier rows.

    Copies have identical distances to every query, so the tie rule matters a lot.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.integers(0, 256, size=(num_vectors, dim)).astype(np.uint8)
    copies = rng.integers(0, num_vectors // 2, size=num_vectors // 3)
    targets = rng.integers(num_vectors // 2, num_vectors, size=num_vectors // 3)
    vectors[targets] = vectors[copies]
    return vectors


def run_knn(vectors, queries, k, metric="l2", device="cpu", block_rows=97, chunk_rows=7):
    ids = np.arange(len(vectors), dtype=np.int64)
    blocks = groundtruth.array_blocks(vectors, block_rows)
    result = groundtruth.exact_knn(blocks, queries, k, metric, device, chunk_rows)
    return result.neighbour_ids.astype(np.int64), result.neighbour_dists


# ---------------------------------------------------------------------------
# key encoding
# ---------------------------------------------------------------------------


def test_sort_key_order_equals_distance_then_id_order():
    rng = np.random.default_rng(0)
    distances = np.concatenate(
        [rng.normal(size=300).astype(np.float32), np.array([0.0, -0.0, 1e-30, -1e-30], np.float32)]
    )
    distances = np.concatenate([distances, distances[:50]])  # exact duplicates -> ties
    ids = rng.permutation(len(distances)).astype(np.int64)

    keys = groundtruth.encode_sort_keys(np, distances[None, :], ids)[0]
    by_key = np.argsort(keys)
    # Reference order: distance ascending, -0.0 equal to +0.0, then smaller id first.
    reference = np.lexsort((ids, distances + np.float32(0.0)))
    assert np.array_equal(by_key, reference)


def test_negative_zero_and_positive_zero_tie_by_id():
    # -0.0 == +0.0, so only the id may decide the order. Both id assignments are checked,
    # because a missing normalisation would order by sign and fail exactly one of them.
    for plus_zero_id, minus_zero_id in [(3, 9), (9, 3)]:
        distances = np.array([[0.0, -0.0]], dtype=np.float32)
        ids = np.array([plus_zero_id, minus_zero_id], dtype=np.int64)
        keys = groundtruth.encode_sort_keys(np, distances, ids)[0]
        ordered_ids = ids[np.argsort(keys)].tolist()
        assert ordered_ids == sorted([plus_zero_id, minus_zero_id])


def test_sort_keys_round_trip():
    rng = np.random.default_rng(1)
    distances = rng.normal(scale=1e6, size=(4, 50)).astype(np.float32)
    ids = rng.integers(0, groundtruth.MAX_GLOBAL_ID, size=50).astype(np.int64)
    keys = groundtruth.encode_sort_keys(np, distances, ids)
    decoded_ids, decoded_dists = groundtruth.decode_sort_keys(np, keys)
    assert np.array_equal(decoded_ids, np.broadcast_to(ids, (4, 50)))
    assert np.array_equal(decoded_dists, distances + np.float32(0.0))


# ---------------------------------------------------------------------------
# correctness against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["l2", "ip"])
def test_cpu_matches_reference_exactly_including_ties(metric):
    vectors = tie_heavy_uint8(600, 16, seed=3)
    queries = np.random.default_rng(4).integers(0, 256, size=(25, 16)).astype(np.uint8)
    ids, dists = run_knn(vectors, queries, k=12, metric=metric)
    ref_ids, ref_dists = reference_knn(vectors, np.arange(600), queries, 12, metric)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(dists.astype(np.float64), ref_dists)


def test_result_does_not_depend_on_block_or_chunk_sizes():
    vectors = tie_heavy_uint8(500, 12, seed=5)
    queries = np.random.default_rng(6).integers(0, 256, size=(40, 12)).astype(np.uint8)
    baseline = run_knn(vectors, queries, 9, block_rows=500, chunk_rows=40)
    for block_rows, chunk_rows in [(1, 1), (13, 5), (499, 39), (64, 64)]:
        ids, dists = run_knn(vectors, queries, 9, block_rows=block_rows, chunk_rows=chunk_rows)
        assert np.array_equal(ids, baseline[0])
        assert np.array_equal(dists, baseline[1])


def test_large_k_over_many_blocks_matches_reference():
    # With k = 40 the merged candidate list is wide, so a sloppy partial-sort boundary
    # (keeping position k-1 without guaranteeing it is the k-th smallest) shows up.
    rng = np.random.default_rng(40)
    vectors = rng.integers(0, 256, size=(800, 8)).astype(np.uint8)
    queries = rng.integers(0, 256, size=(30, 8)).astype(np.uint8)
    ids, dists = run_knn(vectors, queries, k=40, block_rows=64, chunk_rows=11)
    ref_ids, ref_dists = reference_knn(vectors, np.arange(800), queries, 40)
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(dists.astype(np.float64), ref_dists)


def test_equal_distances_pick_the_smaller_id_first():
    # Four identical vectors: every query sees the same distance for all of them.
    vectors = np.tile(np.array([[10, 20, 30]], dtype=np.uint8), (4, 1))
    # A far-away fifth vector must never beat the copies.
    vectors = np.vstack([vectors, np.array([[200, 200, 200]], dtype=np.uint8)])
    ids, _ = run_knn(vectors, np.array([[10, 20, 31]], dtype=np.uint8), k=3, block_rows=2)
    assert ids.tolist() == [[0, 1, 2]]


def test_float_inner_product_matches_reference_on_integer_valued_data():
    rng = np.random.default_rng(7)
    vectors = rng.integers(-20, 20, size=(300, 8)).astype(np.float32)
    queries = rng.integers(-20, 20, size=(15, 8)).astype(np.float32)
    ids, dists = run_knn(vectors, queries, k=6, metric="ip")
    ref_ids, ref_dists = reference_knn(vectors, np.arange(300), queries, 6, "ip")
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(dists.astype(np.float64), ref_dists)


def test_l2_distances_are_never_negative_even_with_float_rounding():
    # Non-integer float data makes ||q||^2 + ||b||^2 - 2 q.b round to tiny negative values
    # for a query that equals a base vector; the accumulator must clamp them at zero.
    rng = np.random.default_rng(30)
    vectors = (rng.normal(size=(400, 64)) * 100.0).astype(np.float32)
    ids, dists = run_knn(vectors, vectors, k=3, block_rows=128, chunk_rows=50)
    assert np.all(dists >= 0.0)
    # Each vector is its own nearest neighbour. The float32 expansion cannot return an
    # exact zero, but the self distance must be tiny compared with the vector norm.
    assert np.array_equal(ids[:, 0], np.arange(400))
    norm_sq = (vectors.astype(np.float64) ** 2).sum(axis=1)
    assert np.all(dists[:, 0] <= 1e-5 * norm_sq)


def test_distances_are_nearest_first():
    vectors = tie_heavy_uint8(300, 8, seed=8)
    queries = tie_heavy_uint8(10, 8, seed=9)
    _, dists = run_knn(vectors, queries, k=10)
    assert np.all(np.diff(dists, axis=1) >= 0)


def test_too_few_vectors_is_an_error():
    vectors = tie_heavy_uint8(6, 4, seed=1)
    with pytest.raises(ValueError):
        run_knn(vectors, vectors[:2], k=10)


def test_no_queries_gives_an_empty_result():
    vectors = tie_heavy_uint8(30, 4, seed=1)
    ids, dists = run_knn(vectors, np.empty((0, 4), dtype=np.uint8), k=5)
    assert ids.shape == (0, 5) and dists.shape == (0, 5)


def test_invalid_arguments():
    queries = np.zeros((2, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        groundtruth.KnnAccumulator(queries, 0, "l2")
    with pytest.raises(ValueError):
        groundtruth.KnnAccumulator(queries, 3, "cosine")
    accumulator = groundtruth.KnnAccumulator(queries, 1, "l2")
    with pytest.raises(ValueError):  # ids that do not fit in 31 bits
        accumulator.add_block(np.array([1 << 31], dtype=np.int64), np.zeros((1, 4), np.uint8))


# ---------------------------------------------------------------------------
# active set: insertions and deletions
# ---------------------------------------------------------------------------


def write_pool(path, vectors):
    binio.write_array(path, vectors)
    return binio.BinReader(path)


def active_reference(base, pool, num_inserted, deleted, queries, k):
    """Reference answer over base + first `num_inserted` pool rows minus deleted ids."""
    all_vectors = np.vstack([base, pool[:num_inserted]])
    alive = np.array([i for i in range(len(all_vectors)) if i not in deleted])
    return reference_knn(all_vectors[alive], alive, queries, k, "l2")


def test_active_set_respects_inserts_and_deletes(tmp_path):
    rng = np.random.default_rng(10)
    base = tie_heavy_uint8(200, 8, seed=11)
    pool = rng.integers(0, 256, size=(50, 8)).astype(np.uint8)
    queries = np.vstack([rng.integers(0, 256, size=(12, 8)).astype(np.uint8), pool[:3]])

    with write_pool(tmp_path / "b.u8bin", base) as base_reader, write_pool(
        tmp_path / "p.u8bin", pool
    ) as pool_reader:
        num_inserted = 20
        deleted = {0, 5, 17, 199, 200 + 3, 200 + 19}
        mask = np.zeros(200 + 50, dtype=bool)
        mask[list(deleted)] = True

        blocks = groundtruth.iter_active_blocks(base_reader, pool_reader, num_inserted, mask, 33)
        result = groundtruth.exact_knn(blocks, queries, 7, "l2", "cpu", 5)

    ref_ids, ref_dists = active_reference(base, pool, num_inserted, deleted, queries, 7)
    assert np.array_equal(result.neighbour_ids.astype(np.int64), ref_ids)
    assert np.array_equal(result.neighbour_dists.astype(np.float64), ref_dists)
    # No deleted id may appear in any answer.
    assert not (set(result.neighbour_ids.ravel().tolist()) & deleted)


def test_inserted_vectors_are_found_and_numbered_after_the_base(tmp_path):
    base = tie_heavy_uint8(100, 8, seed=12)
    pool = np.random.default_rng(13).integers(0, 256, size=(10, 8)).astype(np.uint8)
    with write_pool(tmp_path / "b.u8bin", base) as base_reader, write_pool(
        tmp_path / "p.u8bin", pool
    ) as pool_reader:
        mask = np.zeros(110, dtype=bool)
        # Query = the 4th pool row: after it is inserted it is its own nearest neighbour.
        blocks = groundtruth.iter_active_blocks(base_reader, pool_reader, 5, mask, 16)
        inserted = groundtruth.exact_knn(blocks, pool[3:4], 1, "l2", "cpu")
        # Before any insertion the pool row does not exist.
        blocks = groundtruth.iter_active_blocks(base_reader, pool_reader, 0, mask, 16)
        before = groundtruth.exact_knn(blocks, pool[3:4], 1, "l2", "cpu")

    assert inserted.neighbour_ids[0, 0] == 100 + 3 and inserted.neighbour_dists[0, 0] == 0
    assert before.neighbour_ids[0, 0] < 100


def test_pool_rows_beyond_the_inserted_count_are_invisible(tmp_path):
    base = tie_heavy_uint8(60, 8, seed=14)
    pool = np.random.default_rng(15).integers(0, 256, size=(10, 8)).astype(np.uint8)
    with write_pool(tmp_path / "b.u8bin", base) as base_reader, write_pool(
        tmp_path / "p.u8bin", pool
    ) as pool_reader:
        mask = np.zeros(70, dtype=bool)
        blocks = groundtruth.iter_active_blocks(base_reader, pool_reader, 4, mask, 16)
        result = groundtruth.exact_knn(blocks, pool[8:9], 5, "l2", "cpu")
    assert result.neighbour_ids.max() < 60 + 4


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


@needs_gpu
def test_gpu_equals_cpu_and_reference_on_full_range_uint8():
    # Values over the whole 0..255 range at 128 dimensions make exact squared distances
    # as large as ~8e6; a TF32 matrix product would round them and break equality.
    vectors = tie_heavy_uint8(3000, 128, seed=20)
    queries = np.random.default_rng(21).integers(0, 256, size=(60, 128)).astype(np.uint8)
    cpu = run_knn(vectors, queries, 10, device="cpu", block_rows=700, chunk_rows=32)
    gpu = run_knn(vectors, queries, 10, device="gpu", block_rows=700, chunk_rows=32)
    ref_ids, ref_dists = reference_knn(vectors, np.arange(3000), queries, 10)
    assert np.array_equal(gpu[0], ref_ids)
    assert np.array_equal(gpu[1].astype(np.float64), ref_dists)
    assert np.array_equal(gpu[0], cpu[0]) and np.array_equal(gpu[1], cpu[1])


@needs_gpu
def test_gpu_inner_product_matches_reference():
    rng = np.random.default_rng(22)
    vectors = rng.integers(-100, 100, size=(1500, 32)).astype(np.float32)
    queries = rng.integers(-100, 100, size=(20, 32)).astype(np.float32)
    ids, dists = run_knn(vectors, queries, 8, metric="ip", device="gpu", block_rows=400)
    ref_ids, ref_dists = reference_knn(vectors, np.arange(1500), queries, 8, "ip")
    assert np.array_equal(ids, ref_ids)
    assert np.array_equal(dists.astype(np.float64), ref_dists)


# ---------------------------------------------------------------------------
# running out of memory, and reading blocks ahead
# ---------------------------------------------------------------------------


def test_a_block_that_does_not_fit_is_split_and_gives_the_same_winners(monkeypatch):
    vectors = tie_heavy_uint8(400, 8, seed=11)
    queries = np.random.default_rng(12).integers(0, 256, size=(30, 8)).astype(np.uint8)
    expected = run_knn(vectors, queries, 7, block_rows=400, chunk_rows=10)

    real = groundtruth.pairwise_distance
    attempts = []

    def small_memory(xp, queries_chunk, query_sq, block, metric):
        attempts.append(block.shape[0])
        if block.shape[0] > 60:  # pretend that only 60 rows fit at a time
            raise MemoryError("out of memory")
        return real(xp, queries_chunk, query_sq, block, metric)

    monkeypatch.setattr(groundtruth, "pairwise_distance", small_memory)
    ids, dists = run_knn(vectors, queries, 7, block_rows=400, chunk_rows=10)

    assert np.array_equal(ids, expected[0]) and np.array_equal(dists, expected[1])
    assert max(attempts) == 400 and min(attempts) <= 60  # it really retried with halves
    ref_ids, _ = reference_knn(vectors, np.arange(400), queries, 7)
    assert np.array_equal(ids, ref_ids)


def test_one_row_that_still_does_not_fit_raises_the_error(monkeypatch):
    def never_fits(*args, **kwargs):
        raise MemoryError("out of memory")

    monkeypatch.setattr(groundtruth, "pairwise_distance", never_fits)
    vectors = tie_heavy_uint8(20, 4, seed=1)
    with pytest.raises(MemoryError):
        run_knn(vectors, vectors[:3], 2, block_rows=20, chunk_rows=3)


def numbered_blocks(count, rows=4, dim=3):
    for index in range(count):
        ids = np.arange(index * rows, (index + 1) * rows, dtype=np.int64)
        yield ids, np.full((rows, dim), index, dtype=np.uint8)


def test_prefetch_yields_the_same_blocks_as_float32():
    expected = list(numbered_blocks(7))
    got = list(groundtruth.prefetch_blocks(numbered_blocks(7), depth=2))
    assert len(got) == len(expected)
    for (ids, vectors), (got_ids, got_vectors) in zip(expected, got):
        assert np.array_equal(ids, got_ids)
        assert got_vectors.dtype == np.float32 and np.array_equal(got_vectors, vectors)


def test_prefetch_passes_a_producer_error_on_to_the_consumer():
    def broken():
        yield from numbered_blocks(2)
        raise RuntimeError("disk read failed")

    seen = []
    with pytest.raises(RuntimeError, match="disk read failed"):
        for ids, _ in groundtruth.prefetch_blocks(broken()):
            seen.append(len(ids))
    assert seen == [4, 4]  # the blocks before the error still arrived


def test_closing_the_prefetch_early_stops_the_helper_thread():
    import threading

    before = threading.active_count()
    stream = groundtruth.prefetch_blocks(numbered_blocks(1000), depth=2)
    next(stream)
    stream.close()  # the consumer walks away from an unfinished stream
    assert threading.active_count() == before


def test_scan_with_prefetch_equals_scan_without(monkeypatch):
    vectors = tie_heavy_uint8(300, 6, seed=21)
    queries = np.random.default_rng(22).integers(0, 256, size=(20, 6)).astype(np.uint8)
    plain = groundtruth.exact_knn(
        groundtruth.array_blocks(vectors, 50), queries, 5, "l2", "cpu", 7
    )
    ahead = groundtruth.exact_knn(
        groundtruth.prefetch_blocks(groundtruth.array_blocks(vectors, 50)), queries, 5, "l2", "cpu", 7
    )
    assert np.array_equal(plain.neighbour_ids, ahead.neighbour_ids)
    assert np.array_equal(plain.neighbour_dists, ahead.neighbour_dists)
