"""Exact k-nearest-neighbour ground truth over an "active set" of vectors.

The active set is the initial base plus the vectors inserted so far, minus the deleted
ones. Vectors are streamed in blocks, so the whole set never has to fit in memory.

Deterministic tie rule
----------------------
Equal distances are broken in favour of the SMALLER global id. To make this exact and
cheap, every candidate is encoded as one ``uint64`` sort key

    key = (order_preserving_bits(distance) << 32) | global_id

Keys are all different (ids are unique), so a plain partial sort (``partition``) selects
the same k winners on the CPU and on the GPU, and sorting the keys orders them by
(distance, id).

Distances are "smaller is closer": squared L2, or the negated inner product.

Filtered merge
--------------
Encoding and partitioning every distance of every block is the expensive part of a scan
(measured: about 85% of the time of one query-chunk x block pair, round 00077). Once a query
holds k candidates, an entry whose distance is STRICTLY larger than the distance of its k-th
best candidate can never be one of the final k winners, so only the other entries are
encoded and merged. Entries at exactly the threshold distance are kept, because the smaller
id wins such a tie. The winners are therefore the same as in the dense merge; the dense merge
is still used while a query has fewer than k candidates and whenever the filter would keep
too many entries to be worth it.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterable, Iterator, Tuple

import numpy as np

from semantic_vdb.binio import BinReader, GroundTruth

METRICS = ("l2", "ip")
DEVICES = ("cpu", "gpu")

# Global ids are stored in the low 32 bits of a key and written as int32 in GT files.
MAX_GLOBAL_ID = (1 << 31) - 1
_EMPTY_KEY = np.uint64(0xFFFFFFFFFFFFFFFF)
# The filtered merge gives way to the dense merge when more than 1/8 of the entries of a block
# survive the filter: encoding and scattering that many entries costs about as much as the
# dense merge. A plain implementation constant; both paths give the same winners.
_DENSE_FALLBACK_DIVISOR = 8

# On the GPU one kernel does the whole filtered merge step that follows the matrix product:
# it turns the dot products into distances, compares each distance with its query's threshold
# and writes the key of every survivor straight into that query's row of the candidate
# matrix. The distance matrix, the boolean mask, ``nonzero`` and the scatter of the portable
# implementation are not needed. Every arithmetic step is a separately rounded IEEE float32
# operation in the same order as ``pairwise_distance`` (the ``__f*_rn`` intrinsics cannot be
# contracted into a fused multiply-add), so the distances are bit-identical to the dense merge.
_COLLECT_KERNEL_SOURCE = r"""
extern "C" __global__ void collect_survivors(
    const float* __restrict__ dot,           // (num_queries, num_entries) q . b
    const float* __restrict__ query_sq,      // (num_queries) ||q||^2   (l2 only)
    const float* __restrict__ block_sq,      // (num_entries) ||b||^2   (l2 only)
    const float* __restrict__ threshold,     // (num_queries) k-th best distance
    const long long* __restrict__ global_ids,  // (num_entries)
    const int num_entries,
    const int use_l2,
    const int num_best,                      // k: columns [0, k) of a candidate row hold the best
    const int width,                         // survivor slots per query
    int* __restrict__ survivor_count,        // (num_queries) survivors found, may exceed width
    unsigned long long* __restrict__ candidates)  // (num_queries, num_best + width)
{
    const int row = blockIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= num_entries) {
        return;
    }
    const long long index = (long long)row * num_entries + column;

    float distance;
    if (use_l2) {
        // ||q||^2 + ||b||^2 - 2 q.b, clamped at 0; a NaN stays a NaN (like numpy.maximum).
        const float sum = __fadd_rn(query_sq[row], block_sq[column]);
        const float value = __fsub_rn(sum, __fmul_rn(2.0f, dot[index]));
        distance = (value != value) ? value : (value > 0.0f ? value : 0.0f);
    } else {
        distance = -dot[index];
    }

    // Survive unless strictly farther than the k-th best (so that a NaN survives and an equal
    // distance survives: the smaller id decides between equal distances).
    if (distance > threshold[row]) {
        return;
    }
    const int slot = atomicAdd(&survivor_count[row], 1);
    if (slot >= width) {
        return;  // the host sees survivor_count > width and uses the dense merge instead
    }

    // Same key as encode_sort_keys: order-preserving float bits (-0.0 becomes +0.0), then the id.
    const unsigned int bits = __float_as_uint(__fadd_rn(distance, 0.0f));
    const unsigned int ordered = (bits >> 31) ? ~bits : (bits | 0x80000000u);
    candidates[(long long)row * (num_best + width) + num_best + slot] =
        ((unsigned long long)ordered << 32) | (unsigned long long)global_ids[column];
}
"""
_collect_kernel_cache: dict = {}


def _collect_kernel():
    """The compiled ``collect_survivors`` kernel (compiled on first use, then cached)."""
    kernel = _collect_kernel_cache.get("kernel")
    if kernel is None:
        import cupy

        kernel = cupy.RawKernel(_COLLECT_KERNEL_SOURCE, "collect_survivors")
        _collect_kernel_cache["kernel"] = kernel
    return kernel


def array_module(device: str) -> Any:
    """Return numpy for ``cpu`` or cupy for ``gpu`` (raises if cupy is unavailable)."""
    if device == "cpu":
        return np
    if device == "gpu":
        try:
            import cupy
        except ImportError as error:
            raise RuntimeError("device 'gpu' needs the cupy package") from error
        return cupy
    raise ValueError(f"device must be one of {DEVICES}, got {device!r}")


# ---------------------------------------------------------------------------
# key encoding
# ---------------------------------------------------------------------------


def _order_preserving_bits(xp: Any, distances: Any) -> Any:
    """The float32 bit patterns of ``distances`` as uint32 values that sort like the floats."""
    # Adding +0.0 turns -0.0 into +0.0 so that the two zeros never compare differently.
    distances = xp.ascontiguousarray(distances, dtype=xp.float32) + xp.float32(0.0)
    bits = distances.view(xp.uint32)
    # IEEE-754 bit patterns sort correctly as unsigned integers only for non-negative
    # floats; flip every bit of negative values and the sign bit of the others.
    is_negative = (bits >> xp.uint32(31)) == 1
    return xp.where(is_negative, ~bits, bits | xp.uint32(0x80000000))


def encode_sort_keys(xp: Any, distances: Any, global_ids: Any) -> Any:
    """Pack float32 distances (q, n) and int64 ids (n,) into uint64 keys (q, n)."""
    ordered = _order_preserving_bits(xp, distances)
    return (ordered.astype(xp.uint64) << xp.uint64(32)) | global_ids.astype(xp.uint64)[None, :]


def encode_sort_keys_flat(xp: Any, distances: Any, global_ids: Any) -> Any:
    """Pack float32 distances (m,) and the int64 ids (m,) of the same entries into keys (m,)."""
    ordered = _order_preserving_bits(xp, distances)
    return (ordered.astype(xp.uint64) << xp.uint64(32)) | global_ids.astype(xp.uint64)


def decode_sort_keys(xp: Any, keys: Any) -> Tuple[Any, Any]:
    """Inverse of ``encode_sort_keys``: returns (ids int64, distances float32)."""
    ids = (keys & xp.uint64(0xFFFFFFFF)).astype(xp.int64)
    ordered = (keys >> xp.uint64(32)).astype(xp.uint32)
    was_non_negative = (ordered >> xp.uint32(31)) == 1
    bits = xp.where(was_non_negative, ordered & xp.uint32(0x7FFFFFFF), ~ordered)
    return ids, xp.ascontiguousarray(bits).view(xp.float32)


# ---------------------------------------------------------------------------
# distances and accumulation
# ---------------------------------------------------------------------------


def pairwise_distance(xp: Any, queries: Any, query_sq: Any, block: Any, metric: str) -> Any:
    """Distances (q, n) between float32 queries (q, d) and a float32 block (n, d)."""
    if metric == "l2":
        block_sq = (block * block).sum(axis=1)[None, :]
        # ||q - b||^2 = ||q||^2 + ||b||^2 - 2 q.b ; clamp tiny negative rounding noise.
        distances = query_sq + block_sq - 2.0 * (queries @ block.T)
        return xp.maximum(distances, 0.0)
    if metric == "ip":
        return -(queries @ block.T)
    raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


class KnnAccumulator:
    """Running exact top-k for a fixed set of queries while base blocks stream by."""

    def __init__(
        self,
        queries: np.ndarray,
        k: int,
        metric: str,
        device: str = "cpu",
        query_chunk_rows: int = 1024,
        use_device_kernel: bool = True,
    ) -> None:
        if k < 1:
            raise ValueError("k must be positive")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if query_chunk_rows < 1:
            raise ValueError("query_chunk_rows must be positive")
        if queries.ndim != 2:
            raise ValueError("queries must be a 2-D array")

        self.k = k
        self.metric = metric
        self.xp = array_module(device)
        self.num_queries = queries.shape[0]
        # On the GPU the filtered merge can run as one kernel (``_COLLECT_KERNEL_SOURCE``);
        # ``use_device_kernel=False`` selects the portable implementation, which tests compare
        # with it. A query chunk must fit the kernel's grid (at most 65535 rows).
        self._device_kernel = (
            device == "gpu" and use_device_kernel and min(query_chunk_rows, 65535) == query_chunk_rows
        )
        # Survivor slots per query in the kernel's candidate matrix; a query with more
        # survivors than that sends its block to the dense merge.
        self._survivor_width = max(64, 2 * k)

        xp = self.xp
        # Queries are cut into chunks so the (chunk x block) distance matrix stays small.
        self._bounds = [
            (start, min(start + query_chunk_rows, self.num_queries))
            for start in range(0, self.num_queries, query_chunk_rows)
        ]
        self._queries = [xp.asarray(queries[a:b], dtype=xp.float32) for a, b in self._bounds]
        self._query_sq = [(chunk * chunk).sum(axis=1)[:, None] for chunk in self._queries]
        # Best k keys per query so far; empty slots hold the largest possible key.
        self._best = [xp.full((b - a, k), _EMPTY_KEY, dtype=xp.uint64) for a, b in self._bounds]
        # Distance of the k-th best entry of every query (+inf while it has fewer than k), and
        # whether every query of the chunk already has k entries. Both always describe
        # ``_best`` of the same chunk; ``_merge_into_chunk`` refreshes them after each merge.
        self._threshold = [xp.full((b - a,), xp.inf, dtype=xp.float32) for a, b in self._bounds]
        self._chunk_is_full = [False] * len(self._bounds)
        # How often each merge path ran, and how many entries the filter let through; read by
        # tests and diagnostics, never by the algorithm.
        self.merge_counts = {
            "filtered": 0,
            "kernel": 0,  # filtered merges that ran as the GPU kernel (a subset of "filtered")
            "dense": 0,
            "entries_seen": 0,
            "entries_kept": 0,
        }

    def add_block(self, global_ids: np.ndarray, vectors: np.ndarray) -> None:
        """Offer one block of base vectors (unique global ids) to every query."""
        if global_ids.shape[0] != vectors.shape[0]:
            raise ValueError("global_ids and vectors must have the same number of rows")
        if global_ids.shape[0] == 0:
            return
        if global_ids.min() < 0 or global_ids.max() > MAX_GLOBAL_ID:
            raise ValueError("global ids must be in 0..2^31-1")

        xp = self.xp
        # One host -> device copy of the block; every query chunk below reuses it.
        block = xp.asarray(vectors, dtype=xp.float32)
        ids = xp.asarray(global_ids, dtype=xp.int64)
        # ||b||^2 does not depend on the query chunk: the kernel path computes it once here
        # (the same expression as in ``pairwise_distance``, hence the same bits).
        block_sq = (block * block).sum(axis=1) if self._device_kernel else None

        for chunk_index in range(len(self._queries)):
            self._merge_into_chunk(chunk_index, ids, block, block_sq)

    def _merge_into_chunk(
        self, chunk_index: int, ids: Any, block: Any, block_sq: Any = None
    ) -> None:
        """Merge one block into the running top-k of one query chunk.

        The running best list is replaced only at the very end, so if the temporaries do not
        fit in memory (``MemoryError``; cupy's out-of-memory error is a subclass) nothing has
        changed yet. The block is then cut in two halves that are merged one after the other,
        which gives the same winners because a top-k merge does not depend on the order.
        """
        xp = self.xp
        try:
            # The filtered merge needs a threshold, which exists only once every query of
            # the chunk holds k entries; before that (and when the filter would keep too
            # much) the dense merge runs. Both give the same winners.
            outcome = None
            distances = None
            used_kernel = False
            if self._chunk_is_full[chunk_index] and self._device_kernel:
                # The kernel path never builds the distance matrix, only the dot products.
                outcome = self._merge_collected(chunk_index, ids, block, block_sq)
                used_kernel = outcome is not None
            if outcome is None:
                distances = pairwise_distance(
                    xp, self._queries[chunk_index], self._query_sq[chunk_index], block,
                    self.metric,
                )
                if self._chunk_is_full[chunk_index] and not self._device_kernel:
                    outcome = self._merge_filtered(chunk_index, distances, ids)
            num_entries = len(self._queries[chunk_index]) * block.shape[0]
            if outcome is not None:
                merged, kept = outcome
                path = "filtered"
            else:
                merged, kept = self._merge_dense(chunk_index, distances, ids), num_entries
                path = "dense"
            threshold, is_full = self._kth_best_distance(merged)
        except MemoryError:
            if block.shape[0] < 2:
                raise
            self._release_device_memory()
            half = block.shape[0] // 2
            # The halves recompute ||b||^2 themselves (block_sq of the whole block is not
            # sliced: the sums of a row do not depend on its neighbours, but recomputing keeps
            # the retry independent of the failed attempt).
            self._merge_into_chunk(chunk_index, ids[:half], block[:half])
            self._merge_into_chunk(chunk_index, ids[half:], block[half:])
            return

        # Commit: the best list, its threshold and the counters change together, and only
        # after the whole merge succeeded.
        self._best[chunk_index] = merged
        self._threshold[chunk_index] = threshold
        self._chunk_is_full[chunk_index] = is_full
        self.merge_counts[path] += 1
        self.merge_counts["kernel"] += int(used_kernel)
        self.merge_counts["entries_seen"] += int(num_entries)
        self.merge_counts["entries_kept"] += int(kept)

    def _merge_collected(self, chunk_index: int, ids: Any, block: Any, block_sq: Any):
        """GPU: filter and encode in one kernel; returns ``(merged keys, kept)`` or ``None``.

        ``None`` means that some query had more survivors than the candidate matrix has
        slots (the caller then runs the dense merge on the same block), which is the case
        for the first blocks after a chunk filled up and for data with very many ties.
        """
        xp = self.xp
        k, width = self.k, self._survivor_width
        queries = self._queries[chunk_index]
        num_queries, num_entries = queries.shape[0], block.shape[0]
        if block_sq is None or block_sq.shape[0] != num_entries:
            block_sq = (block * block).sum(axis=1)

        # The same matrix product as in pairwise_distance; the rest happens in the kernel.
        dot = queries @ block.T  # (num_queries, num_entries) float32

        # Candidate matrix: columns [0, k) hold the current winners, the rest the survivors.
        candidates = xp.full((num_queries, k + width), _EMPTY_KEY, dtype=xp.uint64)
        candidates[:, :k] = self._best[chunk_index]
        survivor_count = xp.zeros(num_queries, dtype=xp.int32)

        threads = 256
        grid = ((num_entries + threads - 1) // threads, num_queries)
        _collect_kernel()(
            grid, (threads,),
            (
                dot, self._query_sq[chunk_index].reshape(-1), block_sq,
                self._threshold[chunk_index], ids,
                np.int32(num_entries), np.int32(1 if self.metric == "l2" else 0),
                np.int32(k), np.int32(width), survivor_count, candidates,
            ),
        )
        # Two small device-to-host reads: the most survivors of any query, and their total.
        most = int(survivor_count.max())
        if most > width:
            return None
        total = int(survivor_count.sum())
        if total * _DENSE_FALLBACK_DIVISOR > num_queries * num_entries:
            return None
        if total == 0:
            return self._best[chunk_index], 0
        merged = xp.partition(candidates, k - 1, axis=1)[:, :k]
        return merged, total

    def _merge_dense(self, chunk_index: int, distances: Any, ids: Any) -> Any:
        """Encode and partition every distance of the block, then merge with the best list."""
        xp = self.xp
        keys = encode_sort_keys(xp, distances, ids)
        # Keep only the k best keys of this block before merging with the running best.
        if keys.shape[1] > self.k:
            keys = xp.partition(keys, self.k - 1, axis=1)[:, : self.k]
        merged = xp.concatenate([self._best[chunk_index], keys], axis=1)
        if merged.shape[1] > self.k:
            merged = xp.partition(merged, self.k - 1, axis=1)[:, : self.k]
        return merged

    def _merge_filtered(self, chunk_index: int, distances: Any, ids: Any):
        """Merge only the entries that can still reach the top-k of their query.

        Returns ``(merged keys, number of entries kept)``, or ``None`` when so many entries
        survive that the dense merge is cheaper. An entry survives unless its distance is
        strictly larger than the query's current k-th best distance (``_threshold``): such an
        entry has at least k better candidates and cannot be a winner. Equal distances
        survive because the smaller id decides between them.
        """
        xp = self.xp
        k = self.k
        num_queries, num_entries = distances.shape
        best = self._best[chunk_index]

        # "not (d > t)" rather than "d <= t": a NaN distance then survives, as it does in the
        # dense merge (where it becomes a key that sorts after every number).
        keep = ~(distances > self._threshold[chunk_index][:, None])
        counts = keep.sum(axis=1)  # survivors per query
        total = int(counts.sum())
        if total * _DENSE_FALLBACK_DIVISOR > num_queries * num_entries:
            return None
        if total == 0:
            return best, 0  # no entry of this block can improve any query

        # nonzero walks the matrix row by row, so the survivors of one query are adjacent.
        rows, cols = xp.nonzero(keep)
        keys = encode_sort_keys_flat(xp, distances[rows, cols], ids[cols])

        # Candidate matrix: the k current winners of each query followed by that query's
        # survivors (padded with the largest key); the survivor with running number r of
        # row q goes to column k + (r - first survivor of q).
        first_of_row = xp.cumsum(counts) - counts
        slot = xp.arange(total) - first_of_row[rows]
        width = int(counts.max())
        candidates = xp.full((num_queries, k + width), _EMPTY_KEY, dtype=xp.uint64)
        candidates[:, :k] = best
        candidates[rows, k + slot] = keys
        merged = xp.partition(candidates, k - 1, axis=1)[:, :k]
        return merged, total

    def _kth_best_distance(self, best: Any) -> Tuple[Any, bool]:
        """Per query the distance of its k-th best entry, and whether every query has k."""
        xp = self.xp
        # Of the k kept keys the largest belongs to the k-th best entry.
        kth_key = best.max(axis=1)
        is_empty = kth_key == _EMPTY_KEY
        _, kth_distance = decode_sort_keys(xp, kth_key)
        threshold = xp.where(is_empty, xp.float32(xp.inf), kth_distance).astype(xp.float32)
        return threshold, not bool(is_empty.any())

    def _release_device_memory(self) -> None:
        """Give cached device memory back so that a smaller retry can succeed."""
        if hasattr(self.xp, "get_default_memory_pool"):
            self.xp.get_default_memory_pool().free_all_blocks()

    def finish(self) -> GroundTruth:
        """Sort each query's winners by (distance, id) and return host arrays."""
        xp = self.xp
        id_parts, dist_parts = [], []
        for best in self._best:
            if bool((best == _EMPTY_KEY).any()):
                raise ValueError("fewer than k vectors were offered to the accumulator")
            ids, distances = decode_sort_keys(xp, xp.sort(best, axis=1))
            id_parts.append(_to_host(ids).astype(np.int32))
            dist_parts.append(_to_host(distances).astype(np.float32))

        if not id_parts:
            return GroundTruth(
                neighbour_ids=np.empty((0, self.k), dtype=np.int32),
                neighbour_dists=np.empty((0, self.k), dtype=np.float32),
            )
        return GroundTruth(
            neighbour_ids=np.concatenate(id_parts, axis=0),
            neighbour_dists=np.concatenate(dist_parts, axis=0),
        )


def _to_host(array: Any) -> np.ndarray:
    # cupy arrays expose .get() for the device -> host copy; numpy arrays do not.
    return array.get() if hasattr(array, "get") else array


def prefetch_blocks(
    blocks: Iterable[Tuple[np.ndarray, np.ndarray]], depth: int = 2
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield the same ``(ids, vectors)`` blocks, preparing the next ones in a helper thread.

    Reading a block from disk and converting it to float32 happens in the background while
    the consumer (the GPU or CPU scan) works on the current block. At most ``depth`` blocks
    wait in the queue, which bounds the extra memory. Exceptions of the producer are raised
    in the consumer, and closing the generator early stops the producer.
    """
    ready: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()
    finished = object()

    def put(item) -> bool:
        # Wait for space, but give up as soon as the consumer has gone away.
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for ids, vectors in blocks:
                converted = np.ascontiguousarray(vectors, dtype=np.float32)
                if not put((ids, converted)):
                    return
            put(finished)
        except BaseException as error:  # handed to the consumer, which re-raises it
            put(error)

    thread = threading.Thread(target=produce, name="block-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = ready.get()
            if item is finished:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join(timeout=5.0)


def exact_knn(
    blocks: Iterable[Tuple[np.ndarray, np.ndarray]],
    queries: np.ndarray,
    k: int,
    metric: str,
    device: str = "cpu",
    query_chunk_rows: int = 1024,
) -> GroundTruth:
    """Exact top-k of ``queries`` over all vectors in ``blocks``.

    ``blocks`` yields ``(global_ids, vectors)`` pairs; ids must be unique over the whole
    stream. The result ids are global ids, ordered nearest first, ties by smaller id.
    """
    accumulator = KnnAccumulator(queries, k, metric, device, query_chunk_rows)
    for global_ids, vectors in blocks:
        accumulator.add_block(global_ids, vectors)
    return accumulator.finish()


# ---------------------------------------------------------------------------
# active-set block iteration
# ---------------------------------------------------------------------------


def array_blocks(
    vectors: np.ndarray, block_rows: int, first_id: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Cut an in-memory matrix into blocks whose global ids start at ``first_id``."""
    for start in range(0, vectors.shape[0], block_rows):
        stop = min(start + block_rows, vectors.shape[0])
        yield np.arange(first_id + start, first_id + stop, dtype=np.int64), vectors[start:stop]


def iter_active_blocks(
    base_reader: BinReader,
    insert_reader: BinReader | None,
    num_inserted: int,
    deleted_mask: np.ndarray,
    block_rows: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield the alive vectors as ``(global_ids, vectors)`` blocks.

    Global ids: base row ``i`` has id ``i``; the ``j``-th inserted vector has id
    ``num_base + j``. ``deleted_mask[id]`` is True for deleted ids. Only the first
    ``num_inserted`` rows of the insert pool exist at this point of the stream.
    """
    num_base = base_reader.num_rows
    segments = [(base_reader, 0, num_base)]
    if num_inserted > 0:
        segments.append((insert_reader, num_base, num_inserted))

    for reader, id_offset, count in segments:
        for start in range(0, count, block_rows):
            block_count = min(block_rows, count - start)
            ids = np.arange(id_offset + start, id_offset + start + block_count, dtype=np.int64)
            alive = ~deleted_mask[ids]
            if not alive.any():
                continue
            vectors = reader.read_range(start, block_count)
            if alive.all():
                yield ids, vectors
            else:
                yield ids[alive], vectors[alive]
