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


def encode_sort_keys(xp: Any, distances: Any, global_ids: Any) -> Any:
    """Pack float32 distances (q, n) and int64 ids (n,) into uint64 keys (q, n)."""
    # Adding +0.0 turns -0.0 into +0.0 so that the two zeros never compare differently.
    distances = xp.ascontiguousarray(distances, dtype=xp.float32) + xp.float32(0.0)
    bits = distances.view(xp.uint32)
    # IEEE-754 bit patterns sort correctly as unsigned integers only for non-negative
    # floats; flip every bit of negative values and the sign bit of the others.
    is_negative = (bits >> xp.uint32(31)) == 1
    ordered = xp.where(is_negative, ~bits, bits | xp.uint32(0x80000000))
    return (ordered.astype(xp.uint64) << xp.uint64(32)) | global_ids.astype(xp.uint64)[None, :]


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

        for chunk_index in range(len(self._queries)):
            self._merge_into_chunk(chunk_index, ids, block)

    def _merge_into_chunk(self, chunk_index: int, ids: Any, block: Any) -> None:
        """Merge one block into the running top-k of one query chunk.

        The running best list is replaced only at the very end, so if the temporaries do not
        fit in memory (``MemoryError``; cupy's out-of-memory error is a subclass) nothing has
        changed yet. The block is then cut in two halves that are merged one after the other,
        which gives the same winners because a top-k merge does not depend on the order.
        """
        xp = self.xp
        try:
            distances = pairwise_distance(
                xp, self._queries[chunk_index], self._query_sq[chunk_index], block, self.metric
            )
            keys = encode_sort_keys(xp, distances, ids)
            # Keep only the k best keys of this block before merging with the running best.
            if keys.shape[1] > self.k:
                keys = xp.partition(keys, self.k - 1, axis=1)[:, : self.k]
            merged = xp.concatenate([self._best[chunk_index], keys], axis=1)
            if merged.shape[1] > self.k:
                merged = xp.partition(merged, self.k - 1, axis=1)[:, : self.k]
        except MemoryError:
            if block.shape[0] < 2:
                raise
            self._release_device_memory()
            half = block.shape[0] // 2
            self._merge_into_chunk(chunk_index, ids[:half], block[:half])
            self._merge_into_chunk(chunk_index, ids[half:], block[half:])
            return
        self._best[chunk_index] = merged

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
