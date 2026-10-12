"""Universe lists: one scan of "everything that can ever exist", then cheap filtering.

The ground truth of a request is the exact top-k of the vectors that are ALIVE when the
request runs: the initial base plus the vectors inserted so far, minus the deleted ones.
Every workload set of a family shares the same base and the same insertion order, so all of
their active sets are subsets of one fixed collection, the *universe*:

    universe = base (ids 0 .. num_base - 1) + the first M vectors of the insertion order
               (ids num_base .. num_base + M - 1),   M = most insertions of any set

For every query vector that is requested anywhere we scan the universe ONCE and keep its
``K`` nearest vectors (ordered by distance, ties by the smaller id). The answer for any set
and step is then read from that list:

    keep the entries that are alive in this set at this step, take the first k of them.

Why this is exact: restricting a total order to a subset keeps the relative order, and every
member of the subset that ranks ahead of the last list entry is itself in the list. So if at
least k list entries are alive, the first k of them ARE the k nearest alive vectors. If fewer
are alive the caller scans the active set for that request (the fallback), so a too short
list costs time, never correctness.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from math import exp, lgamma, log, log1p
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from semantic_vdb.binio import BinReader
from semantic_vdb.groundtruth import KnnAccumulator, iter_active_blocks, prefetch_blocks
from semantic_vdb.parallel import (
    ParallelPlan,
    gpu_worker_environment,
    run_cpu_jobs,
    spawn_pool,
    split_ranges,
)
from semantic_vdb.resume import ResumeState, array_digest, sync_path, touch_durable

# Probability that a list is too short for one (query, step) and the fallback is needed.
DEFAULT_FAILURE_PROBABILITY = 1e-9
# Queries scanned together against one pass over the universe.
QUERY_GROUP_ROWS = 1 << 19


# ---------------------------------------------------------------------------
# how long a list must be
# ---------------------------------------------------------------------------


def _probability_fewer_than(needed: int, trials: int, success_probability: float) -> float:
    """P[Binomial(trials, p) < needed]: the chance that fewer than ``needed`` entries live."""
    if success_probability >= 1.0:
        return 0.0 if trials >= needed else 1.0
    log_success = log(success_probability)
    log_failure = log1p(-success_probability)
    log_trials_factorial = lgamma(trials + 1)
    total = 0.0
    for alive in range(min(needed, trials + 1)):
        log_term = (
            log_trials_factorial
            - lgamma(alive + 1)
            - lgamma(trials - alive + 1)
            + alive * log_success
            + (trials - alive) * log_failure
        )
        total += exp(log_term)
    return min(total, 1.0)


def choose_list_length(
    needed: int,
    alive_fraction: float,
    universe_size: int,
    failure_probability: float = DEFAULT_FAILURE_PROBABILITY,
) -> int:
    """Shortest list length ``K`` whose chance of holding fewer than ``needed`` live entries
    is at most ``failure_probability``.

    An entry is alive with probability ``alive_fraction`` (the active set is that fraction of
    the universe, and deletions and insertions do not depend on the rank of an entry). The
    result never exceeds the universe, where the list simply is the whole ranking.
    """
    if needed < 1:
        raise ValueError("needed must be positive")
    if not 0.0 < alive_fraction <= 1.0:
        raise ValueError("alive_fraction must be in (0, 1]")
    if universe_size < 1:
        raise ValueError("universe_size must be positive")
    if universe_size <= needed:
        return universe_size

    def enough(length: int) -> bool:
        return _probability_fewer_than(needed, length, alive_fraction) <= failure_probability

    # Grow geometrically until the list is long enough, then bisect (the tail shrinks as
    # the list grows). `low` is the smallest length that is still possible.
    low = high = needed
    while not enough(high):
        if high >= universe_size:
            return universe_size
        low = high + 1
        high = min(high * 2, universe_size)
    return _bisect_length(enough, low, high)


def _bisect_length(enough: Callable[[int], bool], low: int, high: int) -> int:
    """Smallest length in ``[low, high]`` for which ``enough`` holds (``high`` satisfies it)."""
    while low < high:
        middle = (low + high) // 2
        if enough(middle):
            high = middle
        else:
            low = middle + 1
    return high


# ---------------------------------------------------------------------------
# reading an answer out of a list
# ---------------------------------------------------------------------------


def select_alive_prefix(
    list_ids: np.ndarray,
    list_dists: np.ndarray,
    alive_limit: int,
    deleted_mask: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From each list row keep the entries alive now and return the first ``k`` of them.

    ``list_ids`` and ``list_dists`` are ``(n, K)`` rows ordered nearest first. An entry is
    alive if its id is below ``alive_limit`` (``num_base`` + the vectors inserted so far, so
    later insertions are not there yet) and it is not marked in ``deleted_mask``.

    Returns ``(ids, dists, enough)``: ``(n, k)`` arrays and a boolean per row that is False
    when fewer than ``k`` entries are alive. The arrays of such rows are not meaningful.
    """
    n, length = list_ids.shape
    if k > length:
        raise ValueError(f"cannot take {k} answers from lists of length {length}")

    # Alive = already inserted and not deleted. `deleted_mask` is indexed by global id.
    alive = list_ids < alive_limit
    alive &= ~deleted_mask[list_ids]
    enough = alive.sum(axis=1) >= k

    # A stable sort on "not alive" moves the alive entries to the front and keeps their
    # nearest-first order, so the first k columns are the answer.
    order = np.argsort(~alive, axis=1, kind="stable")[:, :k]
    ids = np.take_along_axis(list_ids, order, axis=1)
    dists = np.take_along_axis(list_dists, order, axis=1)
    return ids, dists, enough


# ---------------------------------------------------------------------------
# scanning the universe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanJob:
    """Everything a worker needs to scan the universe for a range of queries.

    Only paths and numbers, never arrays: the job is pickled to a spawned process and the
    big arrays are shared through memory-mapped ``.npy`` files.
    """

    base_path: str
    insert_path: Optional[str]
    num_inserts: int  # M: how many vectors of the insertion order belong to the universe
    metric: str
    device: str  # "cpu" or "gpu"
    device_ordinal: Optional[int]  # cupy device number inside THIS worker (gpu only)
    queries_path: str
    ids_path: str
    dists_path: str
    start: int
    stop: int
    list_length: int
    base_batch_rows: int
    query_chunk_rows: int
    query_group_rows: int
    # Directory of "<first>_<last>.done" markers, one per finished query group; None = no
    # resume. A group with a marker is skipped, because its rows are already in the list files.
    done_dir: Optional[str] = None


def universe_blocks(
    base: BinReader, insert_pool: Optional[BinReader], num_inserts: int, block_rows: int
):
    """The whole universe as ``(global_ids, vectors)`` blocks: base, then the insertions."""
    total = base.num_rows + num_inserts
    nothing_deleted = np.zeros(total, dtype=bool)
    return iter_active_blocks(
        base, insert_pool if num_inserts > 0 else None, num_inserts, nothing_deleted, block_rows
    )


def run_scan_job(job: ScanJob) -> int:
    """Fill rows ``[start, stop)`` of the list files; returns the number of queries done."""
    queries = np.load(job.queries_path, mmap_mode="r")
    list_ids = np.load(job.ids_path, mmap_mode="r+")
    list_dists = np.load(job.dists_path, mmap_mode="r+")

    if job.device == "gpu":
        import cupy

        scope = cupy.cuda.Device(job.device_ordinal)
    else:
        scope = contextlib.nullcontext()

    base = BinReader(job.base_path)
    insert_pool = BinReader(job.insert_path) if job.insert_path and job.num_inserts > 0 else None
    try:
        with scope:
            for first in range(job.start, job.stop, job.query_group_rows):
                last = min(first + job.query_group_rows, job.stop)
                marker = Path(job.done_dir) / f"{first}_{last}.done" if job.done_dir else None
                if marker is not None and marker.exists():
                    continue  # finished by an earlier run of this build
                group = np.array(queries[first:last])

                accumulator = KnnAccumulator(
                    group, job.list_length, job.metric, job.device, job.query_chunk_rows
                )
                # One pass over the universe per query group; the next block is read and
                # converted by a helper thread while the current one is being scanned.
                blocks = universe_blocks(base, insert_pool, job.num_inserts, job.base_batch_rows)
                for ids, vectors in prefetch_blocks(blocks):
                    accumulator.add_block(ids, vectors)
                result = accumulator.finish()

                list_ids[first:last] = result.neighbour_ids
                list_dists[first:last] = result.neighbour_dists
                if marker is not None:
                    # The marker may only appear once the rows are safely on disk.
                    list_ids.flush()
                    list_dists.flush()
                    sync_path(job.ids_path)
                    sync_path(job.dists_path)
                    touch_durable(marker)
    finally:
        base.close()
        if insert_pool is not None:
            insert_pool.close()
    list_ids.flush()
    list_dists.flush()
    return job.stop - job.start


@dataclass(frozen=True)
class UniverseLists:
    """The finished lists of one group of queries (memory-mapped, read only)."""

    ids: np.ndarray  # (n, K) int32 global ids, nearest first
    dists: np.ndarray  # (n, K) float32 distances
    list_length: int


def _list_files_fit(ids_path: Path, dists_path: Path, count: int, list_length: int) -> bool:
    """Do the list files of an earlier attempt exist with the shape this scan needs?"""
    for path, dtype in ((ids_path, np.int32), (dists_path, np.float32)):
        if not path.is_file():
            return False
        try:
            rows = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            return False
        if rows.shape != (count, list_length) or rows.dtype != dtype:
            return False
    return True


def scan_lists(
    queries: np.ndarray,
    list_length: int,
    base_path: Path,
    insert_path: Optional[Path],
    num_inserts: int,
    metric: str,
    plan: ParallelPlan,
    scratch_dir: Path,
    name: str,
    query_group_rows: int = QUERY_GROUP_ROWS,
    resume: Optional[ResumeState] = None,
) -> UniverseLists:
    """Scan the universe for ``queries`` and return their top-``list_length`` lists.

    The scan is split into contiguous query ranges, one per worker (a GPU worker per GPU
    process, or a CPU worker). With a single worker everything runs in this process.

    With ``resume`` every finished query group leaves a marker, and a repeated call (after an
    interruption) scans only the groups without one. The list files of the earlier call are
    reused only when the queries, the list length and the universe are the same.
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)
    count = len(queries)
    queries_path = scratch_dir / f"{name}_queries.npy"
    ids_path = scratch_dir / f"{name}_ids.npy"
    dists_path = scratch_dir / f"{name}_dists.npy"
    done_dir = scratch_dir / f"{name}_done" if resume is not None else None

    keep_rows = False
    if resume is not None:
        # An earlier attempt of the same build must have scanned exactly these queries.
        resume.check_or_set(
            f"scan_{name}",
            {
                "queries_sha256": array_digest(queries),
                "list_length": list_length,
                "num_inserts": num_inserts,
                "metric": metric,
                "base_bytes": Path(base_path).stat().st_size,
                "insert_bytes": Path(insert_path).stat().st_size if insert_path else 0,
            },
        )
        keep_rows = _list_files_fit(ids_path, dists_path, count, list_length)
        if not keep_rows:
            shutil.rmtree(done_dir, ignore_errors=True)  # markers without rows are worthless
        done_dir.mkdir(parents=True, exist_ok=True)
    np.save(queries_path, np.ascontiguousarray(queries))

    if not keep_rows:
        # The output files exist before the workers start; each worker fills its own rows.
        ids_file = np.lib.format.open_memmap(
            ids_path, mode="w+", dtype=np.int32, shape=(count, list_length)
        )
        dists_file = np.lib.format.open_memmap(
            dists_path, mode="w+", dtype=np.float32, shape=(count, list_length)
        )
        del ids_file, dists_file

    def make_job(start: int, stop: int, ordinal: Optional[int]) -> ScanJob:
        return ScanJob(
            base_path=str(base_path),
            insert_path=str(insert_path) if insert_path else None,
            num_inserts=num_inserts,
            metric=metric,
            device=plan.device,
            device_ordinal=ordinal,
            queries_path=str(queries_path),
            ids_path=str(ids_path),
            dists_path=str(dists_path),
            start=start,
            stop=stop,
            list_length=list_length,
            base_batch_rows=plan.base_batch_rows,
            query_chunk_rows=plan.query_chunk_rows,
            query_group_rows=query_group_rows,
            done_dir=str(done_dir) if done_dir is not None else None,
        )

    ranges = split_ranges(count, plan.scan_workers)
    if plan.device == "cpu":
        run_cpu_jobs(
            run_scan_job, [make_job(a, b, None) for a, b in ranges], plan.cpu_workers, plan.node_cpus
        )
    else:
        per_gpu = plan.workers_per_gpu
        if len(ranges) == 1:
            # One worker on one GPU: run in this process, which already owns a CUDA context.
            run_scan_job(make_job(ranges[0][0], ranges[0][1], plan.gpus[0].ordinal))
        else:
            _scan_on_gpu_workers(plan, ranges, per_gpu, make_job)

    return UniverseLists(
        ids=np.load(ids_path, mmap_mode="r"),
        dists=np.load(dists_path, mmap_mode="r"),
        list_length=list_length,
    )


def _scan_on_gpu_workers(plan: ParallelPlan, ranges, per_gpu: int, make_job) -> None:
    """Run the query ranges on spawned workers: ``per_gpu`` processes for each GPU.

    A worker sees exactly one GPU (``CUDA_VISIBLE_DEVICES``), so inside it the device number
    is always 0. The ranges are dealt out to the GPUs in order, and all pools run at once.
    """
    with contextlib.ExitStack() as stack:
        pending = []
        for gpu_position, gpu in enumerate(plan.gpus):
            first = gpu_position * per_gpu
            mine = ranges[first : first + per_gpu]
            if not mine:
                continue
            pool = stack.enter_context(
                spawn_pool(
                    len(mine),
                    environment=gpu_worker_environment(gpu.info),
                    cpus=plan.gpu_cpus[gpu_position],
                )
            )
            jobs = [make_job(a, b, 0) for a, b in mine]
            pending.append(pool.map_async(run_scan_job, jobs))
        for result in pending:
            result.get()  # raises the worker's exception, if any


# ---------------------------------------------------------------------------
# which vectors need a list (shared between group sizes)
# ---------------------------------------------------------------------------


@dataclass
class PoolRequests:
    """A pool and the rows of it that some stream requests."""

    vectors: np.ndarray  # (pool_size, d)
    requested_rows: np.ndarray  # sorted, distinct pool rows
    # (groups, size) when the pool is "group after group with equal sizes" (N2 pools); only
    # such pools can borrow lists from a larger pool of the same family.
    layout: Optional[Tuple[int, int]] = None


@dataclass
class ListRequests:
    vectors: np.ndarray  # (n, d) the distinct vectors to scan
    list_index: List[np.ndarray]  # per pool: pool row -> row in `vectors` (-1 = not requested)
    shared_rows: List[int]  # per pool: requested rows served by the reference pool's lists
    own_rows: List[int]  # per pool: requested rows that needed a list of their own


def matching_rows(
    vectors: np.ndarray,
    reference: np.ndarray,
    num_groups: int,
    size: int,
    reference_size: int,
    groups_per_step: int = 1024,
) -> np.ndarray:
    """For every row of a pool (``num_groups x size``): is it byte-for-byte the row with the
    same member number in the same group of the reference pool (``reference_size`` members)?"""
    dim = vectors.shape[1]
    equal = np.empty(num_groups * size, dtype=bool)
    for first in range(0, num_groups, groups_per_step):
        last = min(first + groups_per_step, num_groups)
        mine = vectors[first * size : last * size].reshape(last - first, size, dim)
        theirs = reference[first * reference_size : last * reference_size].reshape(
            last - first, reference_size, dim
        )[:, :size, :]
        equal[first * size : last * size] = (mine == theirs).all(axis=2).reshape(-1)
    return equal


def plan_list_requests(pools: Sequence[PoolRequests]) -> ListRequests:
    """Decide which distinct vectors need a universe list and where each pool row finds it.

    The largest N2 pool is the reference. A row of a smaller pool whose vector is exactly
    the same as the matching row of the reference pool (same group, same member number)
    uses the reference's list. Every other requested row gets a list of its own. Nothing
    is assumed about how the pools were generated: equality is checked on the vectors, and
    a row that differs merely costs one more list.
    """
    dim = pools[0].vectors.shape[1]
    layouts = [pool.layout for pool in pools]
    reference_position = None
    for position, layout in enumerate(layouts):
        if layout is not None and (
            reference_position is None or layout[1] > layouts[reference_position][1]
        ):
            reference_position = position

    # Rows (of the reference pool) and extra vectors that need lists.
    reference_rows: List[np.ndarray] = []
    own_rows: Dict[int, np.ndarray] = {}
    served_by_reference: Dict[int, np.ndarray] = {}
    reference_target: Dict[int, np.ndarray] = {}  # pool position -> reference row per requested row

    for position, pool in enumerate(pools):
        requested = pool.requested_rows
        if position == reference_position or reference_position is None or pool.layout is None:
            if position == reference_position:
                reference_rows.append(requested)
                served_by_reference[position] = np.ones(len(requested), dtype=bool)
                reference_target[position] = requested
            else:
                served_by_reference[position] = np.zeros(len(requested), dtype=bool)
            continue

        groups, size = pool.layout
        reference_pool = pools[reference_position]
        _, reference_size = reference_pool.layout
        same = matching_rows(pool.vectors, reference_pool.vectors, groups, size, reference_size)
        matches = same[requested]
        mapped = (requested // size) * reference_size + (requested % size)
        served_by_reference[position] = matches
        reference_target[position] = mapped
        reference_rows.append(mapped[matches])

    needed_reference = (
        np.unique(np.concatenate(reference_rows)) if reference_rows else np.empty(0, np.int64)
    )

    vectors_parts: List[np.ndarray] = []
    if reference_position is not None and len(needed_reference) > 0:
        vectors_parts.append(pools[reference_position].vectors[needed_reference])
    next_index = len(needed_reference)

    list_index: List[np.ndarray] = []
    shared_counts: List[int] = []
    own_counts: List[int] = []
    for position, pool in enumerate(pools):
        index = np.full(len(pool.vectors), -1, dtype=np.int64)
        requested = pool.requested_rows
        served = served_by_reference[position]
        if served.any():
            targets = reference_target[position][served]
            index[requested[served]] = np.searchsorted(needed_reference, targets)
        unserved = requested[~served]
        if len(unserved) > 0:
            index[unserved] = next_index + np.arange(len(unserved))
            vectors_parts.append(pool.vectors[unserved])
            next_index += len(unserved)
        list_index.append(index)
        shared_counts.append(int(served.sum()))
        own_counts.append(int(len(unserved)))

    vectors = (
        np.concatenate(vectors_parts, axis=0)
        if vectors_parts
        else np.empty((0, dim), dtype=pools[0].vectors.dtype)
    )
    return ListRequests(vectors, list_index, shared_counts, own_counts)


# ---------------------------------------------------------------------------
# the fallback: scan the active set
# ---------------------------------------------------------------------------


class ActiveSetScanner:
    """Exact top-k over the vectors alive at one moment of one set (the fallback path).

    This is the scan the ground truth used before the universe lists existed. It is only
    run for the few requests whose list had fewer than k live entries.
    """

    def __init__(self, base_path: Path, insert_path: Optional[Path], metric: str, plan: ParallelPlan):
        self._base = BinReader(base_path)
        self._insert = BinReader(insert_path) if insert_path is not None else None
        self._metric = metric
        self._plan = plan

    def close(self) -> None:
        self._base.close()
        if self._insert is not None:
            self._insert.close()

    def top(self, queries: np.ndarray, k: int, inserted: int, deleted_mask: np.ndarray):
        from semantic_vdb.groundtruth import exact_knn

        blocks = iter_active_blocks(
            self._base,
            self._insert if inserted > 0 else None,
            inserted,
            deleted_mask,
            self._plan.base_batch_rows,
        )
        return exact_knn(
            prefetch_blocks(blocks), queries, k, self._metric, self._plan.device,
            self._plan.query_chunk_rows,
        )
