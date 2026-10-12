"""Workload builder: seeds -> groups -> streams -> updates -> ground truth.

One build makes a FAMILY of workload sets. A set is one pool of group members (one target
overlap and one group size) combined with one update percentage:

    <base_set>/workloads/<name>/
        family.json  build_report.json
        shared/                  results later runs reuse (shared_store.py): neighbours and PCA
                                 bases, noise scale per overlap, evaluation ground truth, ...
        o80/ o50/ ...            N2 pools: one directory per target overlap ("o<percent>")
            g50/ g200/ ...       one directory per group size
                u0/ u25/ ...     one complete workload set per update percentage
        g1/                      pool of given queries (no noise: no overlap level)
        external/                externally grouped vectors (no noise: no overlap level)

A finished family can later receive more pools (``[workload] existing = extend``): see
``extend.py``. Pools that exist are never changed.

Every set holds ``total_requests`` requests: ``updates`` of them are updates (one insertion
plus one deletion each) and the rest are reads (queries). A set directory contains:

    pool/                   all group members (unique vectors) and their group ids
    seeds.<ext>             the original seed queries (not for external groups)
    eval_query.<ext>        the queries used for the periodic recall checkpoints
    stream_zipf_<s>/        per Zipf exponent: the queries executed in every step
    stream_uniform_unique/  (optional) the same layout for the duplicate-free uniform stream
    updates/                global ids deleted in every step
    groundtruth/            exact top-k per step for the stream queries and the evaluation
                            queries (step 0 = the initial base, before any update); with
                            [groundtruth] stream_groundtruth = false only the evaluation
                            queries (groundtruth/eval) have ground truth
    workload.json           parameters, calibration, diagnostics and file hashes

Order inside a step: delete -> insert -> queries. A query's ground truth is computed on the
active set AFTER the step's deletions and insertions.

What is computed once and shared (this is where the build time goes):

* the neighbour search and the PCA bases: once for all pools; the noise scale alpha: once
  per target overlap;
* the Zipf streams: drawn once per pool at the longest read count, every set uses a prefix;
* the ground truth: ONE scan of the "universe" (the base plus every vector any set inserts)
  per distinct query vector; every set and step then reads its answer from that list by
  dropping the vectors that are not alive (``universe.py``);
* the deleted ids and the evaluation ground truth of a percentage: once for all pools;
* files that are identical in several sets are hard-linked (``layout.py``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from semantic_vdb import __version__
from semantic_vdb.base_set import BaseSet, load_base_set
from semantic_vdb.binio import write_array, write_groundtruth, write_ids
from semantic_vdb.config import WorkloadConfig
from semantic_vdb.family import FAMILY_FORMAT_VERSION, ExistingFamily, common_fingerprint
from semantic_vdb.layout import (
    PERSISTENT_SHARED_DIR,
    RESUME_DIR,
    SHARED_DIR,
    UNIVERSE_DIR,
    BoundedWriter,
    hash_files,
    link_tree,
    overlap_dir_name,
    pool_dir_name,
)
from semantic_vdb.manifest import read_json, sha256_file, write_json
from semantic_vdb.n2 import (
    N2Shared,
    OverlapCalibration,
    calibrate_overlap,
    calibration_settings,
    make_pool,
    pool_statistics,
    prepare_neighbours_and_bases,
)
from semantic_vdb.parallel import ParallelPlan, check_host_memory, resolve_plan
from semantic_vdb.resume import ResumeError, ResumeState, array_digest
from semantic_vdb.shared_store import SharedStore
from semantic_vdb.sources import SeedSource, load_eval_queries, load_seed_source
from semantic_vdb.stream import (
    Stream,
    build_stream,
    build_unique_stream,
    repeat_diagnostics,
    step_boundaries,
)
from semantic_vdb.universe import (
    ActiveSetScanner,
    PoolRequests,
    UniverseLists,
    choose_list_length,
    plan_list_requests,
    scan_lists,
    select_alive_prefix,
)
from semantic_vdb.updates import RequestMix, UpdateSchedule, draw_deletions, make_schedule

FORMAT_VERSION = FAMILY_FORMAT_VERSION
# Directory name of the duplicate-free uniform stream, and the second entry of its random
# seed sequence (Zipf streams use their position in the exponent list, 0, 1, 2, ...), so
# adding an exponent never changes the uniform stream.
UNIQUE_STREAM_NAME = "stream_uniform_unique"
UNIQUE_STREAM_SEED_KEY = 1_000_003
# Host bytes per request of a master stream (three int64 arrays), for the memory estimate.
BYTES_PER_STREAM_REQUEST = 24

Log = Callable[[str], None]


class WorkloadError(RuntimeError):
    """The workload cannot be built from the given configuration and data."""


def _no_log(message: str) -> None:
    return None


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _fingerprint(config: WorkloadConfig, base_set: BaseSet) -> dict:
    """What an unfinished build must have in common with a build that continues it.

    The machine resources ([parallel]) are left out: the files do not depend on them.
    """
    description = dataclasses.asdict(config)
    description.pop("parallel", None)
    return {
        "format_version": FORMAT_VERSION,
        "tool_version": __version__,
        "config": description,
        "base_set": {
            "base_row_ids_sha256": base_set.base_row_ids_sha256,
            "insert_row_ids_sha256": base_set.insert_row_ids_sha256,
            "num_base": base_set.num_base,
            "num_inserts": base_set.num_inserts,
            "dtype": str(base_set.dtype),
            "dim": base_set.dim,
            "metric": base_set.metric,
        },
    }


def _open_work_dir(work_dir: Path, fingerprint: dict, log: Log) -> ResumeState:
    """Create the working directory of a build, or open the unfinished one it continues.

    A directory that holds the resume state of an earlier build is reused only when its
    fingerprint is equal to this build's; otherwise the build stops and says what differs.
    A directory without such a state was not made by a build that can be continued and is
    replaced.
    """
    if work_dir.exists() and ResumeState.exists(work_dir / RESUME_DIR):
        resume = ResumeState(work_dir / RESUME_DIR)
        try:
            resume.check_or_set("fingerprint", fingerprint)
        except ResumeError as error:
            raise WorkloadError(
                f"{work_dir} holds an unfinished build that does not match this configuration "
                f"({error}). Delete {work_dir} to start over, or restore the configuration."
            ) from error
        log(f"[resume] continuing the unfinished build in {work_dir}")
        return resume

    if work_dir.exists():
        shutil.rmtree(work_dir)  # not made by a build that can be continued
    work_dir.mkdir(parents=True)
    resume = ResumeState(work_dir / RESUME_DIR)
    resume.check_or_set("fingerprint", fingerprint)
    return resume


def _keep_or_remove(work_dir: Path, resume: ResumeState, log: Log) -> None:
    """After a failed build: keep the finished stages for a retry, or leave no trace."""
    if resume.any_progress():
        # Keep what is finished; the directory is named .building / .extending, so it cannot
        # be mistaken for a finished family.
        log(
            f"[resume] the build stopped; {work_dir} keeps its finished stages "
            f"({', '.join(resume.completed())}). Run the same command again to continue."
        )
    else:
        shutil.rmtree(work_dir, ignore_errors=True)


def build_workload(config: WorkloadConfig, log: Log = _no_log) -> Path:
    """Build the family of workload sets next to its base set; return the family directory.

    The build runs in ``<name>.building``. If it is interrupted after it finished a stage, that
    directory stays, and running the same build again continues with the first unfinished
    stage (``resume.py``). A directory that was made for another configuration is never
    reused or deleted silently: the build stops with an error that says what differs.

    When the family already exists the build stops, unless ``[workload] existing = extend``:
    then only the pools that are missing are added (``extend.py``).
    """
    base_set = load_base_set(config.base_set_dir)
    final_dir = base_set.directory / "workloads" / config.name
    if final_dir.exists():
        if config.existing != "extend":
            raise FileExistsError(
                f"{final_dir} already exists; refusing to overwrite a workload "
                "(set [workload] existing = extend to add missing pools to it)"
            )
        # Imported here because extend.py builds on this module.
        from semantic_vdb.extend import extend_family

        return extend_family(config, base_set, final_dir, log)

    work_dir = final_dir.with_name(config.name + ".building")
    resume = _open_work_dir(work_dir, _fingerprint(config, base_set), log)

    try:
        _build_family(config, base_set, work_dir, log, resume)
        # The staging areas only held data that the sets now link to or have copied. The
        # shared/ directory stays: it holds what a later run reuses.
        for staging in (SHARED_DIR, UNIVERSE_DIR, RESUME_DIR):
            shutil.rmtree(work_dir / staging, ignore_errors=True)
        os.replace(work_dir, final_dir)
    except BaseException:
        _keep_or_remove(work_dir, resume, log)
        raise
    return final_dir


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _stream_name(exponent: float) -> str:
    return f"stream_zipf_{exponent:g}"


def _step_name(step: int, suffix: str) -> str:
    return f"step_{step:05d}{suffix}"


def _checkpoint_steps(num_steps: int, every: int) -> List[int]:
    """Steps after which the evaluation queries are scored (step 0 = initial state)."""
    steps = {0, num_steps}
    steps.update(step for step in range(1, num_steps + 1) if step % every == 0)
    return sorted(steps)


class StageTimer:
    """Wall-clock seconds per stage of the build (written to build_report.json)."""

    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started


def _plan_schedules(
    config: WorkloadConfig, base_set: BaseSet
) -> Tuple[List[RequestMix], Dict[str, UpdateSchedule]]:
    """Validate every set before any data file is read; return the mixes and schedules."""
    mixes = config.request_mixes()
    schedules: Dict[str, UpdateSchedule] = {}
    for mix in mixes:
        try:
            # Every update is one insertion plus one deletion, so both totals are mix.updates.
            schedules[mix.name] = make_schedule(
                base_set.num_base,
                mix.updates,
                mix.updates,
                config.stream.num_steps,
                max_inserts=base_set.num_inserts,
            )
        except ValueError as error:
            raise type(error)(f"set {mix.name}: {error}") from error
    return mixes, schedules


@dataclass(frozen=True)
class PoolSpec:
    """One pool of the family, known before any vector exists."""

    name: str  # directory relative to the family: o80/g200, g1, external
    kind: str  # "n2", "none" or "external"
    group_size: Optional[int]
    num_groups: int
    size: int  # number of vectors the pool will hold
    overlap_name: Optional[str] = None  # "o80" (N2 pools only)
    target_overlap: Optional[float] = None  # 0.8 (N2 pools only)

    @property
    def key(self) -> str:
        """The name as ONE path component, for scratch files and stage markers."""
        return self.name.replace("/", "_")


def _pool_specs(config: WorkloadConfig, source: SeedSource) -> List[PoolSpec]:
    if source.mode == "external":
        sizes = source.external_group_sizes
        return [PoolSpec("external", "external", None, len(sizes), int(sizes.sum()))]
    num_seeds = len(source.seed_vectors)
    if config.groups.variants == "none":
        return [PoolSpec(pool_dir_name("none", 1), "none", 1, num_seeds, num_seeds)]
    # One pool for every (target overlap, group size) pair, overlap by overlap.
    specs: List[PoolSpec] = []
    for target in config.groups.target_overlaps:
        overlap_name = overlap_dir_name(target)
        for size in config.groups.group_sizes:
            specs.append(
                PoolSpec(
                    pool_dir_name("n2", size, overlap_name), "n2", size, num_seeds,
                    num_seeds * size, overlap_name, target,
                )
            )
    return specs


def _check_unique_stream_fits(
    config: WorkloadConfig, specs: List[PoolSpec], longest_reads: int
) -> None:
    """A duplicate-free stream needs at least as many distinct vectors as the longest set reads."""
    if not config.stream.uniform_unique:
        return
    too_small = [spec for spec in specs if spec.size < longest_reads]
    if too_small:
        details = ", ".join(f"{spec.name}: {spec.size} vectors" for spec in too_small)
        raise WorkloadError(
            f"the uniform_unique stream needs {longest_reads} different vectors (the longest "
            f"set has that many reads) but {details}; lower total_requests or raise the "
            "update percentages, drop the small group sizes, or add groups / group_size"
        )


# ---------------------------------------------------------------------------
# pools
# ---------------------------------------------------------------------------


@dataclass
class PoolData:
    spec: PoolSpec
    vectors: np.ndarray  # (pool_size, d), group after group
    group_sizes: np.ndarray  # (num_groups,)
    statistics: dict
    master_streams: Dict[str, Stream]  # drawn at the longest read count
    list_index: Optional[np.ndarray] = None  # pool row -> row in the stream lists (-1 = none)
    shared_rows: int = 0  # requested rows served by the reference pool's lists
    own_rows: int = 0  # requested rows that needed a list of their own

    def requested_rows(self) -> np.ndarray:
        return np.unique(np.concatenate([s.pool_index for s in self.master_streams.values()]))


def _master_streams(
    config: WorkloadConfig, group_sizes: np.ndarray, longest_reads: int
) -> Dict[str, Stream]:
    """Draw every stream of a pool once, at the longest read count; sets use prefixes."""
    streams: Dict[str, Stream] = {}
    for exponent_index, exponent in enumerate(config.stream.zipf_exponents):
        rng = np.random.default_rng(np.random.SeedSequence([config.stream.seed, exponent_index]))
        streams[_stream_name(exponent)] = build_stream(group_sizes, longest_reads, exponent, rng)
    if config.stream.uniform_unique:
        rng = np.random.default_rng(
            np.random.SeedSequence([config.stream.seed, UNIQUE_STREAM_SEED_KEY])
        )
        streams[UNIQUE_STREAM_NAME] = build_unique_stream(group_sizes, longest_reads, rng)
    return streams


def _build_pools(
    config: WorkloadConfig,
    base_set: BaseSet,
    source: SeedSource,
    specs: List[PoolSpec],
    longest_reads: int,
    plan: ParallelPlan,
    scratch: Path,
    log: Log,
    resume: Optional[ResumeState],
    store: SharedStore,
    existing: Optional[ExistingFamily],
) -> List[PoolData]:
    pools: List[PoolData] = []
    shared: Optional[N2Shared] = None
    calibrations: Dict[str, OverlapCalibration] = {}

    if any(spec.kind == "n2" for spec in specs):
        # Neighbours and PCA bases do not depend on the group size or the overlap: once for
        # all pools (or taken from the family that is being extended).
        shared = prepare_neighbours_and_bases(
            config, base_set, source.seed_vectors, plan, scratch, store, log, resume,
            reuse=existing.store if existing is not None else None,
        )
        # One noise scale per target overlap, in the order the overlaps first appear.
        for spec in specs:
            if spec.kind != "n2" or spec.overlap_name in calibrations:
                continue
            stored = existing.store.calibration(spec.overlap_name) if existing is not None else None
            calibration = calibrate_overlap(
                config, base_set, shared, plan, spec.overlap_name, spec.target_overlap, log,
                resume, stored,
            )
            calibrations[spec.overlap_name] = calibration
            if stored is None:
                # Kept in shared/cache.json so that a later run finds alpha without a search.
                settings = calibration_settings(config, spec.target_overlap)
                store.set_calibrations({spec.overlap_name: calibration.to_info(settings)})

    for spec in specs:
        if spec.kind == "external":
            vectors, group_sizes = source.external_vectors, source.external_group_sizes
            statistics = {"external_groups": int(len(group_sizes))}
        elif spec.kind == "none":
            # The given queries are the pool: no neighbour search, no calibration.
            vectors = source.seed_vectors
            group_sizes = np.ones(len(vectors), dtype=np.int64)
            statistics = {"variants": "none", "num_seeds": int(len(vectors))}
        else:
            stage = f"pool_{spec.key}"
            pool_path = scratch / f"pool_{spec.key}.npy"
            statistics_path = scratch / f"pool_{spec.key}_statistics.json"
            calibration = calibrations[spec.overlap_name]
            group_sizes = np.full(spec.num_groups, spec.group_size, dtype=np.int64)
            if resume is not None and resume.done(stage):
                log(f"[resume] pool {spec.name} and its statistics were finished earlier")
                vectors = np.array(np.load(pool_path, mmap_mode="r"))
                statistics = read_json(statistics_path)
            else:
                log(f"[pool {spec.name}] making {spec.size} variants ({plan.cpu_workers} CPU workers)")
                vectors = make_pool(
                    config, shared, spec.num_groups, base_set.dim, spec.group_size,
                    calibration.result.alpha, plan, scratch, spec.key,
                )
                statistics = {
                    "variants": "n2",
                    **pool_statistics(
                        config, base_set, shared, calibration, vectors, spec.group_size, plan
                    ),
                }
                if resume is not None:
                    write_json(statistics_path, statistics)
                    resume.mark(stage, files=[pool_path, statistics_path])
        streams = _master_streams(config, group_sizes, longest_reads)
        pools.append(PoolData(spec, vectors, group_sizes, statistics, streams))
    return pools


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------


def _estimate_host_bytes(
    config: WorkloadConfig,
    base_set: BaseSet,
    specs: List[PoolSpec],
    longest_reads: int,
    stream_list_length: int,
    eval_list_length: int,
    num_eval: int,
    num_seeds: int,
) -> int:
    """A generous estimate of the host memory the build holds at its peak (for preflight)."""
    item = base_set.dtype.itemsize
    streams = len(config.stream.zipf_exponents) + (1 if config.stream.uniform_unique else 0)
    # The pool is also copied once when it is loaded back from its scratch file.
    pool_bytes = 2 * sum(spec.size * base_set.dim * item for spec in specs)
    stream_bytes = len(specs) * streams * longest_reads * BYTES_PER_STREAM_REQUEST
    distinct = min(max(spec.size for spec in specs), longest_reads * streams * len(specs))
    list_bytes = distinct * stream_list_length * 8 + num_eval * eval_list_length * 8
    basis_bytes = num_seeds * config.groups.axes * base_set.dim * 8
    return pool_bytes + stream_bytes + list_bytes + basis_bytes


@dataclass
class _BuildContext:
    """Everything the set-writing code needs, in one place."""

    config: WorkloadConfig
    base_set: BaseSet
    source: SeedSource
    eval_queries: np.ndarray
    plan: ParallelPlan
    pools: List[PoolData]
    mixes: List[RequestMix]
    schedules: Dict[str, UpdateSchedule]
    universe_inserts: int
    stream_lists: Optional[UniverseLists]  # None when [groundtruth] stream_groundtruth = false
    eval_lists: Optional[UniverseLists]  # None when an existing family supplies the answers
    eval_list_length: int  # length of the evaluation lists (also when an existing family has them)
    distinct_queries: Optional[int]  # distinct stream query vectors with a list (None: no lists)
    base_path: Path
    insert_path: Optional[Path]
    work_dir: Path
    shared_dir: Path  # per-pool link sources, removed when the build is done
    store: SharedStore  # shared/ of this build (what a later run reuses)
    existing: Optional[ExistingFamily]  # the family that is being extended, if any
    log: Log
    resume: Optional[ResumeState] = None

    @property
    def link_store(self) -> SharedStore:
        """Where the files that every set links to come from.

        A new family writes them into its own shared/; a run that extends a family reads
        them from that family's shared/ (evaluation ground truth, deleted ids, queries).
        """
        return self.existing.store if self.existing is not None else self.store


def _input_description(source: SeedSource, eval_queries: np.ndarray) -> dict:
    """Digests of every query input, to be sure a continued build reads the same queries."""
    return {
        "original_queries": array_digest(source.original_queries),
        "seed_vectors": array_digest(source.seed_vectors),
        "external": array_digest(source.external_vectors, source.external_group_sizes),
        "eval_queries": array_digest(eval_queries),
    }


@dataclass
class _Plan:
    """Everything that is known before any vector file is read (see ``_plan_family``)."""

    mixes: List[RequestMix]
    schedules: Dict[str, UpdateSchedule]
    source: SeedSource
    eval_queries: np.ndarray
    inputs: dict  # digests of every query input
    specs: List[PoolSpec]  # every pool of the configuration (not only the missing ones)
    longest_reads: int
    universe_inserts: int
    universe_size: int
    stream_list_length: int
    eval_list_length: int


def _plan_family(config: WorkloadConfig, base_set: BaseSet) -> _Plan:
    """Validate everything that can fail without reading vector data, and size the build."""
    gt = config.groundtruth
    mixes, schedules = _plan_schedules(config, base_set)
    source: SeedSource = load_seed_source(config, base_set)
    eval_queries = load_eval_queries(config, source, base_set)
    inputs = _input_description(source, eval_queries)
    specs = _pool_specs(config, source)
    longest_reads = max(mix.reads for mix in mixes)
    _check_unique_stream_fits(config, specs, longest_reads)
    most_neighbours = config.evaluation.eval_k
    if gt.stream_groundtruth:
        most_neighbours = max(gt.stream_k, most_neighbours)
    if base_set.num_base < most_neighbours:
        raise WorkloadError(
            f"the base has {base_set.num_base} vectors, fewer than the {most_neighbours} "
            "neighbours the ground truth stores"
        )

    # The universe: the base plus every vector any set inserts. A vector of it is alive
    # with probability num_base / universe_size (the active set has a constant size).
    universe_inserts = max(schedule.total_inserts for schedule in schedules.values())
    universe_size = base_set.num_base + universe_inserts
    alive_fraction = base_set.num_base / universe_size
    # Without stream ground truth no stream list is made (length 0 also keeps the memory
    # estimate below free of it).
    stream_list_length = (
        choose_list_length(gt.stream_k, alive_fraction, universe_size)
        if gt.stream_groundtruth
        else 0
    )
    eval_list_length = choose_list_length(config.evaluation.eval_k, alive_fraction, universe_size)
    return _Plan(
        mixes, schedules, source, eval_queries, inputs, specs, longest_reads, universe_inserts,
        universe_size, stream_list_length, eval_list_length,
    )


def _plan_stream_lists(pools: List[PoolData]):
    """Which distinct vectors need a stream list, and where every pool row finds its list.

    Pools that share an overlap level are planned together (the largest of them is the
    reference whose lists the smaller ones borrow, ``plan_list_requests``). Pools of another
    overlap level have other noise, so they are planned on their own and their lists follow
    behind. Returns the vectors to scan; sets ``list_index`` and the counters of every pool.
    """
    groups: Dict[Optional[str], List[PoolData]] = {}
    for pool in pools:
        groups.setdefault(pool.spec.overlap_name, []).append(pool)

    vectors_parts: List[np.ndarray] = []
    offset = 0  # lists of the earlier groups; this group's list rows start behind them
    for members in groups.values():
        requests = plan_list_requests(
            [
                PoolRequests(
                    pool.vectors,
                    pool.requested_rows(),
                    # (groups, group size): N2 pools can lend their lists to smaller N2 pools
                    (pool.spec.num_groups, pool.spec.group_size)
                    if pool.spec.kind == "n2"
                    else None,
                )
                for pool in members
            ]
        )
        for pool, index, shared_rows, own_rows in zip(
            members, requests.list_index, requests.shared_rows, requests.own_rows
        ):
            # -1 means "this row has no list" and must stay -1 after the shift.
            pool.list_index = np.where(index >= 0, index + offset, -1)
            pool.shared_rows, pool.own_rows = shared_rows, own_rows
        vectors_parts.append(requests.vectors)
        offset += len(requests.vectors)
    return np.concatenate(vectors_parts, axis=0)


def _build_family(
    config: WorkloadConfig,
    base_set: BaseSet,
    work_dir: Path,
    log: Log,
    resume: ResumeState,
    existing: Optional[ExistingFamily] = None,
    planned: Optional[_Plan] = None,
) -> None:
    """Build the pools of the family into ``work_dir``.

    With ``existing`` (a finished family that is being extended) only the pools that family
    does not have are built; the evaluation ground truth, the deleted ids and the neighbours
    and bases come from its ``shared/`` directory, and nothing of it is written again.
    """
    timer = StageTimer()
    gt = config.groundtruth
    finished_before = resume.completed()  # stages an earlier, interrupted run completed
    shared_dir, universe_dir = work_dir / SHARED_DIR, work_dir / UNIVERSE_DIR
    store = SharedStore(work_dir / PERSISTENT_SHARED_DIR)

    # ---- everything that can fail without reading vector data fails first --------------
    with timer("plan"):
        if planned is None:
            planned = _plan_family(config, base_set)
        try:
            resume.check_or_set("inputs", planned.inputs)
        except ResumeError as error:
            raise WorkloadError(
                f"{work_dir} holds an unfinished build made for other query files ({error}). "
                f"Delete {work_dir} to start over."
            ) from error
        mixes, schedules, source = planned.mixes, planned.schedules, planned.source
        eval_queries, longest_reads = planned.eval_queries, planned.longest_reads
        universe_inserts, universe_size = planned.universe_inserts, planned.universe_size
        stream_list_length, eval_list_length = planned.stream_list_length, planned.eval_list_length
        # A run that extends a family builds only the pools that family does not have.
        specs = planned.specs
        if existing is not None:
            specs = [spec for spec in specs if spec.name not in existing.pool_names]

    with timer("resources"):
        plan = resolve_plan(config.parallel, gt.device, gt.base_batch_rows, gt.query_chunk_rows, log)
        if plan.gpus:
            import cupy

            # The main process uses the first selected GPU for the searches it runs itself.
            cupy.cuda.Device(plan.gpus[0].ordinal).use()
        num_seeds = 0 if source.seed_vectors is None else len(source.seed_vectors)
        check_host_memory(
            _estimate_host_bytes(
                config, base_set, specs, longest_reads, stream_list_length, eval_list_length,
                len(eval_queries), num_seeds,
            )
        )
        log(f"[parallel] {plan.describe()}")
        if not np.issubdtype(base_set.dtype, np.integer):
            # For float data the last bits of a distance can depend on the batch sizes, so a
            # continued build has to use the sizes of the first run.
            try:
                resume.check_or_set(
                    "batch_sizes",
                    {"base_batch_rows": plan.base_batch_rows, "query_chunk_rows": plan.query_chunk_rows},
                )
            except ResumeError as error:
                raise WorkloadError(
                    f"{error}; set [groundtruth] base_batch_rows and query_chunk_rows to the "
                    f"values of the first run, or delete {work_dir} to start over"
                ) from error

    # ---- pools and streams ---------------------------------------------------------------
    with timer("pools"):
        pools = _build_pools(
            config, base_set, source, specs, longest_reads, plan, universe_dir, log, resume,
            store, existing,
        )
        for pool in pools:
            log(f"[pool {pool.spec.name}] {len(pool.vectors)} vectors in {len(pool.group_sizes)} groups")

    # ---- universe lists: one scan for everything that is requested anywhere ---------------
    with timer("universe_lists"):
        base_path = base_set.directory / f"base{base_set.suffix}"
        insert_path = (
            base_set.directory / f"insert_pool{base_set.suffix}" if universe_inserts > 0 else None
        )
        stream_lists: Optional[UniverseLists] = None
        distinct_queries: Optional[int] = None
        if gt.stream_groundtruth:
            stream_vectors = _plan_stream_lists(pools)
            log(
                f"[universe] {len(stream_vectors)} distinct query vectors, lists of "
                f"{stream_list_length} over {universe_size} vectors "
                f"({base_set.num_base} base + {universe_inserts} inserted)"
            )
            stream_lists = scan_lists(
                stream_vectors, stream_list_length, base_path, insert_path, universe_inserts,
                base_set.metric, plan, universe_dir, "stream", resume=resume,
            )
            distinct_queries = int(len(stream_vectors))
        else:
            log("[universe] stream ground truth is off: only the evaluation queries are scanned")
        eval_lists: Optional[UniverseLists] = None
        if existing is None:
            eval_lists = scan_lists(
                eval_queries, eval_list_length, base_path, insert_path, universe_inserts,
                base_set.metric, plan, universe_dir, "eval", resume=resume,
            )
        else:
            # The evaluation answers of every update percentage are already in the family.
            log("[universe] the evaluation ground truth comes from the existing family")

    # ---- the sets ---------------------------------------------------------------------------
    with timer("sets"):
        context = _BuildContext(
            config, base_set, source, eval_queries, plan, pools, mixes, schedules,
            universe_inserts, stream_lists, eval_lists, eval_list_length, distinct_queries,
            base_path, insert_path, work_dir, shared_dir, store, existing, log, resume,
        )
        records = _build_sets(context)

    with timer("manifest"):
        index = _family_index(context, records)
        if existing is None:
            index["common"] = common_fingerprint(config, base_set, planned.inputs)
            write_json(work_dir / "family.json", index)
        else:
            # The index of the extended family is merged and written by extend.py, last.
            write_json(work_dir / "family_fragment.json", index)
    # The report is written last so that it also contains the time of the manifest stage.
    write_json(
        work_dir / "build_report.json",
        {
            "mode": "initial" if existing is None else "extension",
            "pools_built": [pool.spec.name for pool in pools],
            "stage_seconds": timer.seconds,
            "parallel": plan.describe(),
            # Stages that an interrupted earlier run had finished (empty for an uninterrupted
            # build); their time is not in stage_seconds.
            "resumed_from_stages": finished_before,
            "stream_requests_answered_by_scan": sum(r.stream_fallbacks for r in records),
        },
    )


# ---------------------------------------------------------------------------
# answers from the lists
# ---------------------------------------------------------------------------


class _AnswerSource:
    """Turns universe lists into ground truth, with the scan of the active set as fallback."""

    def __init__(self, context: _BuildContext) -> None:
        self._context = context
        self._scanner = ActiveSetScanner(
            context.base_path, context.insert_path, context.base_set.metric, context.plan
        )

    def close(self) -> None:
        self._scanner.close()

    def answer(
        self,
        list_ids: np.ndarray,
        list_dists: np.ndarray,
        query_vectors: Callable[[np.ndarray], np.ndarray],
        k: int,
        inserted: int,
        deleted_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Answers for the given list rows, and how many of them needed the fallback.

        ``query_vectors(rows)`` yields the vectors of the rows that must be scanned because
        their list had fewer than k live entries.
        """
        alive_limit = self._context.base_set.num_base + inserted
        ids, dists, enough = select_alive_prefix(list_ids, list_dists, alive_limit, deleted_mask, k)
        missing = np.flatnonzero(~enough)
        if len(missing) > 0:
            # Requests of the same step that repeat a vector are scanned only once.
            unique_vectors, inverse = np.unique(
                query_vectors(missing), axis=0, return_inverse=True
            )
            result = self._scanner.top(unique_vectors, k, inserted, deleted_mask)
            inverse = inverse.reshape(-1)
            ids[missing] = result.neighbour_ids[inverse]
            dists[missing] = result.neighbour_dists[inverse]
        return ids, dists, int(len(missing))


# ---------------------------------------------------------------------------
# writing the sets
# ---------------------------------------------------------------------------


@dataclass
class _SetRecord:
    pool: PoolData
    mix: RequestMix
    directory: Path
    manifest_sha256: str = ""
    stream_fallbacks: int = 0  # requests of this set answered by scanning the active set
    eval_fallbacks: int = 0  # the same for the evaluation queries (shared by the pools)


def _write_common_files(context: _BuildContext) -> None:
    """Files that are the same in several sets: the queries (new family only) and the pools."""
    suffix = context.base_set.suffix
    if context.existing is None:
        # A new family keeps the queries in shared/common, which every set links to. An
        # extended family already has them there.
        common = context.store.common_dir
        if context.source.mode != "external":
            write_array(common / f"seeds{suffix}", context.source.seed_vectors)
        write_array(common / f"eval_query{suffix}", context.eval_queries)

    for pool in context.pools:
        directory = context.shared_dir / f"pool_{pool.spec.key}" / "pool"
        write_array(directory / f"pool{suffix}", pool.vectors)
        write_ids(
            directory / "pool_group_id.ibin",
            np.repeat(np.arange(len(pool.group_sizes)), pool.group_sizes),
        )


def _build_sets(context: _BuildContext) -> List[_SetRecord]:
    config, base_set, plan = context.config, context.base_set, context.plan
    suffix = base_set.suffix
    num_steps = config.stream.num_steps
    stream_k, eval_k = config.groundtruth.stream_k, config.evaluation.eval_k
    num_base = base_set.num_base

    store, link_store = context.store, context.link_store
    # An extended family takes the evaluation answers and the deleted ids of the existing
    # family (the link store); a new family computes and writes them into its own shared/.
    computing_shared = context.existing is None

    resume = context.resume
    if resume is None or not resume.done("common"):
        _write_common_files(context)
        if resume is not None:
            resume.mark("common")
    answers = _AnswerSource(context)
    writer = BoundedWriter(plan.cpu_workers)
    records: List[_SetRecord] = []

    # The stream lists are gathered row by row; the evaluation lists are used whole at every
    # checkpoint, so they are loaded once.
    with_stream_gt = context.stream_lists is not None
    stream_ids = context.stream_lists.ids if with_stream_gt else None
    stream_dists = context.stream_lists.dists if with_stream_gt else None
    if computing_shared:
        eval_ids = np.array(context.eval_lists.ids)
        eval_dists = np.array(context.eval_lists.dists)

    def eval_answer(
        inserted: int, deleted_mask: np.ndarray, directory: Path, step: int
    ) -> int:
        ids, dists, fallbacks = answers.answer(
            eval_ids, eval_dists, lambda rows: context.eval_queries[rows], eval_k, inserted,
            deleted_mask,
        )
        writer.submit(
            write_groundtruth, directory / "groundtruth" / "eval" / _step_name(step, ".bin"),
            ids, dists,
        )
        return fallbacks

    # Step 0 is the initial base for every set: its evaluation ground truth is shared.
    step0_fallbacks = 0
    if computing_shared:
        no_deletions = np.zeros(num_base + context.universe_inserts, dtype=bool)
        if resume is not None and resume.done("eval_step0"):
            step0_fallbacks = int(resume.info("eval_step0")["fallbacks"])
        else:
            step0_fallbacks = eval_answer(0, no_deletions, store.eval_step0_dir, 0)
            if resume is not None:
                writer.finish()  # the file must be complete before the marker is written
                writer = BoundedWriter(plan.cpu_workers)
                resume.mark("eval_step0", {"fallbacks": step0_fallbacks})

    try:
        for mix in context.mixes:
            stage = f"mix_{mix.name}"
            if resume is not None and resume.done(stage):
                context.log(f"[resume] set {mix.name}: finished earlier, kept")
                records.extend(_finished_records(context, mix))
                continue
            if resume is not None:
                # Whatever an interrupted attempt left of this set is incomplete: start it over.
                for pool in context.pools:
                    shutil.rmtree(context.work_dir / pool.spec.name / mix.name, ignore_errors=True)
                if computing_shared:
                    shutil.rmtree(store.mix_dir(mix.name), ignore_errors=True)
            schedule = context.schedules[mix.name]
            # Deleted ids depend only on the percentage and the seed, never on the pool.
            deletions = draw_deletions(schedule, config.updates.seed)
            mix_shared = link_store.mix_dir(mix.name)  # files every pool's set links to
            if computing_shared:
                for step, ids in enumerate(deletions, start=1):
                    if len(ids) > 0:
                        write_ids(mix_shared / "updates" / f"delete_step_{step:05d}.ibin", ids)

            if mix.updates == 0:
                # Nothing ever changes, so every checkpoint would repeat step 0.
                checkpoints = [0]
            else:
                checkpoints = _checkpoint_steps(num_steps, config.evaluation.checkpoint_every)
            boundaries = step_boundaries(mix.reads, num_steps)

            # Per pool: the prefix of every master stream that this set reads.
            set_dirs = {p.spec.name: context.work_dir / p.spec.name / mix.name for p in context.pools}
            streams = {
                p.spec.name: {name: master.head(mix.reads) for name, master in p.master_streams.items()}
                for p in context.pools
            }
            stream_fallbacks = {p.spec.name: 0 for p in context.pools}
            eval_fallbacks = step0_fallbacks

            deleted_mask = np.zeros(num_base + context.universe_inserts, dtype=bool)
            inserted = 0
            for step in range(1, num_steps + 1):
                # Order inside a step: delete, insert, then the queries of the step.
                deleted_mask[deletions[step - 1]] = True
                inserted += int(schedule.inserts_per_step[step - 1])
                first, last = boundaries[step - 1], boundaries[step]

                for pool in context.pools:
                    name_of_pool = pool.spec.name
                    set_dir = set_dirs[name_of_pool]
                    names = list(streams[name_of_pool])
                    if with_stream_gt:
                        # All streams of the step are answered together.
                        rows = np.concatenate(
                            [streams[name_of_pool][name].pool_index[first:last] for name in names]
                        )
                        list_rows = pool.list_index[rows]
                        if (list_rows < 0).any():
                            raise WorkloadError(f"pool {name_of_pool}: a requested row has no list")
                        ids, dists, fallbacks = answers.answer(
                            stream_ids[list_rows], stream_dists[list_rows],
                            lambda missing, rows=rows, pool=pool: pool.vectors[rows[missing]],
                            stream_k, inserted, deleted_mask,
                        )
                        stream_fallbacks[name_of_pool] += fallbacks

                    step_queries = int(last - first)
                    for index, name in enumerate(names):
                        stream = streams[name_of_pool][name]
                        writer.submit(
                            write_array, set_dir / name / _step_name(step, suffix),
                            pool.vectors[stream.pool_index[first:last]],
                        )
                        writer.submit(
                            write_ids, set_dir / name / _step_name(step, ".group_id.ibin"),
                            stream.group_ids[first:last],
                        )
                        if with_stream_gt:
                            part = slice(index * step_queries, (index + 1) * step_queries)
                            writer.submit(
                                write_groundtruth,
                                set_dir / "groundtruth" / name / _step_name(step, ".bin"),
                                ids[part], dists[part],
                            )
                if computing_shared and step in checkpoints:
                    eval_fallbacks += eval_answer(inserted, deleted_mask, mix_shared, step)
                if step % 10 == 0 or step == num_steps:
                    context.log(f"[groundtruth {mix.name}] step {step}/{num_steps} done")

            writer.finish()
            writer = BoundedWriter(plan.cpu_workers)

            info_path = link_store.mix_info_path(mix.name)
            if computing_shared:
                write_json(info_path, {"eval_requests_answered_by_scan": eval_fallbacks})
            else:
                eval_fallbacks = int(read_json(info_path)["eval_requests_answered_by_scan"])

            for pool in context.pools:
                set_dir = set_dirs[pool.spec.name]
                # Files shared with other sets are linked in; the rest was written above.
                link_tree(link_store.common_dir, set_dir)
                link_tree(context.shared_dir / f"pool_{pool.spec.key}", set_dir)
                link_tree(link_store.eval_step0_dir, set_dir)
                link_tree(mix_shared, set_dir)
                record = _finish_set(
                    context, pool, mix, schedule, checkpoints, streams[pool.spec.name], set_dir,
                    stream_fallbacks[pool.spec.name], eval_fallbacks,
                )
                records.append(record)
            if resume is not None:
                os.sync()  # the sets are many small files: flush them all before the marker
                resume.mark(stage)
    finally:
        answers.close()
        writer.finish()
    return records


def _finished_records(context: _BuildContext, mix: RequestMix) -> List[_SetRecord]:
    """The records of the sets of ``mix`` that an earlier run of this build finished."""
    records = []
    for pool in context.pools:
        set_dir = context.work_dir / pool.spec.name / mix.name
        manifest_path = set_dir / "workload.json"
        if not manifest_path.is_file():
            raise WorkloadError(
                f"{set_dir} is marked as finished but has no workload.json; "
                f"delete {context.work_dir} to start over"
            )
        groundtruth = read_json(manifest_path)["groundtruth"]
        records.append(
            _SetRecord(
                pool, mix, set_dir, sha256_file(manifest_path),
                int(groundtruth["stream_requests_answered_by_scan"] or 0),
                int(groundtruth["eval_requests_answered_by_scan"]),
            )
        )
    return records


def _set_config(config: WorkloadConfig, spec: PoolSpec) -> dict:
    """The configuration recorded in the workload.json of one set.

    The lists that define the pools hold only this pool's own group size and target overlap.
    A set then looks the same whether it was made together with other pools or added to the
    family later, and ``existing`` (what to do about an existing family) is not part of it.
    """
    description = dataclasses.asdict(config)
    description.pop("existing", None)
    if spec.kind == "n2":
        description["groups"]["group_sizes"] = (spec.group_size,)
        description["groups"]["target_overlaps"] = (spec.target_overlap,)
    return description


def _finish_set(
    context: _BuildContext,
    pool: PoolData,
    mix: RequestMix,
    schedule: UpdateSchedule,
    checkpoints: List[int],
    streams: Dict[str, Stream],
    set_dir: Path,
    stream_fallbacks: int,
    eval_fallbacks: int,
) -> _SetRecord:
    """Hash the files of one set and write its workload.json."""
    config, base_set, plan = context.config, context.base_set, context.plan

    stream_info = {}
    for name, stream in streams.items():
        diagnostics = repeat_diagnostics(stream)
        stream_info[name] = {
            "kind": stream.kind,
            "exponent": stream.exponent,
            **dataclasses.asdict(diagnostics),
        }
        context.log(
            f"[stream {pool.spec.name}/{mix.name}] {name}: exact repeats "
            f"{diagnostics.exact_repeat_fraction:.3f}, "
            f"group revisits {diagnostics.group_revisit_fraction:.3f}"
        )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if plan.device == "gpu":
        import cupy

        environment["cupy"] = cupy.__version__

    source = context.source
    # Without stream ground truth there are no stream lists: the counts about them are null.
    stream_gt = context.stream_lists is not None
    manifest = {
        "format_version": FORMAT_VERSION,
        "tool_version": __version__,
        "name": config.name,
        "set_name": mix.name,
        "pool": {
            "name": pool.spec.name,
            "kind": pool.spec.kind,
            "group_size": pool.spec.group_size,
            "overlap": pool.spec.overlap_name,
            "target_overlap": pool.spec.target_overlap,
            "vectors": int(len(pool.vectors)),
            "groups": int(len(pool.group_sizes)),
        },
        "update_set": {
            "update_percent": mix.update_percent,
            "total_requests": config.stream.total_requests,
            "reads": mix.reads,
            "insertions": schedule.total_inserts,
            "deletions": schedule.total_deletes,
        },
        "base_set": {
            "directory": str(base_set.directory),
            "base_row_ids_sha256": base_set.base_row_ids_sha256,
            "insert_row_ids_sha256": base_set.insert_row_ids_sha256,
            "num_base": base_set.num_base,
            "num_inserts_available": base_set.num_inserts,
            "dtype": str(base_set.dtype),
            "dim": base_set.dim,
            "metric": base_set.metric,
        },
        "config": _set_config(config, pool.spec),
        "seed_source": {
            "mode": source.mode,
            "num_original_queries": int(len(source.original_queries)),
            "num_groups": int(len(pool.group_sizes)),
            "pool_vectors": int(len(pool.vectors)),
        },
        "groups": pool.statistics,
        "streams": stream_info,
        "schedule": {
            "inserts_per_step": schedule.inserts_per_step,
            "deletes_per_step": schedule.deletes_per_step,
            "checkpoints": checkpoints,
            "id_rule": "base row i -> id i; j-th inserted vector -> id num_base + j; ids are never reused",
            "step_order": "delete, insert, then queries; ground truth uses the active set after both",
        },
        "groundtruth": {
            "method": "universe_lists",
            "stream_groundtruth": stream_gt,
            "universe_size": base_set.num_base + context.universe_inserts,
            "stream_list_length": context.stream_lists.list_length if stream_gt else None,
            "eval_list_length": context.eval_list_length,
            "distinct_query_vectors": context.distinct_queries,
            "pool_rows_with_own_list": pool.own_rows if stream_gt else None,
            "pool_rows_sharing_a_list": pool.shared_rows if stream_gt else None,
            "stream_requests_answered_by_scan": stream_fallbacks if stream_gt else None,
            "eval_requests_answered_by_scan": eval_fallbacks,
            "base_batch_rows": plan.base_batch_rows,
            "query_chunk_rows": plan.query_chunk_rows,
        },
        "environment": environment,
    }

    # Hash every other file so the exact output can be compared later (hard links once).
    paths = sorted(path for path in set_dir.rglob("*") if path.is_file())
    hashes = hash_files(paths, plan.cpu_workers)
    manifest["files"] = {
        str(path.relative_to(set_dir)): {"bytes": hashes[path][0], "sha256": hashes[path][1]}
        for path in paths
    }
    manifest_path = set_dir / "workload.json"
    write_json(manifest_path, manifest)
    return _SetRecord(
        pool, mix, set_dir, hash_files([manifest_path], 1)[manifest_path][1],
        stream_fallbacks, eval_fallbacks,
    )


def _family_index(context: _BuildContext, records: List[_SetRecord]) -> dict:
    """The pools, overlaps and sets of this build, for programs that walk through the family.

    A new family writes it as ``family.json``; a run that extends a family writes it as
    ``family_fragment.json``, which ``extend.py`` merges into the family's index.
    """
    config = context.config
    # Pool by pool, percentage by percentage, in the order of the configuration.
    pool_order = {pool.spec.name: position for position, pool in enumerate(context.pools)}
    mix_order = {mix.name: position for position, mix in enumerate(context.mixes)}
    ordered = sorted(records, key=lambda r: (pool_order[r.pool.spec.name], mix_order[r.mix.name]))

    overlaps: Dict[str, dict] = {}
    for pool in context.pools:
        if pool.spec.overlap_name is not None and pool.spec.overlap_name not in overlaps:
            overlaps[pool.spec.overlap_name] = {
                "name": pool.spec.overlap_name,
                "target_overlap": pool.statistics["target_overlap"],
                "alpha": pool.statistics["alpha"],
                "achieved_overlap": pool.statistics["calibration_overlap"],
                "tolerance": config.groups.overlap_tolerance,
            }

    return {
        "format_version": FORMAT_VERSION,
        "name": config.name,
        "overlaps": list(overlaps.values()),
        "pools": [
            {
                "name": pool.spec.name,
                "kind": pool.spec.kind,
                "group_size": pool.spec.group_size,
                "overlap": pool.spec.overlap_name,
                "vectors": int(len(pool.vectors)),
                "groups": int(len(pool.group_sizes)),
            }
            for pool in context.pools
        ],
        "sets": [
            {
                "path": f"{record.pool.spec.name}/{record.mix.name}",
                "pool": record.pool.spec.name,
                "set_name": record.mix.name,
                "update_percent": record.mix.update_percent,
                "reads": record.mix.reads,
                "insertions": record.mix.updates,
                "deletions": record.mix.updates,
                "workload_json_sha256": record.manifest_sha256,
            }
            for record in ordered
        ],
    }
