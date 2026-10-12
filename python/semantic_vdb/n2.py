"""N2 groups: local noise bases per seed, the noise scale alpha, and the pools of variants.

The work is split in the order the data flows, and the parts that are independent per seed
run on several CPU processes (``cpu_workers``):

1. ``prepare_neighbours_and_bases``
                        exact neighbours of every seed (a GPU/CPU scan of the base) and one
                        PCA basis per seed (parallel by seed). They depend only on the seeds
                        and the base, NOT on the group size or the overlap, so they are made
                        once for every pool of the family and kept in ``shared/n2``.
2. ``calibrate_overlap`` the noise scale ``alpha`` searched for ONE target overlap (a few
                        scans of a few hundred queries).
3. ``make_pool``       the members of every group for one group size and one alpha (parallel
                        by seed).
4. ``pool_statistics`` how similar the variants really are, measured on all seeds.

A worker never gets arrays through a pipe. It gets file paths and a range of seeds, reads
what it needs from memory-mapped ``.npy`` files and writes its rows into output files. The
random numbers of a seed come from ``SeedSequence([seed, 1, index])`` and do not depend on
which worker handles the seed, so the result is the same for any number of workers.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from semantic_vdb.base_set import BaseSet
from semantic_vdb.binio import BinReader, GroundTruth
from semantic_vdb.calibrate import CalibrationResult, find_alpha, mean_overlap, overlap_per_query
from semantic_vdb.config import WorkloadConfig
from semantic_vdb.groundtruth import exact_knn, iter_active_blocks, prefetch_blocks
from semantic_vdb.parallel import ParallelPlan, run_cpu_jobs, split_ranges
from semantic_vdb.perturb import (
    LocalBasis,
    apply_noise,
    draw_noise,
    fit_local_basis,
    make_variants,
)
from semantic_vdb.resume import ResumeState, describe_differences
from semantic_vdb.shared_store import SharedStore, SharedStoreError, n2_fingerprint

# Neighbour overlap is always measured on the top-10 lists.
OVERLAP_K = 10


class N2Error(RuntimeError):
    """The N2 groups cannot be built from the given configuration and data."""


# ---------------------------------------------------------------------------
# exact neighbours in the initial base
# ---------------------------------------------------------------------------


class BaseScanner:
    """Exact top-k search in the INITIAL base (nothing deleted, nothing inserted)."""

    def __init__(self, base_set: BaseSet, plan: ParallelPlan) -> None:
        self._base_set = base_set
        self._plan = plan
        self._base = base_set.open_base()

    def close(self) -> None:
        self._base.close()

    def read_rows(self, row_ids: np.ndarray) -> np.ndarray:
        return self._base.read_rows(row_ids)

    def top(self, queries: np.ndarray, k: int) -> GroundTruth:
        num_base = self._base_set.num_base
        blocks = iter_active_blocks(
            self._base, None, 0, np.zeros(num_base, dtype=bool), self._plan.base_batch_rows
        )
        # The next block is read and converted while the current one is being scanned.
        return exact_knn(
            prefetch_blocks(blocks),
            queries,
            k,
            self._base_set.metric,
            self._plan.device,
            self._plan.query_chunk_rows,
        )


@contextlib.contextmanager
def base_scanner(base_set: BaseSet, plan: ParallelPlan) -> Iterator[BaseScanner]:
    scanner = BaseScanner(base_set, plan)
    try:
        yield scanner
    finally:
        scanner.close()


# ---------------------------------------------------------------------------
# local bases (parallel by seed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasisJob:
    base_path: str
    neighbour_ids_path: str
    axes_path: str
    std_path: str
    residual_path: str
    start: int
    stop: int
    neighbours: int
    axes: int


def run_basis_job(job: BasisJob) -> int:
    """Fit the PCA basis of the seeds ``[start, stop)`` and store it."""
    neighbour_ids = np.load(job.neighbour_ids_path, mmap_mode="r")
    axes_out = np.load(job.axes_path, mmap_mode="r+")
    std_out = np.load(job.std_path, mmap_mode="r+")
    residual_out = np.load(job.residual_path, mmap_mode="r+")
    with BinReader(job.base_path) as base:
        for index in range(job.start, job.stop):
            # The seed's `neighbours` nearest base vectors are its local neighbourhood.
            ids = np.asarray(neighbour_ids[index, : job.neighbours], dtype=np.int64)
            basis = fit_local_basis(base.read_rows(ids), job.axes)
            axes_out[index] = basis.axes
            std_out[index] = basis.std
            residual_out[index] = basis.residual_sigma
    axes_out.flush()
    std_out.flush()
    residual_out.flush()
    return job.stop - job.start


@dataclass
class N2Shared:
    """What every pool of the family shares: independent of the group size AND the overlap."""

    seeds_path: Path
    axes_path: Path
    std_path: Path
    residual_path: Path
    seed_neighbours: GroundTruth  # top max(neighbours, 10) base neighbours of every seed
    seed_dtype: np.dtype

    def basis(self, index: int) -> LocalBasis:
        axes = np.load(self.axes_path, mmap_mode="r")
        std = np.load(self.std_path, mmap_mode="r")
        residual = np.load(self.residual_path, mmap_mode="r")
        return LocalBasis(
            axes=np.array(axes[index]), std=np.array(std[index]), residual_sigma=float(residual[index])
        )


def _load_shared(directory: Path, seeds_path: Path, seed_dtype: np.dtype) -> N2Shared:
    """Open the neighbours and bases that are stored in ``directory`` (the ``n2`` folder)."""
    return N2Shared(
        seeds_path,
        directory / "basis_axes.npy",
        directory / "basis_std.npy",
        directory / "basis_residual.npy",
        GroundTruth(
            np.load(directory / "seed_neighbour_ids.npy"),
            np.load(directory / "seed_neighbour_dists.npy"),
        ),
        seed_dtype,
    )


def prepare_neighbours_and_bases(
    config: WorkloadConfig,
    base_set: BaseSet,
    seeds: np.ndarray,
    plan: ParallelPlan,
    scratch: Path,
    store: SharedStore,
    log,
    resume: Optional[ResumeState] = None,
    reuse: Optional[SharedStore] = None,
) -> N2Shared:
    """Neighbours and PCA bases for all seeds (done once for every pool of the family).

    Both depend only on the seeds and the base, not on the group size and not on the target
    overlap. They are written to ``store`` (the ``shared/n2`` folder of the family), together
    with their sha256, so that a later run can use them.

    ``reuse`` is the store of an existing family (extension): when it holds the neighbours
    and bases they are verified and used as they are; nothing is computed. With ``resume``
    the two long parts leave a marker when they are finished (the exact neighbours of the
    seeds; the bases), and a repeated call after an interruption loads what is finished.
    """
    groups = config.groups
    num_seeds, dim = len(seeds), base_set.dim
    neighbour_count = max(groups.neighbours, OVERLAP_K)

    if base_set.num_base < neighbour_count:
        raise N2Error(
            f"the base has {base_set.num_base} vectors, fewer than neighbours = {neighbour_count}"
        )
    if groups.axes > min(groups.neighbours - 1, dim):
        raise N2Error(
            f"axes = {groups.axes} exceeds min(neighbours - 1, dim) = "
            f"{min(groups.neighbours - 1, dim)}"
        )

    # The pool workers read the seeds from a file; it is rewritten on every run (a few MB).
    scratch.mkdir(parents=True, exist_ok=True)
    seeds_path = scratch / "seeds.npy"
    np.save(seeds_path, np.ascontiguousarray(seeds))
    fingerprint = n2_fingerprint(config, base_set, seeds)

    if reuse is not None and reuse.has_n2():
        try:
            reuse.verify_n2(fingerprint)
        except SharedStoreError as error:
            raise N2Error(str(error)) from error
        log(f"[shared] using the stored neighbours and PCA bases of {reuse.directory}")
        return _load_shared(reuse.n2_dir, seeds_path, seeds.dtype)

    directory = store.n2_dir
    directory.mkdir(parents=True, exist_ok=True)
    ids_path = directory / "seed_neighbour_ids.npy"
    dists_path = directory / "seed_neighbour_dists.npy"
    axes_path = directory / "basis_axes.npy"
    std_path = directory / "basis_std.npy"
    residual_path = directory / "basis_residual.npy"

    if resume is not None and resume.done("n2_bases"):
        log("[resume] neighbours and PCA bases of the seeds were finished earlier")
        return _load_shared(directory, seeds_path, seeds.dtype)

    with base_scanner(base_set, plan) as scanner:
        # 1. Nearest base vectors of every seed (also gives the seed's own top-10 list).
        if resume is not None and resume.done("seed_neighbours"):
            log("[resume] the exact neighbours of the seeds were found earlier")
            seed_neighbours = GroundTruth(np.load(ids_path), np.load(dists_path))
        else:
            log(f"[groups] exact neighbours of {num_seeds} seeds (k = {neighbour_count})")
            seed_neighbours = scanner.top(seeds, neighbour_count)
            np.save(ids_path, np.ascontiguousarray(seed_neighbours.neighbour_ids))
            np.save(dists_path, np.ascontiguousarray(seed_neighbours.neighbour_dists))
            if resume is not None:
                resume.mark("seed_neighbours", files=[ids_path, dists_path])

    # 2. One local PCA basis per seed, from its `neighbours` nearest base vectors.
    log(f"[groups] fitting local PCA bases ({plan.cpu_workers} CPU workers)")
    for path, shape in (
        (axes_path, (num_seeds, groups.axes, dim)),
        (std_path, (num_seeds, groups.axes)),
        (residual_path, (num_seeds,)),
    ):
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape).flush()
    basis_jobs = [
        BasisJob(
            str(base_set.directory / f"base{base_set.suffix}"), str(ids_path), str(axes_path),
            str(std_path), str(residual_path), start, stop, groups.neighbours, groups.axes,
        )
        for start, stop in split_ranges(num_seeds, plan.cpu_workers)
    ]
    run_cpu_jobs(run_basis_job, basis_jobs, plan.cpu_workers, plan.node_cpus)

    # The sha256 of all five files is recorded only now that every one of them is complete.
    store.write_n2_manifest(fingerprint)
    if resume is not None:
        resume.mark("n2_bases", files=[axes_path, std_path, residual_path])
    return N2Shared(
        seeds_path, axes_path, std_path, residual_path, seed_neighbours, seeds.dtype
    )


@dataclass(frozen=True)
class OverlapCalibration:
    """The noise scale alpha found for one target overlap, and the settings it was found with."""

    overlap_name: str
    target_overlap: float
    result: CalibrationResult
    num_calibration: int

    def to_info(self, settings: dict) -> dict:
        """Plain JSON data (for ``cache.json`` and the resume markers)."""
        return {
            **settings,
            "alpha": self.result.alpha,
            "achieved_overlap": self.result.achieved_overlap,
            "iterations": self.result.iterations,
            "num_calibration": self.num_calibration,
        }


def calibration_settings(config: WorkloadConfig, target_overlap: float) -> dict:
    """Every setting that alpha depends on besides the neighbours, bases and seeds."""
    return {
        "target_overlap": target_overlap,
        "tolerance": config.groups.overlap_tolerance,
        "calibration_queries": config.groups.calibration_queries,
        "residual": config.groups.residual,
        "seed": config.seeds.seed,
    }


def calibration_from_info(overlap_name: str, info: dict, settings: dict) -> OverlapCalibration:
    """Rebuild a stored calibration; it must have been made with the same ``settings``."""
    stored = {key: info.get(key) for key in settings}
    if stored != settings:
        raise N2Error(
            f"the stored noise scale of {overlap_name} was found with other settings "
            f"({describe_differences(stored, settings)})"
        )
    return OverlapCalibration(
        overlap_name,
        settings["target_overlap"],
        CalibrationResult(info["alpha"], info["achieved_overlap"], int(info["iterations"])),
        int(info["num_calibration"]),
    )


def calibrate_overlap(
    config: WorkloadConfig,
    base_set: BaseSet,
    shared: N2Shared,
    plan: ParallelPlan,
    overlap_name: str,
    target_overlap: float,
    log,
    resume: Optional[ResumeState] = None,
    stored: Optional[dict] = None,
) -> OverlapCalibration:
    """The noise scale alpha for ONE target overlap.

    The calibration seeds and the noise draws do not depend on the target, so the result for
    a target is the same whichever other targets are built in the same run or in an earlier
    one. ``stored`` is a calibration of this overlap made by an earlier run (``cache.json``);
    it is used instead of searching again.
    """
    groups = config.groups
    settings = calibration_settings(config, target_overlap)
    stage = f"calibration_{overlap_name}"
    if stored is not None:
        log(f"[groups] using the stored noise scale of {overlap_name}")
        return calibration_from_info(overlap_name, stored, settings)
    if resume is not None and resume.done(stage):
        log(f"[resume] the noise scale of {overlap_name} was found earlier")
        return calibration_from_info(overlap_name, resume.info(stage), settings)

    seeds = np.load(shared.seeds_path, mmap_mode="r")
    num_seeds = len(seeds)

    # Calibrate alpha on a random subset of the seeds, with common random numbers: every
    # candidate alpha reuses the same noise draw of a seed, so the overlap is a smooth
    # function of alpha.
    calibration_rng = np.random.default_rng(np.random.SeedSequence([config.seeds.seed, 0]))
    num_calibration = min(groups.calibration_queries, num_seeds)
    calibration_ids = np.sort(calibration_rng.choice(num_seeds, num_calibration, replace=False))
    bases = [shared.basis(int(i)) for i in calibration_ids]
    draws = [
        draw_noise(
            basis, 1, np.random.default_rng(np.random.SeedSequence([config.seeds.seed, 2, int(i)]))
        )
        for basis, i in zip(bases, calibration_ids)
    ]
    seed_top = shared.seed_neighbours.neighbour_ids[calibration_ids, :OVERLAP_K]

    with base_scanner(base_set, plan) as scanner:

        def overlap_at(alpha: float) -> float:
            variants = np.vstack(
                [
                    apply_noise(
                        np.array(seeds[i]), basis, draw, alpha, groups.residual, shared.seed_dtype
                    )
                    for i, basis, draw in zip(calibration_ids, bases, draws)
                ]
            )
            return mean_overlap(seed_top, scanner.top(variants, OVERLAP_K).neighbour_ids)

        log(f"[groups] calibrating alpha for target overlap {target_overlap} ({overlap_name})")
        result = find_alpha(overlap_at, target_overlap, groups.overlap_tolerance)
        log(f"[groups] alpha = {result.alpha:.6g} (overlap {result.achieved_overlap:.4f})")

    calibration = OverlapCalibration(overlap_name, target_overlap, result, int(num_calibration))
    if resume is not None:
        resume.mark(stage, calibration.to_info(settings))
    return calibration


# ---------------------------------------------------------------------------
# pools (parallel by seed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolJob:
    seeds_path: str
    axes_path: str
    std_path: str
    residual_path: str
    pool_path: str
    start: int
    stop: int
    group_size: int
    include_original: bool
    residual: bool
    alpha: float
    seed: int


def run_pool_job(job: PoolJob) -> int:
    """Write the group members of the seeds ``[start, stop)`` into the pool file."""
    seeds = np.load(job.seeds_path, mmap_mode="r")
    axes = np.load(job.axes_path, mmap_mode="r")
    std = np.load(job.std_path, mmap_mode="r")
    residual = np.load(job.residual_path, mmap_mode="r")
    pool = np.load(job.pool_path, mmap_mode="r+")
    size = job.group_size

    for index in range(job.start, job.stop):
        seed_vector = np.array(seeds[index])
        basis = LocalBasis(
            axes=np.array(axes[index]), std=np.array(std[index]), residual_sigma=float(residual[index])
        )
        # Per-seed random numbers: the same for any number of workers.
        rng = np.random.default_rng(np.random.SeedSequence([job.seed, 1, index]))
        variant_count = size - 1 if job.include_original else size
        parts = []
        if job.include_original:
            parts.append(seed_vector[None, :])
        if variant_count > 0:
            parts.append(
                make_variants(seed_vector, basis, job.alpha, variant_count, rng, job.residual)
            )
        pool[index * size : (index + 1) * size] = np.vstack(parts)
    pool.flush()
    return job.stop - job.start


def make_pool(
    config: WorkloadConfig,
    shared: N2Shared,
    num_seeds: int,
    dim: int,
    group_size: int,
    alpha: float,
    plan: ParallelPlan,
    scratch: Path,
    name: str,
) -> np.ndarray:
    """The pool of one group size and one alpha: group after group, ``group_size`` members each."""
    pool_path = scratch / f"pool_{name}.npy"
    pool = np.lib.format.open_memmap(
        pool_path, mode="w+", dtype=shared.seed_dtype, shape=(num_seeds * group_size, dim)
    )
    pool.flush()
    del pool

    jobs = [
        PoolJob(
            str(shared.seeds_path), str(shared.axes_path), str(shared.std_path),
            str(shared.residual_path), str(pool_path), start, stop, group_size,
            config.groups.include_original, config.groups.residual, alpha,
            config.seeds.seed,
        )
        for start, stop in split_ranges(num_seeds, plan.cpu_workers)
    ]
    run_cpu_jobs(run_pool_job, jobs, plan.cpu_workers, plan.node_cpus)
    return np.array(np.load(pool_path, mmap_mode="r"))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def pool_statistics(
    config: WorkloadConfig,
    base_set: BaseSet,
    shared: N2Shared,
    calibration: OverlapCalibration,
    pool: np.ndarray,
    group_size: int,
    plan: ParallelPlan,
) -> Dict[str, object]:
    """Describe the generated variants: overlap and distance ratio over ALL seeds."""
    groups = config.groups
    num_seeds = len(shared.seed_neighbours.neighbour_ids)
    statistics: Dict[str, object] = {
        "alpha": calibration.result.alpha,
        "target_overlap": calibration.target_overlap,
        "calibration_overlap": calibration.result.achieved_overlap,
        "calibration_queries": calibration.num_calibration,
        "calibration_evaluations": calibration.result.iterations,
    }
    if group_size > 1 or not groups.include_original:
        representative = np.arange(num_seeds) * group_size + (1 if groups.include_original else 0)
        with base_scanner(base_set, plan) as scanner:
            variant_top = scanner.top(pool[representative], OVERLAP_K)
        seed_ids = shared.seed_neighbours.neighbour_ids[:, :OVERLAP_K]
        per_seed = overlap_per_query(seed_ids, variant_top.neighbour_ids)
        statistics["overall_mean_overlap"] = float(per_seed.mean())
        statistics["overlap_quantiles_5_50_95"] = np.quantile(per_seed, [0.05, 0.5, 0.95]).tolist()
        if base_set.metric == "l2":
            # d1 ratio: nearest-neighbour distance of the variant over that of its seed.
            seed_d1 = np.sqrt(shared.seed_neighbours.neighbour_dists[:, 0].astype(np.float64))
            variant_d1 = np.sqrt(variant_top.neighbour_dists[:, 0].astype(np.float64))
            valid = seed_d1 > 0
            statistics["d1_ratio_quantiles_5_50_95"] = (
                np.quantile(variant_d1[valid] / seed_d1[valid], [0.05, 0.5, 0.95]).tolist()
                if valid.any()
                else None
            )
    return statistics
