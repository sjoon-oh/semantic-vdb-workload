"""Seed sources: where the original queries of a workload come from.

* ``public_queries``  a query file shipped with the dataset; every query is a candidate seed.
* ``base_rows``       the rows a base set reserved as queries (``num_query_rows``).
* ``external``        vectors made elsewhere (e.g. paraphrases embedded by a text model) with
                      a group id per vector. They are used as they are; only skew is added.

All vectors must have the dtype and dimension of the base set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from semantic_vdb.base_set import BaseSet
from semantic_vdb.binio import BinReader, read_ids
from semantic_vdb.config import WorkloadConfig


class SourceError(ValueError):
    """An input query file does not fit the base set."""


@dataclass(frozen=True)
class SeedSource:
    mode: str
    # Every original query. These are the evaluation queries when evaluation.source = original.
    original_queries: np.ndarray
    # N2 modes: the seed queries that get a group of variants (rows of original_queries).
    seed_vectors: Optional[np.ndarray]
    seed_indices: Optional[np.ndarray]
    # external mode: ready-made groups, stored group after group.
    external_vectors: Optional[np.ndarray]
    external_group_sizes: Optional[np.ndarray]


def _read_queries(path: Path, base_set: BaseSet, what: str) -> np.ndarray:
    """Read a whole query file and check that it matches the base set."""
    with BinReader(path) as reader:
        if reader.dtype != base_set.dtype:
            raise SourceError(f"{what} {path} has dtype {reader.dtype}, the base set has {base_set.dtype}")
        if reader.num_cols != base_set.dim:
            raise SourceError(f"{what} {path} has dimension {reader.num_cols}, the base set has {base_set.dim}")
        if reader.num_rows == 0:
            raise SourceError(f"{what} {path} is empty")
        return reader.read_range(0, reader.num_rows)


def _choose_subset(count: int, wanted: Optional[int], seed_sequence: np.random.SeedSequence) -> np.ndarray:
    """Indices of ``wanted`` of ``count`` items (sorted); all of them when ``wanted`` is None."""
    if wanted is None or wanted >= count:
        if wanted is not None and wanted > count:
            raise SourceError(f"num_groups = {wanted} but only {count} are available")
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed_sequence)
    return np.sort(rng.choice(count, size=wanted, replace=False)).astype(np.int64)


def load_seed_source(config: WorkloadConfig, base_set: BaseSet) -> SeedSource:
    seeds = config.seeds
    subset_sequence = np.random.SeedSequence([seeds.seed, 3])

    if seeds.mode == "public_queries":
        queries = _read_queries(seeds.query_file, base_set, "query file")
        chosen = _choose_subset(len(queries), seeds.num_groups, subset_sequence)
        return SeedSource(seeds.mode, queries, queries[chosen], chosen, None, None)

    if seeds.mode == "base_rows":
        if base_set.num_query_rows == 0:
            raise SourceError(
                "mode base_rows needs a base set built with num_query_rows > 0 (none were reserved)"
            )
        with base_set.open_query_rows() as reader:
            queries = reader.read_range(0, reader.num_rows)
        chosen = _choose_subset(len(queries), seeds.num_groups, subset_sequence)
        return SeedSource(seeds.mode, queries, queries[chosen], chosen, None, None)

    if seeds.mode == "external":
        vectors = _read_queries(seeds.query_file, base_set, "external query file")
        group_ids = read_ids(seeds.group_id_file)
        if len(group_ids) != len(vectors):
            raise SourceError(
                f"{seeds.group_id_file} has {len(group_ids)} ids for {len(vectors)} vectors"
            )
        # Make every group contiguous (stable sort keeps the order inside a group).
        order = np.argsort(group_ids, kind="stable")
        vectors, group_ids = vectors[order], group_ids[order]
        _, first_rows, sizes = np.unique(group_ids, return_index=True, return_counts=True)
        chosen = _choose_subset(len(sizes), seeds.num_groups, subset_sequence)
        keep_rows = np.concatenate(
            [np.arange(first_rows[g], first_rows[g] + sizes[g]) for g in chosen]
        )
        return SeedSource(
            seeds.mode,
            original_queries=vectors,
            seed_vectors=None,
            seed_indices=None,
            external_vectors=vectors[keep_rows],
            external_group_sizes=sizes[chosen].astype(np.int64),
        )

    raise SourceError(f"unknown seed mode {seeds.mode!r}")


def load_eval_queries(config: WorkloadConfig, source: SeedSource, base_set: BaseSet) -> np.ndarray:
    """The queries used for the periodic recall checkpoints (never perturbed)."""
    if config.evaluation.source == "original":
        if source.mode == "external":
            raise SourceError("evaluation.source = original is not available for mode external")
        return source.original_queries
    return _read_queries(config.evaluation.eval_file, base_set, "evaluation file")
