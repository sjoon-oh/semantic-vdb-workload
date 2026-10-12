"""Request streams over groups of similar queries.

Two kinds of stream exist:

* ``zipf``            ``length`` requests. Each request first picks a GROUP with a Zipf law over
                      group popularity ranks (``p(rank) ~ 1 / rank**exponent``; exponent 0 is
                      uniform) and then one MEMBER of that group uniformly at random, with
                      replacement. The same vector may therefore be requested many times
                      ("exact repeats"); that is intended.
* ``uniform_unique``  ``length`` requests drawn uniformly from ALL pool vectors WITHOUT
                      replacement: every request is a different vector (no exact repeats), but
                      neighbouring requests still come from the same groups. It needs
                      ``length <= pool size``.

Which group holds which popularity rank (Zipf) is a random permutation, so popularity does
not follow the order of the seed queries in their file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semantic_vdb.updates import split_evenly


def zipf_probabilities(num_groups: int, exponent: float) -> np.ndarray:
    """Probability of each popularity rank (rank 0 is the most popular)."""
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if exponent < 0.0:
        raise ValueError("the Zipf exponent must not be negative")
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    # Normalise so that the probabilities sum to one.
    return weights / weights.sum()


@dataclass(frozen=True)
class Stream:
    exponent: float
    group_ids: np.ndarray  # (length,) int64, index of the requested group
    member_ids: np.ndarray  # (length,) int64, member index inside that group
    pool_index: np.ndarray  # (length,) int64, row of the requested vector in the pool
    kind: str = "zipf"  # "zipf" or "uniform_unique"

    @property
    def length(self) -> int:
        return len(self.group_ids)

    def head(self, count: int) -> "Stream":
        """The first ``count`` requests, in order (used to share one stream between sets)."""
        if not 1 <= count <= self.length:
            raise ValueError(f"count must be in 1..{self.length}, got {count}")
        return Stream(
            exponent=self.exponent,
            group_ids=self.group_ids[:count],
            member_ids=self.member_ids[:count],
            pool_index=self.pool_index[:count],
            kind=self.kind,
        )


def build_stream(
    group_sizes: np.ndarray, length: int, exponent: float, rng: np.random.Generator
) -> Stream:
    """Draw a stream of ``length`` requests over groups of the given sizes.

    The pool is laid out group after group: group ``g`` owns the rows
    ``offsets[g] .. offsets[g] + group_sizes[g] - 1``.
    """
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    if group_sizes.ndim != 1 or group_sizes.size == 0 or group_sizes.min() < 1:
        raise ValueError("group_sizes must be a non-empty 1-D array of positive sizes")
    if length < 1:
        raise ValueError("length must be positive")

    num_groups = len(group_sizes)
    probabilities = zipf_probabilities(num_groups, exponent)

    # Random assignment of popularity ranks to groups.
    group_of_rank = rng.permutation(num_groups)
    ranks = rng.choice(num_groups, size=length, p=probabilities)
    group_ids = group_of_rank[ranks].astype(np.int64)

    # Uniform member inside the chosen group (sizes may differ between groups).
    member_ids = np.minimum(
        (rng.random(length) * group_sizes[group_ids]).astype(np.int64),
        group_sizes[group_ids] - 1,
    )
    offsets = np.concatenate([[0], np.cumsum(group_sizes)])[:-1]
    pool_index = offsets[group_ids] + member_ids
    return Stream(exponent=exponent, group_ids=group_ids, member_ids=member_ids, pool_index=pool_index)


def build_unique_stream(group_sizes: np.ndarray, length: int, rng: np.random.Generator) -> Stream:
    """Draw ``length`` DIFFERENT pool vectors, uniformly over the whole pool.

    The pool is laid out group after group (see ``build_stream``). Every pool row is equally
    likely to be requested, so a group receives requests in proportion to its size, and no row
    is requested twice. The first ``count`` requests of the stream are themselves a valid
    duplicate-free uniform stream of length ``count``.
    """
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    if group_sizes.ndim != 1 or group_sizes.size == 0 or group_sizes.min() < 1:
        raise ValueError("group_sizes must be a non-empty 1-D array of positive sizes")
    pool_size = int(group_sizes.sum())
    if length < 1:
        raise ValueError("length must be positive")
    if length > pool_size:
        raise ValueError(
            f"a duplicate-free stream of {length} requests needs {length} different vectors "
            f"but the pool holds only {pool_size}"
        )

    # Uniform sample of pool rows without replacement, in random order.
    pool_index = rng.choice(pool_size, size=length, replace=False).astype(np.int64)

    # Map every pool row back to its group: `offsets[g]` is the first row of group g, so the
    # group of a row is the last offset that is <= the row.
    offsets = np.concatenate([[0], np.cumsum(group_sizes)])[:-1]
    group_ids = (np.searchsorted(offsets, pool_index, side="right") - 1).astype(np.int64)
    member_ids = pool_index - offsets[group_ids]
    return Stream(
        exponent=0.0,  # every vector is equally popular
        group_ids=group_ids,
        member_ids=member_ids,
        pool_index=pool_index,
        kind="uniform_unique",
    )


def step_boundaries(length: int, num_steps: int) -> np.ndarray:
    """Cumulative request counts: step ``i`` covers ``[b[i], b[i + 1])`` (0-based ``i``)."""
    sizes = split_evenly(length, num_steps)
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


@dataclass(frozen=True)
class RepeatDiagnostics:
    # Share of requests whose exact vector was already requested earlier. This is the hit
    # ratio an exact-match cache of unlimited size would reach.
    exact_repeat_fraction: float
    # Share of requests whose GROUP was already requested earlier. This bounds what an
    # ideal approximate (semantic) cache of unlimited size could reach.
    group_revisit_fraction: float
    distinct_vectors: int
    distinct_groups: int


def repeat_diagnostics(stream: Stream) -> RepeatDiagnostics:
    """Measure how repetitive a stream is (a description, never an input to generation)."""
    distinct_vectors = int(np.unique(stream.pool_index).size)
    distinct_groups = int(np.unique(stream.group_ids).size)
    return RepeatDiagnostics(
        exact_repeat_fraction=1.0 - distinct_vectors / stream.length,
        group_revisit_fraction=1.0 - distinct_groups / stream.length,
        distinct_vectors=distinct_vectors,
        distinct_groups=distinct_groups,
    )
