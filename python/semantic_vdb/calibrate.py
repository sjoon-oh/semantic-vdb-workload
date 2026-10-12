"""Calibration of the noise scale ``alpha`` from a target neighbour overlap.

"Overlap" is the mean fraction of a seed's true top-k neighbours that are also top-k
neighbours of its variant (1.0 = identical neighbours, 0.0 = nothing shared). The user
states the overlap they want; this module searches for the ``alpha`` that produces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class CalibrationError(RuntimeError):
    """The target overlap cannot be reached (or the search did not converge)."""


@dataclass(frozen=True)
class CalibrationResult:
    alpha: float
    achieved_overlap: float
    iterations: int


def overlap_per_query(neighbours_a: np.ndarray, neighbours_b: np.ndarray) -> np.ndarray:
    """Fraction of shared ids for every row of two (n, k) neighbour-id matrices."""
    if neighbours_a.shape != neighbours_b.shape:
        raise ValueError("neighbour matrices must have the same shape")
    if neighbours_a.shape[0] == 0:
        raise ValueError("at least one query is needed")
    # (n, k, 1) == (n, 1, k) marks every equal pair; ids inside a row are unique, so the
    # number of True entries per row is the size of the intersection.
    shared = (neighbours_a[:, :, None] == neighbours_b[:, None, :]).sum(axis=(1, 2))
    return shared / neighbours_a.shape[1]


def mean_overlap(neighbours_a: np.ndarray, neighbours_b: np.ndarray) -> float:
    """Mean fraction of shared ids between two (n, k) neighbour-id matrices."""
    return float(overlap_per_query(neighbours_a, neighbours_b).mean())


def find_alpha(
    overlap_at: Callable[[float], float],
    target_overlap: float,
    tolerance: float,
    max_iterations: int = 60,
    initial_alpha: float = 1.0,
    max_doublings: int = 60,
) -> CalibrationResult:
    """Find ``alpha`` with ``|overlap_at(alpha) - target_overlap| <= tolerance``.

    ``overlap_at`` must be (approximately) non-increasing: more noise, less overlap.
    The search first brackets the target by doubling ``alpha`` and then bisects.
    It raises ``CalibrationError`` instead of returning a value that misses the target.
    """
    if not 0.0 < target_overlap < 1.0:
        raise ValueError("target_overlap must be strictly between 0 and 1")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    evaluations = 0

    def measure(alpha: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return overlap_at(alpha)

    # Without noise the variant equals the seed, so the overlap must be above the target.
    low = 0.0
    if measure(low) < target_overlap:
        raise CalibrationError("the overlap without any noise is already below the target")

    # Grow alpha until the overlap falls below the target: [low, high] brackets it.
    high = initial_alpha
    for _ in range(max_doublings):
        if measure(high) <= target_overlap:
            break
        low, high = high, high * 2.0
    else:
        raise CalibrationError("the overlap never fell below the target, even for huge alpha")

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        overlap = measure(middle)
        if abs(overlap - target_overlap) <= tolerance:
            return CalibrationResult(alpha=middle, achieved_overlap=overlap, iterations=evaluations)
        if overlap > target_overlap:
            low = middle  # still too similar: more noise needed
        else:
            high = middle  # too different: less noise

    raise CalibrationError(
        f"no alpha within {tolerance} of the target overlap {target_overlap} after "
        f"{max_iterations} bisection steps"
    )
