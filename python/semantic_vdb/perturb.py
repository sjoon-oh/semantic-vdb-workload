"""N2 perturbation: noise along the directions in which the neighbourhood really varies.

For a seed query ``q`` take its ``m`` nearest base vectors, centre them and run a PCA.
The top ``r`` principal axes (and their standard deviations) describe how nearby data
differ from each other. A variant of ``q`` is

    q' = q + alpha * sum_j z_j * std_j * axis_j ,   z_j ~ N(0, 1)

so the noise is zero-mean around ``q`` (no pull towards any particular neighbour) and
lives in the ``r``-dimensional subspace spanned by the axes. ``alpha`` scales the noise
and is calibrated elsewhere (``calibrate.py``) from a target neighbour overlap.

Assumption (NOT verified): the directions along which different items near ``q`` differ
resemble the directions along which a paraphrase of ``q`` would differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalBasis:
    axes: np.ndarray  # (r, d) float64, orthonormal rows, strongest axis first
    std: np.ndarray  # (r,) float64, standard deviation of the neighbours along each axis
    residual_sigma: float  # per-dimension std of the variance the r axes do not explain


@dataclass(frozen=True)
class NoiseDraw:
    """Standard-normal numbers for ``count`` variants.

    Keeping them separate from ``alpha`` lets the calibration reuse the same random
    numbers for every trial value of ``alpha`` ("common random numbers"), which makes
    the measured overlap a smooth, repeatable function of ``alpha``.
    """

    axis_noise: np.ndarray  # (count, r)
    residual_noise: np.ndarray  # (count, d)


def fit_local_basis(neighbours: np.ndarray, num_axes: int) -> LocalBasis:
    """PCA of a neighbourhood: top ``num_axes`` axes, their std and the residual level."""
    count, dim = neighbours.shape
    if num_axes < 1 or num_axes > min(count - 1, dim):
        raise ValueError(
            f"num_axes must be in 1..min(neighbours - 1, dim) = {min(count - 1, dim)}, "
            f"got {num_axes}"
        )

    centred = neighbours.astype(np.float64)
    centred -= centred.mean(axis=0)
    # Rows of vt are the principal axes; singular values give the variance per axis.
    _, singular_values, vt = np.linalg.svd(centred, full_matrices=False)
    variances = singular_values**2 / (count - 1)

    total_variance = float((centred**2).sum() / (count - 1))
    explained_variance = float(variances[:num_axes].sum())
    # Variance not captured by the chosen axes, spread evenly over all dimensions.
    residual_sigma = float(np.sqrt(max(total_variance - explained_variance, 0.0) / dim))

    return LocalBasis(
        axes=vt[:num_axes].copy(),
        std=np.sqrt(variances[:num_axes]),
        residual_sigma=residual_sigma,
    )


def draw_noise(basis: LocalBasis, count: int, rng: np.random.Generator) -> NoiseDraw:
    num_axes, dim = basis.axes.shape
    return NoiseDraw(
        axis_noise=rng.standard_normal((count, num_axes)),
        residual_noise=rng.standard_normal((count, dim)),
    )


def quantize(vectors: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """Round and clip float vectors to the value range of ``dtype``."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(vectors), info.min, info.max).astype(dtype)
    return vectors.astype(dtype)


def apply_noise(
    seed_vector: np.ndarray,
    basis: LocalBasis,
    draw: NoiseDraw,
    alpha: float,
    use_residual: bool,
    dtype: np.dtype | type,
) -> np.ndarray:
    """Turn a noise draw into variants of ``seed_vector`` (rounded to ``dtype``)."""
    if alpha < 0.0:
        raise ValueError("alpha must not be negative")

    # (count, r) scaled by the per-axis std, then mapped back to d dimensions.
    noise = (draw.axis_noise * (alpha * basis.std)) @ basis.axes
    if use_residual:
        noise = noise + draw.residual_noise * (alpha * basis.residual_sigma)

    variants = seed_vector.astype(np.float64)[None, :] + noise
    return quantize(variants, dtype)


def make_variants(
    seed_vector: np.ndarray,
    basis: LocalBasis,
    alpha: float,
    count: int,
    rng: np.random.Generator,
    use_residual: bool = False,
) -> np.ndarray:
    """Draw ``count`` fresh variants of ``seed_vector`` in the seed's own dtype."""
    draw = draw_noise(basis, count, rng)
    return apply_noise(seed_vector, basis, draw, alpha, use_residual, seed_vector.dtype)
