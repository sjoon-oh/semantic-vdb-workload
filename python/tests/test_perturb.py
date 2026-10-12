"""Tests for semantic_vdb.perturb (the N2 noise model)."""

import numpy as np
import pytest

from semantic_vdb import perturb

DIM = 12
TRUE_AXES = 3


def subspace_neighbours(count=400, seed=0):
    """Neighbours that vary ONLY inside a known 3-D subspace of a 12-D space."""
    rng = np.random.default_rng(seed)
    # Orthonormal basis of the subspace and clearly different spreads along its axes.
    basis, _ = np.linalg.qr(rng.normal(size=(DIM, TRUE_AXES)))
    spreads = np.array([9.0, 4.0, 1.5])
    coordinates = rng.normal(size=(count, TRUE_AXES)) * spreads
    centre = rng.normal(size=DIM) * 5.0
    return centre + coordinates @ basis.T, basis, spreads


def projector_onto(rows):
    """Projection matrix onto the span of the (orthonormal) rows."""
    return rows.T @ rows


# ---------------------------------------------------------------------------
# fit_local_basis
# ---------------------------------------------------------------------------


def test_fitted_axes_span_the_true_subspace_and_are_orthonormal():
    neighbours, true_basis, spreads = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, TRUE_AXES)

    assert basis.axes.shape == (TRUE_AXES, DIM)
    assert np.allclose(basis.axes @ basis.axes.T, np.eye(TRUE_AXES), atol=1e-10)
    # Same subspace: the projector onto the fitted axes equals the one onto the truth.
    assert np.allclose(projector_onto(basis.axes), projector_onto(true_basis.T), atol=1e-8)
    # Strongest axis first, with spreads close to the generating ones.
    assert np.all(np.diff(basis.std) < 0)
    assert np.allclose(basis.std, spreads, rtol=0.15)


def test_residual_sigma_is_zero_when_the_axes_explain_everything():
    neighbours, _, _ = subspace_neighbours()
    assert perturb.fit_local_basis(neighbours, TRUE_AXES).residual_sigma < 1e-6
    # Keeping fewer axes than the true rank leaves unexplained variance behind.
    assert perturb.fit_local_basis(neighbours, 1).residual_sigma > 0.5


def test_too_many_axes_are_rejected():
    neighbours, _, _ = subspace_neighbours(count=10)
    with pytest.raises(ValueError):
        perturb.fit_local_basis(neighbours, 10)  # at most count - 1 = 9
    with pytest.raises(ValueError):
        perturb.fit_local_basis(neighbours, 0)


# ---------------------------------------------------------------------------
# noise properties
# ---------------------------------------------------------------------------


def float_variants(basis, seed_vector, alpha, count, rng_seed=1, residual=False):
    draw = perturb.draw_noise(basis, count, np.random.default_rng(rng_seed))
    return perturb.apply_noise(seed_vector, basis, draw, alpha, residual, np.float64)


def test_noise_stays_inside_the_fitted_subspace():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, 2)  # fewer axes than the true rank
    seed_vector = np.zeros(DIM)
    offsets = float_variants(basis, seed_vector, alpha=1.5, count=500) - seed_vector
    outside = offsets - offsets @ projector_onto(basis.axes)
    assert np.abs(outside).max() < 1e-9
    assert np.abs(offsets).max() > 1.0  # and the noise is not trivially zero


def test_noise_is_centred_on_the_seed_with_the_requested_spread_per_axis():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, TRUE_AXES)
    seed_vector = np.arange(DIM, dtype=np.float64)
    alpha, count = 0.7, 40_000
    offsets = float_variants(basis, seed_vector, alpha, count) - seed_vector

    coordinates = offsets @ basis.axes.T  # (count, r): position along every axis
    assert np.allclose(coordinates.mean(axis=0), 0.0, atol=4 * alpha * basis.std / np.sqrt(count))
    assert np.allclose(coordinates.std(axis=0), alpha * basis.std, rtol=0.02)


def test_alpha_zero_returns_the_seed_and_larger_alpha_moves_further():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, TRUE_AXES)
    seed_vector = np.ones(DIM) * 3.0

    assert np.array_equal(float_variants(basis, seed_vector, 0.0, 5), np.tile(seed_vector, (5, 1)))
    distances = [
        np.linalg.norm(float_variants(basis, seed_vector, alpha, 200) - seed_vector, axis=1).mean()
        for alpha in (0.1, 0.5, 2.0)
    ]
    assert distances[0] < distances[1] < distances[2]
    # Same random numbers: the offset scales exactly linearly with alpha.
    small = float_variants(basis, seed_vector, 0.5, 50) - seed_vector
    large = float_variants(basis, seed_vector, 1.0, 50) - seed_vector
    assert np.allclose(large, 2.0 * small)


def test_residual_option_adds_out_of_subspace_noise():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, 1)
    seed_vector = np.zeros(DIM)
    without = float_variants(basis, seed_vector, 1.0, 300, residual=False)
    with_residual = float_variants(basis, seed_vector, 1.0, 300, residual=True)
    projector = projector_onto(basis.axes)
    assert np.abs(without - without @ projector).max() < 1e-9
    assert np.abs(with_residual - with_residual @ projector).max() > 0.1


def test_negative_alpha_is_rejected():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, 2)
    draw = perturb.draw_noise(basis, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        perturb.apply_noise(np.zeros(DIM), basis, draw, -1.0, False, np.float32)


# ---------------------------------------------------------------------------
# rounding to the vector dtype
# ---------------------------------------------------------------------------


def test_quantize_rounds_to_nearest_and_clips_to_the_dtype_range():
    values = np.array([[-3.2, 0.4, 0.6, 254.4, 254.6, 300.0]])
    assert perturb.quantize(values, np.uint8).tolist() == [[0, 0, 1, 254, 255, 255]]
    assert perturb.quantize(values, np.uint8).dtype == np.uint8
    signed = np.array([[-200.0, -0.6, 127.4, 130.0]])
    assert perturb.quantize(signed, np.int8).tolist() == [[-128, -1, 127, 127]]
    assert perturb.quantize(values, np.float32).dtype == np.float32


def test_make_variants_keeps_the_seed_dtype_and_is_reproducible():
    neighbours, _, _ = subspace_neighbours()
    basis = perturb.fit_local_basis(neighbours, TRUE_AXES)
    seed_vector = np.full(DIM, 100, dtype=np.uint8)

    first = perturb.make_variants(seed_vector, basis, 0.5, 20, np.random.default_rng(4))
    again = perturb.make_variants(seed_vector, basis, 0.5, 20, np.random.default_rng(4))
    other = perturb.make_variants(seed_vector, basis, 0.5, 20, np.random.default_rng(5))
    assert first.dtype == np.uint8 and first.shape == (20, DIM)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
