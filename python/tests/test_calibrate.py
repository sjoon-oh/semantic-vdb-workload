"""Tests for semantic_vdb.calibrate."""

import numpy as np
import pytest

from semantic_vdb import calibrate, groundtruth, perturb
from reference import reference_knn, reference_overlap


# ---------------------------------------------------------------------------
# mean_overlap
# ---------------------------------------------------------------------------


def test_mean_overlap_counts_shared_ids():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    b = np.array([[4, 3, 9, 10], [5, 6, 7, 8]])
    # Row 1 shares {3, 4} = 2 of 4; row 2 shares all 4.
    assert calibrate.mean_overlap(a, b) == pytest.approx((0.5 + 1.0) / 2)
    assert calibrate.mean_overlap(a, a) == 1.0
    assert calibrate.mean_overlap(a, a + 100) == 0.0
    with pytest.raises(ValueError):
        calibrate.mean_overlap(a, b[:1])
    with pytest.raises(ValueError):
        calibrate.mean_overlap(a[:0], b[:0])


# ---------------------------------------------------------------------------
# find_alpha with a synthetic overlap curve
# ---------------------------------------------------------------------------


def test_find_alpha_inverts_a_known_curve_and_larger_targets_need_less_noise():
    def overlap_at(alpha):
        return float(np.exp(-alpha))

    found = {}
    for target in (0.9, 0.5, 0.2):
        result = calibrate.find_alpha(overlap_at, target, tolerance=0.005)
        assert abs(result.achieved_overlap - target) <= 0.005
        assert result.alpha == pytest.approx(-np.log(target), abs=0.02)
        found[target] = result.alpha
    # A higher required overlap means less noise.
    assert found[0.9] < found[0.5] < found[0.2]


def test_find_alpha_brackets_by_doubling_when_the_start_value_is_too_small():
    result = calibrate.find_alpha(lambda alpha: float(np.exp(-alpha / 1000.0)), 0.5, 0.005)
    assert result.alpha == pytest.approx(1000.0 * np.log(2.0), rel=0.02)


def test_unreachable_targets_raise_instead_of_returning_a_wrong_alpha():
    # Each failure mode must be diagnosed with its own message, not by running out of steps.
    with pytest.raises(calibrate.CalibrationError, match="never fell below"):
        calibrate.find_alpha(lambda alpha: 1.0, 0.5, 0.01)
    with pytest.raises(calibrate.CalibrationError, match="without any noise"):
        calibrate.find_alpha(lambda alpha: 0.3, 0.5, 0.01)
    with pytest.raises(calibrate.CalibrationError, match="bisection steps"):
        calibrate.find_alpha(lambda alpha: float(np.exp(-alpha)), 0.5, 1e-9, max_iterations=3)


def test_find_alpha_validates_arguments():
    with pytest.raises(ValueError):
        calibrate.find_alpha(lambda a: 0.5, 1.0, 0.01)
    with pytest.raises(ValueError):
        calibrate.find_alpha(lambda a: 0.5, 0.5, 0.0)


# ---------------------------------------------------------------------------
# end to end: calibrate on real nearest-neighbour overlaps
# ---------------------------------------------------------------------------


def clustered_base(seed=0, clusters=30, per_cluster=120, dim=16):
    rng = np.random.default_rng(seed)
    centres = rng.integers(40, 200, size=(clusters, dim))
    spread = rng.normal(size=(clusters * per_cluster, dim)) * 9.0
    points = np.repeat(centres, per_cluster, axis=0) + spread
    return np.clip(np.rint(points), 0, 255).astype(np.uint8)


def top_ids(base, queries, k):
    blocks = groundtruth.array_blocks(base, 700)
    return groundtruth.exact_knn(blocks, queries, k, "l2").neighbour_ids


def test_calibrated_alpha_reaches_the_target_overlap_on_fresh_noise():
    base = clustered_base()
    rng = np.random.default_rng(1)
    seeds = base[rng.choice(len(base), size=40, replace=False)]
    seed_neighbours = top_ids(base, seeds, 10)

    # One local basis per seed, from its 100 nearest base vectors.
    bases = []
    for seed_vector in seeds:
        ids = top_ids(base, seed_vector[None, :], 100)[0]
        bases.append(perturb.fit_local_basis(base[ids], 8))

    # Common random numbers: one noise draw per seed, reused for every trial alpha.
    draws = [perturb.draw_noise(b, 1, np.random.default_rng(100 + i)) for i, b in enumerate(bases)]

    def overlap_at(alpha):
        variants = np.vstack(
            [
                perturb.apply_noise(seeds[i], bases[i], draws[i], alpha, False, np.uint8)
                for i in range(len(seeds))
            ]
        )
        return calibrate.mean_overlap(seed_neighbours, top_ids(base, variants, 10))

    target = 0.6
    result = calibrate.find_alpha(overlap_at, target, tolerance=0.03)
    assert abs(result.achieved_overlap - target) <= 0.03

    # Fresh noise (different random numbers) at the calibrated alpha lands near the target,
    # measured with the independent float64 reference implementation.
    fresh = np.vstack(
        [
            perturb.make_variants(seeds[i], bases[i], result.alpha, 1, np.random.default_rng(900 + i))
            for i in range(len(seeds))
        ]
    )
    reference_ids, _ = reference_knn(base, np.arange(len(base)), fresh, 10)
    seed_reference_ids, _ = reference_knn(base, np.arange(len(base)), seeds, 10)
    assert abs(reference_overlap(seed_reference_ids, reference_ids) - target) < 0.12
