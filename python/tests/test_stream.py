"""Tests for semantic_vdb.stream."""

import numpy as np
import pytest

from semantic_vdb import stream as stream_module
from semantic_vdb.stream import Stream


def make_stream(sizes, length, exponent, seed=0):
    return stream_module.build_stream(np.asarray(sizes), length, exponent, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Zipf law
# ---------------------------------------------------------------------------


def test_zipf_probabilities_follow_the_power_law():
    probabilities = stream_module.zipf_probabilities(1000, 0.99)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(probabilities) < 0)
    # p(rank i) / p(rank j) = (j / i) ** s, with ranks counted from 1.
    assert probabilities[0] / probabilities[9] == pytest.approx(10 ** 0.99)
    assert probabilities[3] / probabilities[99] == pytest.approx((100 / 4) ** 0.99)


def test_exponent_zero_is_uniform_and_invalid_arguments_are_rejected():
    assert np.allclose(stream_module.zipf_probabilities(8, 0.0), 1 / 8)
    with pytest.raises(ValueError):
        stream_module.zipf_probabilities(0, 1.0)
    with pytest.raises(ValueError):
        stream_module.zipf_probabilities(5, -0.1)


# ---------------------------------------------------------------------------
# build_stream
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# duplicate-free uniform stream
# ---------------------------------------------------------------------------


def make_unique(sizes, length, seed=0):
    return stream_module.build_unique_stream(np.asarray(sizes), length, np.random.default_rng(seed))


def test_unique_stream_never_requests_a_vector_twice():
    stream = make_unique([200] * 50, 6000)
    assert stream.kind == "uniform_unique" and stream.length == 6000
    assert len(np.unique(stream.pool_index)) == 6000  # every request is a different pool row
    assert stream_module.repeat_diagnostics(stream).exact_repeat_fraction == 0.0


def test_unique_stream_indices_follow_the_pool_layout_for_unequal_groups():
    sizes = np.array([3, 1, 7, 2, 5])
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    stream = make_unique(sizes, 18)  # the whole pool of 18 rows
    assert np.array_equal(np.sort(stream.pool_index), np.arange(18))  # each row exactly once
    assert np.array_equal(stream.pool_index, offsets[stream.group_ids] + stream.member_ids)
    assert (stream.member_ids >= 0).all()
    assert (stream.member_ids < sizes[stream.group_ids]).all()


def test_unique_stream_requests_groups_in_proportion_to_their_size():
    sizes = np.array([100, 300, 200, 400])  # pool of 1000 rows, 4 groups
    stream = make_unique(sizes, 500)  # half of the pool
    counts = np.bincount(stream.group_ids, minlength=4)
    # A uniform draw of 500 of 1000 rows gives each group about half of its rows; the spread
    # is hypergeometric (std below 8 for these sizes), so +-40 is a wide safety margin.
    assert np.all(np.abs(counts - sizes / 2) < 40), counts


def test_unique_stream_is_uniform_over_equal_groups():
    stream = make_unique([200] * 40, 4000)
    counts = np.bincount(stream.group_ids, minlength=40)
    # Mean 100 per group; the std is about 6.7, so +-40 (6 sigma) never triggers by chance.
    assert np.all(np.abs(counts - 100) < 40), counts


def test_unique_stream_is_deterministic_and_depends_on_the_seed():
    first = make_unique([20] * 30, 100, seed=1)
    again = make_unique([20] * 30, 100, seed=1)
    other = make_unique([20] * 30, 100, seed=2)
    assert np.array_equal(first.pool_index, again.pool_index)
    assert not np.array_equal(first.pool_index, other.pool_index)


def test_unique_stream_longer_than_the_pool_is_rejected():
    with pytest.raises(ValueError, match="needs 19 different vectors"):
        make_unique([3, 1, 7, 2, 5], 19)
    with pytest.raises(ValueError):
        make_unique([3, 1], 0)
    with pytest.raises(ValueError):
        make_unique([3, 0], 2)


def test_head_of_a_unique_stream_stays_unique_and_keeps_its_kind():
    head = make_unique([50] * 20, 800).head(300)
    assert head.kind == "uniform_unique" and len(np.unique(head.pool_index)) == 300


def test_head_returns_the_first_requests_unchanged():
    stream = make_stream([3, 1, 7, 2, 5], 400, 0.6)
    head = stream.head(150)
    assert head.length == 150 and head.exponent == stream.exponent
    assert np.array_equal(head.group_ids, stream.group_ids[:150])
    assert np.array_equal(head.member_ids, stream.member_ids[:150])
    assert np.array_equal(head.pool_index, stream.pool_index[:150])
    assert stream.head(400).length == 400  # the full length is allowed


@pytest.mark.parametrize("count", [0, -1, 401])
def test_head_rejects_counts_outside_the_stream(count):
    with pytest.raises(ValueError):
        make_stream([3, 1, 7, 2, 5], 400, 0.6).head(count)


def test_indices_are_consistent_with_the_pool_layout():
    sizes = np.array([3, 1, 7, 2, 5])
    offsets = np.concatenate([[0], np.cumsum(sizes)])[:-1]
    stream = make_stream(sizes, 5000, 0.6)

    assert isinstance(stream, Stream) and stream.length == 5000
    assert stream.group_ids.min() >= 0 and stream.group_ids.max() < len(sizes)
    assert np.all(stream.member_ids >= 0)
    assert np.all(stream.member_ids < sizes[stream.group_ids])
    assert np.array_equal(stream.pool_index, offsets[stream.group_ids] + stream.member_ids)
    assert stream.pool_index.max() < sizes.sum()


def test_same_seed_is_deterministic_and_other_seed_differs():
    first = make_stream([4] * 20, 1000, 0.9, seed=3)
    again = make_stream([4] * 20, 1000, 0.9, seed=3)
    other = make_stream([4] * 20, 1000, 0.9, seed=4)
    assert np.array_equal(first.pool_index, again.pool_index)
    assert not np.array_equal(first.pool_index, other.pool_index)


def test_group_popularity_follows_the_zipf_law():
    # Sorted request counts per group must match the expected counts per popularity rank.
    num_groups, length = 12, 400_000
    stream = make_stream([5] * num_groups, length, 1.0, seed=5)
    observed = np.sort(np.bincount(stream.group_ids, minlength=num_groups))[::-1]
    probabilities = stream_module.zipf_probabilities(num_groups, 1.0)
    expected = length * probabilities
    spread = np.sqrt(length * probabilities * (1 - probabilities))
    assert np.all(np.abs(observed - expected) < 4 * spread)


def test_uniform_exponent_spreads_requests_evenly_over_groups():
    num_groups, length = 40, 200_000
    stream = make_stream([3] * num_groups, length, 0.0, seed=6)
    counts = np.bincount(stream.group_ids, minlength=num_groups)
    chi_square = ((counts - length / num_groups) ** 2 / (length / num_groups)).sum()
    # 99.9% critical value of chi-square with 39 degrees of freedom is about 72.1.
    assert chi_square < 72.1


def test_members_are_uniform_inside_each_group_even_for_unequal_sizes():
    sizes = [1, 6, 20]
    stream = make_stream(sizes, 300_000, 0.0, seed=7)
    for group, size in enumerate(sizes):
        members = stream.member_ids[stream.group_ids == group]
        counts = np.bincount(members, minlength=size)
        assert len(counts) == size  # no member index beyond the group size
        expected = len(members) / size
        chi_square = ((counts - expected) ** 2 / expected).sum()
        # Loose bound: far below the 99.99% critical value for up to 19 degrees of freedom.
        assert chi_square < 60.0


def test_popularity_does_not_follow_group_order():
    # The most requested group must not always be group 0.
    winners = {
        int(np.argmax(np.bincount(make_stream([2] * 30, 5000, 1.5, seed=s).group_ids, minlength=30)))
        for s in range(12)
    }
    assert len(winners) > 3


def test_invalid_streams_are_rejected():
    with pytest.raises(ValueError):
        make_stream([], 10, 0.5)
    with pytest.raises(ValueError):
        make_stream([3, 0], 10, 0.5)
    with pytest.raises(ValueError):
        make_stream([3], 0, 0.5)


# ---------------------------------------------------------------------------
# steps and diagnostics
# ---------------------------------------------------------------------------


def test_step_boundaries_cover_the_stream_exactly():
    bounds = stream_module.step_boundaries(10, 4)
    assert bounds.tolist() == [0, 3, 6, 8, 10]
    assert stream_module.step_boundaries(7, 7).tolist() == list(range(8))


def test_repeat_diagnostics_on_a_hand_checked_stream():
    stream = Stream(
        exponent=0.5,
        group_ids=np.array([0, 0, 1, 0]),
        member_ids=np.array([0, 0, 2, 1]),
        pool_index=np.array([0, 0, 5, 1]),
    )
    diagnostics = stream_module.repeat_diagnostics(stream)
    # Request 2 repeats request 1 exactly; requests 2 and 4 revisit group 0.
    assert diagnostics.exact_repeat_fraction == pytest.approx(1 / 4)
    assert diagnostics.group_revisit_fraction == pytest.approx(2 / 4)
    assert (diagnostics.distinct_vectors, diagnostics.distinct_groups) == (3, 2)


def test_repeat_diagnostics_match_a_sequential_python_simulation():
    stream = make_stream([6] * 25, 3000, 0.8, seed=8)
    seen_vectors, seen_groups = set(), set()
    exact_repeats = group_revisits = 0
    for vector, group in zip(stream.pool_index.tolist(), stream.group_ids.tolist()):
        exact_repeats += vector in seen_vectors
        group_revisits += group in seen_groups
        seen_vectors.add(vector)
        seen_groups.add(group)

    diagnostics = stream_module.repeat_diagnostics(stream)
    assert diagnostics.exact_repeat_fraction == pytest.approx(exact_repeats / 3000)
    assert diagnostics.group_revisit_fraction == pytest.approx(group_revisits / 3000)


def test_larger_groups_mean_fewer_exact_repeats():
    small = stream_module.repeat_diagnostics(make_stream([5] * 30, 20_000, 0.6, seed=9))
    large = stream_module.repeat_diagnostics(make_stream([200] * 30, 20_000, 0.6, seed=9))
    assert large.exact_repeat_fraction < small.exact_repeat_fraction
    # The group revisit share is the same: it does not depend on the group size.
    assert large.group_revisit_fraction == pytest.approx(small.group_revisit_fraction, abs=0.005)
