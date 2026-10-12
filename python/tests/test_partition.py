"""Tests for semantic_vdb.partition."""

import numpy as np
import pytest

from semantic_vdb.partition import partition_rows

TOTAL = 10_000


def all_sets(partition):
    return partition.base_row_ids, partition.query_row_ids, partition.insert_row_ids


def test_counts_and_range():
    part = partition_rows(TOTAL, num_base=1000, num_query_rows=200, max_inserts=300, seed=1)
    assert len(part.base_row_ids) == 1000
    assert len(part.query_row_ids) == 200
    assert len(part.insert_row_ids) == 300
    for rows in all_sets(part):
        assert rows.dtype == np.int64
        assert rows.min() >= 0 and rows.max() < TOTAL


def test_sets_are_pairwise_disjoint_and_duplicate_free():
    part = partition_rows(TOTAL, num_base=1500, num_query_rows=500, max_inserts=2000, seed=2)
    combined = np.concatenate(all_sets(part))
    # Every row appears at most once across the three sets.
    assert len(np.unique(combined)) == len(combined)


def test_base_and_query_rows_are_sorted_but_insert_order_is_random():
    part = partition_rows(TOTAL, num_base=1000, num_query_rows=300, max_inserts=2000, seed=3)
    assert np.all(np.diff(part.base_row_ids) > 0)
    assert np.all(np.diff(part.query_row_ids) > 0)
    # A random order of 2000 distinct rows is sorted with negligible probability.
    assert not np.all(np.diff(part.insert_row_ids) > 0)


def test_same_seed_is_deterministic_and_other_seed_differs():
    first = partition_rows(TOTAL, 1000, 100, 500, seed=7)
    again = partition_rows(TOTAL, 1000, 100, 500, seed=7)
    other = partition_rows(TOTAL, 1000, 100, 500, seed=8)
    for left, right in zip(all_sets(first), all_sets(again)):
        assert np.array_equal(left, right)
    assert not np.array_equal(first.base_row_ids, other.base_row_ids)
    assert not np.array_equal(first.insert_row_ids, other.insert_row_ids)


def test_rest_uses_every_remaining_row_exactly_once():
    part = partition_rows(TOTAL, num_base=1000, num_query_rows=250, max_inserts=None, seed=4)
    assert len(part.insert_row_ids) == TOTAL - 1000 - 250
    combined = np.concatenate(all_sets(part))
    # The three sets together are a permutation of the whole file.
    assert np.array_equal(np.sort(combined), np.arange(TOTAL))


def test_prefix_selection_takes_the_first_rows():
    part = partition_rows(
        TOTAL, num_base=700, num_query_rows=100, max_inserts=400, seed=6, base_selection="prefix"
    )
    assert np.array_equal(part.base_row_ids, np.arange(700))
    # Queries and inserts come strictly from behind the prefix and stay disjoint.
    assert part.query_row_ids.min() >= 700
    assert part.insert_row_ids.min() >= 700
    combined = np.concatenate(all_sets(part))
    assert len(np.unique(combined)) == len(combined)


def test_zero_query_rows_and_zero_inserts_are_allowed():
    part = partition_rows(TOTAL, num_base=TOTAL, num_query_rows=0, max_inserts=0, seed=0)
    assert np.array_equal(part.base_row_ids, np.arange(TOTAL))
    assert len(part.query_row_ids) == 0 and len(part.insert_row_ids) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_base=0, num_query_rows=0, max_inserts=0),  # empty base
        dict(num_base=9000, num_query_rows=500, max_inserts=600),  # does not fit
        dict(num_base=10, num_query_rows=-1, max_inserts=0),  # negative count
        dict(num_base=10, num_query_rows=0, max_inserts=-5),  # negative count
        dict(num_base=TOTAL, num_query_rows=1, max_inserts=None),  # rest would be negative
        dict(num_base=10, num_query_rows=0, max_inserts=0, base_selection="sorted"),
    ],
)
def test_invalid_requests_are_rejected(kwargs):
    with pytest.raises(ValueError):
        partition_rows(TOTAL, seed=0, **kwargs)
