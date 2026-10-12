"""Row partition of a raw vector file into base / query / insert row sets.

All three sets are drawn from the same population of raw row numbers and never overlap:

* ``base_row_ids``   rows of the initial index, sorted ascending (so the raw file is read
                      sequentially). The id of a base vector is its position in this array.
* ``query_row_ids``  rows reserved as queries (``base_rows`` mode only), sorted ascending.
* ``insert_row_ids`` rows to be inserted later, in a FIXED RANDOM ORDER. The order matters:
                      the j-th inserted vector is ``insert_row_ids[j]`` for every workload
                      that shares this partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

BASE_SELECTIONS = ("random", "prefix")


@dataclass(frozen=True)
class RowPartition:
    base_row_ids: np.ndarray  # int64, ascending
    query_row_ids: np.ndarray  # int64, ascending (may be empty)
    insert_row_ids: np.ndarray  # int64, random order


def _draw_distinct_rows(
    rng: np.random.Generator, first_row: int, population_size: int, count: int
) -> np.ndarray:
    """Draw ``count`` distinct rows from ``[first_row, first_row + population_size)``.

    The result is in random order. When the whole population is requested a full
    permutation is returned. Note that for large draws numpy materialises the whole
    population (8 bytes per row), which is acceptable on this host but means a
    ``max_inserts`` of "rest" on a billion-row file needs about 8 GB of memory.
    """
    if count == population_size:
        rows = rng.permutation(population_size)
    else:
        rows = rng.choice(population_size, size=count, replace=False)
    return first_row + rows.astype(np.int64)


def partition_rows(
    total_rows: int,
    num_base: int,
    num_query_rows: int,
    max_inserts: Optional[int],
    seed: int,
    base_selection: str = "random",
) -> RowPartition:
    """Split ``total_rows`` raw rows into base / query / insert sets.

    ``max_inserts=None`` means "every remaining row" (the rest of the file).
    ``base_selection="prefix"`` takes the first ``num_base`` rows as the base; the
    other two sets are then drawn from the rows after it.
    """
    if base_selection not in BASE_SELECTIONS:
        raise ValueError(f"base_selection must be one of {BASE_SELECTIONS}, got {base_selection!r}")
    if num_base <= 0:
        raise ValueError("num_base must be positive")
    if num_query_rows < 0 or (max_inserts is not None and max_inserts < 0):
        raise ValueError("num_query_rows and max_inserts must not be negative")

    # Resolve "rest" and check that the three sets fit into the file.
    if max_inserts is None:
        max_inserts = total_rows - num_base - num_query_rows
    needed_rows = num_base + num_query_rows + max_inserts
    if max_inserts < 0 or needed_rows > total_rows:
        raise ValueError(
            f"need {num_base} base + {num_query_rows} query + {max_inserts} insert rows "
            f"= {needed_rows}, but the file has only {total_rows} rows"
        )

    rng = np.random.default_rng(seed)

    if base_selection == "prefix":
        # Base is the file prefix; queries and inserts come from the rows behind it.
        base_row_ids = np.arange(num_base, dtype=np.int64)
        drawn = _draw_distinct_rows(
            rng, num_base, total_rows - num_base, num_query_rows + max_inserts
        )
        query_row_ids = drawn[:num_query_rows]
        insert_row_ids = drawn[num_query_rows:]
    else:
        # One joint draw guarantees that the three sets are disjoint by construction.
        drawn = _draw_distinct_rows(rng, 0, total_rows, needed_rows)
        base_row_ids = drawn[:num_base]
        query_row_ids = drawn[num_base : num_base + num_query_rows]
        insert_row_ids = drawn[num_base + num_query_rows :]

    return RowPartition(
        base_row_ids=np.sort(base_row_ids),
        query_row_ids=np.sort(query_row_ids),
        # Insert order is deliberately NOT sorted: it is the fixed insertion sequence.
        insert_row_ids=insert_row_ids,
    )
