"""Independent reference implementations used only by the tests.

They are deliberately simple and share no code with ``semantic_vdb``: float64 arithmetic,
one query at a time, ties broken with ``numpy.lexsort``.
"""

import numpy as np


def reference_knn(vectors, ids, queries, k, metric="l2"):
    """Exact top-k by brute force in float64; ties go to the smaller id.

    Returns ``(ids, distances)`` arrays of shape (num_queries, k), nearest first.
    """
    vectors = vectors.astype(np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    result_ids = np.empty((len(queries), k), dtype=np.int64)
    result_dists = np.empty((len(queries), k), dtype=np.float64)

    for row, query in enumerate(queries.astype(np.float64)):
        if metric == "l2":
            distance = ((vectors - query) ** 2).sum(axis=1)
        elif metric == "ip":
            distance = -(vectors @ query)
        else:
            raise ValueError(metric)
        # lexsort sorts by the LAST key first: distance, then id.
        order = np.lexsort((ids, distance))[:k]
        result_ids[row] = ids[order]
        result_dists[row] = distance[order]
    return result_ids, result_dists


def reference_overlap(ids_a, ids_b):
    """Mean fraction of shared ids between two (n, k) neighbour matrices."""
    shared = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ids_a, ids_b)]
    return float(np.mean(shared)) / ids_a.shape[1]
