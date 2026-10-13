"""The scan accumulator as it was before round 00077 (every distance is encoded and sorted).

``KnnAccumulator`` now merges only the entries that can still reach the top-k (the filtered
merge). This class keeps the previous, simpler algorithm so that tests can check that both give
byte-identical lists. Unlike ``reference.py`` it deliberately shares the key encoding and the
distance function with ``semantic_vdb``: the question it answers is "did the filter change any
winner", not "is the distance formula right" (``reference.py`` answers that one).
"""

import numpy as np

from semantic_vdb import groundtruth
from semantic_vdb.binio import GroundTruth

EMPTY_KEY = np.uint64(0xFFFFFFFFFFFFFFFF)


class DenseKnnAccumulator:
    """Running exact top-k per query; every block is fully encoded and partitioned."""

    def __init__(self, queries, k, metric, device="cpu", query_chunk_rows=1024):
        self.k = k
        self.metric = metric
        self.xp = groundtruth.array_module(device)
        xp = self.xp
        self.bounds = [
            (start, min(start + query_chunk_rows, len(queries)))
            for start in range(0, len(queries), query_chunk_rows)
        ]
        self.queries = [xp.asarray(queries[a:b], dtype=xp.float32) for a, b in self.bounds]
        self.query_sq = [(chunk * chunk).sum(axis=1)[:, None] for chunk in self.queries]
        self.best = [xp.full((b - a, k), EMPTY_KEY, dtype=xp.uint64) for a, b in self.bounds]

    def add_block(self, global_ids, vectors):
        xp = self.xp
        block = xp.asarray(vectors, dtype=xp.float32)
        ids = xp.asarray(global_ids, dtype=xp.int64)
        for index in range(len(self.queries)):
            distances = groundtruth.pairwise_distance(
                xp, self.queries[index], self.query_sq[index], block, self.metric
            )
            keys = groundtruth.encode_sort_keys(xp, distances, ids)
            # Keep only the k best keys of this block, then merge with the running best.
            if keys.shape[1] > self.k:
                keys = xp.partition(keys, self.k - 1, axis=1)[:, : self.k]
            merged = xp.concatenate([self.best[index], keys], axis=1)
            if merged.shape[1] > self.k:
                merged = xp.partition(merged, self.k - 1, axis=1)[:, : self.k]
            self.best[index] = merged

    def finish(self):
        xp = self.xp
        id_parts, dist_parts = [], []
        for best in self.best:
            ids, distances = groundtruth.decode_sort_keys(xp, xp.sort(best, axis=1))
            id_parts.append(groundtruth._to_host(ids).astype(np.int32))
            dist_parts.append(groundtruth._to_host(distances).astype(np.float32))
        return GroundTruth(
            neighbour_ids=np.concatenate(id_parts, axis=0),
            neighbour_dists=np.concatenate(dist_parts, axis=0),
        )
