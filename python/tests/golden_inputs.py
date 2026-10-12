"""Deterministic synthetic inputs and scenario descriptions for the golden-hash regression.

The golden hashes (``golden/golden_hashes.json``) were produced by the code of round 00066,
BEFORE the universe-list ground truth and the parallel generation of round 00067. They pin
the contract of round 00067: for integer-valued data the output files must not change by a
single bit. This module is shared by the script that made the hashes and by the test that
checks them, so both build exactly the same inputs.

It depends only on numpy and ``semantic_vdb.binio`` (whose API is the same before and after
round 00067), never on the configuration classes: scenarios are plain dictionaries and each
side turns them into its own configuration objects.
"""

from pathlib import Path

import numpy as np

from semantic_vdb import binio

DIM = 12
RAW_ROWS = 1400
NUM_BASE, MAX_INSERTS = 800, 300
NUM_QUERIES = 40

# Parameters shared by every scenario unless the scenario overrides them.
COMMON = dict(
    metric="l2",
    mode="public_queries",
    variants="n2",
    group_size=6,
    percents=(25.0,),
    total_requests=480,
    num_steps=8,
    zipf=(0.5, 0.99),
    uniform_unique=False,
    stream_k=5,
    eval_k=10,
    checkpoint_every=3,
    base_seed=7,
    seeds_seed=11,
    stream_seed=5,
    updates_seed=21,
    neighbours=60,
    axes=4,
    target_overlap=0.7,
    calibration_queries=20,
    overlap_tolerance=0.05,
)

SCENARIOS = {
    "n2_family": dict(percents=(0.0, 25.0, 50.0)),
    "n2_unique": dict(percents=(0.0, 25.0, 50.0), total_requests=240, uniform_unique=True),
    "n2_group3": dict(group_size=3),
    "n2_inner_product": dict(metric="ip"),
    "given_queries": dict(
        variants="none", group_size=1, percents=(0.0, 25.0), total_requests=40, num_steps=4,
        zipf=(0.5,), uniform_unique=True,
    ),
    "external_unique": dict(
        mode="external", percents=(0.0,), total_requests=30, num_steps=3, zipf=(),
        uniform_unique=True, eval_k=7, checkpoint_every=4,
    ),
}


def scenario(name):
    """The full parameter dictionary of one scenario."""
    values = dict(COMMON)
    values.update(SCENARIOS[name])
    return values


def synthetic_rows(count, seed, centres=None):
    """Clustered uint8 vectors; the cluster centres can be shared between calls."""
    rng = np.random.default_rng(seed)
    if centres is None:
        centres = rng.integers(40, 200, size=(25, DIM))
    labels = rng.integers(0, len(centres), size=count)
    points = centres[labels] + rng.normal(size=(count, DIM)) * 8.0
    return np.clip(np.rint(points), 0, 255).astype(np.uint8), centres


def write_inputs(directory):
    """Write the raw file, the query file and the external-mode files; return their paths."""
    directory = Path(directory)
    raw, centres = synthetic_rows(RAW_ROWS, seed=1)
    queries, _ = synthetic_rows(NUM_QUERIES, seed=2, centres=centres)
    paths = {
        "raw": directory / "raw.u8bin",
        "queries": directory / "query.u8bin",
        "external": directory / "external.u8bin",
        "external_groups": directory / "external_groups.ibin",
        "external_eval": directory / "external_eval.u8bin",
    }
    binio.write_array(paths["raw"], raw)
    binio.write_array(paths["queries"], queries)

    # External groups of sizes 4, 10, 1 and 15 with the rows shuffled, as in a real input.
    vectors, _ = synthetic_rows(30, seed=9, centres=synthetic_rows(10, 1)[1])
    group_ids = np.array([5] * 4 + [9] * 10 + [2] * 1 + [7] * 15)
    shuffle = np.random.default_rng(3).permutation(30)
    binio.write_array(paths["external"], vectors[shuffle])
    binio.write_ids(paths["external_groups"], group_ids[shuffle])
    binio.write_array(paths["external_eval"], queries[:15])
    return paths
