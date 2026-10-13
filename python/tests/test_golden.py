"""Golden-hash regression: the output files must not change with round 00067.

``golden/golden_hashes.json`` holds the sha256 of every output file of several small
synthetic scenarios, made by the code of round 00066 (before the universe lists and the
parallel generation). For integer-valued data the new code has to reproduce every hash, for
every update percentage and for any number of CPU workers.

The hashes depend on the random number generators of numpy. They are recorded together with
the numpy version, and the test is skipped (not failed) under another numpy version.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import golden_inputs as gi
from semantic_vdb import base_set as base_set_module
from semantic_vdb import config, manifest, workload

GOLDEN_FILE = Path(__file__).parent / "golden" / "golden_hashes.json"
GOLDEN = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(
    GOLDEN["numpy"] != np.__version__,
    reason=f"golden hashes were made with numpy {GOLDEN['numpy']}, this is {np.__version__}",
)

# No NUMA pinning and no GPU: the golden scenarios run on the CPU in any environment.
def cpu_parallel(workers=1):
    return config.ParallelConfig(gpus="none", cpu_workers=workers, numa_node="none")


def build_scenario(name, tmp_path, workers=1):
    """Build one scenario with the current code; returns the family directory."""
    p = gi.scenario(name)
    paths = gi.write_inputs(tmp_path)
    base = base_set_module.build_base_set(
        config.BaseSetConfig(
            raw_file=paths["raw"], metric=p["metric"], seed=p["base_seed"], num_base=gi.NUM_BASE,
            base_selection="random", num_query_rows=0, max_inserts=gi.MAX_INSERTS,
            output_dir=tmp_path / "out", name="base_" + name,
        )
    )
    external = p["mode"] == "external"
    cfg = config.WorkloadConfig(
        name="wl",
        base_set_dir=base.directory,
        seeds=config.SeedsConfig(
            p["mode"], paths["external"] if external else paths["queries"],
            paths["external_groups"] if external else None, None, seed=p["seeds_seed"],
        ),
        groups=config.GroupsConfig(
            group_sizes=(p["group_size"],), include_original=False, neighbours=p["neighbours"],
            axes=p["axes"], residual=False, target_overlaps=(p["target_overlap"],),
            calibration_queries=p["calibration_queries"], overlap_tolerance=p["overlap_tolerance"],
            variants=p["variants"],
        ),
        stream=config.StreamConfig(
            total_requests=p["total_requests"], num_steps=p["num_steps"],
            zipf_exponents=tuple(p["zipf"]), member_selection="uniform", seed=p["stream_seed"],
            uniform_unique=p["uniform_unique"],
        ),
        updates=config.UpdatesConfig(seed=p["updates_seed"], update_percents=tuple(p["percents"])),
        evaluation=config.EvaluationConfig(
            "external" if external else "original", paths["external_eval"] if external else None,
            checkpoint_every=p["checkpoint_every"], eval_k=p["eval_k"],
        ),
        groundtruth=config.GroundTruthConfig(
            stream_k=p["stream_k"], device="cpu", base_batch_rows=97, query_chunk_rows=13,
            stream_groundtruth=True,
        ),
        parallel=cpu_parallel(workers),
    )
    return workload.build_workload(cfg)


def hashes_of_pool(family, pool_name):
    """{set name: {relative path: sha256}} of one pool directory."""
    return {
        path.name: {
            relative: entry["sha256"]
            for relative, entry in manifest.read_json(path / "workload.json")["files"].items()
        }
        for path in sorted((family / pool_name).iterdir())
        if path.is_dir()
    }


@pytest.mark.parametrize("name", sorted(GOLDEN["scenarios"]))
def test_output_files_equal_the_golden_hashes(name, tmp_path):
    p = gi.scenario(name)
    family = build_scenario(name, tmp_path)
    if p["mode"] == "external":
        pool_name = "external"
    elif p["variants"] == "none":
        pool_name = "g1"
    else:
        # N2 pools live below their overlap level (o70 for the target overlap 0.7).
        pool_name = f"o70/g{p['group_size']}"
    assert hashes_of_pool(family, pool_name) == GOLDEN["scenarios"][name]


@pytest.mark.parametrize("workers", [2, 4])
def test_output_files_do_not_depend_on_the_number_of_cpu_workers(workers, tmp_path):
    family = build_scenario("n2_family", tmp_path, workers=workers)
    assert hashes_of_pool(family, "o70/g6") == GOLDEN["scenarios"]["n2_family"]
