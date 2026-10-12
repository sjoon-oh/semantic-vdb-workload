"""The target overlap as a combination axis: pools o<overlap>/g<size>/u<percent>.

A family built with several target overlaps must contain, for every overlap, exactly the pools
that a build with that single overlap makes (the noise scale of an overlap does not depend on
the other overlaps). Ground truth is always re-derived from the output files with the
independent float64 reference, so these tests do not trust the builder's own bookkeeping.
"""

import numpy as np
import pytest

from semantic_vdb import binio, config, manifest, n2, workload
from reference import reference_knn
from test_end_to_end import (
    NUM_STEPS,
    StageCounter,
    build_base,
    dataset,  # noqa: F401  (pytest fixture)
    eval_only_groundtruth,
    inode,
    interrupt,
    named,
    read_all,
    set_files,
    verify_workload,
    workload_config,
    workload_manifest,
)


def overlap_config(
    base_set, query_path, overlaps=(0.7, 0.5), sizes=(3,), percents=(25.0,), **more
):
    """A small family configuration with the given overlap and group-size axes."""
    groups = config.GroupsConfig(
        group_sizes=tuple(sizes),
        include_original=False,
        neighbours=60,
        axes=4,
        residual=False,
        target_overlaps=tuple(overlaps),
        calibration_queries=20,
        overlap_tolerance=0.05,
        variants=more.pop("variants", "n2"),
    )
    updates = config.UpdatesConfig(seed=21, update_percents=tuple(percents))
    return workload_config(base_set, query_path, groups=groups, updates=updates, **more)


def build(dataset, base_set, **kwargs):
    """Build a family from ``overlap_config`` (a shorthand for the many calls below)."""
    return workload.build_workload(overlap_config(base_set, dataset["query_path"], **kwargs))


# ---------------------------------------------------------------------------
# what is built
# ---------------------------------------------------------------------------


def test_every_overlap_and_group_size_is_a_pool_and_every_set_is_exact(dataset):
    base_set = build_base(dataset)
    family = build(dataset, base_set, overlaps=(0.7, 0.5), sizes=(3, 6))

    # Overlap level first, then the group size, then the update percentage.
    top_level = sorted(path.name for path in family.iterdir() if path.is_dir())
    assert top_level == ["o50", "o70", "shared"]
    targets = {"o70": 0.7, "o50": 0.5}
    for overlap, target in targets.items():
        assert sorted(path.name for path in (family / overlap).iterdir()) == ["g3", "g6"]
        for size in (3, 6):
            directory = family / overlap / f"g{size}" / "u25"
            # Ground truth of every stream and of the evaluation queries, from the files alone.
            verify_workload(directory, base_set, NUM_STEPS, group_size=size)
            info = workload_manifest(directory)
            assert info["pool"]["name"] == f"{overlap}/g{size}"
            assert info["pool"]["overlap"] == overlap
            assert info["pool"]["target_overlap"] == target
            assert info["groups"]["target_overlap"] == target
            # The set describes its own pool only, not the other overlaps of the build.
            assert info["config"]["groups"]["target_overlaps"] == [target]
            assert info["config"]["groups"]["group_sizes"] == [size]


def test_the_family_index_lists_pools_in_the_order_of_the_configuration(dataset):
    base_set = build_base(dataset)
    family = build(dataset, base_set, overlaps=(0.5, 0.7), sizes=(6, 3))
    index = manifest.read_json(family / "family.json")

    assert [pool["name"] for pool in index["pools"]] == [
        "o50/g6", "o50/g3", "o70/g6", "o70/g3",
    ]
    assert [entry["name"] for entry in index["overlaps"]] == ["o50", "o70"]
    assert [entry["path"] for entry in index["sets"]] == [
        "o50/g6/u25", "o50/g3/u25", "o70/g6/u25", "o70/g3/u25",
    ]
    for pool in index["pools"]:
        assert pool["overlap"] == pool["name"].split("/")[0]


def test_a_smaller_target_overlap_needs_a_larger_alpha_and_hits_its_target(dataset):
    base_set = build_base(dataset)
    targets = (0.9, 0.7, 0.5, 0.4)
    family = build(dataset, base_set, overlaps=targets)
    index = manifest.read_json(family / "family.json")

    by_target = {entry["target_overlap"]: entry for entry in index["overlaps"]}
    alphas = [by_target[target]["alpha"] for target in targets]
    # More noise means a lower overlap, so alpha grows as the target shrinks.
    assert alphas == sorted(alphas) and len(set(alphas)) == len(alphas)
    for target in targets:
        entry = by_target[target]
        assert abs(entry["achieved_overlap"] - target) <= entry["tolerance"]


def test_the_measured_overlap_of_every_pool_is_the_one_the_statistics_report(dataset):
    """Recompute the top-10 overlap of seed and variant from the files, with the reference."""
    base_set = build_base(dataset)
    family = build(dataset, base_set, overlaps=(0.7, 0.5))
    base = read_all(base_set.directory / "base.u8bin")
    base_ids = np.arange(len(base))
    seeds = read_all(family / "shared" / "common" / "seeds.u8bin")
    seed_top, _ = reference_knn(base, base_ids, seeds, 10)

    measured = {}
    for overlap in ("o70", "o50"):
        directory = family / overlap / "g3" / "u25"
        pool = read_all(directory / "pool" / "pool.u8bin")
        first_variants = pool[0::3]  # include_original = false: member 0 is a variant
        variant_top, _ = reference_knn(base, base_ids, first_variants, 10)
        shared_ids = [len(set(a) & set(b)) / 10 for a, b in zip(seed_top, variant_top)]
        measured[overlap] = float(np.mean(shared_ids))
        stored = workload_manifest(directory)["groups"]["overall_mean_overlap"]
        assert measured[overlap] == pytest.approx(stored, abs=1e-12), overlap
    # The two pools really differ in how close their variants are to the seeds.
    assert measured["o70"] > measured["o50"]


def test_pools_without_noise_have_no_overlap_level(dataset, monkeypatch):
    """With variants = none the overlap list is ignored: one pool g1, no noise scale."""
    base_set = build_base(dataset)
    monkeypatch.setattr(n2, "find_alpha", lambda *a, **k: pytest.fail("no alpha is searched"))
    stream = config.StreamConfig(
        total_requests=40, num_steps=4, zipf_exponents=(0.5,), member_selection="uniform",
        seed=5,
    )
    family = build(
        dataset, base_set, overlaps=(0.7, 0.5), sizes=(1,), variants="none", stream=stream
    )
    index = manifest.read_json(family / "family.json")
    assert [pool["name"] for pool in index["pools"]] == ["g1"]
    assert index["pools"][0]["overlap"] is None
    assert index["overlaps"] == []


# ---------------------------------------------------------------------------
# the pools of a joint build are the pools of separate builds
# ---------------------------------------------------------------------------


def test_every_overlap_of_a_joint_build_equals_a_build_of_that_overlap_alone(dataset):
    """The noise scale of an overlap must not depend on which other overlaps are built."""
    base_set = build_base(dataset)
    query_path = dataset["query_path"]

    def alone(overlaps, name):
        cfg = overlap_config(base_set, query_path, overlaps=overlaps, sizes=(3, 6))
        return workload.build_workload(named(cfg, name))

    joint = alone((0.7, 0.5), "joint")
    only70, only50 = alone((0.7,), "only70"), alone((0.5,), "only50")

    joint_files = set_files(joint)
    alone_files = {**set_files(only70), **set_files(only50)}
    assert set(joint_files) == set(alone_files) == {
        f"{overlap}/g{size}/u25" for overlap in ("o70", "o50") for size in (3, 6)
    }
    for name in joint_files:
        # Every file of the set: the pool, the streams, the deleted ids, all ground truth.
        assert joint_files[name] == alone_files[name], name
    # Also the noise scale itself.
    joint_index = manifest.read_json(joint / "family.json")
    joint_alpha = {entry["name"]: entry["alpha"] for entry in joint_index["overlaps"]}
    for family in (only70, only50):
        for entry in manifest.read_json(family / "family.json")["overlaps"]:
            assert joint_alpha[entry["name"]] == entry["alpha"]


def test_the_order_of_the_overlaps_does_not_change_any_set(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    forward = overlap_config(base_set, query_path, (0.7, 0.5))
    backward = overlap_config(base_set, query_path, (0.5, 0.7))
    first = workload.build_workload(named(forward, "forward"))
    second = workload.build_workload(named(backward, "backward"))
    assert set_files(first) == set_files(second)


# ---------------------------------------------------------------------------
# what the overlaps share
# ---------------------------------------------------------------------------


def test_overlaps_share_the_files_that_do_not_depend_on_the_noise(dataset):
    base_set = build_base(dataset)
    family = build(dataset, base_set, percents=(0.0, 25.0))
    first, second = family / "o70" / "g3" / "u25", family / "o50" / "g3" / "u25"
    store = family / "shared"
    eval_dir = store / "mixes" / "u25" / "groundtruth" / "eval"

    # Queries, evaluation ground truth and deleted ids: ONE file for both overlaps, and the
    # same file as in shared/, where a later run finds it.
    for relative, shared_path in (
        ("seeds.u8bin", store / "common" / "seeds.u8bin"),
        ("eval_query.u8bin", store / "common" / "eval_query.u8bin"),
        (
            "groundtruth/eval/step_00000.bin",
            store / "eval_step0" / "groundtruth" / "eval" / "step_00000.bin",
        ),
        ("groundtruth/eval/step_00008.bin", eval_dir / "step_00008.bin"),
        (
            "updates/delete_step_00001.ibin",
            store / "mixes" / "u25" / "updates" / "delete_step_00001.ibin",
        ),
    ):
        assert inode(first / relative) == inode(second / relative) == inode(shared_path), relative

    # The pool differs (other noise), but the requests visit the same groups and members.
    first_pool, second_pool = first / "pool" / "pool.u8bin", second / "pool" / "pool.u8bin"
    assert inode(first_pool) != inode(second_pool)
    assert not np.array_equal(read_all(first_pool), read_all(second_pool))
    group_file = "stream_zipf_0.5/step_00001.group_id.ibin"
    assert (
        binio.read_ids(first / group_file).tolist() == binio.read_ids(second / group_file).tolist()
    )


def test_the_neighbours_and_bases_are_made_once_and_every_alpha_is_stored(dataset, monkeypatch):
    base_set = build_base(dataset)
    searches = []
    original_top = n2.BaseScanner.top

    def counting_top(self, queries, k):
        searches.append(k)
        return original_top(self, queries, k)

    monkeypatch.setattr(n2.BaseScanner, "top", counting_top)
    family = build(dataset, base_set, overlaps=(0.7, 0.5, 0.9))

    # The neighbours of the seeds (k = neighbours = 60): ONE scan for three overlaps.
    assert searches.count(60) == 1
    cache = manifest.read_json(family / "shared" / "cache.json")
    assert sorted(cache["calibrations"]) == ["o50", "o70", "o90"]
    index = manifest.read_json(family / "family.json")
    for entry in index["overlaps"]:
        stored = cache["calibrations"][entry["name"]]
        assert stored["alpha"] == entry["alpha"]
        assert stored["target_overlap"] == entry["target_overlap"]
    assert sorted(cache["n2"]["files"]) == [
        "basis_axes.npy", "basis_residual.npy", "basis_std.npy",
        "seed_neighbour_dists.npy", "seed_neighbour_ids.npy",
    ]
    for name in cache["n2"]["files"]:
        assert (family / "shared" / "n2" / name).is_file()


def test_stream_ground_truth_off_scans_only_the_evaluation_queries(dataset, monkeypatch):
    base_set = build_base(dataset)
    scans = []
    original_scan = workload.scan_lists

    def recording_scan(queries, list_length, *args, **kwargs):
        scans.append((args[6], len(queries)))
        return original_scan(queries, list_length, *args, **kwargs)

    monkeypatch.setattr(workload, "scan_lists", recording_scan)
    family = build(
        dataset, base_set, overlaps=(0.7, 0.5), groundtruth=eval_only_groundtruth()
    )
    assert [name for name, _ in scans] == ["eval"]  # the evaluation queries, once
    for overlap in ("o70", "o50"):
        verify_workload(family / overlap / "g3" / "u25", base_set, NUM_STEPS, group_size=3)


# ---------------------------------------------------------------------------
# interruption
# ---------------------------------------------------------------------------


def test_a_build_interrupted_after_the_first_overlap_keeps_its_noise_scale(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = overlap_config(base_set, dataset["query_path"], overlaps=(0.7, 0.5), sizes=(3, 6))
    clean = workload.build_workload(named(cfg, "clean"))

    # The statistics of the third pool (first pool of the second overlap) are never reached.
    original = interrupt(monkeypatch, "pool_statistics", after_calls=2)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(named(cfg, "resumed"))
    monkeypatch.setattr(workload, "pool_statistics", original)

    counter = StageCounter(monkeypatch)
    family = workload.build_workload(named(cfg, "resumed"))
    assert counter.calls["find_alpha"] == 0  # both noise scales were found before the pools
    assert counter.calls["basis_job"] == 0  # the PCA bases were kept as well
    assert counter.calls["make_pool"] == 2  # only the pools of o50 are made now
    assert counter.calls["pool_statistics"] == 2
    assert set_files(family) == set_files(clean)
    report = manifest.read_json(family / "build_report.json")
    kept = {"n2_bases", "calibration_o70", "calibration_o50", "pool_o70_g3", "pool_o70_g6"}
    assert kept <= set(report["resumed_from_stages"])


def test_a_build_with_several_overlaps_leaves_nothing_but_the_family(dataset):
    base_set = build_base(dataset)
    family = build(dataset, base_set)
    assert not list(family.parent.glob("*.building"))
    assert not any((family / name).exists() for name in ("_shared", "_universe", "_resume"))
    # shared/ is the one staging directory that stays: it holds what a later run reuses.
    assert (family / "shared" / "cache.json").is_file()
    assert sorted(path.name for path in (family / "shared").iterdir()) == [
        "cache.json", "common", "eval_step0", "mix_info", "mixes", "n2",
    ]
