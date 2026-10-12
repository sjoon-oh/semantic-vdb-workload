"""End-to-end tests: build a base set and workloads on small synthetic data.

Verification re-derives everything from the OUTPUT FILES with the independent float64
reference, so these tests do not trust any intermediate state of the builder.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from semantic_vdb import base_set as base_set_module
from semantic_vdb import binio, config, layout, manifest, n2, parallel, workload
from reference import reference_knn
from test_groundtruth import gpu_available

DIM = 12
BASE_ROWS = 1400
NUM_BASE, MAX_INSERTS, NUM_QUERY_ROWS = 800, 300, 60
# Requests per set (reads + updates) and steps. At 25% updates a set has 360 reads and
# 120 updates, i.e. 120 insertions and 120 deletions, 15 of each per step.
TOTAL_REQUESTS, NUM_STEPS = 480, 8
POOL = "o70/g6"  # the pool directory of the default group size 6 at target overlap 0.7


# ---------------------------------------------------------------------------
# fixtures and builders
# ---------------------------------------------------------------------------


def synthetic_rows(count, seed, centres=None):
    rng = np.random.default_rng(seed)
    if centres is None:
        centres = rng.integers(40, 200, size=(25, DIM))
    labels = rng.integers(0, len(centres), size=count)
    points = centres[labels] + rng.normal(size=(count, DIM)) * 8.0
    return np.clip(np.rint(points), 0, 255).astype(np.uint8), centres


@pytest.fixture()
def dataset(tmp_path):
    raw, centres = synthetic_rows(BASE_ROWS, seed=1)
    queries, _ = synthetic_rows(40, seed=2, centres=centres)
    raw_path = tmp_path / "raw.u8bin"
    query_path = tmp_path / "query.u8bin"
    binio.write_array(raw_path, raw)
    binio.write_array(query_path, queries)
    return {"tmp": tmp_path, "raw": raw, "raw_path": raw_path, "queries": queries, "query_path": query_path}


def build_base(dataset, name="base_a", **overrides):
    values = dict(
        raw_file=dataset["raw_path"], metric="l2", seed=7, num_base=NUM_BASE,
        base_selection="random", num_query_rows=0, max_inserts=MAX_INSERTS,
        output_dir=dataset["tmp"] / "out", name=name,
    )
    values.update(overrides)
    return base_set_module.build_base_set(config.BaseSetConfig(**values))


def cpu_parallel(workers=1):
    """No GPU, no NUMA pinning: the tests must not depend on the machine they run on."""
    return config.ParallelConfig(gpus="none", cpu_workers=workers, numa_node="none")


def workload_config(base_set, query_path, name="wl", mode="public_queries", device="cpu", **overrides):
    values = dict(
        name=name,
        base_set_dir=base_set.directory,
        seeds=config.SeedsConfig(mode, query_path, None, None, seed=11),
        groups=config.GroupsConfig(
            group_sizes=(6,), include_original=False, neighbours=60, axes=4, residual=False,
            target_overlaps=(0.7,), calibration_queries=20, overlap_tolerance=0.05,
        ),
        stream=config.StreamConfig(
            total_requests=TOTAL_REQUESTS, num_steps=NUM_STEPS, zipf_exponents=(0.5, 0.99),
            member_selection="uniform", seed=5,
        ),
        updates=config.UpdatesConfig(seed=21, update_percents=(25.0,)),
        evaluation=config.EvaluationConfig("original", None, checkpoint_every=3, eval_k=10),
        groundtruth=config.GroundTruthConfig(
            stream_k=5, device=device, base_batch_rows=97, query_chunk_rows=13
        ),
        # The only GPU test uses GPU 1 (see test_gpu_and_cpu_produce_identical_workloads).
        parallel=cpu_parallel() if device == "cpu" else config.ParallelConfig(
            gpus=(1,), cpu_workers=1, numa_node="none"
        ),
    )
    values.update(overrides)
    return config.WorkloadConfig(**values)


def pool_name_of(workload_cfg):
    """Directory name of the (first) pool of a configuration."""
    if workload_cfg.seeds.mode == "external":
        return "external"
    return (
        f"{layout.overlap_dir_name(workload_cfg.groups.target_overlaps[0])}/"
        f"g{workload_cfg.groups.group_sizes[0]}"
    )


def build_set(workload_cfg, set_name="u25"):
    """Build the family and return the directory of one of its sets (default: 25% updates)."""
    family = workload.build_workload(workload_cfg)
    return family / pool_name_of(workload_cfg) / set_name


def workload_manifest(directory):
    return manifest.read_json(directory / "workload.json")


# ---------------------------------------------------------------------------
# independent verification from the files
# ---------------------------------------------------------------------------


def read_all(path):
    return binio.read_array(path)


def verify_workload(directory, base_set, num_steps, group_size):
    """Recompute every ground-truth file with the float64 reference and compare."""
    info = workload_manifest(directory)
    schedule = info["schedule"]
    suffix = ".u8bin"

    base = read_all(base_set.directory / f"base{suffix}")
    inserts = read_all(base_set.directory / f"insert_pool{suffix}")
    num_base = len(base)
    pool = read_all(directory / "pool" / f"pool{suffix}")
    pool_groups = binio.read_ids(directory / "pool" / "pool_group_id.ibin")
    eval_queries = read_all(directory / f"eval_query{suffix}")
    stream_k = info["config"]["groundtruth"]["stream_k"]
    stream_gt = info["config"]["groundtruth"]["stream_groundtruth"]
    assert info["groundtruth"]["stream_groundtruth"] == stream_gt
    eval_k = info["config"]["evaluation"]["eval_k"]

    # Independent re-execution of the update schedule from the delete files.
    alive = {i: base[i] for i in range(num_base)}
    inserted = 0
    deleted_so_far = set()

    def check_eval(step):
        ids = np.array(sorted(alive))
        vectors = np.array([alive[i] for i in ids])
        stored = binio.read_groundtruth(directory / "groundtruth" / "eval" / f"step_{step:05d}.bin")
        expected_ids, expected_dists = reference_knn(vectors, ids, eval_queries, eval_k)
        assert np.array_equal(stored.neighbour_ids.astype(np.int64), expected_ids), f"eval step {step}"
        assert np.array_equal(stored.neighbour_dists.astype(np.float64), expected_dists)

    check_eval(0)  # initial state, before any update
    # Every stream directory: the Zipf streams and, when present, the uniform_unique stream.
    stream_names = sorted(path.name for path in directory.glob("stream_*"))
    assert stream_names

    for step in range(1, num_steps + 1):
        delete_path = directory / "updates" / f"delete_step_{step:05d}.ibin"
        deleted = binio.read_ids(delete_path) if delete_path.exists() else np.empty(0, np.int64)
        assert len(deleted) == schedule["deletes_per_step"][step - 1]
        for identifier in deleted.tolist():
            assert identifier in alive, f"step {step}: id {identifier} deleted but not alive"
            assert identifier not in deleted_so_far
            deleted_so_far.add(identifier)
            del alive[identifier]
        for _ in range(schedule["inserts_per_step"][step - 1]):
            alive[num_base + inserted] = inserts[inserted]
            inserted += 1
        # Deletions equal insertions, so the alive count stays at the base size at every step.
        assert len(alive) == num_base, f"step {step}"

        ids = np.array(sorted(alive))
        vectors = np.array([alive[i] for i in ids])
        for name in stream_names:
            queries = read_all(directory / name / f"step_{step:05d}{suffix}")
            groups = binio.read_ids(directory / name / f"step_{step:05d}.group_id.ibin")
            # Every executed query is a member of the group it claims to belong to.
            for query, group in zip(queries, groups):
                members = pool[group * group_size : (group + 1) * group_size]
                assert (pool_groups[group * group_size : (group + 1) * group_size] == group).all()
                assert (members == query).all(axis=1).any()
            if not stream_gt:
                # [groundtruth] stream_groundtruth = false: the queries exist, their answers do not.
                assert not (directory / "groundtruth" / name).exists(), name
                continue
            stored = binio.read_groundtruth(directory / "groundtruth" / name / f"step_{step:05d}.bin")
            expected_ids, expected_dists = reference_knn(vectors, ids, queries, stream_k)
            assert np.array_equal(stored.neighbour_ids.astype(np.int64), expected_ids), (name, step)
            assert np.array_equal(stored.neighbour_dists.astype(np.float64), expected_dists)
        if step in schedule["checkpoints"]:
            check_eval(step)

    # Eval GT files exist exactly for the checkpoints.
    eval_files = sorted(path.name for path in (directory / "groundtruth" / "eval").glob("*.bin"))
    assert eval_files == [f"step_{s:05d}.bin" for s in schedule["checkpoints"]]
    update_set = info["update_set"]
    assert inserted == update_set["insertions"] == len(deleted_so_far) == update_set["deletions"]
    # Reads plus updates make up the requested total, and every stream holds exactly `reads`.
    assert update_set["reads"] + inserted == update_set["total_requests"]
    for name in stream_names:
        rows = sum(len(read_all(path)) for path in (directory / name).glob("step_*.u8bin"))
        assert rows == update_set["reads"], name


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_public_queries_workload_matches_the_independent_reference(dataset):
    base_set = build_base(dataset)
    directory = build_set(workload_config(base_set, dataset["query_path"]))

    verify_workload(directory, base_set, num_steps=NUM_STEPS, group_size=6)
    info = workload_manifest(directory)
    assert info["seed_source"] == {
        "mode": "public_queries", "num_original_queries": 40, "num_groups": 40, "pool_vectors": 240,
    }
    # The calibrated variants really have (about) the requested neighbour overlap.
    assert abs(info["groups"]["overall_mean_overlap"] - 0.7) < 0.12
    assert info["groups"]["alpha"] > 0
    # Checkpoints for 8 steps with checkpoint_every = 3: initial state, 3, 6 and the last step.
    assert info["schedule"]["checkpoints"] == [0, 3, 6, 8]
    # 25% of 480 requests = 120 updates = 120 insertions + 120 deletions, 15 of each per step.
    assert info["schedule"]["inserts_per_step"] == [15] * 8
    assert info["schedule"]["deletes_per_step"] == [15] * 8
    assert info["update_set"] == {
        "update_percent": 25.0, "total_requests": 480, "reads": 360, "insertions": 120, "deletions": 120,
    }
    assert info["set_name"] == "u25" and info["name"] == "wl"
    assert 0.0 <= info["streams"]["stream_zipf_0.99"]["exact_repeat_fraction"] <= 1.0


def test_manifest_diagnostics_match_the_stored_streams(dataset):
    base_set = build_base(dataset)
    # Several sets: the shorter ones use a prefix of the longest stream, and their
    # diagnostics must describe that prefix, not the whole shared stream.
    family = workload.build_workload(family_config(base_set, dataset["query_path"]))
    for set_name in ("u0", "u25", "u50"):
        check_diagnostics_of_one_set(family / POOL / set_name)


def check_diagnostics_of_one_set(directory):
    info = workload_manifest(directory)
    pool = read_all(directory / "pool" / "pool.u8bin")
    reads = info["update_set"]["reads"]
    for name, stream_info in info["streams"].items():
        # Rebuild the stream from the step files and count repeats independently.
        steps = sorted((directory / name).glob("step_*.u8bin"))
        rows = np.vstack([read_all(path) for path in steps])
        groups = np.concatenate(
            [binio.read_ids(path.with_name(path.name.replace(".u8bin", ".group_id.ibin"))) for path in steps]
        )
        assert len(rows) == reads
        seen, repeats, seen_groups, revisits = set(), 0, set(), 0
        for row, group in zip(rows, groups.tolist()):
            key = row.tobytes()
            repeats += key in seen
            revisits += group in seen_groups
            seen.add(key)
            seen_groups.add(group)
        # The manifest counts repeats of pool ROWS. Two different pool rows can hold the same
        # bytes after rounding, which makes the byte-level count the larger one; when all pool
        # rows are different bytes the two counts must agree exactly.
        assert stream_info["exact_repeat_fraction"] <= repeats / reads + 1e-12
        if len(np.unique(pool, axis=0)) == len(pool):
            assert stream_info["exact_repeat_fraction"] == pytest.approx(repeats / reads)
        assert stream_info["group_revisit_fraction"] == pytest.approx(revisits / reads)


def test_same_inputs_give_identical_outputs(dataset):
    first_base = build_base(dataset, "base_one")
    second_base = build_base(dataset, "base_two")
    first = build_set(workload_config(first_base, dataset["query_path"]))
    second = build_set(workload_config(second_base, dataset["query_path"]))
    files_first = workload_manifest(first)["files"]
    files_second = workload_manifest(second)["files"]
    assert files_first == files_second
    assert workload_manifest(first)["groups"] == workload_manifest(second)["groups"]


def test_workloads_sharing_a_base_set_see_the_same_inserted_vectors(dataset):
    base_set = build_base(dataset)
    many = build_set(workload_config(base_set, dataset["query_path"], name="many"))
    few = build_set(
        workload_config(
            base_set, dataset["query_path"], name="few",
            updates=config.UpdatesConfig(seed=22, update_percents=(10.0,)),
        ),
        "u10",
    )
    verify_workload(many, base_set, 8, 6)
    verify_workload(few, base_set, 8, 6)
    info_many, info_few = workload_manifest(many), workload_manifest(few)
    assert info_many["base_set"]["base_row_ids_sha256"] == info_few["base_set"]["base_row_ids_sha256"]
    # Both sets consume the base set's fixed insert order from its beginning; only the number
    # of insertions differs.
    assert info_few["schedule"]["inserts_per_step"] == [6] * 8  # 10% of 480 = 48 updates
    assert info_many["schedule"]["inserts_per_step"] == [15] * 8  # 25% of 480 = 120 updates


# ---------------------------------------------------------------------------
# several update percentages in one family
# ---------------------------------------------------------------------------


def family_config(base_set, query_path, percents=(0.0, 25.0, 50.0), **overrides):
    updates = config.UpdatesConfig(seed=21, update_percents=percents)
    return workload_config(base_set, query_path, updates=updates, **overrides)


def read_stream(set_directory, name):
    """All queries of one stream, step after step, as one array."""
    steps = sorted((set_directory / name).glob("step_*.u8bin"))
    return np.vstack([read_all(path) for path in steps])


def test_several_update_percentages_make_one_family_of_complete_sets(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(family_config(base_set, dataset["query_path"]))

    # The overlap level and the shared/ directory that a later run reuses (_shared is gone).
    assert sorted(path.name for path in family.iterdir() if path.is_dir()) == ["o70", "shared"]
    assert sorted(path.name for path in (family / POOL).iterdir()) == ["u0", "u25", "u50"]
    assert (family / "family.json").exists() and (family / "build_report.json").exists()
    assert not (family / "_shared").exists() and not (family / "_universe").exists()
    assert not list(family.parent.glob("*.building"))
    expected = {  # percent -> (reads, insertions, deletions) for 480 requests
        "u0": (480, 0, 0), "u25": (360, 120, 120), "u50": (240, 240, 240),
    }
    for set_name, (reads, insertions, deletions) in expected.items():
        directory = family / POOL / set_name
        verify_workload(directory, base_set, NUM_STEPS, group_size=6)
        info = workload_manifest(directory)
        assert info["set_name"] == set_name
        assert (info["update_set"]["reads"], info["update_set"]["insertions"],
                info["update_set"]["deletions"]) == (reads, insertions, deletions)
        assert sum(info["schedule"]["inserts_per_step"]) == insertions
        assert sum(info["schedule"]["deletes_per_step"]) == deletions


def test_a_read_only_set_has_no_updates_and_a_single_evaluation_checkpoint(dataset):
    base_set = build_base(dataset)
    directory = build_set(family_config(base_set, dataset["query_path"]), "u0")

    # No delete files are written, because nothing is deleted.
    assert not (directory / "updates").exists() or not list((directory / "updates").iterdir())
    eval_files = sorted(path.name for path in (directory / "groundtruth" / "eval").glob("*.bin"))
    assert eval_files == ["step_00000.bin"]  # nothing changes, so later checkpoints would repeat it
    assert workload_manifest(directory)["schedule"]["checkpoints"] == [0]
    # The stream queries still get their own ground truth in every step.
    for step in range(1, NUM_STEPS + 1):
        assert (directory / "groundtruth" / "stream_zipf_0.5" / f"step_{step:05d}.bin").exists()


def test_sets_of_a_family_share_the_pool_the_alpha_and_the_stream_prefix(dataset, monkeypatch):
    base_set = build_base(dataset)

    calibrations = []
    original_find_alpha = n2.find_alpha

    def counting_find_alpha(*args, **kwargs):
        calibrations.append(1)
        return original_find_alpha(*args, **kwargs)

    monkeypatch.setattr(n2, "find_alpha", counting_find_alpha)
    family = workload.build_workload(family_config(base_set, dataset["query_path"]))
    assert len(calibrations) == 1  # alpha is searched once for the whole family

    sets = {name: family / POOL / name for name in ("u0", "u25", "u50")}
    infos = {name: workload_manifest(directory) for name, directory in sets.items()}
    shared_files = ["pool/pool.u8bin", "pool/pool_group_id.ibin", "seeds.u8bin", "eval_query.u8bin"]
    for relative in shared_files:
        assert len({infos[name]["files"][relative]["sha256"] for name in sets}) == 1, relative
    assert len({repr(infos[name]["groups"]) for name in sets}) == 1  # same alpha and statistics

    # Every set uses the first `reads` requests of the same Zipf stream: u50 is a prefix of
    # u25, which is a prefix of u0 (compared on the stored query files).
    for stream_name in infos["u0"]["streams"]:
        full = read_stream(sets["u0"], stream_name)
        middle = read_stream(sets["u25"], stream_name)
        short = read_stream(sets["u50"], stream_name)
        assert (len(full), len(middle), len(short)) == (480, 360, 240)
        assert np.array_equal(middle, full[:360])
        assert np.array_equal(short, full[:240])


def test_sets_of_a_family_differ_in_their_deletions_and_ground_truth(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(family_config(base_set, dataset["query_path"]))
    first_deletions = {
        name: binio.read_ids(family / POOL / name / "updates" / "delete_step_00001.ibin").tolist()
        for name in ("u25", "u50")
    }
    # u25 deletes 15 vectors in step 1, u50 deletes 30; each set draws its own victims.
    assert (len(first_deletions["u25"]), len(first_deletions["u50"])) == (15, 30)


def test_a_failure_in_a_later_set_leaves_no_finished_family(dataset, monkeypatch):
    base_set = build_base(dataset)
    original_finish_set = workload._finish_set
    calls = []

    def failing_second_set(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return original_finish_set(*args, **kwargs)

    monkeypatch.setattr(workload, "_finish_set", failing_second_set)
    with pytest.raises(RuntimeError, match="disk full"):
        workload.build_workload(family_config(base_set, dataset["query_path"]))

    workloads = base_set.directory / "workloads"
    # No finished family; only the .building directory with the finished stages stays, so that
    # the same build can continue (see the resume tests below).
    assert [path.name for path in workloads.iterdir()] == ["wl.building"]
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# interrupting and continuing a build
# ---------------------------------------------------------------------------


def named(cfg, name):
    return dataclasses.replace(cfg, name=name)


def set_files(family):
    """{pool/set: {relative path: sha256}} of every set of a finished family."""
    result = {}
    # A set is a directory with a workload.json (o70/g6/u25, g1/u0, external/u0, ...).
    for manifest_path in sorted(family.rglob("workload.json")):
        set_dir = manifest_path.parent
        result[set_dir.relative_to(family).as_posix()] = {
            relative: entry["sha256"]
            for relative, entry in workload_manifest(set_dir)["files"].items()
        }
    return result


class StageCounter:
    """Counts the calls of the expensive parts of a build."""

    def __init__(self, monkeypatch):
        self.calls = {
            "find_alpha": 0, "make_pool": 0, "pool_statistics": 0, "scan_lists": [],
            "finish_set": 0, "basis_job": 0,
        }
        find_alpha, make_pool = n2.find_alpha, workload.make_pool
        pool_statistics, scan_lists = workload.pool_statistics, workload.scan_lists
        finish_set = workload._finish_set

        def counted(name, function):
            def wrapper(*args, **kwargs):
                self.calls[name] += 1
                return function(*args, **kwargs)
            return wrapper

        def counted_scan(queries, list_length, *args, **kwargs):
            self.calls["scan_lists"].append(args[6])
            return scan_lists(queries, list_length, *args, **kwargs)

        monkeypatch.setattr(n2, "find_alpha", counted("find_alpha", find_alpha))
        # One job per CPU worker; the tests run with a single worker, so this runs in-process.
        monkeypatch.setattr(n2, "run_basis_job", counted("basis_job", n2.run_basis_job))
        monkeypatch.setattr(workload, "make_pool", counted("make_pool", make_pool))
        monkeypatch.setattr(workload, "pool_statistics", counted("pool_statistics", pool_statistics))
        monkeypatch.setattr(workload, "scan_lists", counted_scan)
        monkeypatch.setattr(workload, "_finish_set", counted("finish_set", finish_set))


def interrupt(monkeypatch, target, after_calls, error=KeyboardInterrupt):
    """Make ``workload.<target>`` raise ``error`` on its call number ``after_calls + 1``."""
    original = getattr(workload, target)
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) > after_calls:
            raise error("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(workload, target, interrupted)
    return original


def test_a_build_interrupted_between_sets_continues_with_the_same_files(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = sizes_config(base_set, dataset["query_path"])  # pools g3, g6; sets u0, u25, u50
    clean = workload.build_workload(named(cfg, "clean"))

    # Ctrl-C while the third set is finished: the sets of u0 (two pools) are complete.
    original_finish = interrupt(monkeypatch, "_finish_set", after_calls=2)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(named(cfg, "resumed"))
    building = base_set.directory / "workloads" / "resumed.building"
    assert building.is_dir() and not (base_set.directory / "workloads" / "resumed").exists()

    monkeypatch.setattr(workload, "_finish_set", original_finish)
    counter = StageCounter(monkeypatch)
    family = workload.build_workload(named(cfg, "resumed"))

    # Nothing before the sets is computed again, and the two finished sets are kept.
    assert counter.calls["find_alpha"] == 0 and counter.calls["make_pool"] == 0
    assert counter.calls["pool_statistics"] == 0 and counter.calls["scan_lists"] == ["stream", "eval"]
    assert counter.calls["finish_set"] == 4  # 2 pools x (u25, u50)
    assert set_files(family) == set_files(clean)
    for pool in ("g3", "g6"):
        for set_name in ("u0", "u25", "u50"):
            verify_workload(family / "o70" / pool / set_name, base_set, NUM_STEPS, group_size=int(pool[1:]))
    # The files that the sets share are still ONE hard-linked file, also between a set that was
    # finished before the interruption and one that was built after it.
    for relative in ("pool/pool.u8bin", "seeds.u8bin", "groundtruth/eval/step_00000.bin"):
        assert len({inode(family / "o70" / "g3" / name / relative) for name in ("u0", "u25", "u50")}) == 1, relative
    report = manifest.read_json(family / "build_report.json")
    assert {"n2_bases", "calibration_o70", "pool_o70_g3", "pool_o70_g6", "eval_step0", "mix_u0"} <= set(report["resumed_from_stages"])
    # No trace of the build is left behind.
    assert not list(family.parent.glob("*.building"))
    assert not any((family / name).exists() for name in ("_shared", "_universe", "_resume"))


def test_a_build_interrupted_in_the_pool_stage_keeps_the_neighbours_and_finished_pools(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = sizes_config(base_set, dataset["query_path"])
    clean = workload.build_workload(named(cfg, "clean"))

    original_statistics = interrupt(monkeypatch, "pool_statistics", after_calls=1)  # dies in pool g6
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(named(cfg, "resumed"))
    monkeypatch.setattr(workload, "pool_statistics", original_statistics)

    counter = StageCounter(monkeypatch)
    family = workload.build_workload(named(cfg, "resumed"))
    assert counter.calls["find_alpha"] == 0  # alpha was kept
    assert counter.calls["basis_job"] == 0  # and so were the PCA bases of the seeds
    assert counter.calls["make_pool"] == 1 and counter.calls["pool_statistics"] == 1  # only g6
    assert set_files(family) == set_files(clean)
    assert (manifest.read_json(family / "o70" / "g3" / "u0" / "workload.json")["groups"]
            == manifest.read_json(clean / "o70" / "g3" / "u0" / "workload.json")["groups"])


def test_a_build_killed_in_the_neighbour_search_redoes_only_that_stage(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = family_config(base_set, dataset["query_path"])
    clean = workload.build_workload(named(cfg, "clean"))

    original_find_alpha = n2.find_alpha

    def dying_find_alpha(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(n2, "find_alpha", dying_find_alpha)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(named(cfg, "resumed"))
    monkeypatch.setattr(n2, "find_alpha", original_find_alpha)

    searched = []
    original_top = n2.BaseScanner.top

    def counting_top(self, queries, k):
        searched.append(k)
        return original_top(self, queries, k)

    monkeypatch.setattr(n2.BaseScanner, "top", counting_top)
    family = workload.build_workload(named(cfg, "resumed"))
    # The exact neighbours of the seeds (k = neighbours = 60) were finished before the build
    # died; the top-10 searches of the calibration and of the pool statistics still run.
    assert 60 not in searched and 10 in searched
    assert set_files(family) == set_files(clean)


def test_an_unfinished_build_of_another_configuration_is_refused_and_kept(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = family_config(base_set, dataset["query_path"])
    original = interrupt(monkeypatch, "_finish_set", after_calls=1)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(cfg)
    monkeypatch.setattr(workload, "_finish_set", original)
    building = base_set.directory / "workloads" / "wl.building"
    before = sorted(path.relative_to(building).as_posix() for path in building.rglob("*"))

    changed = dataclasses.replace(cfg, stream=dataclasses.replace(cfg.stream, seed=cfg.stream.seed + 1))
    with pytest.raises(workload.WorkloadError, match=r"does not match.*stream"):
        workload.build_workload(changed)
    # Nothing was deleted, and the original configuration can still continue.
    assert sorted(path.relative_to(building).as_posix() for path in building.rglob("*")) == before
    family = workload.build_workload(cfg)
    assert (family / "family.json").exists()


def test_other_query_files_than_the_unfinished_build_are_refused(dataset, monkeypatch, tmp_path):
    base_set = build_base(dataset)
    cfg = family_config(base_set, dataset["query_path"])
    original = interrupt(monkeypatch, "_finish_set", after_calls=1)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(cfg)
    monkeypatch.setattr(workload, "_finish_set", original)

    # The same path now holds different queries: the stored digests no longer match.
    other, _ = synthetic_rows(40, seed=99)
    binio.write_array(dataset["query_path"], other)
    with pytest.raises(workload.WorkloadError, match="other query files"):
        workload.build_workload(cfg)
    assert (base_set.directory / "workloads" / "wl.building").is_dir()


def test_a_failure_before_any_stage_finished_leaves_nothing_behind(dataset, monkeypatch):
    base_set = build_base(dataset)

    def no_pools(*args, **kwargs):
        raise RuntimeError("no memory")

    monkeypatch.setattr(workload, "prepare_neighbours_and_bases", no_pools)
    with pytest.raises(RuntimeError, match="no memory"):
        workload.build_workload(family_config(base_set, dataset["query_path"]))
    workloads = base_set.directory / "workloads"
    assert not workloads.exists() or not list(workloads.iterdir())


def test_a_resumed_build_with_stream_groundtruth_off_equals_a_clean_one(dataset, monkeypatch):
    base_set = build_base(dataset)
    cfg = sizes_config(base_set, dataset["query_path"], groundtruth=eval_only_groundtruth())
    clean = workload.build_workload(named(cfg, "clean"))
    original = interrupt(monkeypatch, "_finish_set", after_calls=3)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(named(cfg, "resumed"))
    monkeypatch.setattr(workload, "_finish_set", original)
    family = workload.build_workload(named(cfg, "resumed"))
    assert set_files(family) == set_files(clean)
    for pool in ("g3", "g6"):
        verify_workload(family / "o70" / pool / "u50", base_set, NUM_STEPS, group_size=int(pool[1:]))


# ---------------------------------------------------------------------------
# duplicate-free uniform stream and variants = none
# ---------------------------------------------------------------------------


def stream_config(**overrides):
    values = dict(
        total_requests=TOTAL_REQUESTS, num_steps=NUM_STEPS, zipf_exponents=(0.5, 0.99),
        member_selection="uniform", seed=5, uniform_unique=True,
    )
    values.update(overrides)
    return config.StreamConfig(**values)


def test_uniform_unique_stream_is_built_for_every_set_and_is_exact(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(
        family_config(base_set, dataset["query_path"], stream=stream_config(total_requests=240))
    )
    # 240 requests: u0 has 240 reads, u25 180, u50 120; the pool holds 40 groups x 6 = 240.
    for set_name, reads in (("u0", 240), ("u25", 180), ("u50", 120)):
        directory = family / POOL / set_name
        verify_workload(directory, base_set, NUM_STEPS, group_size=6)  # all stream_* dirs
        info = workload_manifest(directory)
        unique = info["streams"]["stream_uniform_unique"]
        assert unique["kind"] == "uniform_unique" and unique["exponent"] == 0.0
        # No vector is requested twice: no exact repeats, as many distinct vectors as reads.
        assert unique["exact_repeat_fraction"] == 0.0
        assert unique["distinct_vectors"] == reads
        assert info["streams"]["stream_zipf_0.5"]["kind"] == "zipf"


def test_uniform_unique_streams_of_a_family_are_prefixes_of_each_other(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(
        family_config(base_set, dataset["query_path"], stream=stream_config(total_requests=240))
    )
    full = read_stream(family / POOL / "u0", "stream_uniform_unique")
    middle = read_stream(family / POOL / "u25", "stream_uniform_unique")
    short = read_stream(family / POOL / "u50", "stream_uniform_unique")
    assert (len(full), len(middle), len(short)) == (240, 180, 120)
    assert np.array_equal(middle, full[:180]) and np.array_equal(short, full[:120])


def test_a_uniform_unique_stream_as_long_as_the_pool_uses_every_vector_once(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(
        family_config(
            base_set, dataset["query_path"], percents=(0.0,), stream=stream_config(total_requests=240)
        )
    )
    pool = read_all(family / POOL / "u0" / "pool" / "pool.u8bin")
    requested = read_stream(family / POOL / "u0", "stream_uniform_unique")
    assert sorted(row.tobytes() for row in requested) == sorted(row.tobytes() for row in pool)


def test_uniform_unique_longer_than_the_pool_fails_before_anything_is_written(dataset):
    base_set = build_base(dataset)
    # 480 reads for u0 but only 240 vectors in the pool (40 groups x 6).
    with pytest.raises(workload.WorkloadError, match="needs 480 different vectors"):
        workload.build_workload(family_config(base_set, dataset["query_path"], stream=stream_config()))
    workloads = base_set.directory / "workloads"
    assert not workloads.exists() or not list(workloads.iterdir())


def test_only_the_uniform_unique_stream_when_there_are_no_zipf_exponents(dataset):
    base_set = build_base(dataset)
    stream = stream_config(total_requests=240, zipf_exponents=())
    directory = build_set(family_config(base_set, dataset["query_path"], stream=stream), "u25")
    assert sorted(path.name for path in directory.glob("stream_*")) == ["stream_uniform_unique"]
    verify_workload(directory, base_set, NUM_STEPS, group_size=6)


def test_variants_none_uses_the_given_queries_as_they_are(dataset, monkeypatch):
    base_set = build_base(dataset)

    def no_calibration(*args, **kwargs):
        raise AssertionError("variants = none must not search a noise scale")

    monkeypatch.setattr(n2, "find_alpha", no_calibration)
    monkeypatch.setattr(n2, "prepare_neighbours_and_bases", no_calibration)
    monkeypatch.setattr(workload, "prepare_neighbours_and_bases", no_calibration)
    monkeypatch.setattr(workload, "calibrate_overlap", no_calibration)

    groups = config.GroupsConfig(
        group_sizes=(1,), include_original=False, neighbours=60, axes=4, residual=False,
        target_overlaps=(0.7,), calibration_queries=20, overlap_tolerance=0.05, variants="none",
    )
    # 40 given queries; u0 reads every one of them exactly once, u25 reads 30 of them.
    stream = stream_config(total_requests=40, num_steps=4, zipf_exponents=(0.5,))
    family = workload.build_workload(
        family_config(
            base_set, dataset["query_path"], percents=(0.0, 25.0), groups=groups, stream=stream
        )
    )
    for set_name in ("u0", "u25"):
        directory = family / "g1" / set_name
        verify_workload(directory, base_set, 4, group_size=1)
        pool = read_all(directory / "pool" / "pool.u8bin")
        assert np.array_equal(pool, dataset["queries"])  # the pool IS the query file
        assert binio.read_ids(directory / "pool" / "pool_group_id.ibin").tolist() == list(range(40))
        info = workload_manifest(directory)
        assert info["groups"] == {"variants": "none", "num_seeds": 40}

    one_pass = read_stream(family / "g1" / "u0", "stream_uniform_unique")
    assert sorted(row.tobytes() for row in one_pass) == sorted(
        row.tobytes() for row in dataset["queries"]
    )  # every given query exactly once, in a random order


def test_uniform_unique_works_with_unequal_external_groups(dataset):
    base_set = build_base(dataset)
    vectors, _ = synthetic_rows(30, seed=9, centres=synthetic_rows(10, 1)[1])
    group_ids = np.array([5] * 4 + [9] * 10 + [2] * 1 + [7] * 15)
    external_path, group_path = dataset["tmp"] / "external.u8bin", dataset["tmp"] / "groups.ibin"
    eval_path = dataset["tmp"] / "eval.u8bin"
    binio.write_array(external_path, vectors)
    binio.write_ids(group_path, group_ids)
    binio.write_array(eval_path, dataset["queries"][:15])

    seeds = config.SeedsConfig("external", external_path, group_path, None, seed=11)
    evaluation = config.EvaluationConfig("external", eval_path, checkpoint_every=4, eval_k=7)
    stream = stream_config(total_requests=30, num_steps=3, zipf_exponents=())
    directory = build_set(
        family_config(
            base_set, None, percents=(0.0,), mode="external", seeds=seeds, evaluation=evaluation,
            stream=stream,
        ),
        "u0",
    )
    pool = read_all(directory / "pool" / "pool.u8bin")
    pool_groups = binio.read_ids(directory / "pool" / "pool_group_id.ibin")
    starts = np.concatenate([[0], np.cumsum(np.bincount(pool_groups))])
    seen = []
    for step_path in sorted((directory / "stream_uniform_unique").glob("step_*.u8bin")):
        queries = read_all(step_path)
        groups = binio.read_ids(step_path.with_name(step_path.name.replace(".u8bin", ".group_id.ibin")))
        for query, group in zip(queries, groups):
            # The reported group really contains the query (groups have sizes 1, 4, 15, 10).
            assert (pool[starts[group] : starts[group + 1]] == query).all(axis=1).any()
            seen.append(query.tobytes())
    assert sorted(seen) == sorted(row.tobytes() for row in pool)  # all 30, once each


def test_include_original_puts_the_seed_into_every_group(dataset):
    base_set = build_base(dataset)
    groups = config.GroupsConfig(
        group_sizes=(4,), include_original=True, neighbours=60, axes=4, residual=False,
        target_overlaps=(0.7,), calibration_queries=20, overlap_tolerance=0.05,
    )
    directory = build_set(workload_config(base_set, dataset["query_path"], groups=groups))
    pool = read_all(directory / "pool" / "pool.u8bin")
    seeds = read_all(directory / "seeds.u8bin")
    assert np.array_equal(pool[0::4], seeds)  # member 0 of every group is the seed itself
    verify_workload(directory, base_set, 8, 4)


def test_num_groups_selects_a_subset_of_the_queries_but_evaluation_keeps_all(dataset):
    base_set = build_base(dataset)
    seeds = config.SeedsConfig("public_queries", dataset["query_path"], None, num_groups=10, seed=11)
    directory = build_set(workload_config(base_set, dataset["query_path"], seeds=seeds))
    assert len(read_all(directory / "seeds.u8bin")) == 10
    assert len(read_all(directory / "eval_query.u8bin")) == 40  # evaluation uses every query
    assert workload_manifest(directory)["seed_source"]["num_groups"] == 10


def test_base_rows_mode_uses_the_reserved_query_rows(dataset):
    base_set = build_base(dataset, num_query_rows=NUM_QUERY_ROWS)
    seeds = config.SeedsConfig("base_rows", None, None, None, seed=11)
    directory = build_set(workload_config(base_set, None, mode="base_rows", seeds=seeds))
    with base_set.open_query_rows() as reader:
        reserved = reader.read_range(0, reader.num_rows)
    assert np.array_equal(read_all(directory / "seeds.u8bin"), reserved)
    verify_workload(directory, base_set, 8, 6)
    # Reserved rows are not part of the base: no base vector equals a seed row id.
    assert not set(base_set_module.read_query_row_ids(base_set).tolist()) & set(
        base_set_module.read_base_row_ids(base_set).tolist()
    )


def test_base_rows_mode_without_reserved_rows_is_rejected(dataset):
    base_set = build_base(dataset)
    seeds = config.SeedsConfig("base_rows", None, None, None, seed=11)
    with pytest.raises(ValueError, match="num_query_rows"):
        build_set(workload_config(base_set, None, mode="base_rows", seeds=seeds))


def test_external_groups_are_used_as_given_with_unequal_sizes(dataset):
    base_set = build_base(dataset)
    rng = np.random.default_rng(3)
    vectors, _ = synthetic_rows(30, seed=9, centres=synthetic_rows(10, 1)[1])
    group_ids = np.array([5] * 4 + [9] * 10 + [2] * 1 + [7] * 15)
    shuffle = rng.permutation(30)
    external_path = dataset["tmp"] / "external.u8bin"
    group_path = dataset["tmp"] / "external_groups.ibin"
    binio.write_array(external_path, vectors[shuffle])
    binio.write_ids(group_path, group_ids[shuffle])
    eval_path = dataset["tmp"] / "eval.u8bin"
    binio.write_array(eval_path, dataset["queries"][:15])

    seeds = config.SeedsConfig("external", external_path, group_path, None, seed=11)
    evaluation = config.EvaluationConfig("external", eval_path, checkpoint_every=4, eval_k=7)
    directory = build_set(
        workload_config(base_set, None, mode="external", seeds=seeds, evaluation=evaluation)
    )
    pool = read_all(directory / "pool" / "pool.u8bin")
    pool_groups = binio.read_ids(directory / "pool" / "pool_group_id.ibin")
    assert len(pool) == 30
    # Group sizes 1, 4, 15, 10 in ascending id order (2, 5, 7, 9).
    assert np.bincount(pool_groups).tolist() == [1, 4, 15, 10]
    assert len(read_all(directory / "eval_query.u8bin")) == 15
    info = workload_manifest(directory)
    assert info["seed_source"]["mode"] == "external"

    # Each pool group must hold exactly the vectors that carried its label in the input
    # (compared as sets of rows: the order inside a group is not part of the contract).
    starts_of_groups = np.concatenate([[0], np.cumsum(np.bincount(pool_groups))])
    for position, label in enumerate(sorted(set(group_ids.tolist()))):
        in_pool = pool[starts_of_groups[position] : starts_of_groups[position + 1]]
        expected = vectors[group_ids == label]
        assert sorted(row.tobytes() for row in in_pool) == sorted(row.tobytes() for row in expected)

    # Every stream query is a member of its group, and every GT file is exact.
    stream_k = info["config"]["groundtruth"]["stream_k"]
    base = read_all(base_set.directory / "base.u8bin")
    for name in info["streams"]:
        for step_path in sorted((directory / name).glob("step_*.u8bin")):
            queries = read_all(step_path)
            groups = binio.read_ids(step_path.with_name(step_path.name.replace(".u8bin", ".group_id.ibin")))
            starts = np.concatenate([[0], np.cumsum(np.bincount(pool_groups))])
            for query, group in zip(queries, groups):
                assert (pool[starts[group] : starts[group + 1]] == query).all(axis=1).any()
    assert stream_k == 5 and len(base) == NUM_BASE


def test_infeasible_requests_fail_before_anything_is_written(dataset):
    base_set = build_base(dataset)

    def attempt(percents):
        updates = config.UpdatesConfig(seed=1, update_percents=percents)
        return workload.build_workload(workload_config(base_set, dataset["query_path"], updates=updates))

    # 70% of 480 requests = 336 insertions, but the base set holds only 300 insertable vectors.
    with pytest.raises(ValueError, match="set u70"):
        attempt((0.0, 70.0))
    # 0.1% of 480 requests = 0.48 updates: not a whole number.
    with pytest.raises(ValueError, match="not a whole number"):
        attempt((0.1,))
    # 98.75% of 480 requests = 474 updates, which leaves 6 reads for 8 steps.
    with pytest.raises(ValueError, match="at least one read"):
        attempt((98.75,))
    assert not (base_set.directory / "workloads").exists() or not list(
        (base_set.directory / "workloads").iterdir()
    )


def test_existing_workload_is_never_overwritten(dataset):
    base_set = build_base(dataset)
    build_set(workload_config(base_set, dataset["query_path"]))
    with pytest.raises(FileExistsError):
        build_set(workload_config(base_set, dataset["query_path"]))


def test_query_file_with_the_wrong_shape_is_rejected(dataset):
    base_set = build_base(dataset)
    wrong_dim = dataset["tmp"] / "wrong.u8bin"
    binio.write_array(wrong_dim, np.zeros((5, DIM + 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="query file .* has dimension"):
        build_set(workload_config(base_set, wrong_dim))
    wrong_type = dataset["tmp"] / "wrong.fbin"
    binio.write_array(wrong_type, np.zeros((5, DIM), dtype=np.float32))
    with pytest.raises(ValueError, match="query file .* has dtype"):
        build_set(workload_config(base_set, wrong_type))


def test_inner_product_metric_works(dataset):
    base_set = build_base(dataset, "base_ip", metric="ip")
    directory = build_set(workload_config(base_set, dataset["query_path"]))
    # Re-verify one stream/step against the reference with the inner-product metric.
    info = workload_manifest(directory)
    assert info["base_set"]["metric"] == "ip"
    assert "d1_ratio_quantiles_5_50_95" not in info["groups"]  # only defined for L2
    stored = binio.read_groundtruth(directory / "groundtruth" / "eval" / "step_00000.bin")
    base = read_all(base_set.directory / "base.u8bin")
    expected_ids, _ = reference_knn(base, np.arange(len(base)), read_all(directory / "eval_query.u8bin"), 10, "ip")
    assert np.array_equal(stored.neighbour_ids.astype(np.int64), expected_ids)


@pytest.mark.skipif(not gpu_available(), reason="no usable cupy/GPU")
def test_gpu_and_cpu_produce_identical_workloads(dataset):
    base_set = build_base(dataset)
    cpu = build_set(workload_config(base_set, dataset["query_path"], name="cpu_run"))
    gpu = build_set(workload_config(base_set, dataset["query_path"], name="gpu_run", device="gpu"))
    files_cpu = workload_manifest(cpu)["files"]
    files_gpu = workload_manifest(gpu)["files"]
    assert files_cpu == files_gpu


# ---------------------------------------------------------------------------
# group sizes as an axis: pools x update percentages
# ---------------------------------------------------------------------------


def sizes_config(base_set, query_path, sizes=(3, 6), percents=(0.0, 25.0, 50.0), **overrides):
    groups = config.GroupsConfig(
        group_sizes=tuple(sizes), include_original=False, neighbours=60, axes=4, residual=False,
        target_overlaps=(0.7,), calibration_queries=20, overlap_tolerance=0.05,
    )
    updates = config.UpdatesConfig(seed=21, update_percents=percents)
    return workload_config(base_set, query_path, groups=groups, updates=updates, **overrides)


def inode(path):
    return path.stat().st_ino


def test_every_pair_of_group_size_and_percentage_is_a_complete_verified_set(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(sizes_config(base_set, dataset["query_path"]))

    assert sorted(path.name for path in (family / "o70").iterdir()) == ["g3", "g6"]
    for size in (3, 6):
        pool_dir = family / "o70" / f"g{size}"
        assert sorted(path.name for path in pool_dir.iterdir()) == ["u0", "u25", "u50"]
        for set_name in ("u0", "u25", "u50"):
            verify_workload(pool_dir / set_name, base_set, NUM_STEPS, group_size=size)
            info = workload_manifest(pool_dir / set_name)
            assert info["pool"] == {
                "name": f"o70/g{size}", "kind": "n2", "group_size": size,
                "overlap": "o70", "target_overlap": 0.7, "vectors": 40 * size, "groups": 40,
            }
            groundtruth = info["groundtruth"]
            assert groundtruth["method"] == "universe_lists"
            # With the default list length no request needs the slow scan.
            assert groundtruth["stream_requests_answered_by_scan"] == 0
            assert groundtruth["eval_requests_answered_by_scan"] == 0


def eval_only_groundtruth(**overrides):
    """[groundtruth] with stream_groundtruth = false (same batch sizes as workload_config)."""
    return config.GroundTruthConfig(
        stream_k=5, device="cpu", base_batch_rows=97, query_chunk_rows=13,
        stream_groundtruth=False, **overrides,
    )


def test_without_stream_groundtruth_only_the_evaluation_queries_are_scored(dataset, monkeypatch):
    base_set = build_base(dataset)
    scans = []
    original_scan = workload.scan_lists

    def recording_scan(queries, list_length, *args, **kwargs):
        scans.append((args[6], len(queries)))  # (name, number of query vectors)
        return original_scan(queries, list_length, *args, **kwargs)

    monkeypatch.setattr(workload, "scan_lists", recording_scan)
    full = workload.build_workload(family_config(base_set, dataset["query_path"]))
    assert [name for name, _ in scans] == ["stream", "eval"]
    scans.clear()
    light = workload.build_workload(
        family_config(
            base_set, dataset["query_path"], name="wl_eval_only", groundtruth=eval_only_groundtruth()
        )
    )
    # The universe is scanned for the evaluation queries only (the 40 original queries).
    assert scans == [("eval", 40)]

    for set_name in ("u0", "u25", "u50"):
        full_set, light_set = full / POOL / set_name, light / POOL / set_name
        # Evaluation answers are checked against the float64 reference; stream answers are absent.
        verify_workload(light_set, base_set, NUM_STEPS, group_size=6)
        assert sorted(path.name for path in (light_set / "groundtruth").iterdir()) == ["eval"]

        # Everything else is byte for byte what the full build writes: the queries, group ids,
        # deleted ids, pool, seeds and the evaluation ground truth.
        full_files = workload_manifest(full_set)["files"]
        light_files = workload_manifest(light_set)["files"]
        assert set(light_files) == {
            relative for relative in full_files if not relative.startswith("groundtruth/stream_")
        }
        for relative, entry in light_files.items():
            assert entry["sha256"] == full_files[relative]["sha256"], relative

        info = workload_manifest(light_set)["groundtruth"]
        assert info["stream_groundtruth"] is False
        assert info["stream_list_length"] is None and info["distinct_query_vectors"] is None
        assert info["pool_rows_with_own_list"] is None and info["stream_requests_answered_by_scan"] is None
        assert info["eval_list_length"] == workload_manifest(full_set)["groundtruth"]["eval_list_length"]
        assert workload_manifest(full_set)["groundtruth"]["stream_groundtruth"] is True


def test_stream_groundtruth_off_works_for_several_group_sizes_and_the_unique_stream(dataset):
    base_set = build_base(dataset)
    stream = config.StreamConfig(
        total_requests=TOTAL_REQUESTS, num_steps=NUM_STEPS, zipf_exponents=(0.5,),
        member_selection="uniform", seed=5, uniform_unique=True,
    )
    family = workload.build_workload(
        sizes_config(
            base_set, dataset["query_path"], sizes=(12, 16), stream=stream,
            groundtruth=eval_only_groundtruth(),
        )
    )
    for size in (12, 16):  # 40 groups x 12 = 480 vectors: just enough for the 480 distinct reads
        for set_name in ("u0", "u25", "u50"):
            directory = family / "o70" / f"g{size}" / set_name
            verify_workload(directory, base_set, NUM_STEPS, group_size=size)
            assert (directory / "stream_uniform_unique").is_dir()
            assert not list(directory.glob("groundtruth/stream_*"))
    assert not (family / "_universe").exists() and not list(family.parent.glob("*.building"))


def test_family_json_indexes_every_pool_and_set(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(sizes_config(base_set, dataset["query_path"]))
    index = manifest.read_json(family / "family.json")

    assert [pool["name"] for pool in index["pools"]] == ["o70/g3", "o70/g6"]
    assert [(entry["path"]) for entry in index["sets"]] == [
        f"o70/g{size}/u{percent}" for size in (3, 6) for percent in (0, 25, 50)
    ]
    for entry in index["sets"]:
        directory = family / entry["path"]
        assert directory.is_dir()
        assert entry["workload_json_sha256"] == manifest.sha256_file(directory / "workload.json")
        info = workload_manifest(directory)
        assert (entry["reads"], entry["insertions"]) == (
            info["update_set"]["reads"], info["update_set"]["insertions"],
        )
    # One entry per overlap level: the noise scale that was found for it.
    assert [entry["name"] for entry in index["overlaps"]] == ["o70"]
    assert index["overlaps"][0]["target_overlap"] == 0.7 and index["overlaps"][0]["alpha"] > 0
    assert abs(index["overlaps"][0]["achieved_overlap"] - 0.7) <= index["overlaps"][0]["tolerance"]

    report = manifest.read_json(family / "build_report.json")
    assert set(report["stage_seconds"]) >= {
        "plan", "resources", "pools", "universe_lists", "sets", "manifest",
    }
    assert all(seconds >= 0 for seconds in report["stage_seconds"].values())
    assert report["parallel"]["cpu_workers"] == 1 and report["parallel"]["device"] == "cpu"


def test_files_that_are_the_same_in_several_sets_are_one_hard_linked_file(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(sizes_config(base_set, dataset["query_path"]))
    sets = {(size, name): family / "o70" / f"g{size}" / name for size in (3, 6) for name in ("u0", "u25", "u50")}

    # Seeds and evaluation queries: identical in all six sets.
    for relative in ("seeds.u8bin", "eval_query.u8bin"):
        assert len({inode(directory / relative) for directory in sets.values()}) == 1
    # The pool: identical across the percentages of one group size, different between sizes.
    for size in (3, 6):
        pool_files = {inode(sets[(size, name)] / "pool" / "pool.u8bin") for name in ("u0", "u25", "u50")}
        assert len(pool_files) == 1
    assert inode(sets[(3, "u0")] / "pool" / "pool.u8bin") != inode(sets[(6, "u0")] / "pool" / "pool.u8bin")
    # Deleted ids and evaluation ground truth: identical across the pools of one percentage.
    for name in ("u25", "u50"):
        for relative in ("updates/delete_step_00001.ibin", "groundtruth/eval/step_00000.bin"):
            low, high = sets[(3, name)] / relative, sets[(6, name)] / relative
            assert inode(low) == inode(high) and low.read_bytes() == high.read_bytes()
    # Stream ground truth belongs to one pool and is never shared.
    assert inode(sets[(3, "u25")] / "groundtruth/stream_zipf_0.5/step_00001.bin") != inode(
        sets[(6, "u25")] / "groundtruth/stream_zipf_0.5/step_00001.bin"
    )


def test_alpha_is_searched_once_for_all_group_sizes(dataset, monkeypatch):
    base_set = build_base(dataset)
    calls = []
    original = n2.find_alpha
    monkeypatch.setattr(n2, "find_alpha", lambda *a, **k: calls.append(1) or original(*a, **k))
    family = workload.build_workload(sizes_config(base_set, dataset["query_path"]))

    assert len(calls) == 1
    alphas = {workload_manifest(family / "o70" / f"g{size}" / "u25")["groups"]["alpha"] for size in (3, 6)}
    assert len(alphas) == 1


def test_a_smaller_pool_borrows_the_lists_of_the_larger_one(dataset):
    base_set = build_base(dataset)
    family = workload.build_workload(sizes_config(base_set, dataset["query_path"]))
    small = workload_manifest(family / "o70" / "g3" / "u25")["groundtruth"]
    large = workload_manifest(family / "o70" / "g6" / "u25")["groundtruth"]
    # The first three members of a group are the same vectors in both pools.
    assert small["pool_rows_sharing_a_list"] > 0
    assert small["pool_rows_with_own_list"] + small["pool_rows_sharing_a_list"] > 0
    assert large["pool_rows_sharing_a_list"] > 0 and large["pool_rows_with_own_list"] == 0
    assert small["distinct_query_vectors"] == large["distinct_query_vectors"]  # one scan for both


def test_every_pool_of_a_family_matches_a_build_of_that_size_alone(dataset):
    base_set = build_base(dataset)
    both = workload.build_workload(sizes_config(base_set, dataset["query_path"], percents=(25.0,)))
    alone = workload.build_workload(
        sizes_config(base_set, dataset["query_path"], sizes=(3,), percents=(25.0,), name="alone")
    )
    together_files = workload_manifest(both / "o70" / "g3" / "u25")["files"]
    alone_files = workload_manifest(alone / "o70" / "g3" / "u25")["files"]
    assert together_files == alone_files


def test_uniform_unique_names_the_pool_that_is_too_small(dataset):
    base_set = build_base(dataset)
    stream = stream_config(total_requests=240)  # 240 distinct requests per read-only set
    with pytest.raises(workload.WorkloadError, match=r"g3: 120 vectors"):
        workload.build_workload(
            sizes_config(base_set, dataset["query_path"], percents=(0.0,), stream=stream)
        )
    assert not (base_set.directory / "workloads").exists() or not list(
        (base_set.directory / "workloads").iterdir()
    )
    # The small pool is checked wherever it stands in the list, not only when it is first.
    with pytest.raises(workload.WorkloadError, match=r"g3: 120 vectors"):
        workload.build_workload(
            sizes_config(base_set, dataset["query_path"], sizes=(6, 3), percents=(0.0,), stream=stream)
        )


# ---------------------------------------------------------------------------
# the scan of the active set (fallback), links, workers and resources
# ---------------------------------------------------------------------------


def test_lists_that_are_too_short_fall_back_to_scanning_and_stay_exact(dataset, monkeypatch):
    base_set = build_base(dataset)
    # Lists of exactly k entries: with 60% updates only about a third of the universe is
    # alive, so most lists lose entries and the fallback has to answer.
    monkeypatch.setattr(workload, "choose_list_length", lambda needed, alive, size, *a, **k: needed)
    family = workload.build_workload(
        sizes_config(base_set, dataset["query_path"], sizes=(6,), percents=(0.0, 60.0))
    )
    for set_name in ("u0", "u60"):
        directory = family / "o70" / "g6" / set_name
        verify_workload(directory, base_set, NUM_STEPS, group_size=6)
        groundtruth = workload_manifest(directory)["groundtruth"]
        assert groundtruth["stream_requests_answered_by_scan"] > 0, set_name
        assert groundtruth["eval_requests_answered_by_scan"] > 0, set_name


def test_without_hard_links_the_sets_are_copied_and_still_correct(dataset, monkeypatch):
    base_set = build_base(dataset)

    def refuse(*args, **kwargs):
        raise OSError("file system without hard links")

    monkeypatch.setattr(layout.os, "link", refuse)
    family = workload.build_workload(
        sizes_config(base_set, dataset["query_path"], sizes=(6,), percents=(0.0, 25.0))
    )
    low, high = family / "o70" / "g6" / "u0", family / "o70" / "g6" / "u25"
    assert inode(low / "pool" / "pool.u8bin") != inode(high / "pool" / "pool.u8bin")
    assert (low / "pool" / "pool.u8bin").read_bytes() == (high / "pool" / "pool.u8bin").read_bytes()
    verify_workload(low, base_set, NUM_STEPS, group_size=6)
    verify_workload(high, base_set, NUM_STEPS, group_size=6)


def test_a_copied_set_directory_is_independent_of_the_family(dataset, tmp_path):
    import shutil

    base_set = build_base(dataset)
    family = workload.build_workload(
        sizes_config(base_set, dataset["query_path"], sizes=(6,), percents=(0.0, 25.0))
    )
    copy = tmp_path / "moved_u25"
    shutil.copytree(family / "o70" / "g6" / "u25", copy)
    shutil.rmtree(family)  # the original family is gone
    assert read_all(copy / "pool" / "pool.u8bin").shape == (240, DIM)
    info = manifest.read_json(copy / "workload.json")
    for relative, entry in info["files"].items():
        assert manifest.sha256_file(copy / relative) == entry["sha256"], relative


def test_the_number_of_cpu_workers_never_changes_the_files(dataset):
    base_set = build_base(dataset)
    one = workload.build_workload(sizes_config(base_set, dataset["query_path"], name="one"))
    many = workload.build_workload(
        sizes_config(
            base_set, dataset["query_path"], name="many", parallel=cpu_parallel(workers=3),
        )
    )
    for size in (3, 6):
        for set_name in ("u0", "u25", "u50"):
            first = workload_manifest(one / "o70" / f"g{size}" / set_name)["files"]
            second = workload_manifest(many / "o70" / f"g{size}" / set_name)["files"]
            assert first == second, (size, set_name)
    assert manifest.read_json(many / "build_report.json")["parallel"]["cpu_workers"] == 3


def test_a_machine_without_enough_memory_stops_before_any_work(dataset, monkeypatch):
    base_set = build_base(dataset)
    monkeypatch.setattr(parallel, "host_available_bytes", lambda *a, **k: 1000)
    with pytest.raises(parallel.ResourceError, match="host memory"):
        workload.build_workload(sizes_config(base_set, dataset["query_path"]))
    assert not (base_set.directory / "workloads").exists() or not list(
        (base_set.directory / "workloads").iterdir()
    )


def test_a_numa_node_that_does_not_exist_is_an_error(dataset):
    base_set = build_base(dataset)
    missing = config.ParallelConfig(gpus="none", cpu_workers=1, numa_node=99)
    with pytest.raises(parallel.ResourceError, match="NUMA node 99"):
        workload.build_workload(sizes_config(base_set, dataset["query_path"], parallel=missing))


def test_the_resolved_batch_sizes_are_recorded(dataset):
    base_set = build_base(dataset)
    auto = config.GroundTruthConfig(stream_k=5, device="cpu", base_batch_rows=None, query_chunk_rows=None)
    family = workload.build_workload(
        sizes_config(base_set, dataset["query_path"], sizes=(6,), percents=(25.0,), groundtruth=auto)
    )
    groundtruth = workload_manifest(family / "o70" / "g6" / "u25")["groundtruth"]
    assert (groundtruth["base_batch_rows"], groundtruth["query_chunk_rows"]) == (100_000, 1024)


# ---------------------------------------------------------------------------
# float data: not bit-exact, but the same neighbours up to rounding
# ---------------------------------------------------------------------------


def test_float32_data_gives_the_same_neighbours_up_to_rounding(tmp_path):
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(1400, DIM)).astype(np.float32)
    queries = rng.normal(size=(40, DIM)).astype(np.float32)
    raw_path, query_path = tmp_path / "raw.fbin", tmp_path / "query.fbin"
    binio.write_array(raw_path, raw)
    binio.write_array(query_path, queries)
    base_set = base_set_module.build_base_set(
        config.BaseSetConfig(
            raw_file=raw_path, metric="l2", seed=7, num_base=NUM_BASE, base_selection="random",
            num_query_rows=0, max_inserts=MAX_INSERTS, output_dir=tmp_path / "out", name="base_f",
        )
    )
    family = workload.build_workload(
        sizes_config(base_set, query_path, sizes=(6,), percents=(25.0,))
    )
    directory = family / "o70" / "g6" / "u25"
    info = workload_manifest(directory)
    base = read_float(base_set.directory / "base.fbin")
    inserts = read_float(base_set.directory / "insert_pool.fbin")

    alive = {i: base[i] for i in range(len(base))}
    inserted = 0
    for step in range(1, NUM_STEPS + 1):
        delete_path = directory / "updates" / f"delete_step_{step:05d}.ibin"
        for identifier in (binio.read_ids(delete_path).tolist() if delete_path.exists() else []):
            del alive[identifier]
        for _ in range(info["schedule"]["inserts_per_step"][step - 1]):
            alive[len(base) + inserted] = inserts[inserted]
            inserted += 1
        ids = np.array(sorted(alive))
        vectors = np.array([alive[i] for i in ids])
        for name in ("stream_zipf_0.5", "stream_zipf_0.99"):
            step_queries = read_float(directory / name / f"step_{step:05d}.fbin")
            stored = binio.read_groundtruth(directory / "groundtruth" / name / f"step_{step:05d}.bin")
            expected_ids, expected_dists = reference_knn(vectors, ids, step_queries, 5)
            assert_same_neighbours_up_to_rounding(stored, expected_ids, expected_dists)


def read_float(path):
    return read_all(path).astype(np.float64)


def assert_same_neighbours_up_to_rounding(stored, expected_ids, expected_dists):
    """Equal neighbour ids, except where two distances are equal up to float32 rounding."""
    stored_ids = stored.neighbour_ids.astype(np.int64)
    different = np.flatnonzero((stored_ids != expected_ids).any(axis=1))
    # Random normal data has no exact ties, so a different list needs a near-tie.
    assert len(different) <= 0.02 * len(stored_ids)
    for row in different:
        assert np.allclose(stored.neighbour_dists[row].astype(np.float64), expected_dists[row], rtol=1e-4)
    close = np.isclose(stored.neighbour_dists.astype(np.float64), expected_dists, rtol=1e-4)
    assert close.all()


@pytest.mark.skipif(not gpu_available(), reason="no usable cupy/GPU")
@pytest.mark.parametrize("workers_per_gpu", [1, 2])
def test_gpu_workers_give_the_same_files_as_the_cpu(dataset, workers_per_gpu):
    from gpu_helpers import gpu_plan_or_skip

    gpu_plan_or_skip((1,), workers_per_gpu)  # skips when GPU 1 is not free
    base_set = build_base(dataset)
    cpu = workload.build_workload(sizes_config(base_set, dataset["query_path"], name="cpu_run"))
    parallel_config = config.ParallelConfig(
        gpus=(1,), workers_per_gpu=workers_per_gpu, cpu_workers=2, numa_node="none"
    )
    gpu = workload.build_workload(
        sizes_config(
            base_set, dataset["query_path"], name="gpu_run", device="gpu", parallel=parallel_config
        )
    )
    for size in (3, 6):
        for set_name in ("u0", "u25", "u50"):
            assert (
                workload_manifest(cpu / "o70" / f"g{size}" / set_name)["files"]
                == workload_manifest(gpu / "o70" / f"g{size}" / set_name)["files"]
            ), (size, set_name)
    report = manifest.read_json(gpu / "build_report.json")
    assert report["parallel"]["gpus"] == [1] and report["parallel"]["workers_per_gpu"] == workers_per_gpu
