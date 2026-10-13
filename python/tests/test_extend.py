"""Adding pools to a finished family (``[workload] existing = extend``).

The properties that matter, each checked on the output files:

* a family that was extended is, file for file, the family that one build with all pools makes;
* pools that exist are skipped and never touched (inode, size and modification time unchanged);
* anything the pools have in common that changed is refused, and the family stays as it was;
* the work of an interrupted extension is not lost and the family is never left half-updated.
"""

import copy
import dataclasses
import os
import shutil
import subprocess
import sys

import pytest

from semantic_vdb import binio, config, extend, family, manifest, n2, stream_gt, workload
from test_end_to_end import (
    NUM_STEPS,
    build_base,
    dataset,  # noqa: F401  (pytest fixture)
    eval_only_groundtruth,
    inode,
    interrupt,
    named,
    set_files,
    synthetic_rows,
    verify_workload,
    workload_config,
    workload_manifest,
)

def make_config(
    base_set, query_path, overlaps=(0.7,), sizes=(3,), percents=(0.0, 25.0), name="fam",
    existing="error", **more,
):
    """A small family configuration (stream ground truth on unless ``groundtruth`` is given)."""
    groups = config.GroupsConfig(
        group_sizes=tuple(sizes),
        include_original=False,
        neighbours=60,
        axes=4,
        residual=False,
        target_overlaps=tuple(overlaps),
        calibration_queries=20,
        overlap_tolerance=0.05,
    )
    updates = config.UpdatesConfig(seed=21, update_percents=tuple(percents))
    return workload_config(
        base_set, query_path, name=name, groups=groups, updates=updates, existing=existing,
        **more,
    )


def with_pools(cfg, overlaps=None, sizes=None):
    """The same configuration with other pool axes, asking to extend an existing family."""
    groups = dataclasses.replace(
        cfg.groups,
        target_overlaps=tuple(overlaps) if overlaps is not None else cfg.groups.target_overlaps,
        group_sizes=tuple(sizes) if sizes is not None else cfg.groups.group_sizes,
    )
    return dataclasses.replace(cfg, groups=groups, existing="extend")


def one_build(base_set, query_path, overlaps, sizes=(3,), **more):
    """The family that ONE build with all these pools makes (the reference for an extension)."""
    cfg = make_config(base_set, query_path, overlaps=overlaps, sizes=sizes, **more)
    return workload.build_workload(named(cfg, "once"))


def tree_state(directory):
    """{relative path: (size, mtime, inode)} of every file below ``directory``."""
    state = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            info = path.stat()
            relative = path.relative_to(directory).as_posix()
            state[relative] = (info.st_size, info.st_mtime_ns, info.st_ino)
    return state


def normalized_manifest(directory, ignore=()):
    """workload.json without what legitimately differs between two families of equal content."""
    info = copy.deepcopy(workload_manifest(directory))
    info["name"] = info["config"]["name"] = None
    info["base_set"]["directory"] = info["config"]["base_set_dir"] = None
    for key in ignore:
        info["groundtruth"].pop(key, None)
    return info


def same_family(left, right, ignore=()):
    """Every set has the same files, the same workload.json and the same family.json entry."""
    assert set_files(left) == set_files(right)
    for manifest_path in sorted(left.rglob("workload.json")):
        relative = manifest_path.parent.relative_to(left)
        assert normalized_manifest(left / relative, ignore) == normalized_manifest(
            right / relative, ignore
        ), relative

    indexes = [manifest.read_json(directory / "family.json") for directory in (left, right)]
    for index in indexes:
        index["name"] = None
        for entry in index["sets"]:
            # The hash of workload.json differs with the family name inside it; the content
            # itself was compared above.
            entry["workload_json_sha256"] = None
    assert indexes[0] == indexes[1]  # pools, overlaps, sets and the common fingerprint
    caches = [
        manifest.read_json(directory / "shared" / "cache.json") for directory in (left, right)
    ]
    assert caches[0] == caches[1]


# ---------------------------------------------------------------------------
# an extended family is the family of one build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "then",
    [
        dict(overlaps=(0.7, 0.5)),  # a new overlap
        dict(sizes=(3, 6)),  # a new group size
        dict(overlaps=(0.7, 0.5), sizes=(3, 6)),  # both at once
        dict(overlaps=(0.5, 0.7)),  # a new overlap in front of the old one
    ],
)
def test_an_extended_family_equals_the_family_of_one_build_with_all_pools(dataset, then):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = make_config(base_set, query_path, name="grown")
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, **then))

    reference = with_pools(make_config(base_set, query_path), **then)
    once = workload.build_workload(named(reference, "once"))
    same_family(grown, once)


def test_the_new_sets_of_an_extension_are_exact(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], percents=(0.0, 25.0, 50.0))
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5), sizes=(3, 6)))
    # Ground truth of every new set against the float64 reference, from the files alone: the
    # evaluation answers are the ones linked from the family, the stream answers are new.
    for pool, size in (("o70/g6", 6), ("o50/g3", 3), ("o50/g6", 6)):
        for set_name in ("u0", "u25", "u50"):
            verify_workload(grown / pool / set_name, base_set, NUM_STEPS, group_size=size)


def test_without_stream_ground_truth_the_extension_equals_one_build_exactly(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    without = dict(groundtruth=eval_only_groundtruth())
    start = make_config(base_set, query_path, name="grown", **without)
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    once = one_build(base_set, query_path, overlaps=(0.7, 0.5), **without)
    same_family(grown, once)  # nothing is ignored: workload.json is equal as a whole


def test_an_extension_can_be_followed_by_another_one(dataset):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = make_config(base_set, query_path, name="grown")
    workload.build_workload(start)
    second = with_pools(start, overlaps=(0.7, 0.5))
    workload.build_workload(second)
    grown = workload.build_workload(with_pools(second, overlaps=(0.7, 0.5, 0.9), sizes=(3, 6)))

    once = one_build(base_set, query_path, overlaps=(0.7, 0.5, 0.9), sizes=(3, 6))
    same_family(grown, once)
    # One report per build or extension (the stream answering stage adds its own reports).
    assert sorted(
        path.name for path in grown.glob("build_report*.json") if "stream_groundtruth" not in path.name
    ) == [
        "build_report.json",
        "build_report_extension_1.json",
        "build_report_extension_2.json",
    ]


def test_scan_counts_of_the_evaluation_answers_come_from_the_family(dataset, monkeypatch):
    """With lists too short for the answers, the scan counts must survive an extension."""
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    short = workload.choose_list_length

    # k + 1 entries: with a fifth of the universe dead most requests need the slow fallback.
    monkeypatch.setattr(workload, "choose_list_length", lambda k, *args: k + 1)
    start = make_config(base_set, query_path, name="grown", percents=(0.0, 50.0))
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    once = one_build(base_set, query_path, overlaps=(0.7, 0.5), percents=(0.0, 50.0))
    monkeypatch.setattr(workload, "choose_list_length", short)

    counts = [
        workload_manifest(grown / "o50" / "g3" / name)["groundtruth"][
            "eval_requests_answered_by_scan"
        ]
        for name in ("u0", "u50")
    ]
    assert counts[1] > 0  # the fallback really was needed
    same_family(grown, once)


# ---------------------------------------------------------------------------
# skipping
# ---------------------------------------------------------------------------


def test_running_again_with_the_same_pools_changes_nothing(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7, 0.5))
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)

    messages = []
    extending = dataclasses.replace(start, existing="extend")
    again = workload.build_workload(extending, log=messages.append)

    assert again == family_dir
    # Not one file touched, not even its modification time.
    assert tree_state(family_dir) == before
    assert any("nothing to add" in message for message in messages)
    assert not list(family_dir.parent.glob("fam.extending*"))


def test_the_pools_that_exist_are_never_touched_by_an_extension(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7,), sizes=(3, 6))
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir / "o70")

    messages = []
    workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)), log=messages.append)

    assert tree_state(family_dir / "o70") == before
    assert [m for m in messages if "exists: skipped" in m] == [
        "[extend] the pool o70/g3 exists: skipped",
        "[extend] the pool o70/g6 exists: skipped",
    ]
    assert not list(family_dir.parent.glob("fam.extending*"))


def test_pools_that_the_configuration_no_longer_lists_stay_in_the_family(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7,))
    family_dir = workload.build_workload(start)
    old_files = set_files(family_dir)

    # The configuration now names only the NEW overlap: o70 is neither rebuilt nor removed.
    grown = workload.build_workload(with_pools(start, overlaps=(0.5,)))

    index = manifest.read_json(grown / "family.json")
    assert {pool["name"] for pool in index["pools"]} == {"o70/g3", "o50/g3"}
    # Entries the configuration does not mention keep their place behind the ones it does.
    assert [entry["name"] for entry in index["overlaps"]] == ["o50", "o70"]
    kept = {key: value for key, value in set_files(grown).items() if key.startswith("o70")}
    assert kept == old_files


def test_the_default_is_to_refuse_an_existing_family(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)

    wider = dataclasses.replace(start.groups, target_overlaps=(0.7, 0.5))
    with pytest.raises(FileExistsError, match="existing = extend"):
        workload.build_workload(dataclasses.replace(start, groups=wider))
    assert tree_state(family_dir) == before


def test_extend_on_a_family_that_does_not_exist_just_builds_it(dataset):
    base_set = build_base(dataset)
    cfg = make_config(base_set, dataset["query_path"], existing="extend")
    family_dir = workload.build_workload(cfg)
    assert (family_dir / "family.json").is_file()
    assert not list(family_dir.parent.glob("*.extending*"))


# ---------------------------------------------------------------------------
# what is shared with the family
# ---------------------------------------------------------------------------


def test_an_extension_does_not_repeat_what_the_family_stored(dataset, monkeypatch):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7,))
    workload.build_workload(start)

    searched, alphas, scans = [], [], []
    original_top, original_find = n2.BaseScanner.top, n2.find_alpha
    original_scan = stream_gt.scan_lists

    def counting_top(self, queries, k):
        searched.append(k)
        return original_top(self, queries, k)

    def counting_find(*args, **kwargs):
        alphas.append(1)
        return original_find(*args, **kwargs)

    def counting_scan(queries, length, *args, **kwargs):
        scans.append(args[6])
        return original_scan(queries, length, *args, **kwargs)

    monkeypatch.setattr(n2.BaseScanner, "top", counting_top)
    monkeypatch.setattr(n2, "find_alpha", counting_find)
    monkeypatch.setattr(stream_gt, "scan_lists", counting_scan)

    # A new group size at the existing overlap: no neighbour scan (k = 60), no noise scale
    # search, and no scan for the evaluation queries; only the new pool's own work. The
    # follow-up stage scans only vectors that have no list yet (one scan here).
    workload.build_workload(with_pools(start, sizes=(3, 6)))
    assert 60 not in searched
    assert alphas == []
    assert scans == ["list"]

    searched.clear(), alphas.clear(), scans.clear()
    # A new overlap needs its noise scale, but the neighbours still come from shared/.
    workload.build_workload(with_pools(start, overlaps=(0.7, 0.5), sizes=(3, 6)))
    assert 60 not in searched
    assert len(alphas) > 0
    assert scans == ["list"]


def test_the_evaluation_answers_of_new_sets_are_the_files_of_the_family(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7,))
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))

    old_set, new_set = grown / "o70" / "g3" / "u25", grown / "o50" / "g3" / "u25"
    store = grown / "shared"
    eval_dir = store / "mixes" / "u25" / "groundtruth" / "eval"
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
        # ONE file: the old set, the new set and shared/ are three names of it.
        names = (old_set / relative, new_set / relative, shared_path)
        assert len({inode(path) for path in names}) == 1, relative
    assert inode(old_set / "pool" / "pool.u8bin") != inode(new_set / "pool" / "pool.u8bin")


def test_the_report_of_an_extension_names_the_pools_it_built_and_skipped(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], overlaps=(0.7,))
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))

    report = manifest.read_json(grown / "build_report_extension_1.json")
    assert report["mode"] == "extension"
    assert report["pools_built"] == ["o50/g3"]
    assert report["pools_skipped"] == ["o70/g3"]
    assert {"plan", "pools", "sets", "manifest"} <= set(report["stage_seconds"])
    assert manifest.read_json(grown / "build_report.json")["mode"] == "initial"  # untouched
    # The noise scale of the new overlap is in the family's shared results.
    cache = manifest.read_json(grown / "shared" / "cache.json")
    assert sorted(cache["calibrations"]) == ["o50", "o70"]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def replace_in(cfg, section, **fields):
    """The configuration with some fields of one of its sections changed."""
    updated = dataclasses.replace(getattr(cfg, section), **fields)
    return dataclasses.replace(cfg, **{section: updated})


# Changes of what every pool of a family shares: each must be refused.
SHARED_CHANGES = {
    "stream seed": lambda c: replace_in(c, "stream", seed=c.stream.seed + 1),
    "zipf exponents": lambda c: replace_in(c, "stream", zipf_exponents=(0.5,)),
    "total requests": lambda c: replace_in(
        c, "stream", total_requests=c.stream.total_requests + 48
    ),
    "number of steps": lambda c: replace_in(c, "stream", num_steps=4),
    "update percentages": lambda c: replace_in(c, "updates", update_percents=(0.0, 25.0, 50.0)),
    "update seed": lambda c: replace_in(c, "updates", seed=c.updates.seed + 1),
    "seed of the seeds": lambda c: replace_in(c, "seeds", seed=c.seeds.seed + 1),
    "number of groups": lambda c: replace_in(c, "seeds", num_groups=30),
    "neighbours": lambda c: replace_in(c, "groups", neighbours=50),
    "axes": lambda c: replace_in(c, "groups", axes=3),
    "residual": lambda c: replace_in(c, "groups", residual=True),
    "include_original": lambda c: replace_in(c, "groups", include_original=True),
    "calibration queries": lambda c: replace_in(c, "groups", calibration_queries=10),
    "overlap tolerance": lambda c: replace_in(c, "groups", overlap_tolerance=0.04),
    "evaluation k": lambda c: replace_in(c, "evaluation", eval_k=7),
    "checkpoint interval": lambda c: replace_in(c, "evaluation", checkpoint_every=2),
}


@pytest.mark.parametrize("what", sorted(SHARED_CHANGES))
def test_a_change_of_what_all_pools_share_is_refused_and_the_family_stays_as_it_was(dataset, what):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)

    wanted = SHARED_CHANGES[what](with_pools(start, overlaps=(0.7, 0.5)))
    with pytest.raises(workload.WorkloadError, match="all its pools share"):
        workload.build_workload(wanted)
    assert tree_state(family_dir) == before
    assert not list(family_dir.parent.glob("fam.extending*"))


def test_the_message_names_the_setting_that_differs(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    workload.build_workload(start)
    with pytest.raises(workload.WorkloadError) as caught:
        workload.build_workload(SHARED_CHANGES["axes"](with_pools(start, overlaps=(0.7, 0.5))))
    message = str(caught.value)
    assert "groups.axes" in message and "was 4" in message and "now 3" in message
    assert "new family" in message  # what to do instead


def test_other_query_files_than_the_ones_of_the_family_are_refused(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)

    other, _ = synthetic_rows(40, seed=99)  # the same path now holds other queries
    binio.write_array(dataset["query_path"], other)
    with pytest.raises(workload.WorkloadError, match="inputs"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    assert tree_state(family_dir) == before


def test_a_moved_data_set_does_not_block_an_extension(dataset):
    """Where the query file is does not matter, what it holds does (e.g. another machine)."""
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    workload.build_workload(start)

    moved = dataset["tmp"] / "moved" / "queries.u8bin"
    moved.parent.mkdir()
    shutil.copy(dataset["query_path"], moved)
    new_place = dataclasses.replace(
        with_pools(start, overlaps=(0.7, 0.5)),
        seeds=dataclasses.replace(start.seeds, query_file=moved),
    )
    grown = workload.build_workload(new_place)
    assert (grown / "o50" / "g3" / "u25" / "workload.json").is_file()


def test_a_family_of_the_old_format_cannot_be_extended(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    index = manifest.read_json(family_dir / "family.json")
    index["format_version"] = 2
    manifest.write_json(family_dir / "family.json", index)
    before = tree_state(family_dir)

    with pytest.raises(workload.WorkloadError, match="format 2.*new name"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    assert tree_state(family_dir) == before


def test_a_directory_that_is_not_a_family_is_refused(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    (base_set.directory / "workloads" / "fam").mkdir(parents=True)
    with pytest.raises(workload.WorkloadError, match="no family.json"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))


# ---------------------------------------------------------------------------
# the stored shared results are checked before they are used
# ---------------------------------------------------------------------------


def test_damaged_stored_neighbours_are_refused_not_used(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    path = family_dir / "shared" / "n2" / "basis_axes.npy"
    data = bytearray(path.read_bytes())
    data[-3] ^= 0xFF  # one wrong byte inside the stored bases
    path.write_bytes(bytes(data))
    before = tree_state(family_dir)

    with pytest.raises(n2.N2Error, match="sha256"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    assert tree_state(family_dir) == before
    assert not list(family_dir.parent.glob("fam.extending*"))


def test_stored_neighbours_made_for_other_seeds_are_refused(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    cache_path = family_dir / "shared" / "cache.json"
    cache = manifest.read_json(cache_path)
    cache["n2"]["fingerprint"]["seeds_sha256"] = "0" * 64
    manifest.write_json(cache_path, cache)
    with pytest.raises(n2.N2Error, match="other inputs"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))


def test_a_family_without_stored_neighbours_makes_them_again_and_keeps_them(dataset, monkeypatch):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = make_config(base_set, query_path)
    family_dir = workload.build_workload(start)
    # Remove the neighbours and bases from the family (the noise scale stays in cache.json).
    shutil.rmtree(family_dir / "shared" / "n2")
    cache_path = family_dir / "shared" / "cache.json"
    cache = manifest.read_json(cache_path)
    del cache["n2"]
    manifest.write_json(cache_path, cache)

    searched = []
    original_top = n2.BaseScanner.top

    def counting_top(self, queries, k):
        searched.append(k)
        return original_top(self, queries, k)

    monkeypatch.setattr(n2.BaseScanner, "top", counting_top)
    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))

    assert searched.count(60) == 1  # made again, once
    once = one_build(base_set, query_path, overlaps=(0.7, 0.5))
    same_family(grown, once)  # also cache.json: n2 is back
    # The recreated files are the ones recorded in cache.json.
    cache = manifest.read_json(grown / "shared" / "cache.json")
    for name, entry in cache["n2"]["files"].items():
        assert manifest.sha256_file(grown / "shared" / "n2" / name) == entry["sha256"]


# ---------------------------------------------------------------------------
# interruption: nothing is lost, the family is never half updated
# ---------------------------------------------------------------------------


def test_an_interrupted_build_of_the_new_pools_continues_and_the_family_is_untouched(
    dataset, monkeypatch
):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = make_config(base_set, query_path, sizes=(3,))
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)
    grow = with_pools(start, overlaps=(0.7, 0.5), sizes=(3, 6))

    original = interrupt(monkeypatch, "_finish_set", after_calls=1)  # dies while it writes sets
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(grow)
    monkeypatch.setattr(workload, "_finish_set", original)
    # The family is exactly what it was; the unfinished work waits next to it.
    assert tree_state(family_dir) == before
    assert (family_dir.parent / "fam.extending").is_dir()
    assert not (family_dir.parent / "fam.extending.lock").exists()  # the lock is released

    grown = workload.build_workload(grow)
    once = one_build(base_set, query_path, overlaps=(0.7, 0.5), sizes=(3, 6))
    same_family(grown, once)
    assert not (family_dir.parent / "fam.extending").exists()


def test_an_unfinished_extension_of_a_family_that_changed_since_is_refused(dataset, monkeypatch):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    first = with_pools(start, overlaps=(0.7, 0.5))
    original = interrupt(monkeypatch, "_finish_set", after_calls=1)
    with pytest.raises(KeyboardInterrupt):
        workload.build_workload(first)
    monkeypatch.setattr(workload, "_finish_set", original)

    # The family gains a pool by another run before the unfinished work is continued.
    index = manifest.read_json(family_dir / "family.json")
    index["pools"].append(
        {
            "name": "o90/g3", "kind": "n2", "group_size": 3, "overlap": "o90",
            "vectors": 0, "groups": 0,
        }
    )
    manifest.write_json(family_dir / "family.json", index)
    with pytest.raises(workload.WorkloadError, match="unfinished build"):
        workload.build_workload(first)


class Died(Exception):
    """Stands for a process that dies at a chosen step of the commit."""


COMMIT_STEPS = ["first move", "second move", "cache", "report", "index", "cleanup"]


@pytest.mark.parametrize("fail_at", COMMIT_STEPS)
def test_a_commit_that_dies_at_any_step_is_completed_by_running_again(
    dataset, monkeypatch, fail_at
):
    base_set = build_base(dataset)
    query_path = dataset["query_path"]
    start = make_config(base_set, query_path, overlaps=(0.7,), sizes=(3,))
    family_dir = workload.build_workload(start)
    old_index_text = (family_dir / "family.json").read_text()
    grow = with_pools(start, overlaps=(0.7, 0.5), sizes=(3, 6))  # three new pools: three moves

    real_move, real_write = extend._move_directory, extend._write_text
    real_rmtree = extend.shutil.rmtree
    moves = []

    def moving(source, destination):
        moves.append(destination.name)
        if fail_at == "first move" and len(moves) == 1:
            raise Died
        if fail_at == "second move" and len(moves) == 2:
            raise Died
        return real_move(source, destination)

    def writing(path, text):
        step = {"cache.json": "cache", "family.json": "index"}.get(path.name)
        if path.name.startswith("build_report_extension"):
            step = "report"
        if step == fail_at:
            raise Died
        return real_write(path, text)

    def removing(path, *args, **kwargs):
        if fail_at == "cleanup" and str(path).endswith("fam.extending"):
            raise Died
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(extend, "_move_directory", moving)
    monkeypatch.setattr(extend, "_write_text", writing)
    monkeypatch.setattr(extend.shutil, "rmtree", removing)
    with pytest.raises(Died):
        workload.build_workload(grow)
    if fail_at not in ("cleanup", "index"):
        # Until family.json is replaced the family lists exactly the pools it had.
        assert (family_dir / "family.json").read_text() == old_index_text
    # The journal and the built pools are kept for the next run.
    assert (family_dir.parent / "fam.extending" / "journal.json").is_file()

    monkeypatch.setattr(extend, "_move_directory", real_move)
    monkeypatch.setattr(extend, "_write_text", real_write)
    monkeypatch.setattr(extend.shutil, "rmtree", real_rmtree)
    messages = []
    grown = workload.build_workload(grow, log=messages.append)

    assert any("completing the commit" in message for message in messages)
    once = one_build(base_set, query_path, overlaps=(0.7, 0.5), sizes=(3, 6))
    same_family(grown, once)
    assert not (family_dir.parent / "fam.extending").exists()


def test_a_pool_directory_that_is_in_the_way_is_never_overwritten(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    intruder = family_dir / "o50" / "g3"  # not listed in family.json, but there
    intruder.mkdir(parents=True)
    (intruder / "keep.txt").write_text("precious")
    grow = with_pools(start, overlaps=(0.7, 0.5))

    with pytest.raises(workload.WorkloadError, match="already exists"):
        workload.build_workload(grow)
    assert (intruder / "keep.txt").read_text() == "precious"
    listed = [pool["name"] for pool in manifest.read_json(family_dir / "family.json")["pools"]]
    assert listed == ["o70/g3"]  # the index still lists what it listed

    # Once the obstacle is removed the same command finishes the work.
    shutil.rmtree(family_dir / "o50")
    grown = workload.build_workload(grow)
    listed = {pool["name"] for pool in manifest.read_json(grown / "family.json")["pools"]}
    assert listed == {"o70/g3", "o50/g3"}


# ---------------------------------------------------------------------------
# one extension at a time
# ---------------------------------------------------------------------------


def test_a_second_extension_is_refused_while_the_first_one_runs(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    before = tree_state(family_dir)
    lock = family_dir.parent / "fam.extending.lock"
    lock.write_text(f"{os.getpid()}\n")  # this very process "is running" an extension

    with pytest.raises(workload.WorkloadError, match="another extension"):
        workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    assert lock.read_text() == f"{os.getpid()}\n"  # the owner's lock is left alone
    assert tree_state(family_dir) == before


def test_a_lock_of_a_process_that_no_longer_exists_is_replaced(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    finished = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True
    )
    (family_dir.parent / "fam.extending.lock").write_text(finished.stdout)  # a dead process id

    grown = workload.build_workload(with_pools(start, overlaps=(0.7, 0.5)))
    assert (grown / "o50" / "g3" / "u25" / "workload.json").is_file()
    assert not (family_dir.parent / "fam.extending.lock").exists()


def test_the_lock_is_released_when_the_extension_fails(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"])
    family_dir = workload.build_workload(start)
    with pytest.raises(workload.WorkloadError):
        workload.build_workload(SHARED_CHANGES["axes"](with_pools(start, overlaps=(0.7, 0.5))))
    assert not (family_dir.parent / "fam.extending.lock").exists()


# ---------------------------------------------------------------------------
# the merged index
# ---------------------------------------------------------------------------


def test_family_index_of_an_extension_lists_pools_and_sets_in_configuration_order(dataset):
    base_set = build_base(dataset)
    start = make_config(base_set, dataset["query_path"], percents=(0.0, 25.0))
    workload.build_workload(start)
    grown = workload.build_workload(with_pools(start, overlaps=(0.5, 0.7), sizes=(6, 3)))

    index = manifest.read_json(grown / "family.json")
    pools = ["o50/g6", "o50/g3", "o70/g6", "o70/g3"]
    assert [pool["name"] for pool in index["pools"]] == pools
    assert [entry["name"] for entry in index["overlaps"]] == ["o50", "o70"]
    assert [entry["path"] for entry in index["sets"]] == [
        f"{pool}/{name}" for pool in pools for name in ("u0", "u25")
    ]
    for entry in index["sets"]:
        info_path = grown / entry["path"] / "workload.json"
        assert entry["workload_json_sha256"] == manifest.sha256_file(info_path)
    assert index["common"] == family.read_existing_family(grown).index["common"]
