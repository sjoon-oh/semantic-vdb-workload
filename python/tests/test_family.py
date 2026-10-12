"""family.json: the common fingerprint and the merge of an extension into the index."""

import dataclasses
import json
from pathlib import Path

import pytest

from semantic_vdb import config, family
from semantic_vdb.layout import overlap_dir_name
from test_end_to_end import build_base, dataset, workload_config  # noqa: F401  (fixture)


@pytest.fixture()
def cfg(dataset):
    """(base set, workload configuration) of a small family."""
    base_set = build_base(dataset)
    groups = config.GroupsConfig(
        group_sizes=(3,),
        include_original=False,
        neighbours=60,
        axes=4,
        residual=False,
        target_overlaps=(0.7,),
        calibration_queries=20,
        overlap_tolerance=0.05,
    )
    return base_set, workload_config(base_set, dataset["query_path"], groups=groups)


def fingerprint(base_and_config, **changes):
    base_set, original = base_and_config
    changed = dataclasses.replace(original, **changes)
    return family.common_fingerprint(changed, base_set, {"inputs": "x"})


# ---------------------------------------------------------------------------
# the common fingerprint
# ---------------------------------------------------------------------------


def test_the_fingerprint_ignores_what_may_differ_between_a_family_and_its_extension(cfg):
    base = fingerprint(cfg)
    _, original = cfg
    assert fingerprint(cfg, name="another") == base  # the directory is the name
    assert fingerprint(cfg, existing="extend") == base
    other_machine = config.ParallelConfig(gpus="none", cpu_workers=3, numa_node="none")
    assert fingerprint(cfg, parallel=other_machine) == base
    # The two axes that an extension adds to.
    wider = dataclasses.replace(original.groups, group_sizes=(3, 9), target_overlaps=(0.7, 0.4))
    assert fingerprint(cfg, groups=wider) == base


def test_the_fingerprint_does_not_depend_on_where_the_files_are(cfg):
    """A moved data set (or another machine) must not block an extension; the inputs' digests
    in the fingerprint tell whether the files hold something else."""
    base_set, original = cfg
    base = fingerprint(cfg)
    moved_seeds = dataclasses.replace(original.seeds, query_file=Path("/elsewhere/queries.u8bin"))
    moved_eval = dataclasses.replace(original.evaluation, eval_file=Path("/elsewhere/eval.u8bin"))
    assert fingerprint(cfg, seeds=moved_seeds) == base
    assert fingerprint(cfg, evaluation=moved_eval) == base
    assert fingerprint(cfg, base_set_dir=Path("/elsewhere/base")) == base


def test_the_fingerprint_sees_everything_else(cfg):
    base = fingerprint(cfg)
    base_set, original = cfg
    other_stream = dataclasses.replace(original.stream, seed=original.stream.seed + 1)
    assert fingerprint(cfg, stream=other_stream) != base
    assert fingerprint(cfg, groups=dataclasses.replace(original.groups, axes=3)) != base
    assert fingerprint(cfg, evaluation=dataclasses.replace(original.evaluation, eval_k=3)) != base
    # Other query files change the digests of the inputs.
    other_inputs = family.common_fingerprint(original, base_set, {"inputs": "y"})
    assert other_inputs != base
    # The tool version is not part of it: a newer tool of the same format may extend.
    assert "tool_version" not in str(base)


def test_the_fingerprint_is_plain_json_data(cfg):
    assert json.loads(json.dumps(fingerprint(cfg))) == fingerprint(cfg)


# ---------------------------------------------------------------------------
# merging an extension into the index
# ---------------------------------------------------------------------------


def set_entry(pool, set_name, percent):
    return {
        "path": f"{pool}/{set_name}",
        "pool": pool,
        "set_name": set_name,
        "update_percent": percent,
    }


def pool_entry(name):
    overlap, size = name.split("/")
    return {"name": name, "overlap": overlap, "group_size": int(size[1:])}


def with_axes(original, overlaps, sizes=None, percents=None):
    """The configuration with other pool axes (and other percentages)."""
    groups = dataclasses.replace(
        original.groups,
        target_overlaps=overlaps,
        group_sizes=sizes if sizes is not None else original.groups.group_sizes,
    )
    updates = original.updates
    if percents is not None:
        updates = dataclasses.replace(updates, update_percents=percents)
    return dataclasses.replace(original, groups=groups, updates=updates)


def test_merge_puts_everything_in_the_order_of_the_configuration(cfg):
    _, original = cfg
    wanted = with_axes(original, (0.5, 0.7), sizes=(6, 3), percents=(25.0, 0.0))
    old = {
        "format_version": 3,
        "name": "n",
        "common": {},
        "overlaps": [{"name": "o70"}],
        "pools": [pool_entry("o70/g3")],
        "sets": [set_entry("o70/g3", "u0", 0.0), set_entry("o70/g3", "u25", 25.0)],
    }
    new_pools = ["o50/g6", "o50/g3", "o70/g6"]
    fragment = {
        "overlaps": [{"name": "o70"}, {"name": "o50"}],  # o70 again: a pool at an old overlap
        "pools": [pool_entry(name) for name in new_pools],
        "sets": [
            set_entry(pool, name, percent)
            for pool in new_pools
            for name, percent in (("u0", 0.0), ("u25", 25.0))
        ],
    }
    merged = family.merge_family_index(old, fragment, wanted)

    assert [pool["name"] for pool in merged["pools"]] == [
        "o50/g6", "o50/g3", "o70/g6", "o70/g3",
    ]
    assert [item["name"] for item in merged["overlaps"]] == ["o50", "o70"]  # o70 once
    assert [item["path"] for item in merged["sets"]] == [
        f"{pool}/{name}"
        for pool in ("o50/g6", "o50/g3", "o70/g6", "o70/g3")
        for name in ("u25", "u0")
    ]
    assert merged["common"] == {} and merged["name"] == "n"  # the rest of the index is kept
    assert old["pools"] == [pool_entry("o70/g3")]  # the argument is not modified


def test_merge_keeps_what_the_configuration_does_not_mention_behind_the_rest(cfg):
    _, original = cfg
    wanted = with_axes(original, (0.5,))
    old = {
        "overlaps": [{"name": "o90"}],
        "pools": [pool_entry("o90/g3")],
        "sets": [set_entry("o90/g3", "u0", 0.0)],
    }
    fragment = {
        "overlaps": [{"name": "o50"}],
        "pools": [pool_entry("o50/g3")],
        "sets": [set_entry("o50/g3", "u0", 0.0)],
    }
    merged = family.merge_family_index(old, fragment, wanted)
    assert [pool["name"] for pool in merged["pools"]] == ["o50/g3", "o90/g3"]
    assert [item["name"] for item in merged["overlaps"]] == ["o50", "o90"]


# ---------------------------------------------------------------------------
# opening a family
# ---------------------------------------------------------------------------


def test_only_a_family_of_the_current_format_is_opened(tmp_path):
    (tmp_path / "family.json").write_text('{"format_version": 2}')
    with pytest.raises(family.FamilyError, match="format 2"):
        family.read_existing_family(tmp_path)
    with pytest.raises(family.FamilyError, match="no family.json"):
        family.read_existing_family(tmp_path / "nowhere")


def test_check_common_names_the_difference_and_what_to_do_instead():
    existing = family.ExistingFamily(Path("/x/fam"), {"common": {"a": 1, "b": 2}}, None)
    family.check_common(existing, {"a": 1, "b": 2})  # equal: no error
    with pytest.raises(family.FamilyError, match="a: was 1, now 5") as caught:
        family.check_common(existing, {"a": 5, "b": 2})
    assert "new family" in str(caught.value)


def test_overlap_names_used_by_the_merge_match_the_directories():
    assert [overlap_dir_name(value) for value in (0.5, 0.7)] == ["o50", "o70"]
