"""Tests for semantic_vdb.config and semantic_vdb.manifest."""

from pathlib import Path

import numpy as np
import pytest

from semantic_vdb import config, manifest

BASE_SET_INI = """
[input]
raw_file = /data/base.u8bin
metric = l2

[partition]
seed = 1
num_base = 1000
base_selection = random
num_query_rows = 0
max_inserts = 500   ; inline comment

[output]
dir = /out
name = base_a
"""

WORKLOAD_INI = """
[workload]
name = wl_a

[base_set]
dir = /out/base_a

[seeds]
mode = public_queries
query_file = /data/query.u8bin
seed = 11

[groups]
group_size = 200
target_overlap = 0.8

[stream]
total_requests = 1000
num_steps = 10
zipf_exponents = 0.3, 0.6, 0.99
seed = 5

[updates]
seed = 21
update_percents = 0, 25, 50

[evaluation]
source = original
checkpoint_every = 5

[groundtruth]
device = cpu
"""


def write(tmp_path, text, name="c.ini"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def replace(text, old, new):
    assert old in text
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# base_set.ini
# ---------------------------------------------------------------------------


def test_base_set_config_parses_values_and_ignores_inline_comments(tmp_path):
    cfg = config.load_base_set_config(write(tmp_path, BASE_SET_INI))
    assert cfg.raw_file == Path("/data/base.u8bin")
    assert (cfg.num_base, cfg.num_query_rows, cfg.max_inserts) == (1000, 0, 500)
    assert cfg.base_selection == "random" and cfg.metric == "l2"
    assert cfg.name == "base_a"


def test_max_inserts_rest_means_none(tmp_path):
    text = replace(BASE_SET_INI, "max_inserts = 500   ; inline comment", "max_inserts = rest")
    assert config.load_base_set_config(write(tmp_path, text)).max_inserts is None


@pytest.mark.parametrize(
    "old,new",
    [
        ("num_base = 1000", "num_base = 0"),
        ("num_base = 1000", "num_base = many"),
        ("max_inserts = 500   ; inline comment", "max_inserts = -3"),
        ("base_selection = random", "base_selection = sorted"),
        ("metric = l2", "metric = cosine"),
        ("name = base_a", "name = ../escape"),
        ("seed = 1\n", ""),  # required key missing
    ],
)
def test_bad_base_set_settings_are_rejected(tmp_path, old, new):
    with pytest.raises(config.ConfigError):
        config.load_base_set_config(write(tmp_path, replace(BASE_SET_INI, old, new)))


def test_unreadable_file_is_rejected(tmp_path):
    with pytest.raises(config.ConfigError):
        config.load_base_set_config(tmp_path / "missing.ini")


# ---------------------------------------------------------------------------
# workload.ini
# ---------------------------------------------------------------------------


def test_workload_config_defaults_and_values(tmp_path):
    cfg = config.load_workload_config(write(tmp_path, WORKLOAD_INI))
    assert cfg.seeds.num_groups is None  # "all" by default
    assert cfg.groups.group_sizes == (200,)
    assert cfg.groups.target_overlaps == (0.8,)
    assert cfg.existing == "error"
    assert cfg.groups.include_original is False
    assert cfg.stream.zipf_exponents == (0.3, 0.6, 0.99)
    assert cfg.groundtruth.stream_k == 10 and cfg.evaluation.eval_k == 100
    assert cfg.stream.total_requests == 1000
    assert cfg.updates.update_percents == (0.0, 25.0, 50.0)
    assert [mix.name for mix in cfg.request_mixes()] == ["u0", "u25", "u50"]
    assert [(mix.reads, mix.updates) for mix in cfg.request_mixes()] == [(1000, 0), (750, 250), (500, 500)]
    assert cfg.groundtruth.device == "cpu"


def test_stream_groundtruth_is_on_by_default_and_can_be_switched_off(tmp_path):
    assert config.load_workload_config(write(tmp_path, WORKLOAD_INI)).groundtruth.stream_groundtruth is True
    text = replace(WORKLOAD_INI, "device = cpu", "device = cpu\nstream_groundtruth = false")
    off = config.load_workload_config(write(tmp_path, text))
    assert off.groundtruth.stream_groundtruth is False
    assert off.groundtruth.stream_k == 10  # the other settings are untouched


def test_stream_groundtruth_must_be_true_or_false(tmp_path):
    text = replace(WORKLOAD_INI, "device = cpu", "device = cpu\nstream_groundtruth = maybe")
    with pytest.raises(config.ConfigError, match="stream_groundtruth"):
        config.load_workload_config(write(tmp_path, text))


def test_target_overlap_is_a_configurable_parameter(tmp_path):
    text = replace(WORKLOAD_INI, "target_overlap = 0.8", "target_overlap = 0.95")
    assert config.load_workload_config(write(tmp_path, text)).groups.target_overlaps == (0.95,)


def test_target_overlap_can_be_a_list_that_makes_it_a_combination_axis(tmp_path):
    text = replace(WORKLOAD_INI, "target_overlap = 0.8", "target_overlap = 0.8, 0.5, 0.125")
    loaded = config.load_workload_config(write(tmp_path, text))
    assert loaded.groups.target_overlaps == (0.8, 0.5, 0.125)


def test_target_overlap_defaults_to_one_value_when_it_is_left_out(tmp_path):
    text = replace(WORKLOAD_INI, "target_overlap = 0.8\n", "")
    assert config.load_workload_config(write(tmp_path, text)).groups.target_overlaps == (0.8,)


@pytest.mark.parametrize(
    "value, message",
    [
        ("0.8, 0.8", "repeat"),
        ("0.5, 0.50", "repeat"),  # written differently, the same overlap
        ("0.8, 1.0", "strictly between"),
        ("0.8, 0", "strictly between"),
        ("0.8, -0.1", "strictly between"),
        ("0.8, abc", "numbers"),
        ("0.8, nan", "strictly between"),
    ],
)
def test_bad_target_overlap_lists_are_rejected(tmp_path, value, message):
    text = replace(WORKLOAD_INI, "target_overlap = 0.8", f"target_overlap = {value}")
    with pytest.raises(config.ConfigError, match=message):
        config.load_workload_config(write(tmp_path, text))


def test_existing_policy_is_error_by_default_and_can_be_extend(tmp_path):
    assert config.load_workload_config(write(tmp_path, WORKLOAD_INI)).existing == "error"
    text = replace(WORKLOAD_INI, "[workload]\n", "[workload]\nexisting = extend\n")
    assert config.load_workload_config(write(tmp_path, text)).existing == "extend"


def test_unknown_existing_policy_is_rejected(tmp_path):
    text = replace(WORKLOAD_INI, "[workload]\n", "[workload]\nexisting = overwrite\n")
    with pytest.raises(config.ConfigError, match="existing"):
        config.load_workload_config(write(tmp_path, text))


def test_num_groups_integer_is_accepted(tmp_path):
    text = replace(WORKLOAD_INI, "seed = 11\n", "seed = 11\nnum_groups = 100\n")
    assert config.load_workload_config(write(tmp_path, text)).seeds.num_groups == 100


@pytest.mark.parametrize(
    "old,new",
    [
        ("group_size = 200", "group_size = 0"),
        ("target_overlap = 0.8", "target_overlap = 1.0"),
        ("target_overlap = 0.8", "target_overlap = 0"),
        ("total_requests = 1000", "total_requests = 5"),  # fewer requests than steps
        ("update_percents = 0, 25, 50", "update_percents = 25, 25"),  # repeated percentage
        ("update_percents = 0, 25, 50", "update_percents = 25, 25.0"),  # same set written twice
        ("update_percents = 0, 25, 50", "update_percents = 100"),  # no reads left
        ("update_percents = 0, 25, 50", "update_percents = -5"),
        ("update_percents = 0, 25, 50", "update_percents = 0.05"),  # 0.5 updates: not whole
        ("update_percents = 0, 25, 50", "update_percents = 99.5"),  # 5 reads for 10 steps
        ("zipf_exponents = 0.3, 0.6, 0.99", "zipf_exponents = -1"),
        ("zipf_exponents = 0.3, 0.6, 0.99", "zipf_exponents = 0.3, 0.3"),
        ("zipf_exponents = 0.3, 0.6, 0.99", "zipf_exponents = fast"),
        ("mode = public_queries", "mode = telepathy"),
        ("device = cpu", "device = tpu"),
        ("checkpoint_every = 5", "checkpoint_every = 0"),
        ("name = wl_a", "name = a/b"),
    ],
)
def test_bad_workload_settings_are_rejected(tmp_path, old, new):
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, replace(WORKLOAD_INI, old, new)))


def test_fewer_requests_than_steps_names_the_step_count(tmp_path):
    # The request-mix check would reject this too; the direct check gives the clearer message.
    text = replace(WORKLOAD_INI, "total_requests = 1000", "total_requests = 5")
    with pytest.raises(config.ConfigError, match="num_steps .* cannot exceed total_requests"):
        config.load_workload_config(write(tmp_path, text))


def test_a_percentage_that_leaves_one_read_per_step_is_accepted(tmp_path):
    # 99% of 1000 leaves exactly 10 reads for 10 steps.
    text = replace(WORKLOAD_INI, "update_percents = 0, 25, 50", "update_percents = 99")
    cfg = config.load_workload_config(write(tmp_path, text))
    assert [(mix.reads, mix.updates) for mix in cfg.request_mixes()] == [(10, 990)]


def test_settings_of_the_previous_version_are_refused_not_ignored(tmp_path):
    # `length`, `total_inserts` and `total_deletes` would build a different workload than the
    # user wrote if they were silently dropped.
    old_length = replace(WORKLOAD_INI, "total_requests = 1000", "length = 1000")
    with pytest.raises(config.ConfigError, match="no longer supported"):
        config.load_workload_config(write(tmp_path, old_length))
    for key in ("total_inserts", "total_deletes"):
        text = replace(WORKLOAD_INI, "seed = 21\n", f"seed = 21\n{key} = 10\n")
        with pytest.raises(config.ConfigError, match="no longer supported"):
            config.load_workload_config(write(tmp_path, text))


def test_update_percents_are_required(tmp_path):
    text = replace(WORKLOAD_INI, "update_percents = 0, 25, 50\n", "")
    with pytest.raises(config.ConfigError, match="update_percents"):
        config.load_workload_config(write(tmp_path, text))


# ---------------------------------------------------------------------------
# uniform_unique stream and variants = none
# ---------------------------------------------------------------------------


def test_uniform_unique_and_variants_default_to_the_previous_behaviour(tmp_path):
    cfg = config.load_workload_config(write(tmp_path, WORKLOAD_INI))
    assert cfg.stream.uniform_unique is False
    assert cfg.groups.variants == "n2" and cfg.groups.group_sizes == (200,)


def test_uniform_unique_can_be_added_to_the_zipf_streams(tmp_path):
    text = replace(WORKLOAD_INI, "seed = 5\n", "seed = 5\nuniform_unique = true\n")
    cfg = config.load_workload_config(write(tmp_path, text))
    assert cfg.stream.uniform_unique is True
    assert cfg.stream.zipf_exponents == (0.3, 0.6, 0.99)


def test_zipf_exponents_none_needs_the_uniform_unique_stream(tmp_path):
    only_unique = replace(WORKLOAD_INI, "zipf_exponents = 0.3, 0.6, 0.99", "zipf_exponents = none")
    with pytest.raises(config.ConfigError, match="no stream was requested"):
        config.load_workload_config(write(tmp_path, only_unique))

    text = replace(only_unique, "seed = 5\n", "seed = 5\nuniform_unique = true\n")
    cfg = config.load_workload_config(write(tmp_path, text, "ok.ini"))
    assert cfg.stream.zipf_exponents == () and cfg.stream.uniform_unique is True


def test_variants_none_uses_group_size_one(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "variants = none")
    cfg = config.load_workload_config(write(tmp_path, text))
    assert cfg.groups.variants == "none" and cfg.groups.group_sizes == (1,)


def test_variants_none_with_explicit_group_size_one_is_accepted(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "variants = none\ngroup_size = 1")
    assert config.load_workload_config(write(tmp_path, text)).groups.group_sizes == (1,)


def test_variants_none_with_a_larger_group_size_is_rejected_not_ignored(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "variants = none\ngroup_size = 200")
    with pytest.raises(config.ConfigError, match="group_size must be 1"):
        config.load_workload_config(write(tmp_path, text))


def test_unknown_variants_are_rejected(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "group_size = 200\nvariants = paraphrase")
    with pytest.raises(config.ConfigError, match="variants"):
        config.load_workload_config(write(tmp_path, text))


def test_dataclasses_built_in_code_get_the_same_checks():
    groups = dict(
        group_sizes=(5,), include_original=False, neighbours=60, axes=4, residual=False,
        target_overlaps=(0.7,), calibration_queries=20, overlap_tolerance=0.05,
    )
    with pytest.raises(config.ConfigError):
        config.GroupsConfig(variants="none", **groups)
    with pytest.raises(config.ConfigError):
        config.GroupsConfig(variants="bogus", **groups)
    with pytest.raises(config.ConfigError):
        config.StreamConfig(
            total_requests=10, num_steps=2, zipf_exponents=(), member_selection="uniform", seed=1
        )


# ---------------------------------------------------------------------------
# group sizes as a combination axis, [parallel], automatic batch sizes
# ---------------------------------------------------------------------------


def test_group_size_accepts_a_list(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "group_size = 50, 200")
    assert config.load_workload_config(write(tmp_path, text)).groups.group_sizes == (50, 200)
    text = replace(WORKLOAD_INI, "group_size = 200", "group_size = 7")
    assert config.load_workload_config(write(tmp_path, text, "one.ini")).groups.group_sizes == (7,)


@pytest.mark.parametrize("sizes", ["50, 50", "0, 200", "200, -1", "fifty", "50 200"])
def test_bad_group_size_lists_are_rejected(tmp_path, sizes):
    text = replace(WORKLOAD_INI, "group_size = 200", f"group_size = {sizes}")
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, text))


def test_variants_none_does_not_allow_a_list_of_sizes(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "variants = none\ngroup_size = 1, 2")
    with pytest.raises(config.ConfigError, match="group_size must be 1"):
        config.load_workload_config(write(tmp_path, text))


def test_parallel_defaults_are_the_values_of_the_development_machine(tmp_path):
    cfg = config.load_workload_config(write(tmp_path, WORKLOAD_INI))
    assert cfg.parallel == config.ParallelConfig()
    assert cfg.parallel.gpus == (1,) and cfg.parallel.numa_node == 1
    assert cfg.parallel.cpu_workers == "auto" and cfg.parallel.workers_per_gpu == 1
    assert cfg.parallel.gpu_max_used_mb == 1024


def test_parallel_section_values(tmp_path):
    section = (
        "\n[parallel]\ngpus = 0, 1\ngpu_max_used_mb = 2048\nworkers_per_gpu = 2\n"
        "cpu_workers = 8\nnuma_node = auto\n"
    )
    cfg = config.load_workload_config(write(tmp_path, WORKLOAD_INI + section))
    assert cfg.parallel == config.ParallelConfig(
        gpus=(0, 1), gpu_max_used_mb=2048, workers_per_gpu=2, cpu_workers=8, numa_node="auto"
    )
    text = WORKLOAD_INI + "\n[parallel]\ngpus = auto\nnuma_node = none\ncpu_workers = auto\n"
    cfg = config.load_workload_config(write(tmp_path, text, "auto.ini"))
    assert (cfg.parallel.gpus, cfg.parallel.numa_node, cfg.parallel.cpu_workers) == ("auto", "none", "auto")
    text = WORKLOAD_INI + "\n[parallel]\ngpus = none\nnuma_node = 0\n"
    cfg = config.load_workload_config(write(tmp_path, text, "none.ini"))
    assert (cfg.parallel.gpus, cfg.parallel.numa_node) == ("none", 0)


@pytest.mark.parametrize(
    "line",
    ["gpus = 1, 1", "gpus = -1", "gpus = some", "workers_per_gpu = 0", "cpu_workers = 0",
     "cpu_workers = many", "numa_node = -2", "numa_node = left", "gpu_max_used_mb = -5"],
)
def test_bad_parallel_settings_are_rejected(tmp_path, line):
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, WORKLOAD_INI + f"\n[parallel]\n{line}\n"))


def test_batch_sizes_default_to_auto_and_accept_numbers(tmp_path):
    cfg = config.load_workload_config(write(tmp_path, WORKLOAD_INI))
    assert cfg.groundtruth.base_batch_rows is None and cfg.groundtruth.query_chunk_rows is None
    text = replace(WORKLOAD_INI, "device = cpu", "device = cpu\nbase_batch_rows = 4096\nquery_chunk_rows = auto")
    cfg = config.load_workload_config(write(tmp_path, text, "numbers.ini"))
    assert cfg.groundtruth.base_batch_rows == 4096 and cfg.groundtruth.query_chunk_rows is None
    bad = replace(WORKLOAD_INI, "device = cpu", "device = cpu\nbase_batch_rows = 0")
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, bad, "bad.ini"))


def test_axes_cannot_exceed_neighbours_minus_one(tmp_path):
    text = replace(WORKLOAD_INI, "group_size = 200", "group_size = 200\nneighbours = 10\naxes = 10")
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, text))


def test_mode_specific_files_are_required(tmp_path):
    no_query_file = replace(WORKLOAD_INI, "query_file = /data/query.u8bin\n", "")
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, no_query_file))

    external = replace(WORKLOAD_INI, "mode = public_queries", "mode = external")
    with pytest.raises(config.ConfigError):  # group_id_file missing
        config.load_workload_config(write(tmp_path, external))

    external += "\n"
    external = replace(external, "query_file =", "group_id_file = /data/g.ibin\nquery_file =")
    with pytest.raises(config.ConfigError):  # evaluation.source must be external
        config.load_workload_config(write(tmp_path, external))

    external_ok = replace(external, "source = original", "source = external\neval_file = /data/e.fbin")
    cfg = config.load_workload_config(write(tmp_path, external_ok))
    assert cfg.seeds.mode == "external" and cfg.evaluation.eval_file == Path("/data/e.fbin")


def test_external_evaluation_requires_a_file(tmp_path):
    text = replace(WORKLOAD_INI, "source = original", "source = external")
    with pytest.raises(config.ConfigError):
        config.load_workload_config(write(tmp_path, text))


# ---------------------------------------------------------------------------
# manifest helpers
# ---------------------------------------------------------------------------


def test_json_round_trip_handles_numpy_and_paths(tmp_path):
    payload = {
        "count": np.int64(7),
        "ratio": np.float32(0.5),
        "vector": np.arange(3),
        "path": Path("/a/b"),
        "nested": {"flag": True},
    }
    path = tmp_path / "m.json"
    manifest.write_json(path, payload)
    loaded = manifest.read_json(path)
    assert loaded == {
        "count": 7,
        "ratio": 0.5,
        "vector": [0, 1, 2],
        "path": "/a/b",
        "nested": {"flag": True},
    }
    assert not (tmp_path / "m.json.partial").exists()


def test_sha256_matches_hashlib(tmp_path):
    import hashlib

    data = bytes(range(256)) * 1000
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert manifest.sha256_file(path, chunk_bytes=1000) == hashlib.sha256(data).hexdigest()
