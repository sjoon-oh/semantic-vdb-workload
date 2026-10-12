"""Tests for semantic_vdb.parallel: CPUs, NUMA, GPUs, memory budgets and worker pools.

The machine is made up: fake ``/sys`` trees and fake ``nvidia-smi`` output, so the tests
give the same result on any computer.
"""

import os
import subprocess
from pathlib import Path

import pytest

from semantic_vdb import parallel
from semantic_vdb.config import ConfigError, ParallelConfig
from semantic_vdb.parallel import GpuInfo, ResourceError


# ---------------------------------------------------------------------------
# a made-up machine: 2 NUMA nodes, 8 logical CPUs each, hyper-thread siblings (i, i + 8)
# ---------------------------------------------------------------------------


@pytest.fixture()
def machine(tmp_path):
    node_root, cpu_root, pci_root = tmp_path / "node", tmp_path / "cpu", tmp_path / "pci"
    nodes = {0: "0-3,8-11", 1: "4-7,12-15"}
    for node, cpulist in nodes.items():
        (node_root / f"node{node}").mkdir(parents=True)
        (node_root / f"node{node}" / "cpulist").write_text(cpulist + "\n")
    for cpu in range(16):
        topology = cpu_root / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        low = cpu % 8
        (topology / "thread_siblings_list").write_text(f"{low},{low + 8}\n")
    for bus_id, node in (("0000:17:00.0", 0), ("0000:c1:00.0", 1)):
        (pci_root / bus_id).mkdir(parents=True)
        (pci_root / bus_id / "numa_node").write_text(f"{node}\n")
    return {"node_root": node_root, "cpu_root": cpu_root, "pci_root": pci_root}


def gpus(used0=0, used1=0):
    return [
        GpuInfo(0, "0000:17:00.0", used0, 98304),
        GpuInfo(1, "0000:c1:00.0", used1, 98304),
    ]


# ---------------------------------------------------------------------------
# CPUs and NUMA
# ---------------------------------------------------------------------------


def test_parse_cpu_list():
    assert parallel.parse_cpu_list("16-31,48-63") == set(range(16, 32)) | set(range(48, 64))
    assert parallel.parse_cpu_list("3") == {3}
    assert parallel.parse_cpu_list("0,2-3\n") == {0, 2, 3}
    assert parallel.parse_cpu_list("") == set()


def test_node_cpus_and_a_missing_node(machine):
    assert parallel.node_cpus(1, machine["node_root"]) == {4, 5, 6, 7, 12, 13, 14, 15}
    with pytest.raises(ResourceError, match=r"NUMA node 7 does not exist.*\[0, 1\]"):
        parallel.node_cpus(7, machine["node_root"])


def test_hyper_thread_siblings_form_one_physical_core(machine):
    cpus = parallel.node_cpus(1, machine["node_root"])
    cores = parallel.physical_cores(cpus, machine["cpu_root"])
    assert len(cores) == 4  # eight logical CPUs, four cores
    assert cores[0] == {4, 12} and cores[-1] == {7, 15}


def test_a_core_counts_when_only_one_of_its_threads_is_usable(machine):
    # CPUs 4, 5 and 14 belong to three different cores (their siblings 12, 13 and 6 are not
    # in the set), so each of them counts as one core.
    cores = parallel.physical_cores({4, 5, 14}, machine["cpu_root"])
    assert [min(core) for core in cores] == [4, 5, 14] and all(len(core) == 1 for core in cores)
    # CPUs 5 and 13 are the two threads of one core.
    assert len(parallel.physical_cores({5, 13}, machine["cpu_root"])) == 1


def test_physical_cores_without_topology_files_treats_each_cpu_as_a_core(tmp_path):
    assert len(parallel.physical_cores({1, 2, 3}, tmp_path / "nothing")) == 3


def test_usable_cpus_follow_the_affinity_mask(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 0})
    assert parallel.usable_cpus(1, machine["node_root"]) == {4, 5, 6}
    assert parallel.usable_cpus(None) == {4, 5, 6, 0}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    with pytest.raises(ResourceError, match="none of the CPUs of NUMA node 1"):
        parallel.usable_cpus(1, machine["node_root"])


def test_cpu_workers_auto_is_the_physical_core_count_and_a_number_is_capped():
    messages = []
    assert parallel.resolve_cpu_workers("auto", 16) == 16
    assert parallel.resolve_cpu_workers(4, 16) == 4
    assert parallel.resolve_cpu_workers(64, 16, messages.append) == 16
    assert messages and "16 physical cores" in messages[0]
    assert parallel.resolve_cpu_workers("auto", 0) == 1


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def test_pci_bus_ids_are_comparable_across_formats():
    assert parallel.normalise_pci_bus_id("00000000:C1:00.0") == "0000:c1:00.0"
    assert parallel.normalise_pci_bus_id("0000:C1:00.0") == "0000:c1:00.0"


def test_query_gpus_parses_nvidia_smi_output():
    def run(command, **kwargs):
        text = "0, 00000000:17:00.0, 3, 97887\n1, 00000000:C1:00.0, 25000, 97887\n"
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    found = parallel.query_gpus(run)
    assert [(g.index, g.pci_bus_id, g.used_mb, g.total_mb) for g in found] == [
        (0, "0000:17:00.0", 3, 97887), (1, "0000:c1:00.0", 25000, 97887),
    ]
    assert found[0].free_bytes == (97887 - 3) << 20


def test_query_gpus_without_nvidia_smi_or_on_error_finds_none():
    def missing(command, **kwargs):
        raise FileNotFoundError

    def failing(command, **kwargs):
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="no driver")

    assert parallel.query_gpus(missing) == [] and parallel.query_gpus(failing) == []


def test_select_gpus_modes():
    assert [g.index for g in parallel.select_gpus("auto", gpus(), 1024)] == [0, 1]
    assert [g.index for g in parallel.select_gpus("auto", gpus(used1=40000), 1024)] == [0]
    assert parallel.select_gpus("none", gpus(), 1024) == []
    assert [g.index for g in parallel.select_gpus((1,), gpus(), 1024)] == [1]
    assert [g.index for g in parallel.select_gpus((1, 0), gpus(), 1024)] == [1, 0]


def test_a_named_gpu_that_is_in_use_or_missing_is_an_error_not_a_fallback():
    with pytest.raises(ResourceError, match="GPU 1 is in use"):
        parallel.select_gpus((1,), gpus(used1=5000), 1024)
    with pytest.raises(ResourceError, match="GPU 5 does not exist"):
        parallel.select_gpus((5,), gpus(), 1024)
    with pytest.raises(ResourceError, match="no GPU is free"):
        parallel.select_gpus("auto", gpus(7000, 9000), 1024)
    # The threshold is configurable: 5000 MiB used is fine when the limit is 6000.
    assert [g.index for g in parallel.select_gpus((1,), gpus(used1=5000), 6000)] == [1]


def test_gpu_numa_node_from_sysfs(machine):
    assert parallel.gpu_numa_node("0000:c1:00.0", machine["pci_root"]) == 1
    assert parallel.gpu_numa_node("0000:ff:00.0", machine["pci_root"]) is None  # unknown
    (machine["pci_root"] / "0000:17:00.0" / "numa_node").write_text("-1\n")
    assert parallel.gpu_numa_node("0000:17:00.0", machine["pci_root"]) is None


# ---------------------------------------------------------------------------
# memory budgets
# ---------------------------------------------------------------------------


def test_auto_batch_sizes_stay_inside_the_memory_budget():
    for free_gib in (2, 8, 30, 90):
        for workers in (1, 2, 4):
            base, chunk = parallel.auto_batch_sizes(free_gib << 30, workers)
            budget = min(free_gib * (1 << 30) * parallel.GPU_MEMORY_FRACTION / workers,
                         parallel.PER_WORKER_CAP_BYTES)
            assert base * chunk * parallel.BYTES_PER_PAIR <= budget
            assert 1 <= chunk <= 1024


def test_auto_batch_sizes_shrink_with_more_workers_and_never_exceed_the_cap():
    one = parallel.auto_batch_sizes(90 << 30, 1)
    four = parallel.auto_batch_sizes(30 << 30, 4)
    assert four[0] * four[1] < one[0] * one[1]
    assert one[0] * one[1] * parallel.BYTES_PER_PAIR <= parallel.PER_WORKER_CAP_BYTES
    assert one[1] == 1024  # a roomy GPU gets the full query chunk


def test_auto_batch_sizes_refuse_a_hopeless_budget():
    with pytest.raises(ResourceError, match="too little"):
        parallel.auto_batch_sizes(10_000, 1)


def test_host_memory_check(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1000 kB\nMemAvailable:    2097152 kB\n")  # 2 GiB
    assert parallel.host_available_bytes(meminfo) == 2 << 30
    parallel.check_host_memory(1 << 30, 2 << 30)  # 1 GiB x 1.2 fits into 2 GiB
    with pytest.raises(ResourceError, match="needs about 1.9 GiB"):
        parallel.check_host_memory(int(1.9 * (1 << 30)), 2 << 30)
    meminfo.write_text("MemTotal: 1 kB\n")
    with pytest.raises(ResourceError):
        parallel.host_available_bytes(meminfo)


# ---------------------------------------------------------------------------
# the resolved plan
# ---------------------------------------------------------------------------


def resolve(machine, parallel_config, device="gpu", base_rows=None, chunk_rows=None, gpu_list=None,
            pins=None):
    return parallel.resolve_plan(
        parallel_config, device, base_rows, chunk_rows,
        query=lambda: gpus() if gpu_list is None else gpu_list,
        ordinal_of=lambda bus_id: {"0000:17:00.0": 0, "0000:c1:00.0": 1}[bus_id],
        pin=(pins.append if pins is not None else (lambda cpus: None)),
        **machine,
    )


def test_plan_for_gpu_1_on_numa_node_1(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    pins = []
    plan = resolve(machine, ParallelConfig(gpus=(1,), numa_node=1), pins=pins)
    assert [g.info.index for g in plan.gpus] == [1] and plan.gpus[0].ordinal == 1
    assert plan.numa_node == 1 and plan.cpu_workers == 4 and plan.physical_cores == 4
    assert pins == [{4, 5, 6, 7, 12, 13, 14, 15}]  # pinned to the whole node
    assert plan.gpu_cpus == ((4, 5, 6, 7, 12, 13, 14, 15),)
    assert plan.scan_workers == 1
    assert plan.base_batch_rows * plan.query_chunk_rows * parallel.BYTES_PER_PAIR <= (
        parallel.PER_WORKER_CAP_BYTES
    )


def test_numa_auto_follows_each_gpu(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    pins = []
    plan = resolve(machine, ParallelConfig(gpus=(0, 1), numa_node="auto"), pins=pins)
    assert plan.numa_node == 0 and pins == [{0, 1, 2, 3, 8, 9, 10, 11}]  # the main process: GPU 0's node
    assert plan.gpu_cpus == ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15))
    assert plan.scan_workers == 2


def test_two_workers_per_gpu_double_the_scan_workers_and_halve_the_budget(machine, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(16)))
    single = resolve(machine, ParallelConfig(gpus=(1,), numa_node="none"))
    double = resolve(machine, ParallelConfig(gpus=(1,), numa_node="none", workers_per_gpu=2))
    assert double.scan_workers == 2
    assert double.base_batch_rows * double.query_chunk_rows <= single.base_batch_rows * single.query_chunk_rows


def test_explicit_batch_sizes_are_kept_and_a_half_explicit_one_is_completed(machine):
    config = ParallelConfig(gpus=(1,), numa_node="none")
    plan = resolve(machine, config, base_rows=1234, chunk_rows=56)
    assert (plan.base_batch_rows, plan.query_chunk_rows) == (1234, 56)
    half = resolve(machine, config, base_rows=1234, chunk_rows=None)
    assert half.base_batch_rows == 1234 and half.query_chunk_rows == 1024


def test_a_cpu_build_ignores_gpu_settings_and_uses_default_batches(machine):
    # GPU 9 does not exist, but with device = cpu nobody looks at it.
    plan = resolve(machine, ParallelConfig(gpus=(9,), numa_node="none", cpu_workers=3), device="cpu")
    assert plan.gpus == () and plan.cpu_workers == 3 and plan.scan_workers == 3
    assert (plan.base_batch_rows, plan.query_chunk_rows) == (100_000, 1024)


def test_the_plan_refuses_what_the_machine_cannot_give(machine):
    with pytest.raises(ResourceError, match="GPU 1 is in use"):
        resolve(machine, ParallelConfig(gpus=(1,), numa_node="none"), gpu_list=gpus(used1=9000))
    with pytest.raises(ResourceError, match="NUMA node 3 does not exist"):
        resolve(machine, ParallelConfig(gpus=(1,), numa_node=3))
    with pytest.raises(ResourceError, match="no GPU"):
        resolve(machine, ParallelConfig(gpus=(1,), numa_node="none"), gpu_list=[])
    with pytest.raises(ResourceError, match="selects no GPU"):
        resolve(machine, ParallelConfig(gpus="none", numa_node="none"))


def test_parallel_config_validation():
    for bad in (dict(gpus=()), dict(gpus=(1, 1)), dict(gpus=(-1,)), dict(gpus="some"),
                dict(workers_per_gpu=0), dict(cpu_workers=0), dict(cpu_workers="many"),
                dict(numa_node=-1), dict(numa_node="x"), dict(gpu_max_used_mb=-1)):
        with pytest.raises(ConfigError):
            ParallelConfig(**bad)
    defaults = ParallelConfig()
    assert defaults.gpus == (1,) and defaults.numa_node == 1 and defaults.cpu_workers == "auto"


# ---------------------------------------------------------------------------
# worker processes
# ---------------------------------------------------------------------------


def test_environment_is_set_for_the_block_only(monkeypatch):
    monkeypatch.delenv("PROBE_VARIABLE", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with parallel.temporary_environment({"PROBE_VARIABLE": "x", "OMP_NUM_THREADS": "1"}):
        assert os.environ["PROBE_VARIABLE"] == "x" and os.environ["OMP_NUM_THREADS"] == "1"
    assert "PROBE_VARIABLE" not in os.environ and os.environ["OMP_NUM_THREADS"] == "8"


def test_spawned_workers_are_single_threaded_and_pinned():
    from worker_probe import probe

    cpu = min(os.sched_getaffinity(0))
    with parallel.spawn_pool(2, environment={"PROBE_VARIABLE": "seen"}, cpus=[cpu]) as pool:
        results = pool.map(probe, [0, 1, 2])
    assert all(r["omp"] == "1" and r["openblas"] == "1" for r in results)
    assert all(r["probe"] == "seen" and r["cpus"] == [cpu] for r in results)
    assert os.environ.get("PROBE_VARIABLE") is None  # the parent keeps its environment


def test_a_gpu_worker_sees_exactly_one_gpu_numbered_like_nvidia_smi():
    assert parallel.gpu_worker_environment(GpuInfo(1, "0000:c1:00.0", 0, 1)) == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
    }


def test_run_cpu_jobs_keeps_the_order_in_process_and_in_workers():
    from worker_probe import probe

    jobs = list(range(5))
    assert parallel.run_cpu_jobs(abs, jobs, 1, None) == jobs
    results = parallel.run_cpu_jobs(probe, jobs, 3, None)
    assert len(results) == 5 and all(r["omp"] == "1" for r in results)
