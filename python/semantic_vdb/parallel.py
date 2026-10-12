"""Machine resources of a build: GPUs, NUMA node, physical cores, memory budgets, worker pools.

Everything that looks at the machine lives here, behind small functions that take the
places they read from (``/sys``, ``nvidia-smi``, ``/proc/meminfo``) as arguments, so that
the tests can feed them made-up machines.

Rules this module follows:

* A resource the user NAMES but the machine cannot give (a GPU that is in use, a NUMA node
  that does not exist) is an error. It is never skipped silently, because a silent fallback
  (for example to the CPU) would make a build many times slower without telling anyone.
* Worker processes are always started with ``spawn``. A process that has already used CUDA
  must not be forked, and a spawned worker starts from a clean state.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from semantic_vdb.config import ParallelConfig


class ResourceError(RuntimeError):
    """The machine cannot provide what the configuration asks for."""


SYSFS_NODE_ROOT = Path("/sys/devices/system/node")
SYSFS_CPU_ROOT = Path("/sys/devices/system/cpu")
SYSFS_PCI_ROOT = Path("/sys/bus/pci/devices")
PROC_MEMINFO = Path("/proc/meminfo")

# Peak device memory of one distance block, measured with the ground-truth scan: about 38
# bytes per (query, base vector) pair; 40 leaves a little room.
BYTES_PER_PAIR = 40
# Share of the free GPU memory the batch sizes may use, and a cap per worker so that a
# shared 96 GB card is not filled just because it is empty.
GPU_MEMORY_FRACTION = 0.6
PER_WORKER_CAP_BYTES = 8 << 30
# Host memory must exceed the estimated need by this factor.
HOST_MEMORY_MARGIN = 1.2


# ---------------------------------------------------------------------------
# CPUs and NUMA
# ---------------------------------------------------------------------------


def parse_cpu_list(text: str) -> Set[int]:
    """Parse a kernel CPU list such as ``16-31,48-63`` into a set of CPU numbers."""
    cpus: Set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def node_cpus(node: int, node_root: Path = SYSFS_NODE_ROOT) -> Set[int]:
    """The logical CPUs of a NUMA node (an error if the node does not exist)."""
    cpulist = node_root / f"node{node}" / "cpulist"
    if not cpulist.exists():
        existing = sorted(
            int(path.name[4:]) for path in node_root.glob("node[0-9]*") if path.name[4:].isdigit()
        )
        raise ResourceError(
            f"NUMA node {node} does not exist on this machine (nodes: {existing}); "
            "set [parallel] numa_node to one of them or to none"
        )
    return parse_cpu_list(cpulist.read_text())


def physical_cores(cpus: Set[int], cpu_root: Path = SYSFS_CPU_ROOT) -> List[Set[int]]:
    """Group logical CPUs into physical cores (hyper-thread siblings form one core).

    A sibling that is not in ``cpus`` is left out, so a core counts as soon as one of its
    threads is usable. Cores are returned in order of their lowest CPU number.
    """
    cores: Dict[frozenset, Set[int]] = {}
    for cpu in sorted(cpus):
        siblings_file = cpu_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        if siblings_file.exists():
            siblings = frozenset(parse_cpu_list(siblings_file.read_text()))
        else:
            # Without topology information every CPU is treated as its own core.
            siblings = frozenset({cpu})
        cores.setdefault(siblings, set()).add(cpu)
    return [cores[key] for key in sorted(cores, key=min)]


def usable_cpus(node: Optional[int], node_root: Path = SYSFS_NODE_ROOT) -> Set[int]:
    """CPUs this process may use: the node's CPUs (or all) intersected with the affinity."""
    allowed = set(os.sched_getaffinity(0))
    if node is None:
        return allowed
    usable = node_cpus(node, node_root) & allowed
    if not usable:
        raise ResourceError(
            f"none of the CPUs of NUMA node {node} is allowed for this process "
            "(taskset / cgroup restriction?)"
        )
    return usable


def resolve_cpu_workers(
    setting: Union[str, int], cores: int, log: Callable[[str], None] = lambda message: None
) -> int:
    """``auto`` is the number of usable physical cores; a number is capped at that."""
    if setting == "auto":
        return max(1, cores)
    if setting > cores:
        log(f"[parallel] cpu_workers = {setting} exceeds the {cores} physical cores; using {cores}")
        return max(1, cores)
    return int(setting)


def pin_to_cpus(cpus: Set[int]) -> None:
    """Restrict the calling process (and the processes it starts later) to ``cpus``."""
    os.sched_setaffinity(0, cpus)


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuInfo:
    index: int  # nvidia-smi index (PCI bus order)
    pci_bus_id: str  # normalised, e.g. "0000:c1:00.0"
    used_mb: int
    total_mb: int

    @property
    def free_bytes(self) -> int:
        return (self.total_mb - self.used_mb) << 20


def normalise_pci_bus_id(bus_id: str) -> str:
    """Bring ``00000000:C1:00.0`` and ``0000:c1:00.0`` to the same lower-case form."""
    domain, _, rest = bus_id.strip().lower().partition(":")
    return f"{domain[-4:]}:{rest}"


def query_gpus(run: Callable = subprocess.run) -> List[GpuInfo]:
    """Ask ``nvidia-smi`` for every GPU (an empty list if there is none)."""
    try:
        result = run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        index, bus_id, used, total = [field.strip() for field in line.split(",")]
        gpus.append(
            GpuInfo(int(index), normalise_pci_bus_id(bus_id), int(float(used)), int(float(total)))
        )
    return gpus


def select_gpus(
    setting: Union[str, Tuple[int, ...]], gpus: Sequence[GpuInfo], max_used_mb: int
) -> List[GpuInfo]:
    """Pick the GPUs to use; an unusable named GPU, or no free GPU at all, is an error."""
    by_index = {gpu.index: gpu for gpu in gpus}
    free = [gpu for gpu in gpus if gpu.used_mb < max_used_mb]

    if setting == "none":
        return []
    if setting == "auto":
        if not free:
            raise ResourceError(
                f"no GPU is free (all use at least {max_used_mb} MiB): "
                + ", ".join(f"GPU {gpu.index}: {gpu.used_mb} MiB" for gpu in gpus)
            )
        return free

    chosen = []
    for index in setting:
        if index not in by_index:
            raise ResourceError(
                f"GPU {index} does not exist on this machine (GPUs: {sorted(by_index)}); "
                "set [parallel] gpus to an existing GPU"
            )
        gpu = by_index[index]
        if gpu.used_mb >= max_used_mb:
            raise ResourceError(
                f"GPU {index} is in use ({gpu.used_mb} MiB used, limit {max_used_mb} MiB); "
                "wait for it, name another GPU, or raise [parallel] gpu_max_used_mb"
            )
        chosen.append(gpu)
    return chosen


def cupy_ordinal(pci_bus_id: str) -> int:
    """The cupy device number of the GPU with this PCI bus id."""
    import cupy

    for ordinal in range(cupy.cuda.runtime.getDeviceCount()):
        if normalise_pci_bus_id(cupy.cuda.runtime.deviceGetPCIBusId(ordinal)) == pci_bus_id:
            return ordinal
    raise ResourceError(f"cupy does not see the GPU with PCI bus id {pci_bus_id}")


def gpu_numa_node(pci_bus_id: str, pci_root: Path = SYSFS_PCI_ROOT) -> Optional[int]:
    """The NUMA node a GPU is attached to (None if the kernel does not say)."""
    numa_file = pci_root / pci_bus_id / "numa_node"
    if not numa_file.exists():
        return None
    node = int(numa_file.read_text().strip())
    return node if node >= 0 else None


# ---------------------------------------------------------------------------
# memory budgets
# ---------------------------------------------------------------------------


def auto_batch_sizes(
    free_bytes: int,
    workers: int,
    max_chunk: int = 1024,
    max_base: int = 1_000_000,
) -> Tuple[int, int]:
    """Choose ``(base_batch_rows, query_chunk_rows)`` for one worker from free GPU memory.

    A distance block of ``base_batch_rows x query_chunk_rows`` pairs needs about
    ``BYTES_PER_PAIR`` bytes per pair. The budget is a fraction of the free memory divided by
    the workers that share the card, and never more than ``PER_WORKER_CAP_BYTES``.
    """
    budget = min(free_bytes * GPU_MEMORY_FRACTION / max(workers, 1), PER_WORKER_CAP_BYTES)
    pairs = int(budget // BYTES_PER_PAIR)
    if pairs < 1024:
        raise ResourceError(
            f"only {int(budget)} bytes of GPU memory are available per worker, too little for "
            "the distance blocks; free the GPU or use fewer workers per GPU"
        )
    # Prefer the full query chunk and spend the rest of the budget on the base block.
    chunk = min(max_chunk, max(1, pairs // 1024))
    base = max(1, min(max_base, pairs // chunk))
    return base, chunk


def host_available_bytes(meminfo: Path = PROC_MEMINFO) -> int:
    """``MemAvailable`` in bytes."""
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ResourceError(f"{meminfo} has no MemAvailable line")


def check_host_memory(required_bytes: int, available_bytes: Optional[int] = None) -> None:
    """Stop before any data is read when the estimated need does not fit in RAM."""
    available = host_available_bytes() if available_bytes is None else available_bytes
    if required_bytes * HOST_MEMORY_MARGIN > available:
        raise ResourceError(
            f"the build needs about {required_bytes / 2**30:.1f} GiB of host memory "
            f"(x{HOST_MEMORY_MARGIN} margin) but only {available / 2**30:.1f} GiB is available; "
            "use a smaller total_requests, fewer group sizes or a machine with more memory"
        )


# ---------------------------------------------------------------------------
# the resolved plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedGpu:
    info: GpuInfo
    ordinal: int  # cupy device number in THIS process


@dataclass(frozen=True)
class ParallelPlan:
    """What the build will actually use, after looking at the machine."""

    device: str  # "cpu" or "gpu"
    gpus: Tuple[ResolvedGpu, ...]
    workers_per_gpu: int
    cpu_workers: int
    physical_cores: int
    numa_node: Optional[int]
    node_cpus: Optional[Tuple[int, ...]]  # CPUs this process is pinned to (None = not pinned)
    gpu_cpus: Tuple[Optional[Tuple[int, ...]], ...]  # CPUs for the workers of each GPU
    base_batch_rows: int
    query_chunk_rows: int

    @property
    def scan_workers(self) -> int:
        """Processes that share the ground-truth scans."""
        return len(self.gpus) * self.workers_per_gpu if self.device == "gpu" else self.cpu_workers

    def describe(self) -> dict:
        return {
            "device": self.device,
            "gpus": [gpu.info.index for gpu in self.gpus],
            "workers_per_gpu": self.workers_per_gpu,
            "cpu_workers": self.cpu_workers,
            "physical_cores": self.physical_cores,
            "numa_node": self.numa_node,
            "base_batch_rows": self.base_batch_rows,
            "query_chunk_rows": self.query_chunk_rows,
        }


def resolve_plan(
    parallel: ParallelConfig,
    device: str,
    base_batch_rows: Optional[int],
    query_chunk_rows: Optional[int],
    log: Callable[[str], None] = lambda message: None,
    *,
    query: Callable[[], List[GpuInfo]] = query_gpus,
    ordinal_of: Callable[[str], int] = cupy_ordinal,
    node_root: Path = SYSFS_NODE_ROOT,
    cpu_root: Path = SYSFS_CPU_ROOT,
    pci_root: Path = SYSFS_PCI_ROOT,
    pin: Callable[[Set[int]], None] = pin_to_cpus,
) -> ParallelPlan:
    """Look at the machine, validate the request and pin this process to its NUMA node."""
    # 1. GPUs. With device = cpu the GPU settings are not looked at at all.
    resolved_gpus: List[ResolvedGpu] = []
    if device == "gpu":
        all_gpus = query()
        if not all_gpus:
            raise ResourceError("device = gpu but nvidia-smi reports no GPU on this machine")
        selected = select_gpus(parallel.gpus, all_gpus, parallel.gpu_max_used_mb)
        if not selected:
            raise ResourceError("device = gpu but [parallel] gpus = none selects no GPU")
        resolved_gpus = [ResolvedGpu(gpu, ordinal_of(gpu.pci_bus_id)) for gpu in selected]

    # 2. NUMA node: a number, or the node of the first selected GPU, or none.
    numa_node: Optional[int]
    if parallel.numa_node == "none":
        numa_node = None
    elif parallel.numa_node == "auto":
        numa_node = (
            gpu_numa_node(resolved_gpus[0].info.pci_bus_id, pci_root) if resolved_gpus else None
        )
    else:
        numa_node = int(parallel.numa_node)

    # 3. CPUs: look at the node BEFORE pinning, then pin this process to the whole node.
    cpus = usable_cpus(numa_node, node_root)
    cores = physical_cores(cpus, cpu_root)
    cpu_workers = resolve_cpu_workers(parallel.cpu_workers, len(cores), log)
    node_cpu_list = None
    if numa_node is not None:
        node_cpu_list = tuple(sorted(cpus))

    # The workers of a GPU run on the CPUs of the node they are told to use: the one node
    # of a numbered setting, or, for "auto", the node each GPU is attached to.
    gpu_cpus: List[Optional[Tuple[int, ...]]] = []
    for gpu in resolved_gpus:
        if parallel.numa_node == "auto":
            gpu_node = gpu_numa_node(gpu.info.pci_bus_id, pci_root)
            gpu_cpus.append(
                tuple(sorted(usable_cpus(gpu_node, node_root))) if gpu_node is not None else None
            )
        else:
            gpu_cpus.append(node_cpu_list)
    if numa_node is not None:
        pin(cpus)
        log(f"[parallel] pinned to NUMA node {numa_node} ({len(cpus)} CPUs, {len(cores)} cores)")

    # 4. Batch sizes: explicit numbers are kept, "auto" comes from the free GPU memory.
    if base_batch_rows is not None and query_chunk_rows is not None:
        base_rows, chunk_rows = base_batch_rows, query_chunk_rows
    elif resolved_gpus:
        workers = parallel.workers_per_gpu
        smallest_free = min(gpu.info.free_bytes for gpu in resolved_gpus)
        auto_base, auto_chunk = auto_batch_sizes(smallest_free, workers)
        base_rows = auto_base if base_batch_rows is None else base_batch_rows
        chunk_rows = auto_chunk if query_chunk_rows is None else query_chunk_rows
    else:
        # CPU scans: moderate blocks keep every worker's temporaries small.
        base_rows = 100_000 if base_batch_rows is None else base_batch_rows
        chunk_rows = 1024 if query_chunk_rows is None else query_chunk_rows

    return ParallelPlan(
        device=device,
        gpus=tuple(resolved_gpus),
        workers_per_gpu=parallel.workers_per_gpu,
        cpu_workers=cpu_workers,
        physical_cores=len(cores),
        numa_node=numa_node,
        node_cpus=node_cpu_list,
        gpu_cpus=tuple(gpu_cpus),
        base_batch_rows=base_rows,
        query_chunk_rows=chunk_rows,
    )


# ---------------------------------------------------------------------------
# worker processes
# ---------------------------------------------------------------------------

# Thread counts of the numeric libraries. A worker must use ONE thread: the parallelism
# comes from the number of workers, and several threads per worker would also make the
# last bits of a result depend on the thread count.
SINGLE_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@contextlib.contextmanager
def temporary_environment(values: Dict[str, str]) -> Iterator[None]:
    """Set environment variables for a ``with`` block and restore them afterwards.

    Used around the creation of a worker pool: spawned workers copy the environment of
    their parent at start-up, before any of their own code runs.
    """
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialise_worker(cpus: Optional[Tuple[int, ...]]) -> None:
    # Runs once in every worker: restrict it to the NUMA node of the parent.
    if cpus:
        os.sched_setaffinity(0, set(cpus))


@contextlib.contextmanager
def spawn_pool(
    processes: int,
    environment: Optional[Dict[str, str]] = None,
    cpus: Optional[Sequence[int]] = None,
) -> Iterator["multiprocessing.pool.Pool"]:
    """A pool of ``processes`` spawned workers, single-threaded, optionally CPU-pinned."""
    context = multiprocessing.get_context("spawn")
    values = dict(SINGLE_THREAD_ENVIRONMENT)
    values.update(environment or {})
    # Pool() starts every worker in its constructor, so the environment only has to be
    # set while the constructor runs.
    with temporary_environment(values):
        pool = context.Pool(
            processes=processes,
            initializer=_initialise_worker,
            initargs=(tuple(cpus) if cpus else None,),
        )
    try:
        yield pool
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise


def gpu_worker_environment(gpu: GpuInfo) -> Dict[str, str]:
    """Make a worker see exactly one GPU, numbered like nvidia-smi does."""
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu.index)}


def split_ranges(count: int, parts: int) -> List[Tuple[int, int]]:
    """Cut ``range(count)`` into at most ``parts`` contiguous, near-equal, non-empty ranges."""
    parts = max(1, min(parts, count)) if count > 0 else 1
    edges = np.linspace(0, count, parts + 1).astype(np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def run_cpu_jobs(
    function: Callable[[Any], Any], jobs: Sequence[Any], workers: int, cpus: Optional[Sequence[int]]
) -> List[Any]:
    """Run ``function`` on every job, in this process or on ``workers`` spawned processes.

    The results keep the order of ``jobs``. Jobs and results must be picklable, and
    ``function`` must be defined at module level (a spawned worker imports it by name).
    """
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    with spawn_pool(min(workers, len(jobs)), cpus=cpus) as pool:
        return pool.map(function, list(jobs))
