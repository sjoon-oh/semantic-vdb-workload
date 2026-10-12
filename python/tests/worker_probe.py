"""A function for spawned test workers (a worker imports its function by module name)."""

import os


def probe(_):
    """What a spawned worker sees: thread-count variables, a test variable and its CPUs."""
    return {
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "openblas": os.environ.get("OPENBLAS_NUM_THREADS"),
        "probe": os.environ.get("PROBE_VARIABLE"),
        "cpus": sorted(os.sched_getaffinity(0)),
    }
