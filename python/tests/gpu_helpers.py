"""Helpers for the tests that need a real, free GPU (skipped when there is none)."""

import pytest

from semantic_vdb import parallel
from semantic_vdb.config import ParallelConfig


def gpu_plan_or_skip(gpus=(1,), workers_per_gpu=1, base_rows=97, chunk_rows=13):
    """A real ParallelPlan for the named GPUs, or skip the test if the machine cannot give it."""
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no usable GPU")
        return parallel.resolve_plan(
            ParallelConfig(gpus=gpus, workers_per_gpu=workers_per_gpu, numa_node="none"),
            "gpu", base_rows, chunk_rows,
        )
    except parallel.ResourceError as error:
        pytest.skip(f"GPU not available for the test: {error}")
    except Exception as error:  # noqa: BLE001 - no cupy, no driver, ...
        pytest.skip(f"no usable GPU: {error}")
