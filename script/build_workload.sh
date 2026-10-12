#!/usr/bin/env bash
# Build a family of workload sets (one per update percentage) from an .ini file.
#
# Usage:  script/build_workload.sh <workload.ini>
#
# With "device = gpu" in the .ini the exact ground truth runs on a GPU. On this shared
# machine run it only when the machine is free, for example:
#     eval "$(<fork>/scripts/claude_machine_guard.sh)"     # exports GPU_ID
#     CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU_ID}" script/build_workload.sh x.ini
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <workload.ini>" >&2
    exit 64
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/sukjoon/.local/miniconda3/envs/toucann/bin/python}"

PYTHONPATH="${ROOT}/python" "${PYTHON}" -m semantic_vdb workload --config "$1"
