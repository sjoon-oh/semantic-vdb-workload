#!/usr/bin/env bash
# Build the shared base set from an .ini file.
#
# Usage:  script/build_base_set.sh <base_set.ini>
#
# Uses the conda environment "toucann" unless PYTHON is set. This only reads the raw file
# and writes the base set; it needs no GPU.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <base_set.ini>" >&2
    exit 64
fi

# Resolve the repository root from the location of this script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/sukjoon/.local/miniconda3/envs/toucann/bin/python}"

PYTHONPATH="${ROOT}/python" "${PYTHON}" -m semantic_vdb base-set --config "$1"
