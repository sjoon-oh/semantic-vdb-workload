#!/usr/bin/env bash
# Build the SIFT1B base set and its workload family in one go. The same script builds any
# other data set whose two configuration files are given (see BASE_INI and WORKLOAD_INI).
#
# Usage:  script/build_sift1b.sh [-n|--dry-run] [base|workload|all]
#
#   all        (default) base set, then workload; a base set that already exists is kept
#   base       only the base set          (config/sift1b_base_set.ini)
#   workload   only the workload family   (config/sift1b_workload.ini)
#   -n         print what would run and check the inputs, but build nothing
#
# Environment overrides:
#   PYTHON        interpreter with numpy and cupy   (default: conda env "toucann")
#   BASE_INI      base set configuration            (default: config/sift1b_base_set.ini)
#   WORKLOAD_INI  workload configuration            (default: config/sift1b_workload.ini)
#   LOG_DIR       where the log of this run goes    (default: output/logs)
#
# For SPACEV1B give its two files:
#     BASE_INI=config/spacev1b_base_set.ini WORKLOAD_INI=config/spacev1b_workload.ini \
#         script/build_sift1b.sh
# Check first that the machine is free (<fork>/scripts/claude_machine_guard.sh --check). The
# workload step pins itself to the NUMA node and the GPU of [parallel] in its .ini file.
#
# An interrupted workload step is continued by running the SAME command again: the finished
# stages are kept in <family>.building and skipped (see "중단과 재개" in README.md). If the
# configuration was changed in the meantime, the build stops and says what differs; delete
# the .building directory to start over. The base set step is not resumable (a half-built
# base set is removed), but an existing complete base set is reused. Start long builds in
# tmux or with nohup rather than in a plain ssh shell:
#     tmux new -s sift1b 'script/build_sift1b.sh'
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-${HOME}/.local/miniconda3/envs/toucann/bin/python}"
BASE_INI="${BASE_INI:-${ROOT}/config/sift1b_base_set.ini}"
WORKLOAD_INI="${WORKLOAD_INI:-${ROOT}/config/sift1b_workload.ini}"
LOG_DIR="${LOG_DIR:-${ROOT}/output/logs}"

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

DRY_RUN=0
STEP=all
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        base|workload|all) STEP="$1" ;;
        *) usage >&2; exit 64 ;;
    esac
    shift
done

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "error: $*" >&2; exit 1; }

# ---- inputs that must exist before anything is started ---------------------------------
[[ -x "${PYTHON}" ]] || die "PYTHON=${PYTHON} is not executable (set PYTHON to an env with numpy and cupy)"
if [[ "${STEP}" != workload ]]; then [[ -f "${BASE_INI}" ]] || die "missing ${BASE_INI}"; fi
if [[ "${STEP}" != base ]]; then [[ -f "${WORKLOAD_INI}" ]] || die "missing ${WORKLOAD_INI}"; fi

# Directory the base set step writes to (<output.dir>/<output.name>), read with the tool's own parser.
base_set_dir() {
    PYTHONPATH="${ROOT}/python" "${PYTHON}" -c '
import sys
from semantic_vdb.config import load_base_set_config
config = load_base_set_config(sys.argv[1])
print(config.output_dir / config.name)' "${BASE_INI}"
}

# ---- log of this run -------------------------------------------------------------------
if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${LOG_DIR}"
    LOG_FILE="${LOG_DIR}/sift1b_$(date '+%Y%m%d_%H%M%S').log"
    exec > >(tee -a "${LOG_FILE}") 2>&1
    log "log: ${LOG_FILE}"
fi
log "step=${STEP} dry_run=${DRY_RUN} python=${PYTHON}"

run() {
    log "run: $*"
    if [[ "${DRY_RUN}" -eq 0 ]]; then "$@"; fi
}

# ---- 1) base set -----------------------------------------------------------------------
if [[ "${STEP}" == all || "${STEP}" == base ]]; then
    BASE_DIR="$(base_set_dir)"
    if [[ -f "${BASE_DIR}/base_set.json" ]]; then
        # The tool never overwrites a base set, so an existing one is reused.
        log "base set already built, skipping: ${BASE_DIR}"
    else
        SECONDS=0
        run "${ROOT}/script/build_base_set.sh" "${BASE_INI}"
        [[ "${DRY_RUN}" -eq 1 ]] || log "base set done in ${SECONDS}s: ${BASE_DIR}"
    fi
fi

# ---- 2) workload family ----------------------------------------------------------------
if [[ "${STEP}" == all || "${STEP}" == workload ]]; then
    SECONDS=0
    run "${ROOT}/script/build_workload.sh" "${WORKLOAD_INI}"
    [[ "${DRY_RUN}" -eq 1 ]] || log "workload done in ${SECONDS}s"
fi
