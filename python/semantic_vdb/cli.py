"""Command line entry point.

    python -m semantic_vdb base-set --config base_set.ini
    python -m semantic_vdb workload --config workload.ini

Progress messages go to stderr; the path of the created directory is printed to stdout (for
``workload`` it is the family directory that holds one sub-directory per target overlap).

Exit codes: 0 done, 1 a refusal (an existing family without ``existing = extend``, an extension
that does not fit its family, damaged stored results), 2 a configuration error.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from semantic_vdb.base_set import build_base_set
from semantic_vdb.config import ConfigError, load_base_set_config, load_workload_config
from semantic_vdb.n2 import N2Error
from semantic_vdb.workload import WorkloadError, build_workload


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="semantic_vdb", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    base_parser = commands.add_parser("base-set", help="build the shared base set")
    base_parser.add_argument("--config", required=True, help="path of base_set.ini")

    workload_parser = commands.add_parser(
        "workload",
        help="build the workload sets of one update-percentage family (u0, u25, ...)",
    )
    workload_parser.add_argument("--config", required=True, help="path of workload.ini")

    args = parser.parse_args(argv)

    try:
        if args.command == "base-set":
            base_set = build_base_set(load_base_set_config(args.config), log=_log)
            print(base_set.directory)
        else:
            directory = build_workload(load_workload_config(args.config), log=_log)
            print(directory)
    except ConfigError as error:
        _log(f"configuration error: {error}")
        return 2
    except (WorkloadError, N2Error, FileExistsError) as error:
        # A refusal (an existing family, a configuration that does not fit the family it would
        # extend, damaged stored results) is a message for the user, not a stack trace.
        _log(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
