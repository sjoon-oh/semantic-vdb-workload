#!/usr/bin/env python3
"""Run the mutation checks: every deliberate bug must make the test-suite fail.

Usage (from the repository root, inside the ``toucann`` conda environment):

    python python/tests/mutation/run_mutations.py [--only B1,B2] [--scratch DIR] [--log FILE]
                                                  [--jobs N]

For each mutation the runner copies ``python/semantic_vdb`` and ``python/tests`` into a
scratch directory, applies the wrong snippet, runs pytest there and records whether the
suite failed (CAUGHT) or still passed (SURVIVED). The unmodified copy is run first and
must pass, otherwise "failure" would not prove anything. The real sources are never
modified.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON_DIR = HERE.parent.parent
sys.path.insert(0, str(HERE))

from mutations import MUTATIONS  # noqa: E402


def copy_tree(destination: Path) -> Path:
    """Copy the code base into ``destination`` with the repository's layout.

    The copy holds ``python/`` (package, tests without the mutation directory, conftest)
    and ``config/`` (the example files that some tests read). Returns the ``python/`` copy.
    """
    python_copy = destination / "python"
    python_copy.mkdir(parents=True)
    shutil.copytree(
        PYTHON_DIR / "semantic_vdb",
        python_copy / "semantic_vdb",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        PYTHON_DIR / "tests",
        python_copy / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "mutation"),
    )
    shutil.copy2(PYTHON_DIR / "conftest.py", python_copy / "conftest.py")
    shutil.copytree(PYTHON_DIR.parent / "config", destination / "config")
    return python_copy


def run_pytest(tree: Path) -> subprocess.CompletedProcess:
    """Run the whole suite inside one scratch copy."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated mutation ids")
    parser.add_argument("--scratch", default=None, help="scratch directory (default: temp)")
    parser.add_argument("--log", default=None, help="also write the report to this file")
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="mutations tested at the same time (every one has its own scratch copy)",
    )
    args = parser.parse_args()

    selected = [item for item in args.only.split(",") if item]
    mutations = [m for m in MUTATIONS if not selected or m.mutation_id in selected]
    if selected and len(mutations) != len(selected):
        known = {m.mutation_id for m in MUTATIONS}
        raise SystemExit(f"unknown mutation ids: {sorted(set(selected) - known)}")

    scratch_root = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="mut-"))
    # Start from an empty scratch directory: copies left by an earlier run could hide a
    # source change or a new test.
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True)
    report = []

    # 1. The untouched copy must pass; otherwise a failing mutant proves nothing.
    baseline_tree = copy_tree(scratch_root / "baseline")
    baseline = run_pytest(baseline_tree)
    report.append(f"baseline: {'PASS' if baseline.returncode == 0 else 'FAIL'}")
    if baseline.returncode != 0:
        print("\n".join(report))
        print(baseline.stdout[-3000:])
        return 2

    # 2. Apply each mutation to its own fresh copy (several at a time with --jobs).
    def evaluate(mutation):
        """Returns ``(report line, survived)`` for one mutation."""
        tree = copy_tree(scratch_root / mutation.mutation_id)
        edits = [(mutation.file, mutation.old, mutation.new), *mutation.also]
        for file_name, old_text, new_text in edits:
            target = tree / file_name
            source = target.read_text(encoding="utf-8")
            occurrences = source.count(old_text)
            if occurrences != 1:
                line = (
                    f"{mutation.mutation_id}: INVALID (snippet found {occurrences} times) "
                    f"- {mutation.description}"
                )
                return line, True
            target.write_text(source.replace(old_text, new_text), encoding="utf-8")

        result = run_pytest(tree)
        # pytest exit code 1 = tests failed; 0 = all passed (mutation not noticed).
        if result.returncode == 0:
            status, survived = "SURVIVED", True
        elif result.returncode == 1:
            status, survived = "CAUGHT", False
        else:
            # Collection or internal errors are not evidence of a caught bug.
            status, survived = f"ERROR(rc={result.returncode})", True
        return f"{mutation.mutation_id}: {status} - {mutation.description}", survived

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        outcomes = list(executor.map(evaluate, mutations))  # keeps the order of the registry

    survived = []
    for mutation, (line, did_survive) in zip(mutations, outcomes):
        report.append(line)
        if did_survive:
            survived.append(mutation.mutation_id)

    summary = (
        f"{len(mutations) - len(survived)}/{len(mutations)} mutations caught"
        + (f"; NOT caught: {', '.join(survived)}" if survived else "")
    )
    report.append(summary)
    text = "\n".join(report)
    print(text)
    if args.log:
        Path(args.log).write_text(text + "\n", encoding="utf-8")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
