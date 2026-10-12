"""Add pools to a finished family: ``[workload] existing = extend``.

A run with this setting finds the family directory ``<base_set>/workloads/<name>/`` and

1. compares the pools of the configuration (target overlaps x group sizes) with the pools in
   ``family.json``: the pools that exist are **skipped**, the others are built;
2. refuses, naming the difference, when anything the pools have in common changed
   (``family.check_common``): only new target overlaps and new group sizes can be added;
3. builds the new pools in ``<name>.extending`` next to the family, which is never touched
   while this happens. The neighbours and PCA bases, the evaluation ground truth and the
   deleted ids come from ``<name>/shared`` (``shared_store.py``), so a new pool costs only its
   own generation, statistics and files;
4. **commits**: the new pool directories are moved into the family, the shared results are
   updated, and ``family.json`` is replaced LAST and atomically. A family whose
   ``family.json`` lists a pool therefore has that pool completely.

The commit is described in a journal file before anything is moved. If the process dies in the
middle, running the same command again finishes the journal (every step can be repeated), and
an interrupted build of the new pools continues where it stopped (``resume.py``).

Only one extension may run at a time: it holds a lock file next to the family. A lock left by
a process that no longer exists is replaced.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, List

from semantic_vdb import workload as workload_module
from semantic_vdb.base_set import BaseSet
from semantic_vdb.config import WorkloadConfig
from semantic_vdb.family import (
    ExistingFamily,
    FamilyError,
    check_common,
    common_fingerprint,
    merge_family_index,
    read_existing_family,
)
from semantic_vdb.layout import PERSISTENT_SHARED_DIR
from semantic_vdb.manifest import json_text, read_json, sha256_file
from semantic_vdb.resume import canonical, sync_path, write_text_durable
from semantic_vdb.shared_store import SharedStore
from semantic_vdb.workload import WorkloadError, _no_log

Log = Callable[[str], None]

JOURNAL_FILE = "journal.json"
FRAGMENT_FILE = "family_fragment.json"


# ---------------------------------------------------------------------------
# one extension at a time
# ---------------------------------------------------------------------------


def _process_exists(pid: int) -> bool:
    """True when a process with this id is alive (on this machine)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists but belongs to someone else
    return True


class ExtensionLock:
    """A lock file ``<name>.extending.lock`` that holds the process id of its owner."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def acquire(self) -> None:
        for _ in range(2):  # the second round is after a stale lock was removed
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner = self._owner()
                if owner is not None and _process_exists(owner):
                    raise WorkloadError(
                        f"another extension of this family is running (process {owner}, "
                        f"lock {self.path}); wait for it to finish"
                    ) from None
                # The owner is gone (killed or crashed): its lock is stale.
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w") as handle:
                handle.write(f"{os.getpid()}\n")
            return
        raise WorkloadError(f"cannot take the lock {self.path}")

    def release(self) -> None:
        if self._owner() == os.getpid():
            self.path.unlink(missing_ok=True)

    def _owner(self):
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None


# ---------------------------------------------------------------------------
# the commit (journal)
# ---------------------------------------------------------------------------


def _move_directory(source: Path, destination: Path) -> None:
    """Move a directory into the family; repeating a finished move is a no-op.

    A source that is gone while the destination exists means an earlier attempt of this same
    commit already moved it. Both existing is a conflict: something else created the
    destination, and nothing is overwritten.
    """
    if not source.exists():
        if destination.exists():
            return
        raise WorkloadError(f"{source} is missing and {destination} does not exist")
    if destination.exists():
        raise WorkloadError(
            f"{destination} already exists although the family does not list it; "
            f"remove or rename it, then run the extension again"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    sync_path(destination.parent)


def _write_text(path: Path, text: str) -> None:
    """Replace a small file atomically and durably."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_durable(path, text)


def _apply_journal(work_dir: Path, final_dir: Path, journal: dict, log: Log) -> None:
    """Carry out the commit described by ``journal``; every step may be repeated."""
    # 1. The new pool directories (o<overlap>/g<size>, with all their sets).
    for relative in journal["pool_moves"]:
        _move_directory(work_dir / relative, final_dir / relative)
        log(f"[extend] added the pool {relative}")
    # 2. The neighbours and bases, when the family had none (they were computed now).
    if journal["move_n2"]:
        _move_directory(
            work_dir / PERSISTENT_SHARED_DIR / "n2", final_dir / PERSISTENT_SHARED_DIR / "n2"
        )
    # 3. What the shared results now hold (the noise scale of every new overlap).
    _write_text(final_dir / PERSISTENT_SHARED_DIR / "cache.json", journal["cache_json"])
    # 4. The report of this run, then the index: family.json is the commit point.
    _write_text(final_dir / journal["report_name"], journal["report_json"])
    _write_text(final_dir / "family.json", journal["family_json"])


def _write_journal_and_commit(
    work_dir: Path,
    final_dir: Path,
    existing: ExistingFamily,
    config: WorkloadConfig,
    skipped: List[str],
    log: Log,
) -> None:
    """Plan the commit, write the journal, then carry it out and remove the working directory."""
    fragment = read_json(work_dir / FRAGMENT_FILE)
    merged_index = merge_family_index(existing.index, fragment, config)

    # The shared results after the run: the family's own plus what this run added.
    staged = SharedStore(work_dir / PERSISTENT_SHARED_DIR)
    cache = existing.store.read_cache() or {"format_version": 1, "calibrations": {}}
    move_n2 = "n2" not in cache and staged.has_n2()
    if move_n2:
        cache["n2"] = staged.read_cache()["n2"]
    calibrations = dict(cache.get("calibrations", {}))
    calibrations.update(staged.calibrations())
    cache["calibrations"] = calibrations

    report = read_json(work_dir / "build_report.json")
    report["pools_skipped"] = skipped
    report["family_json_sha256_before"] = sha256_file(final_dir / "family.json")
    number = 1 + len(list(final_dir.glob("build_report_extension_*.json")))

    journal = {
        "pool_moves": [pool["name"] for pool in fragment["pools"]],
        "move_n2": move_n2,
        "cache_json": json_text(canonical(cache)),
        "report_name": f"build_report_extension_{number}.json",
        "report_json": json_text(report),
        "family_json": json_text(merged_index),
    }
    # From here on the commit is a plan on disk: it can be repeated until it is complete.
    write_text_durable(work_dir / JOURNAL_FILE, json_text(journal))
    _apply_journal(work_dir, final_dir, journal, log)
    shutil.rmtree(work_dir, ignore_errors=True)


def _finish_unfinished_commit(work_dir: Path, final_dir: Path, log: Log) -> None:
    """A previous run died while it committed: complete its journal."""
    journal = json.loads((work_dir / JOURNAL_FILE).read_text(encoding="utf-8"))
    log(f"[extend] completing the commit that an earlier run started in {work_dir}")
    _apply_journal(work_dir, final_dir, journal, log)
    shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def extend_family(
    config: WorkloadConfig, base_set: BaseSet, final_dir: Path, log: Log = _no_log
) -> Path:
    """Add the pools of ``config`` that the family in ``final_dir`` does not have yet."""
    try:
        existing = read_existing_family(final_dir)
        planned = workload_module._plan_family(config, base_set)
        check_common(existing, common_fingerprint(config, base_set, planned.inputs))
    except FamilyError as error:
        raise WorkloadError(str(error)) from error

    wanted = [spec.name for spec in planned.specs]
    skipped = [name for name in wanted if name in existing.pool_names]
    missing = [name for name in wanted if name not in existing.pool_names]
    for name in skipped:
        log(f"[extend] the pool {name} exists: skipped")

    work_dir = final_dir.with_name(config.name + ".extending")
    lock = ExtensionLock(final_dir.with_name(config.name + ".extending.lock"))
    lock.acquire()
    try:
        if (work_dir / JOURNAL_FILE).is_file():
            # An earlier run built everything and died while it moved the pools in.
            _finish_unfinished_commit(work_dir, final_dir, log)
            return final_dir
        if not missing:
            log(f"[extend] nothing to add: all {len(wanted)} pools of the configuration exist")
            return final_dir

        fingerprint = workload_module._fingerprint(config, base_set)
        # The family must not have changed since an unfinished build of these pools started.
        fingerprint["extension"] = {
            "family_json_sha256": sha256_file(final_dir / "family.json"),
            "new_pools": missing,
        }
        resume = workload_module._open_work_dir(work_dir, fingerprint, log)
        try:
            workload_module._build_family(
                config, base_set, work_dir, log, resume, existing=existing, planned=planned
            )
            _write_journal_and_commit(work_dir, final_dir, existing, config, skipped, log)
        except BaseException:
            if not (work_dir / JOURNAL_FILE).is_file():
                workload_module._keep_or_remove(work_dir, resume, log)
            raise
        log(f"[extend] added {len(missing)} pools, skipped {len(skipped)}")
        return final_dir
    finally:
        lock.release()
