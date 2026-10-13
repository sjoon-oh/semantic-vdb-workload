"""Names and paths of the family directory, hard links, and hashing of the output files.

    <base_set>/workloads/<name>/
        family.json  build_report.json
        shared/                  results that later runs reuse (see shared_store.py)
        o80/ o50/ ...            one directory per target overlap of the N2 noise
            g50/ g200/ ...       one pool per group size
                u0/ u25/ ...     one complete workload set per update percentage
        g1/                      given queries (no noise, so no overlap level)
        external/                externally grouped vectors (no noise either)

Files whose content is identical in several sets (the pool, the seeds, the evaluation
queries, the deleted ids, the evaluation ground truth) are written once and hard-linked into
the other sets. A hard link looks like an ordinary file; copying a set directory elsewhere
(``cp -r``) produces independent copies, so each set stays usable on its own.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from semantic_vdb.manifest import sha256_file

SHARED_DIR = "_shared"  # per-pool link sources inside the .building directory, removed at the end
PERSISTENT_SHARED_DIR = "shared"  # results of the family that stay and are reused (extension)
UNIVERSE_DIR = "_universe"  # memory-mapped universe lists, removed at the end
RESUME_DIR = "_resume"  # finished-stage markers of an unfinished build (see resume.py)
# Universe lists of the stream queries, below shared/ (see stream_gt.py), and the prefix of the
# directories that the follow-up stage fills before it renames them to their final names.
STREAM_LISTS_DIR = "stream_lists"
TMP_PREFIX = ".tmp_"


def overlap_dir_name(target_overlap: float) -> str:
    """Directory name of an overlap level: ``o`` + the percentage (``o80``; 12.5% is ``o12p5``).

    Same rule as the update sets (``u12p5``): the shortest exact decimal text of the
    percentage, so two different overlaps can never share a name.
    """
    percent = Decimal(repr(float(target_overlap))) * 100
    return "o" + format(percent.normalize(), "f").replace(".", "p")


def pool_dir_name(
    kind: str, group_size: Optional[int], overlap_name: Optional[str] = None
) -> str:
    """Directory of a pool relative to the family: ``o80/g200`` (N2), ``g1``, ``external``.

    Only N2 pools have an overlap level, because only they add noise.
    """
    if kind == "external":
        return "external"
    if group_size is None:
        raise ValueError(f"pool kind {kind!r} needs a group size")
    if kind == "n2":
        if overlap_name is None:
            raise ValueError("an N2 pool needs the name of its overlap level")
        return f"{overlap_name}/g{group_size}"
    return f"g{group_size}"


def link_or_copy(source: Path, destination: Path) -> bool:
    """Make ``destination`` the same file as ``source``; returns True for a hard link.

    A file system that cannot hard link (or a link across devices) gets a normal copy, so
    the build never fails just because of the link.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return True
    except OSError:
        shutil.copyfile(source, destination)
        return False


def link_tree(source_dir: Path, destination_dir: Path) -> int:
    """Link every file below ``source_dir`` to the same relative path below ``destination_dir``."""
    count = 0
    if not source_dir.exists():
        return count
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            link_or_copy(path, destination_dir / path.relative_to(source_dir))
            count += 1
    return count


def hash_files(paths: Iterable[Path], workers: int) -> Dict[Path, Tuple[int, str]]:
    """``{path: (size in bytes, sha256)}``; hard links to one file are hashed only once.

    ``hashlib`` releases the GIL while it hashes a large buffer, so threads give a real
    speed-up.
    """
    paths = list(paths)
    by_inode: Dict[Tuple[int, int], Path] = {}
    for path in paths:
        info = path.stat()
        by_inode.setdefault((info.st_dev, info.st_ino), path)

    def digest(item: Tuple[Tuple[int, int], Path]) -> Tuple[Tuple[int, int], str]:
        return item[0], sha256_file(item[1])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        hashes = dict(executor.map(digest, by_inode.items()))

    result: Dict[Path, Tuple[int, str]] = {}
    for path in paths:
        info = path.stat()
        result[path] = (info.st_size, hashes[(info.st_dev, info.st_ino)])
    return result


class BoundedWriter:
    """Run file-writing jobs on threads, but never let more than ``limit`` wait at once.

    The jobs hold the arrays they write, so an unbounded queue could keep a whole set in
    memory. ``submit`` blocks while ``limit`` jobs are pending; ``finish`` waits for the rest
    and raises the first error.
    """

    def __init__(self, workers: int, limit: int = 64) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self._limit = limit
        self._pending: List = []

    def submit(self, function, *args) -> None:
        self._pending.append(self._executor.submit(function, *args))
        while len(self._pending) >= self._limit:
            self._pending.pop(0).result()

    def finish(self) -> None:
        try:
            for future in self._pending:
                future.result()
        finally:
            self._pending = []
            self._executor.shutdown(wait=True)
