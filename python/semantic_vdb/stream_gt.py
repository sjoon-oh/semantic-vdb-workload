"""Follow-up stage: the exact answers of the stream queries of a finished family.

A workload build writes the queries of every stream (the requests of the Zipf and
``uniform_unique`` streams) but not their answers: the answers need one scan of the whole
universe per distinct query vector, which on a large data set takes hours to days. This
stage computes them afterwards, for a family that already exists, and can be repeated:

1. **Scope.** The sets of the pools named in ``[groundtruth] stream_pools`` (default ``all``)
   are looked at; a (set, stream) pair that already has its answers is skipped. A pair whose
   answers were made for another ``stream_k`` stops the stage (nothing is deleted).
2. **Distinct vectors.** The query files of the pending pairs are read and every query vector
   gets a 128-bit digest of its bytes. Equal vectors share one list, whichever pool, set or
   stream they come from, so nothing has to be known about how the pools were generated.
3. **Lists.** For every vector that has no list yet the universe is scanned once and its
   ``L`` nearest entries are kept (``universe.py``). The lists are stored in segments below
   ``<family>/shared/stream_lists/``; a finished segment is never recomputed, so an
   interrupted stage continues with the next one.
4. **Answers.** For every pending pair and step the answer is read out of the lists (keep the
   entries that are alive in that set at that step, take the first ``stream_k``) and written
   to ``groundtruth/<stream>/step_NNNNN.bin``. The files are written into a ``.tmp_`` folder,
   renamed, and only then recorded in the ``workload.json`` of the set, which is the commit
   point of the pair.

The ids in the answer files are the same ids as everywhere else (base row ``i`` is id ``i``,
the ``j``-th inserted vector is ``num_base + j``); the digests only find the list of a query.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from semantic_vdb import __version__
from semantic_vdb.base_set import BaseSet
from semantic_vdb.binio import BinReader, read_ids, write_groundtruth
from semantic_vdb.config import WorkloadConfig
from semantic_vdb.extend import ExtensionLock
from semantic_vdb.layout import (
    PERSISTENT_SHARED_DIR,
    STREAM_LISTS_DIR,
    TMP_PREFIX,
    BoundedWriter,
    hash_files,
)
from semantic_vdb.manifest import json_text, read_json, sha256_file, write_json
from semantic_vdb.parallel import ParallelPlan, check_host_memory, resolve_plan
from semantic_vdb.resume import sync_path, write_text_durable
from semantic_vdb.universe import AnswerSource, choose_list_length, scan_lists
from semantic_vdb.workload import WorkloadError

Log = Callable[[str], None]

# Query vectors per stored segment of lists: one pass over the universe each, so a segment is
# also the unit of progress that survives an interruption. A multiple of the usual
# query_chunk_rows (1024).
SEGMENT_VECTORS = 1 << 18
SEGMENT_FORMAT = 1
DIGEST_BYTES = 16  # 128-bit digest of a query vector
# Length of the window over which the "recent" speed of the progress lines is measured.
RECENT_WINDOW_SECONDS = 300.0


class StreamGtError(WorkloadError):
    """The stream answers cannot be computed for this family and configuration."""


# ---------------------------------------------------------------------------
# digests of query vectors
# ---------------------------------------------------------------------------


def row_digests(rows: np.ndarray) -> bytes:
    """The 16-byte digests of all rows, concatenated (row ``i`` is ``result[16*i : 16*i+16]``).

    The digest covers the bytes of the row and its element type and length, so a vector of
    another dtype or dimension can never be mistaken for it.
    """
    rows = np.ascontiguousarray(rows)
    if rows.ndim != 2:
        raise ValueError("expected a 2-D array of query vectors")
    prefix = f"{rows.dtype.str}:{rows.shape[1]}:".encode()
    raw = rows.tobytes()
    stride = rows.shape[1] * rows.dtype.itemsize
    blake = hashlib.blake2b
    return b"".join(
        blake(prefix + raw[start : start + stride], digest_size=DIGEST_BYTES).digest()
        for start in range(0, len(raw), stride)
    )


def digests_as_array(joined: bytes) -> np.ndarray:
    """``(n, 2)`` uint64 view of concatenated digests (two halves of 8 bytes each)."""
    return np.frombuffer(joined, dtype="<u8").reshape(-1, 2)


class DigestIndex:
    """Position of a digest in a collection, found by a binary search over the sorted digests."""

    def __init__(self, digests: np.ndarray) -> None:
        digests = np.asarray(digests, dtype=np.uint64).reshape(-1, 2)
        # Sorted by the first half, then by the second; ``position`` maps back to the input.
        order = np.lexsort((digests[:, 1], digests[:, 0]))
        self._first = np.ascontiguousarray(digests[order, 0])
        self._second = np.ascontiguousarray(digests[order, 1])
        self._position = order.astype(np.int64)

    def __len__(self) -> int:
        return len(self._first)

    def lookup(self, digests: np.ndarray) -> np.ndarray:
        """Input position of every digest, or -1 where it is not in the collection."""
        digests = np.asarray(digests, dtype=np.uint64).reshape(-1, 2)
        result = np.full(len(digests), -1, dtype=np.int64)
        if len(self._first) == 0 or len(digests) == 0:
            return result
        low = np.searchsorted(self._first, digests[:, 0], side="left")
        high = np.searchsorted(self._first, digests[:, 0], side="right")
        run = high - low

        # Almost always exactly one entry has the same first half: compare the second half.
        single = np.flatnonzero(run == 1)
        match = self._second[low[single]] == digests[single, 1]
        result[single[match]] = self._position[low[single[match]]]

        # Two different vectors with the same first half are possible (2^-64 per pair):
        # look through that short run for the second half.
        for index in np.flatnonzero(run > 1):
            for candidate in range(low[index], high[index]):
                if self._second[candidate] == digests[index, 1]:
                    result[index] = self._position[candidate]
                    break
        return result


# ---------------------------------------------------------------------------
# the stored lists
# ---------------------------------------------------------------------------


def universe_description(base_set: BaseSet, universe_size: int, list_length: int) -> dict:
    """What the lists of a store depend on: the universe, the metric and the list length."""
    return {
        "num_base": base_set.num_base,
        "universe_size": int(universe_size),
        "metric": base_set.metric,
        "dtype": str(base_set.dtype),
        "dim": base_set.dim,
        "base_row_ids_sha256": base_set.base_row_ids_sha256,
        "insert_row_ids_sha256": base_set.insert_row_ids_sha256,
        "list_length": int(list_length),
    }


def _describe_universe_difference(stored: dict, current: dict) -> str:
    keys = sorted(set(stored) | set(current))
    return ", ".join(
        f"{key}: was {stored.get(key)}, now {current.get(key)}"
        for key in keys
        if stored.get(key) != current.get(key)
    )


@dataclass
class Segment:
    directory: Path
    count: int
    keys: np.ndarray  # (count, 2) uint64 digests, in memory
    ids: np.ndarray  # (count, L) int32, memory-mapped
    dists: np.ndarray  # (count, L) float32, memory-mapped


class ListStore:
    """The universe lists of the stream queries, in segments below one directory."""

    FILES = ("keys.npy", "ids.npy", "dists.npy")

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.segments: List[Segment] = []
        self._offsets = np.zeros(1, dtype=np.int64)  # first global row of every segment
        self._index: Optional[DigestIndex] = None

    # ---- opening ----------------------------------------------------------------------

    def open(self, universe: dict, workers: int, batch: Optional[dict] = None) -> None:
        """Load the finished segments; refuse any that were made for another universe.

        Folders that a stage left unfinished (``.tmp_`` prefix) are removed. Every file is
        checked against the size and sha256 recorded when its segment was written.
        """
        self.segments = []
        if not self.directory.exists():
            self._refresh()
            return
        for leftover in sorted(self.directory.glob(TMP_PREFIX + "*")):
            shutil.rmtree(leftover, ignore_errors=True)

        names = sorted(path for path in self.directory.glob("segment_*") if path.is_dir())
        checks: List[Tuple[Path, Path, str]] = []
        for directory in names:
            meta = read_json(directory / "meta.json")
            if meta["universe"] != universe:
                raise StreamGtError(
                    f"{directory} holds lists made for another universe or list length "
                    f"({_describe_universe_difference(meta['universe'], universe)}); the stream "
                    f"answers cannot be mixed. Delete {self.directory} to compute them again."
                )
            if batch is not None and meta.get("batch") not in (None, batch):
                raise StreamGtError(
                    f"{directory} was made with other batch sizes ({meta['batch']}, now {batch}); "
                    "for float data they change the last bits of the distances. Set "
                    "[groundtruth] base_batch_rows and query_chunk_rows to the first values."
                )
            for name in self.FILES:
                checks.append((directory, directory / name, meta["files"][name]))
        self._verify(checks, workers)

        for directory in names:
            meta = read_json(directory / "meta.json")
            self.segments.append(
                Segment(
                    directory,
                    int(meta["count"]),
                    np.load(directory / "keys.npy"),
                    np.load(directory / "ids.npy", mmap_mode="r"),
                    np.load(directory / "dists.npy", mmap_mode="r"),
                )
            )
        self._refresh()

    @staticmethod
    def _verify(checks: List[Tuple[Path, Path, str]], workers: int) -> None:
        def check(item) -> Optional[str]:
            directory, path, recorded = item
            if not path.is_file() or path.stat().st_size != recorded["bytes"]:
                return f"{path} is missing or has another size than recorded"
            if sha256_file(path) != recorded["sha256"]:
                return f"{path} does not match its recorded sha256"
            return None

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            problems = [text for text in executor.map(check, checks) if text]
        if problems:
            raise StreamGtError(
                "damaged stream lists: " + "; ".join(problems[:3])
                + f" (and {len(problems) - 3} more)" * (len(problems) > 3)
                + ". Delete the segment folder to compute its lists again."
            )

    def _refresh(self) -> None:
        counts = [segment.count for segment in self.segments]
        self._offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        keys = (
            np.concatenate([segment.keys for segment in self.segments])
            if self.segments
            else np.empty((0, 2), dtype=np.uint64)
        )
        self._index = DigestIndex(keys)

    # ---- reading ------------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def find(self, digests: np.ndarray) -> np.ndarray:
        """Global list row of every digest, or -1 where the store has no list for it."""
        return self._index.lookup(digests)

    def gather(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The lists ``(ids, dists)`` of the given global rows, in the given order."""
        length = self.segments[0].ids.shape[1]
        ids = np.empty((len(rows), length), dtype=np.int32)
        dists = np.empty((len(rows), length), dtype=np.float32)
        segment_of = np.searchsorted(self._offsets, rows, side="right") - 1
        for position in np.unique(segment_of):
            chosen = np.flatnonzero(segment_of == position)
            local = rows[chosen] - self._offsets[position]
            ids[chosen] = self.segments[position].ids[local]
            dists[chosen] = self.segments[position].dists[local]
        return ids, dists

    # ---- writing --------------------------------------------------------------------------

    def next_segment_name(self) -> str:
        return f"segment_{len(self.segments) + 1:06d}"

    def add_segment(
        self, keys: np.ndarray, scratch: Path, universe: dict, batch: Optional[dict]
    ) -> None:
        """Make the lists scanned into ``scratch`` (list_ids.npy, list_dists.npy) a segment.

        The files get their final names, the digests and a ``meta.json`` with sizes and
        sha256 are added, everything is flushed, and only then the folder gets the name that
        ``open`` recognises: a crash leaves either no segment or a complete one.
        """
        np.save(scratch / "keys.npy", np.ascontiguousarray(keys, dtype=np.uint64))
        os.replace(scratch / "list_ids.npy", scratch / "ids.npy")
        os.replace(scratch / "list_dists.npy", scratch / "dists.npy")
        (scratch / "list_queries.npy").unlink(missing_ok=True)
        files = {}
        for name in self.FILES:
            path = scratch / name
            sync_path(path)
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        meta = {
            "format": SEGMENT_FORMAT,
            "tool_version": __version__,
            "count": int(len(keys)),
            "universe": universe,
            "batch": batch,
            "files": files,
        }
        write_text_durable(scratch / "meta.json", json_text(meta))
        final = self.directory / self.next_segment_name()
        os.replace(scratch, final)
        sync_path(self.directory)
        self.segments.append(
            Segment(
                final, int(len(keys)), np.load(final / "keys.npy"),
                np.load(final / "ids.npy", mmap_mode="r"),
                np.load(final / "dists.npy", mmap_mode="r"),
            )
        )
        self._refresh()


# ---------------------------------------------------------------------------
# progress lines
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """``3h 02m``, ``12m 05s`` or ``41s`` (hours and minutes only from one hour on)."""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class Progress:
    """Prints a progress line at most every ``interval`` seconds, with speed and time left.

    The speed is measured on the last ``RECENT_WINDOW_SECONDS`` (and on the whole run); the
    time left uses the recent speed. Every number comes from the clock and the counts given
    to it. The clock is a parameter so that tests can drive it.
    """

    def __init__(
        self, log: Log, interval: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._log = log
        self._interval = interval
        self._clock = clock
        self._label = ""
        self._unit = ""
        self._total = 0
        self._done = 0
        self._start_done = 0
        self._started = 0.0
        self._last_printed = 0.0
        self._samples: Deque[Tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def start(self, label: str, total: int, unit: str, already_done: int = 0) -> None:
        with self._lock:
            now = self._clock()
            self._label, self._unit, self._total = label, unit, total
            self._done = already_done
            self._start_done = already_done
            self._started = self._last_printed = now
            self._samples = deque([(now, already_done)])

    def update(self, done: int, note: str = "") -> None:
        """Record progress; prints when the interval has passed since the last line."""
        with self._lock:
            now = self._clock()
            self._done = done
            self._samples.append((now, done))
            while len(self._samples) > 2 and now - self._samples[1][0] > RECENT_WINDOW_SECONDS:
                self._samples.popleft()
            if now - self._last_printed >= self._interval:
                self._print(now, note)

    def report_if_due(self, note: str = "") -> None:
        """Print a line if the interval has passed (called by a timer while work is running)."""
        with self._lock:
            now = self._clock()
            if now - self._last_printed >= self._interval:
                self._print(now, note)

    def finish(self, note: str = "") -> None:
        with self._lock:
            self._print(self._clock(), note)

    def _print(self, now: float, note: str) -> None:
        self._last_printed = now
        elapsed = now - self._started
        first_time, first_done = self._samples[0]
        recent_span = now - first_time
        advanced = self._done - first_done
        recent = advanced / recent_span if recent_span > 0 and advanced > 0 else 0.0
        overall = (self._done - self._start_done) / elapsed if elapsed > 0 else 0.0
        percent = 100.0 * self._done / self._total if self._total else 100.0
        left = (
            format_duration((self._total - self._done) / recent)
            if recent > 0 and self._done < self._total
            else ("0s" if self._done >= self._total else "unknown")
        )
        line = (
            f"[stream gt] {self._label} {self._done:,}/{self._total:,} ({percent:.1f}%) · "
            f"{recent:.1f} {self._unit}/s recent, {overall:.1f} overall · eta {left} · "
            f"elapsed {format_duration(elapsed)}"
        )
        self._log(line + (f" · {note}" if note else ""))


class Ticker:
    """Calls ``function`` every ``period`` seconds on a helper thread until it is closed."""

    def __init__(self, function: Callable[[], None], period: float = 1.0) -> None:
        self._function = function
        self._period = period
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stream-gt-progress", daemon=True)

    def __enter__(self) -> "Ticker":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.wait(self._period):
            self._function()


# ---------------------------------------------------------------------------
# which (set, stream) pairs there are
# ---------------------------------------------------------------------------


@dataclass
class StreamPair:
    """The answers of one stream of one set."""

    pool: str  # o50/g200
    set_name: str  # u25
    stream: str  # stream_zipf_0.9
    set_dir: Path
    manifest: dict  # the parsed workload.json of the set (shared by the pairs of one set)
    num_steps: int
    requests: int

    @property
    def label(self) -> str:
        return f"{self.pool}/{self.set_name} {self.stream}"

    @property
    def final_dir(self) -> Path:
        return self.set_dir / "groundtruth" / self.stream

    @property
    def tmp_dir(self) -> Path:
        return self.set_dir / "groundtruth" / (TMP_PREFIX + self.stream)


def _matching_pools(patterns: Tuple[str, ...], pool_names: List[str]) -> List[str]:
    """The pools named by ``stream_pools`` (``all`` or names / globs), in family order."""
    if patterns == ("all",):
        return list(pool_names)
    chosen: List[str] = []
    for pattern in patterns:
        matches = [name for name in pool_names if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            raise StreamGtError(
                f"[groundtruth] stream_pools: {pattern!r} matches no pool of the family "
                f"(pools: {', '.join(pool_names)})"
            )
        chosen.extend(name for name in matches if name not in chosen)
    return [name for name in pool_names if name in chosen]


def stream_state(manifest: dict, stream: str, stream_k: int) -> str:
    """``complete``, ``missing`` or ``mismatch`` for one stream of one set.

    A set made by an earlier version that computed the answers during the build has
    ``groundtruth.stream_groundtruth = true`` and no per-stream record: its streams are
    complete for the ``stream_k`` that the set recorded in its configuration.
    """
    groundtruth = manifest["groundtruth"]
    record = groundtruth.get("stream", {}).get(stream)
    if record is not None and record.get("complete"):
        return "complete" if record["stream_k"] == stream_k else "mismatch"
    if groundtruth.get("stream_groundtruth") and "stream" not in groundtruth:
        recorded_k = manifest["config"]["groundtruth"].get("stream_k")
        return "complete" if recorded_k == stream_k else "mismatch"
    return "missing"


def select_pairs(
    config: WorkloadConfig, family_dir: Path, index: dict, stream_k: int
) -> Tuple[List[StreamPair], List[StreamPair], List[str]]:
    """``(pending pairs, complete pairs, mismatch messages)`` for the pools in scope."""
    pool_names = [pool["name"] for pool in index["pools"]]
    in_scope = set(_matching_pools(config.groundtruth.stream_pools, pool_names))
    pending: List[StreamPair] = []
    complete: List[StreamPair] = []
    mismatches: List[str] = []
    for entry in index["sets"]:
        if entry["pool"] not in in_scope:
            continue
        set_dir = family_dir / entry["path"]
        manifest = read_json(set_dir / "workload.json")
        num_steps = len(manifest["schedule"]["inserts_per_step"])
        requests = int(manifest["update_set"]["reads"])
        for stream in manifest["streams"]:
            pair = StreamPair(
                entry["pool"], entry["set_name"], stream, set_dir, manifest, num_steps, requests
            )
            state = stream_state(manifest, stream, stream_k)
            if state == "complete":
                complete.append(pair)
            elif state == "mismatch":
                mismatches.append(pair.label)
            else:
                pending.append(pair)
    return pending, complete, mismatches


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


@dataclass
class StreamGtReport:
    pairs_answered: int
    pairs_skipped: int
    vectors_scanned: int
    vectors_reused: int
    fallback_requests: int
    seconds: float


def _sync_family_hashes(family_dir: Path) -> int:
    """Make ``family.json`` carry the current sha256 of every set's workload.json."""
    index_path = family_dir / "family.json"
    index = read_json(index_path)
    changed = 0
    for entry in index["sets"]:
        current = sha256_file(family_dir / entry["path"] / "workload.json")
        if entry.get("workload_json_sha256") != current:
            entry["workload_json_sha256"] = current
            changed += 1
    if changed:
        write_json(index_path, index)
    return changed


def _step_queries(pair: StreamPair, step: int, suffix: str) -> np.ndarray:
    reader = BinReader(pair.set_dir / pair.stream / f"step_{step:05d}{suffix}")
    try:
        return reader.read_range(0, reader.num_rows)
    finally:
        reader.close()


def _collect_new_vectors(
    pending: List[StreamPair], store: ListStore, suffix: str, progress: Progress
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Distinct query vectors of the pending pairs that the store has no list for.

    Returns ``(vectors, digests (n, 2), reused)``: the vectors in the order of their first
    request, their digests, and how many distinct vectors the store already covered.
    """
    first_seen: Dict[bytes, int] = {}
    covered: set = set()
    parts: List[np.ndarray] = []
    digest_parts: List[bytes] = []
    total_files = sum(pair.num_steps for pair in pending)
    progress.start("reading query files", total_files, "files")
    done_files = 0
    for pair in pending:
        for step in range(1, pair.num_steps + 1):
            rows = _step_queries(pair, step, suffix)
            joined = row_digests(rows)
            in_store = store.find(digests_as_array(joined)) >= 0
            new_positions = []
            for position in range(len(rows)):
                key = joined[position * DIGEST_BYTES : (position + 1) * DIGEST_BYTES]
                if in_store[position]:
                    covered.add(key)
                elif key not in first_seen:
                    first_seen[key] = len(first_seen)
                    new_positions.append(position)
            if new_positions:
                parts.append(rows[new_positions])
                digest_parts.append(
                    b"".join(
                        joined[p * DIGEST_BYTES : (p + 1) * DIGEST_BYTES] for p in new_positions
                    )
                )
            done_files += 1
            progress.update(done_files)
    progress.finish()
    if not parts:
        empty = np.empty((0, 0))
        return empty, np.empty((0, 2), dtype=np.uint64), len(covered)
    vectors = np.concatenate(parts, axis=0)
    digests = digests_as_array(b"".join(digest_parts)).copy()
    return vectors, digests, len(covered)


def _fsync_files(paths: List[Path], workers: int) -> None:
    def sync(path: Path) -> None:
        sync_path(path)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        list(executor.map(sync, paths))


def _answer_pair(
    pair: StreamPair,
    store: ListStore,
    answers: AnswerSource,
    stream_k: int,
    universe_size: int,
    suffix: str,
    plan: ParallelPlan,
) -> int:
    """Write the answer files of one pair into its ``.tmp_`` folder; returns fallback count."""
    schedule = pair.manifest["schedule"]
    deleted_mask = np.zeros(universe_size, dtype=bool)
    inserted = 0
    fallbacks = 0
    shutil.rmtree(pair.tmp_dir, ignore_errors=True)
    pair.tmp_dir.mkdir(parents=True)
    writer = BoundedWriter(plan.cpu_workers)
    try:
        for step in range(1, pair.num_steps + 1):
            # Order inside a step: delete, insert, then the queries of the step.
            delete_path = pair.set_dir / "updates" / f"delete_step_{step:05d}.ibin"
            if delete_path.exists():
                deleted_mask[read_ids(delete_path)] = True
            inserted += int(schedule["inserts_per_step"][step - 1])

            queries = _step_queries(pair, step, suffix)
            rows = store.find(digests_as_array(row_digests(queries)))
            if (rows < 0).any():
                raise StreamGtError(f"{pair.label}: a query of step {step} has no list")
            list_ids, list_dists = store.gather(rows)
            ids, dists, missing = answers.answer(
                list_ids, list_dists, lambda wanted, queries=queries: queries[wanted],
                stream_k, inserted, deleted_mask,
            )
            fallbacks += missing
            writer.submit(
                write_groundtruth, pair.tmp_dir / f"step_{step:05d}.bin", ids, dists
            )
    finally:
        writer.finish()
    return fallbacks


def _commit_pair(
    pair: StreamPair, fallbacks: int, stream_k: int, list_length: int, plan: ParallelPlan
) -> None:
    """Make the answers of a pair visible: rename the folder, then record it (the commit)."""
    files = sorted(pair.tmp_dir.glob("step_*.bin"))
    _fsync_files(files, plan.cpu_workers)
    sync_path(pair.tmp_dir)
    # A final folder without a record is the leftover of a stage that died between the rename
    # and the record: it is not trusted and is replaced.
    shutil.rmtree(pair.final_dir, ignore_errors=True)
    os.replace(pair.tmp_dir, pair.final_dir)
    sync_path(pair.final_dir.parent)

    final_files = sorted(pair.final_dir.glob("step_*.bin"))
    hashes = hash_files(final_files, plan.cpu_workers)
    manifest = pair.manifest
    groundtruth = manifest["groundtruth"]
    groundtruth.setdefault("stream", {})[pair.stream] = {
        "complete": True,
        "stream_k": stream_k,
        "list_length": list_length,
        "requests": pair.requests,
        "requests_answered_by_scan": fallbacks,
        "steps": pair.num_steps,
    }
    for path in final_files:
        relative = str(path.relative_to(pair.set_dir))
        manifest["files"][relative] = {"bytes": hashes[path][0], "sha256": hashes[path][1]}
    done = all(
        groundtruth["stream"].get(stream, {}).get("complete") for stream in manifest["streams"]
    )
    groundtruth["stream_groundtruth"] = bool(done)
    groundtruth["stream_k"] = stream_k
    groundtruth["stream_list_length"] = list_length
    write_json(pair.set_dir / "workload.json", manifest)


def run_stream_groundtruth(
    config: WorkloadConfig,
    base_set: BaseSet,
    family_dir: Path,
    log: Log,
    clock: Callable[[], float] = time.monotonic,
) -> StreamGtReport:
    """Compute the missing stream answers of a finished family (safe to repeat)."""
    started = clock()
    lock = ExtensionLock(family_dir.with_name(family_dir.name + ".extending.lock"))
    lock.acquire()
    try:
        return _run_locked(config, base_set, family_dir, log, clock, started)
    finally:
        lock.release()


def release_device_memory() -> None:
    """Give back the GPU memory that this process caches (and nothing else).

    The build runs in the same process as the stage that follows it. Its CUDA memory pool
    keeps the blocks of the build (10 GB were seen after a build on SIFT100M), and a GPU
    that shows that much memory in use is not "free" to ``resolve_plan``. Freed, the process
    holds only its CUDA context (about half a GiB).
    """
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


def _run_locked(config, base_set, family_dir, log, clock, started) -> StreamGtReport:
    gt = config.groundtruth
    index = read_json(family_dir / "family.json")
    # An earlier run may have stopped between the record of a set and the family index.
    _sync_family_hashes(family_dir)

    first_manifest = read_json(family_dir / index["sets"][0]["path"] / "workload.json")
    universe_size = int(first_manifest["groundtruth"]["universe_size"])
    universe_inserts = universe_size - base_set.num_base
    alive_fraction = base_set.num_base / universe_size
    list_length = choose_list_length(gt.stream_k, alive_fraction, universe_size)
    if base_set.num_base < gt.stream_k:
        raise StreamGtError(
            f"the base has {base_set.num_base} vectors, fewer than stream_k = {gt.stream_k}"
        )

    pending, complete, mismatches = select_pairs(config, family_dir, index, gt.stream_k)
    if mismatches:
        shown = ", ".join(mismatches[:3]) + (" ..." if len(mismatches) > 3 else "")
        raise StreamGtError(
            f"{len(mismatches)} answer sets exist for another stream_k than the configured "
            f"{gt.stream_k} ({shown}); delete their groundtruth/stream_* folders to compute "
            "them again, or set stream_k to the value they were made with"
        )
    log(
        f"[stream gt] scope: {len(pending) + len(complete)} (set, stream) pairs, "
        f"{len(complete)} already answered, {len(pending)} to do"
    )
    if not pending:
        _sync_family_hashes(family_dir)
        return StreamGtReport(0, len(complete), 0, 0, 0, clock() - started)

    release_device_memory()  # the build's cached blocks must not make the GPU look busy
    plan = resolve_plan(config.parallel, gt.device, gt.base_batch_rows, gt.query_chunk_rows, log)
    if plan.gpus:
        import cupy

        cupy.cuda.Device(plan.gpus[0].ordinal).use()
    float_batch = (
        None
        if np.issubdtype(base_set.dtype, np.integer)
        else {"base_batch_rows": plan.base_batch_rows, "query_chunk_rows": plan.query_chunk_rows}
    )
    universe = universe_description(base_set, universe_size, list_length)
    suffix = base_set.suffix
    base_path = base_set.directory / f"base{suffix}"
    insert_path = base_set.directory / f"insert_pool{suffix}" if universe_inserts > 0 else None

    store = ListStore(family_dir / PERSISTENT_SHARED_DIR / STREAM_LISTS_DIR)
    store.directory.mkdir(parents=True, exist_ok=True)
    store.open(universe, plan.cpu_workers, float_batch)

    progress = Progress(log, gt.progress_seconds, clock)
    vectors, digests, reused = _collect_new_vectors(pending, store, suffix, progress)
    num_new = len(digests)

    requests = sum(pair.requests for pair in pending)
    answer_bytes = requests * gt.stream_k * 8
    list_bytes = num_new * (list_length * 8 + DIGEST_BYTES)
    log(
        f"[stream gt] distinct query vectors without a list: {num_new:,} "
        f"({reused:,} already in the store); list length {list_length}, universe "
        f"{universe_size:,} vectors; disk needed: answers {answer_bytes / 2**30:.1f} GiB, "
        f"lists {list_bytes / 2**30:.1f} GiB"
    )
    free = shutil.disk_usage(family_dir).free
    if (answer_bytes + list_bytes) * 1.05 > free:
        raise StreamGtError(
            f"not enough disk space below {family_dir}: about "
            f"{(answer_bytes + list_bytes) / 2**30:.1f} GiB are needed, "
            f"{free / 2**30:.1f} GiB are free"
        )
    check_host_memory(int(vectors.nbytes + num_new * 160))

    # ---- lists ---------------------------------------------------------------------------
    if num_new:
        progress.start("lists", num_new, "vectors")
        with Ticker(lambda: progress.report_if_due("segment running")):
            for first in range(0, num_new, SEGMENT_VECTORS):
                last = min(first + SEGMENT_VECTORS, num_new)
                scratch = store.directory / (TMP_PREFIX + store.next_segment_name())
                shutil.rmtree(scratch, ignore_errors=True)
                scratch.mkdir()
                scan_lists(
                    vectors[first:last], list_length, base_path, insert_path, universe_inserts,
                    base_set.metric, plan, scratch, "list",
                )
                store.add_segment(digests[first:last], scratch, universe, float_batch)
                progress.update(last)
        progress.finish()
    del vectors

    # ---- answers -------------------------------------------------------------------------
    answers = AnswerSource(base_path, insert_path, base_set.num_base, base_set.metric, plan)
    fallback_total = 0
    try:
        progress.start("answers", len(pending), "pairs")
        for position, pair in enumerate(pending, start=1):
            fallbacks = _answer_pair(
                pair, store, answers, gt.stream_k, universe_size, suffix, plan
            )
            _commit_pair(pair, fallbacks, gt.stream_k, list_length, plan)
            fallback_total += fallbacks
            progress.update(position, pair.label)
        progress.finish()
    finally:
        answers.close()

    _sync_family_hashes(family_dir)
    seconds = clock() - started
    report = {
        "stage": "stream_groundtruth",
        "tool_version": __version__,
        "stream_k": gt.stream_k,
        "list_length": list_length,
        "pairs_answered": len(pending),
        "pairs_skipped": len(complete),
        "vectors_scanned": num_new,
        "vectors_reused": reused,
        "requests_answered_by_scan": fallback_total,
        "seconds": seconds,
        "parallel": plan.describe(),
    }
    number = 1 + len(list(family_dir.glob("build_report_stream_groundtruth_*.json")))
    write_json(family_dir / f"build_report_stream_groundtruth_{number}.json", report)
    log(f"[stream gt] done in {format_duration(seconds)}: {len(pending)} pairs answered")
    return StreamGtReport(
        len(pending), len(complete), num_new, reused, fallback_total, seconds
    )
