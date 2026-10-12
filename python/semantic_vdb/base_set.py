"""The shared base set: initial base vectors plus the fixed insertion sequence.

A base set is built once from a raw vector file and then referenced by any number of
workload sets (see ``workload.py``). It holds

    base.<ext>           initial base vectors; the id of a vector is its row in this file
    base_row_ids.ibin    raw-file row number of every base row
    insert_pool.<ext>    the first ``max_inserts`` vectors of the fixed random insert order
    insert_row_ids.ibin  raw-file row numbers of those vectors
    query_rows.<ext>     (optional) rows reserved as queries, with query_row_ids.ibin
    base_set.json        parameters, counts and hashes of the id files

Workloads insert the pool in order: the ``j``-th inserted vector is row ``j`` of
``insert_pool.<ext>`` and gets the global id ``num_base + j`` in every workload.

The raw rows are chosen first (``partition.py``) and then copied out of the raw file in ONE
sweep from the first row to the last (``_extract_rows``). A raw file of a billion vectors
lives on a hard disk, where a seek costs as much as reading hundreds of kilobytes: reading the
rows in the random order of the insertion sequence, or through a memory map, would take
hours, a front-to-back sweep takes the time of reading the file once.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from semantic_vdb import __version__
from semantic_vdb.binio import BinReader, BinWriter, read_ids, suffix_for_dtype, write_ids
from semantic_vdb.config import BaseSetConfig
from semantic_vdb.manifest import read_json, sha256_file, write_json
from semantic_vdb.partition import partition_rows

FORMAT_VERSION = 1
# int32 id files and the 32-bit id field of the ground-truth keys limit the raw file size.
MAX_RAW_ROWS = (1 << 31) - 1


class BaseSetError(RuntimeError):
    """A base set is missing files, inconsistent, or was modified after it was built."""


@dataclass(frozen=True)
class BaseSet:
    directory: Path
    metric: str
    dtype: np.dtype
    dim: int
    num_base: int
    num_query_rows: int
    num_inserts: int  # rows available in insert_pool (the base set's max_inserts)
    info: dict

    @property
    def suffix(self) -> str:
        return suffix_for_dtype(self.dtype)

    @property
    def base_row_ids_sha256(self) -> str:
        return self.info["files"]["base_row_ids.ibin"]["sha256"]

    @property
    def insert_row_ids_sha256(self) -> str:
        return self.info["files"]["insert_row_ids.ibin"]["sha256"]

    def open_base(self) -> BinReader:
        return BinReader(self.directory / f"base{self.suffix}")

    def open_insert_pool(self) -> BinReader:
        return BinReader(self.directory / f"insert_pool{self.suffix}")

    def open_query_rows(self) -> BinReader:
        if self.num_query_rows == 0:
            raise BaseSetError("this base set reserved no query rows (num_query_rows = 0)")
        return BinReader(self.directory / f"query_rows{self.suffix}")

    def insert_vectors(self, first: int, count: int) -> np.ndarray:
        """Vectors ``first .. first + count - 1`` of the insertion sequence."""
        if first < 0 or count < 0 or first + count > self.num_inserts:
            raise BaseSetError(
                f"insertion range [{first}, {first + count}) is outside the {self.num_inserts} "
                "vectors of this base set"
            )
        with self.open_insert_pool() as pool:
            return pool.read_range(first, count)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


# One read of the raw file covers this many bytes (the span between the first and the last
# row wanted in a stretch of the file).
SWEEP_CHUNK_BYTES = 256 << 20
# The insertion rows are collected in memory up to this size, in a temporary file beyond it.
IN_MEMORY_INSERT_BYTES = 4 << 30
# Progress is reported after every this many percent of the raw file.
PROGRESS_PERCENT = 5

Log = Callable[[str], None]


def _no_log(message: str) -> None:
    return None


def _rows_of(block: np.ndarray, block_first: int, row_ids: np.ndarray) -> np.ndarray:
    """The rows ``row_ids`` (ascending raw row numbers) of a block that starts at ``block_first``."""
    first, last = int(row_ids[0]), int(row_ids[-1])
    if last - first + 1 == len(row_ids):  # consecutive rows: a view, no copy
        return block[first - block_first : last - block_first + 1]
    return block[row_ids - block_first]


def _extract_rows(
    raw: BinReader,
    base_ids: np.ndarray,
    query_ids: np.ndarray,
    insert_ids: np.ndarray,
    work_dir: Path,
    suffix: str,
    chunk_rows: int,
    log: Log,
) -> None:
    """Write ``base``, ``query_rows`` and ``insert_pool`` by one front-to-back sweep of ``raw``.

    ``base_ids`` and ``query_ids`` are ascending, so their rows are written in the order they
    are met. ``insert_ids`` is in the fixed random insertion order: the rows are met in raw
    order and stored at their position of the insertion order, and written at the end.
    Stretches of the raw file that hold none of the wanted rows are not read at all.
    """
    for name, ids in (("base", base_ids), ("query", query_ids)):
        if len(ids) > 1 and not bool((np.diff(ids) > 0).all()):
            raise BaseSetError(f"the {name} row ids must be strictly ascending")
    dim, dtype = raw.num_cols, raw.dtype
    row_bytes = dim * dtype.itemsize

    # Positions of the insertion order, sorted by raw row: sorted_inserts[k] is the raw row
    # that belongs to insertion number order[k].
    order = np.argsort(insert_ids, kind="stable")
    sorted_inserts = insert_ids[order]
    if len(insert_ids) * row_bytes <= IN_MEMORY_INSERT_BYTES:
        inserts = np.empty((len(insert_ids), dim), dtype=dtype)
        temporary = None
    else:
        temporary = work_dir / "insert_pool.tmp"
        inserts = np.memmap(temporary, dtype=dtype, mode="w+", shape=(len(insert_ids), dim))

    started = time.perf_counter()
    reported = -1
    rows_read = 0
    try:
        with contextlib.ExitStack() as stack:
            base_writer = stack.enter_context(
                BinWriter(work_dir / f"base{suffix}", len(base_ids), dim, dtype)
            )
            query_writer = (
                stack.enter_context(BinWriter(work_dir / f"query_rows{suffix}", len(query_ids), dim, dtype))
                if len(query_ids) > 0
                else None
            )
            for chunk_first in range(0, raw.num_rows, chunk_rows):
                chunk_last = min(chunk_first + chunk_rows, raw.num_rows)
                b_lo, b_hi = np.searchsorted(base_ids, [chunk_first, chunk_last])
                q_lo, q_hi = np.searchsorted(query_ids, [chunk_first, chunk_last])
                i_lo, i_hi = np.searchsorted(sorted_inserts, [chunk_first, chunk_last])
                wanted = [
                    ids[lo:hi]
                    for ids, lo, hi in (
                        (base_ids, b_lo, b_hi), (query_ids, q_lo, q_hi), (sorted_inserts, i_lo, i_hi)
                    )
                    if hi > lo
                ]
                if wanted:
                    # Read only the span from the first to the last wanted row of this stretch.
                    span_first = int(min(ids[0] for ids in wanted))
                    span_last = int(max(ids[-1] for ids in wanted)) + 1
                    block = raw.read_block(span_first, span_last - span_first)
                    rows_read += span_last - span_first
                    if b_hi > b_lo:
                        base_writer.write_rows(_rows_of(block, span_first, base_ids[b_lo:b_hi]))
                    if q_hi > q_lo:
                        query_writer.write_rows(_rows_of(block, span_first, query_ids[q_lo:q_hi]))
                    if i_hi > i_lo:
                        inserts[order[i_lo:i_hi]] = block[sorted_inserts[i_lo:i_hi] - span_first]
                percent = 100 * chunk_last // raw.num_rows
                if percent // PROGRESS_PERCENT > reported:
                    reported = percent // PROGRESS_PERCENT
                    log(
                        f"[base-set] raw file sweep {percent}%: {rows_read} of {raw.num_rows} rows "
                        f"read in {time.perf_counter() - started:.0f}s"
                    )

        with BinWriter(work_dir / f"insert_pool{suffix}", len(insert_ids), dim, dtype) as pool_writer:
            for first in range(0, len(insert_ids), 1_000_000):
                pool_writer.write_rows(np.asarray(inserts[first : first + 1_000_000]))
    finally:
        del inserts
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_base_set(
    config: BaseSetConfig, batch_rows: Optional[int] = None, log: Log = _no_log
) -> BaseSet:
    """Create the base set directory ``<output_dir>/<name>`` (never overwrites one).

    ``batch_rows`` is the number of raw rows that one stretch of the sweep covers (default:
    about 256 MiB of the raw file); ``log`` receives progress messages.
    """
    final_dir = config.output_dir / config.name
    if final_dir.exists():
        raise FileExistsError(f"{final_dir} already exists; refusing to overwrite a base set")
    # Build next to the final location and rename at the end, so a crash leaves no
    # directory that looks like a finished base set.
    work_dir = config.output_dir / (config.name + ".building")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    try:
        _build_into(config, work_dir, batch_rows, log)
        os.replace(work_dir, final_dir)
    except BaseException:
        # Never leave a half-built directory behind, whatever went wrong.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    return load_base_set(final_dir)


def _build_into(
    config: BaseSetConfig, work_dir: Path, batch_rows: Optional[int], log: Log
) -> None:
    """Write all base set files into ``work_dir``."""
    with BinReader(config.raw_file) as raw:
        if raw.num_rows > MAX_RAW_ROWS:
            raise BaseSetError("raw files with 2^31 or more rows are not supported")
        partition = partition_rows(
            raw.num_rows,
            config.num_base,
            config.num_query_rows,
            config.max_inserts,
            config.seed,
            config.base_selection,
        )
        suffix = suffix_for_dtype(raw.dtype)

        id_files = {
            "base_row_ids.ibin": partition.base_row_ids,
            "insert_row_ids.ibin": partition.insert_row_ids,
        }
        if len(partition.query_row_ids) > 0:
            id_files["query_row_ids.ibin"] = partition.query_row_ids
        for name, ids in id_files.items():
            write_ids(work_dir / name, ids)

        # Vector files: the wanted rows are taken out of the raw file in one sweep.
        chunk_rows = batch_rows or max(1, SWEEP_CHUNK_BYTES // (raw.num_cols * raw.dtype.itemsize))
        log(
            f"[base-set] {len(partition.base_row_ids)} base + {len(partition.insert_row_ids)} insert "
            f"+ {len(partition.query_row_ids)} query rows out of {raw.num_rows}; "
            f"sweeping the raw file in stretches of {chunk_rows} rows"
        )
        _extract_rows(
            raw, partition.base_row_ids, partition.query_row_ids, partition.insert_row_ids,
            work_dir, suffix, chunk_rows, log,
        )

        raw_rows, raw_cols = raw.num_rows, raw.num_cols
        raw_dtype = raw.dtype

    # Only the id files are hashed: they identify the content, and hashing the vector
    # files of a 100M-vector base would be slow for little gain.
    files = {
        name: {"sha256": sha256_file(work_dir / name), "bytes": (work_dir / name).stat().st_size}
        for name in id_files
    }
    info = {
        "format_version": FORMAT_VERSION,
        "tool_version": __version__,
        "metric": config.metric,
        "dtype": str(raw_dtype),
        "dim": raw_cols,
        "raw_file": str(config.raw_file),
        "raw_file_rows": raw_rows,
        "raw_file_bytes": config.raw_file.stat().st_size,
        "seed": config.seed,
        "base_selection": config.base_selection,
        "num_base": len(partition.base_row_ids),
        "num_query_rows": len(partition.query_row_ids),
        "num_inserts": len(partition.insert_row_ids),
        "files": files,
    }
    write_json(work_dir / "base_set.json", info)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def load_base_set(directory: str | os.PathLike) -> BaseSet:
    """Open a base set and verify that it is complete and unmodified."""
    directory = Path(directory)
    info_path = directory / "base_set.json"
    if not info_path.exists():
        raise BaseSetError(f"{directory} has no base_set.json")
    info = read_json(info_path)
    if info.get("format_version") != FORMAT_VERSION:
        raise BaseSetError(f"{directory}: unsupported base set format {info.get('format_version')}")

    dtype = np.dtype(info["dtype"])
    suffix = suffix_for_dtype(dtype)

    # The id files identify the content; a different hash means the base set was altered.
    for name, expected in info["files"].items():
        path = directory / name
        if not path.exists():
            raise BaseSetError(f"{directory}: missing {name}")
        if sha256_file(path) != expected["sha256"]:
            raise BaseSetError(f"{directory}: {name} does not match the hash in base_set.json")

    # Every vector file must exist and agree with the counts recorded at build time.
    expected_rows = {
        f"base{suffix}": info["num_base"],
        f"insert_pool{suffix}": info["num_inserts"],
    }
    if info["num_query_rows"] > 0:
        expected_rows[f"query_rows{suffix}"] = info["num_query_rows"]
    for name, rows in expected_rows.items():
        path = directory / name
        if not path.exists():
            raise BaseSetError(f"{directory}: missing {name}")
        with BinReader(path, dtype) as reader:
            if reader.num_rows != rows or reader.num_cols != info["dim"]:
                raise BaseSetError(
                    f"{path}: has {reader.num_rows} x {reader.num_cols}, expected {rows} x {info['dim']}"
                )

    return BaseSet(
        directory=directory,
        metric=info["metric"],
        dtype=dtype,
        dim=info["dim"],
        num_base=info["num_base"],
        num_query_rows=info["num_query_rows"],
        num_inserts=info["num_inserts"],
        info=info,
    )


def read_base_row_ids(base_set: BaseSet) -> np.ndarray:
    return read_ids(base_set.directory / "base_row_ids.ibin")


def read_insert_row_ids(base_set: BaseSet) -> np.ndarray:
    return read_ids(base_set.directory / "insert_row_ids.ibin")


def read_query_row_ids(base_set: BaseSet) -> Optional[np.ndarray]:
    if base_set.num_query_rows == 0:
        return None
    return read_ids(base_set.directory / "query_row_ids.ibin")
