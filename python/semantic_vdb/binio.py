"""Binary file I/O for vectors, id lists and ground truth.

Vector files use the big-ann / cuVS "bin" layout:

    [uint32 num_rows][uint32 num_cols][num_rows * num_cols values, row-major]

The value type is implied by the file extension (``.u8bin`` = uint8, ``.fbin`` =
float32, ...). Ground-truth files hold, in one file:

    [uint32 num_queries][uint32 k][int32 ids (num_queries x k)][float32 dists (same shape)]

All integers are little-endian. Files are written to ``<name>.partial`` and renamed
only after the declared number of rows has been written, so a crashed writer never
leaves something that looks like a finished file.
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

if sys.byteorder != "little":
    raise ImportError("semantic_vdb.binio assumes a little-endian host")

# Two uint32 fields (rows, cols) precede the payload in every file.
HEADER_FORMAT = "<II"
HEADER_BYTES = struct.calcsize(HEADER_FORMAT)

UINT32_MAX = (1 << 32) - 1

# File extension <-> element type. Only types that real ANN datasets ship in.
_SUFFIX_TO_DTYPE = {
    ".fbin": np.dtype(np.float32),
    ".f16bin": np.dtype(np.float16),
    ".u8bin": np.dtype(np.uint8),
    ".i8bin": np.dtype(np.int8),
    ".ibin": np.dtype(np.int32),
}
_DTYPE_TO_SUFFIX = {dtype: suffix for suffix, dtype in _SUFFIX_TO_DTYPE.items()}

GROUNDTRUTH_ID_DTYPE = np.dtype(np.int32)
GROUNDTRUTH_DIST_DTYPE = np.dtype(np.float32)


def dtype_for_suffix(path: str | os.PathLike) -> np.dtype:
    """Return the element type implied by the file extension of ``path``."""
    suffix = Path(path).suffix
    if suffix not in _SUFFIX_TO_DTYPE:
        raise ValueError(
            f"unsupported bin extension {suffix!r}; expected one of {sorted(_SUFFIX_TO_DTYPE)}"
        )
    return _SUFFIX_TO_DTYPE[suffix]


def suffix_for_dtype(dtype: np.dtype | type) -> str:
    """Return the conventional file extension for an element type."""
    key = np.dtype(dtype)
    if key not in _DTYPE_TO_SUFFIX:
        raise ValueError(f"no bin extension is defined for dtype {key}")
    return _DTYPE_TO_SUFFIX[key]


@dataclass(frozen=True)
class BinHeader:
    num_rows: int
    num_cols: int


def read_header(path: str | os.PathLike, dtype: np.dtype | type) -> BinHeader:
    """Read the 8-byte header and check it against the actual file size.

    The size check (``8 + rows * cols * itemsize == file size``) catches truncated
    files, trailing garbage and a wrong ``dtype`` guess in a single comparison.
    """
    item_size = np.dtype(dtype).itemsize
    file_size = os.path.getsize(path)
    if file_size < HEADER_BYTES:
        raise ValueError(f"{path}: file is smaller than the {HEADER_BYTES}-byte header")

    with open(path, "rb") as file:
        num_rows, num_cols = struct.unpack(HEADER_FORMAT, file.read(HEADER_BYTES))

    expected_size = HEADER_BYTES + num_rows * num_cols * item_size
    if file_size != expected_size:
        raise ValueError(
            f"{path}: header says {num_rows} x {num_cols} of {np.dtype(dtype)} "
            f"({expected_size} bytes) but the file has {file_size} bytes"
        )
    return BinHeader(num_rows=num_rows, num_cols=num_cols)


class BinReader:
    """Random-access reader over a bin file (memory-mapped, read-only)."""

    def __init__(self, path: str | os.PathLike, dtype: np.dtype | type | None = None) -> None:
        self.path = Path(path)
        # Default to the element type implied by the extension.
        self.dtype = np.dtype(dtype) if dtype is not None else dtype_for_suffix(self.path)
        header = read_header(self.path, self.dtype)
        self.num_rows = header.num_rows
        self.num_cols = header.num_cols

        if self.num_rows * self.num_cols == 0:
            # np.memmap cannot map a zero-length payload; use an empty array instead.
            self._rows = np.empty((self.num_rows, self.num_cols), dtype=self.dtype)
        else:
            self._rows = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                offset=HEADER_BYTES,
                shape=(self.num_rows, self.num_cols),
            )

    def __enter__(self) -> "BinReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        # Dropping the reference releases the mapping.
        self._rows = np.empty((0, self.num_cols), dtype=self.dtype)

    def read_range(self, start: int, count: int) -> np.ndarray:
        """Return rows ``[start, start + count)`` as an in-memory copy."""
        if start < 0 or count < 0 or start + count > self.num_rows:
            raise IndexError(
                f"{self.path}: range [{start}, {start + count}) is outside 0..{self.num_rows}"
            )
        return np.array(self._rows[start : start + count])

    def read_block(self, start: int, count: int) -> np.ndarray:
        """Rows ``[start, start + count)`` read with large ``pread`` calls, not via the map.

        Same result as ``read_range``, but the disk is asked for the whole block at once. On a
        hard disk this streams at the sequential speed, while touching a memory map page by
        page is several times slower. Use it to sweep a big file from front to back.
        """
        if start < 0 or count < 0 or start + count > self.num_rows:
            raise IndexError(
                f"{self.path}: range [{start}, {start + count}) is outside 0..{self.num_rows}"
            )
        block = np.empty((count, self.num_cols), dtype=self.dtype)
        if block.size == 0:
            return block
        view = memoryview(block).cast("B")
        offset = HEADER_BYTES + start * self.num_cols * self.dtype.itemsize
        descriptor = os.open(self.path, os.O_RDONLY)
        try:
            done = 0
            while done < view.nbytes:
                got = os.preadv(descriptor, [view[done:]], offset + done)
                if got == 0:
                    raise OSError(f"{self.path}: the file ended before row {start + count}")
                done += got
        finally:
            os.close(descriptor)
        return block

    def read_rows(self, row_ids: np.ndarray) -> np.ndarray:
        """Return the rows listed in ``row_ids`` (any order, duplicates allowed)."""
        row_ids = np.asarray(row_ids)
        if row_ids.ndim != 1 or not np.issubdtype(row_ids.dtype, np.integer):
            raise ValueError("row_ids must be a 1-D integer array")
        if row_ids.size and (row_ids.min() < 0 or row_ids.max() >= self.num_rows):
            raise IndexError(f"{self.path}: row id outside 0..{self.num_rows - 1}")
        return np.array(self._rows[row_ids])

    def iter_batches(self, batch_rows: int) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(first_row, rows)`` pieces that together cover the whole file."""
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        for start in range(0, self.num_rows, batch_rows):
            count = min(batch_rows, self.num_rows - start)
            yield start, self.read_range(start, count)


class BinWriter:
    """Streaming writer: the total row count is declared up front.

    The file only becomes visible under its final name when ``close()`` finds that
    exactly ``num_rows`` rows were written.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        num_rows: int,
        num_cols: int,
        dtype: np.dtype | type | None = None,
    ) -> None:
        self.path = Path(path)
        self.dtype = np.dtype(dtype) if dtype is not None else dtype_for_suffix(self.path)
        if self.dtype != dtype_for_suffix(self.path):
            raise ValueError(
                f"{self.path}: dtype {self.dtype} does not match the file extension"
            )
        if not (0 <= num_rows <= UINT32_MAX and 0 <= num_cols <= UINT32_MAX):
            raise ValueError("row and column counts must fit in uint32")

        self.num_rows = num_rows
        self.num_cols = num_cols
        self._rows_written = 0
        self._partial_path = self.path.with_name(self.path.name + ".partial")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._partial_path, "wb")
        self._file.write(struct.pack(HEADER_FORMAT, num_rows, num_cols))

    def write_rows(self, rows: np.ndarray) -> None:
        """Append rows. The array must already have the file's element type."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != self.num_cols:
            raise ValueError(f"expected rows of shape (m, {self.num_cols}), got {rows.shape}")
        if rows.dtype != self.dtype:
            # No silent casting: a lossy cast would corrupt vectors unnoticed.
            raise ValueError(f"expected dtype {self.dtype}, got {rows.dtype}")
        if self._rows_written + rows.shape[0] > self.num_rows:
            raise ValueError("more rows written than declared in the header")

        np.ascontiguousarray(rows).tofile(self._file)
        self._rows_written += rows.shape[0]

    def close(self) -> None:
        """Finish the file; raises if the declared row count was not reached."""
        if self._file.closed:
            return
        self._file.close()
        if self._rows_written != self.num_rows:
            self._partial_path.unlink(missing_ok=True)
            raise ValueError(
                f"{self.path}: declared {self.num_rows} rows but wrote {self._rows_written}"
            )
        os.replace(self._partial_path, self.path)

    def __enter__(self) -> "BinWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            # A failed block must not leave a file that could be mistaken for valid data.
            self._file.close()
            self._partial_path.unlink(missing_ok=True)
            return
        self.close()


def write_array(path: str | os.PathLike, array: np.ndarray) -> None:
    """Write a whole 2-D array as one bin file."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    with BinWriter(path, array.shape[0], array.shape[1], array.dtype) as writer:
        writer.write_rows(array)


def read_array(path: str | os.PathLike, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Read a whole bin file into memory."""
    with BinReader(path, dtype) as reader:
        return reader.read_range(0, reader.num_rows)


def write_ids(path: str | os.PathLike, ids: np.ndarray) -> None:
    """Write a 1-D int32 id list as an ``n x 1`` ``.ibin`` file."""
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"expected a 1-D id list, got shape {ids.shape}")
    if ids.size and (ids.min() < np.iinfo(np.int32).min or ids.max() > np.iinfo(np.int32).max):
        raise ValueError("ids do not fit in int32")
    write_array(path, ids.astype(np.int32).reshape(-1, 1))


def read_ids(path: str | os.PathLike) -> np.ndarray:
    """Read an ``n x 1`` ``.ibin`` file back as a 1-D int32 array."""
    column = read_array(path, np.int32)
    if column.shape[1] != 1:
        raise ValueError(f"{path}: expected one column, found {column.shape[1]}")
    return column.reshape(-1)


@dataclass(frozen=True)
class GroundTruth:
    # neighbour_ids[q, r] is the global id of the r-th nearest vector of query q.
    neighbour_ids: np.ndarray  # shape (num_queries, k), int32
    # neighbour_dists[q, r] is its distance (squared L2, or negated inner product).
    neighbour_dists: np.ndarray  # shape (num_queries, k), float32


def write_groundtruth(
    path: str | os.PathLike, neighbour_ids: np.ndarray, neighbour_dists: np.ndarray
) -> None:
    """Write ids and distances into one ground-truth file."""
    if neighbour_ids.shape != neighbour_dists.shape or neighbour_ids.ndim != 2:
        raise ValueError("ids and distances must be 2-D arrays of the same shape")
    if neighbour_ids.dtype != GROUNDTRUTH_ID_DTYPE:
        raise ValueError(f"ids must be {GROUNDTRUTH_ID_DTYPE}, got {neighbour_ids.dtype}")
    if neighbour_dists.dtype != GROUNDTRUTH_DIST_DTYPE:
        raise ValueError(f"dists must be {GROUNDTRUTH_DIST_DTYPE}, got {neighbour_dists.dtype}")

    num_queries, k = neighbour_ids.shape
    path = Path(path)
    partial_path = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(partial_path, "wb") as file:
        file.write(struct.pack(HEADER_FORMAT, num_queries, k))
        # Ids first, then distances: the layout used by big-ann ground-truth files.
        np.ascontiguousarray(neighbour_ids).tofile(file)
        np.ascontiguousarray(neighbour_dists).tofile(file)
    os.replace(partial_path, path)


def read_groundtruth(path: str | os.PathLike) -> GroundTruth:
    """Read a ground-truth file written by ``write_groundtruth``."""
    file_size = os.path.getsize(path)
    if file_size < HEADER_BYTES:
        raise ValueError(f"{path}: file is smaller than the {HEADER_BYTES}-byte header")

    with open(path, "rb") as file:
        num_queries, k = struct.unpack(HEADER_FORMAT, file.read(HEADER_BYTES))
        expected_size = HEADER_BYTES + num_queries * k * (
            GROUNDTRUTH_ID_DTYPE.itemsize + GROUNDTRUTH_DIST_DTYPE.itemsize
        )
        if file_size != expected_size:
            raise ValueError(
                f"{path}: header says {num_queries} x {k} but the file has {file_size} bytes, "
                f"expected {expected_size}"
            )
        ids = np.fromfile(file, dtype=GROUNDTRUTH_ID_DTYPE, count=num_queries * k)
        dists = np.fromfile(file, dtype=GROUNDTRUTH_DIST_DTYPE, count=num_queries * k)

    return GroundTruth(
        neighbour_ids=ids.reshape(num_queries, k),
        neighbour_dists=dists.reshape(num_queries, k),
    )
