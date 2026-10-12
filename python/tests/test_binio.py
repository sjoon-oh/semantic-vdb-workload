"""Tests for semantic_vdb.binio.

The byte-layout tests build the expected file contents with ``struct`` directly, so they
do not depend on the writer being correct.
"""

import struct

import numpy as np
import pytest

from semantic_vdb import binio

ALL_DTYPES = [np.uint8, np.int8, np.float32, np.float16, np.int32]


def make_matrix(dtype, rows=5, cols=7, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return rng.normal(size=(rows, cols)).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=(rows, cols), endpoint=True).astype(dtype)


# ---------------------------------------------------------------------------
# extension <-> dtype mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_suffix_round_trip(dtype):
    suffix = binio.suffix_for_dtype(dtype)
    assert binio.dtype_for_suffix(f"some/dir/data{suffix}") == np.dtype(dtype)


def test_known_suffixes():
    assert binio.suffix_for_dtype(np.uint8) == ".u8bin"
    assert binio.suffix_for_dtype(np.int8) == ".i8bin"
    assert binio.suffix_for_dtype(np.float32) == ".fbin"
    assert binio.suffix_for_dtype(np.int32) == ".ibin"


def test_unsupported_suffix_and_dtype_are_rejected():
    with pytest.raises(ValueError):
        binio.dtype_for_suffix("data.csv")
    with pytest.raises(ValueError):
        binio.suffix_for_dtype(np.float64)


# ---------------------------------------------------------------------------
# byte layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_file_layout_matches_the_documented_format(tmp_path, dtype):
    matrix = make_matrix(dtype, rows=4, cols=3)
    path = tmp_path / f"m{binio.suffix_for_dtype(dtype)}"
    binio.write_array(path, matrix)

    raw = path.read_bytes()
    # Header: rows first, then cols, both little-endian uint32.
    assert struct.unpack("<II", raw[:8]) == (4, 3)
    # Payload: row-major values with no padding.
    expected_payload = matrix.astype(matrix.dtype.newbyteorder("<")).tobytes(order="C")
    assert raw[8:] == expected_payload
    assert len(raw) == 8 + 4 * 3 * np.dtype(dtype).itemsize


def test_groundtruth_file_layout(tmp_path):
    ids = np.array([[3, 1], [0, 2], [5, 4]], dtype=np.int32)
    dists = np.array([[0.5, 1.5], [2.0, 2.5], [3.0, 3.5]], dtype=np.float32)
    path = tmp_path / "gt.bin"
    binio.write_groundtruth(path, ids, dists)

    raw = path.read_bytes()
    assert struct.unpack("<II", raw[:8]) == (3, 2)
    # Ids come first, then distances.
    id_bytes = raw[8 : 8 + ids.nbytes]
    dist_bytes = raw[8 + ids.nbytes :]
    assert id_bytes == ids.tobytes()
    assert dist_bytes == dists.tobytes()
    assert len(raw) == 8 + ids.nbytes + dists.nbytes


# ---------------------------------------------------------------------------
# round trips and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_array_round_trip(tmp_path, dtype):
    matrix = make_matrix(dtype)
    path = tmp_path / f"m{binio.suffix_for_dtype(dtype)}"
    binio.write_array(path, matrix)
    loaded = binio.read_array(path)
    assert loaded.dtype == np.dtype(dtype)
    assert np.array_equal(loaded, matrix)


def test_empty_file_round_trip(tmp_path):
    path = tmp_path / "empty.u8bin"
    binio.write_array(path, np.empty((0, 16), dtype=np.uint8))
    with binio.BinReader(path) as reader:
        assert (reader.num_rows, reader.num_cols) == (0, 16)
        assert reader.read_range(0, 0).shape == (0, 16)
        assert list(reader.iter_batches(10)) == []


def test_truncated_and_padded_files_are_rejected(tmp_path):
    path = tmp_path / "m.u8bin"
    binio.write_array(path, make_matrix(np.uint8))
    good = path.read_bytes()

    path.write_bytes(good[:-1])
    with pytest.raises(ValueError):
        binio.BinReader(path)

    path.write_bytes(good + b"\x00")
    with pytest.raises(ValueError):
        binio.BinReader(path)

    path.write_bytes(good[:4])
    with pytest.raises(ValueError):
        binio.BinReader(path)


def test_wrong_dtype_guess_is_detected_by_the_size_check(tmp_path):
    path = tmp_path / "m.u8bin"
    binio.write_array(path, make_matrix(np.uint8, rows=6, cols=4))
    with pytest.raises(ValueError):
        binio.BinReader(path, np.float32)


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------


def test_read_range_and_read_rows(tmp_path):
    matrix = make_matrix(np.float32, rows=20, cols=5)
    path = tmp_path / "m.fbin"
    binio.write_array(path, matrix)

    with binio.BinReader(path) as reader:
        assert np.array_equal(reader.read_range(3, 4), matrix[3:7])
        # Unsorted ids with a duplicate must come back in the requested order.
        ids = np.array([7, 2, 7, 19, 0])
        assert np.array_equal(reader.read_rows(ids), matrix[ids])
        with pytest.raises(IndexError):
            reader.read_range(18, 5)
        with pytest.raises(IndexError):
            reader.read_rows(np.array([20]))
        with pytest.raises(IndexError):
            reader.read_rows(np.array([-1]))
        with pytest.raises(ValueError):
            reader.read_rows(np.array([1.5]))


def test_iter_batches_covers_every_row_once(tmp_path):
    matrix = make_matrix(np.uint8, rows=23, cols=4)
    path = tmp_path / "m.u8bin"
    binio.write_array(path, matrix)

    with binio.BinReader(path) as reader:
        pieces = list(reader.iter_batches(10))
    assert [start for start, _ in pieces] == [0, 10, 20]
    assert np.array_equal(np.concatenate([rows for _, rows in pieces]), matrix)


# ---------------------------------------------------------------------------
# streaming writer
# ---------------------------------------------------------------------------


def test_streaming_writer_equals_one_shot_write(tmp_path):
    matrix = make_matrix(np.uint8, rows=11, cols=6)
    one_shot = tmp_path / "a.u8bin"
    streamed = tmp_path / "b.u8bin"
    binio.write_array(one_shot, matrix)

    with binio.BinWriter(streamed, num_rows=11, num_cols=6) as writer:
        writer.write_rows(matrix[:4])
        writer.write_rows(matrix[4:5])
        writer.write_rows(matrix[5:])
    assert streamed.read_bytes() == one_shot.read_bytes()


def test_writer_leaves_no_file_when_row_count_is_wrong(tmp_path):
    path = tmp_path / "short.u8bin"
    writer = binio.BinWriter(path, num_rows=5, num_cols=3)
    writer.write_rows(make_matrix(np.uint8, rows=4, cols=3))
    with pytest.raises(ValueError):
        writer.close()
    assert not path.exists()
    assert not (tmp_path / "short.u8bin.partial").exists()


def test_writer_rejects_bad_input(tmp_path):
    path = tmp_path / "m.u8bin"
    with binio.BinWriter(path, num_rows=4, num_cols=3) as writer:
        with pytest.raises(ValueError):
            writer.write_rows(make_matrix(np.uint8, rows=2, cols=4))  # wrong width
        with pytest.raises(ValueError):
            writer.write_rows(make_matrix(np.float32, rows=2, cols=3))  # wrong dtype
        writer.write_rows(make_matrix(np.uint8, rows=4, cols=3))
        with pytest.raises(ValueError):
            writer.write_rows(make_matrix(np.uint8, rows=1, cols=3))  # too many rows


def test_writer_dtype_must_match_extension(tmp_path):
    with pytest.raises(ValueError):
        binio.BinWriter(tmp_path / "m.u8bin", 1, 1, np.float32)


def test_failed_block_leaves_nothing_behind(tmp_path):
    path = tmp_path / "m.u8bin"
    with pytest.raises(RuntimeError):
        with binio.BinWriter(path, num_rows=2, num_cols=2) as writer:
            writer.write_rows(make_matrix(np.uint8, rows=1, cols=2))
            raise RuntimeError("boom")
    assert not path.exists()
    assert not (tmp_path / "m.u8bin.partial").exists()


# ---------------------------------------------------------------------------
# id lists and ground truth
# ---------------------------------------------------------------------------


def test_id_list_round_trip(tmp_path):
    ids = np.array([5, 0, 99, 5, 2_000_000], dtype=np.int64)
    path = tmp_path / "ids.ibin"
    binio.write_ids(path, ids)
    loaded = binio.read_ids(path)
    assert loaded.dtype == np.int32
    assert np.array_equal(loaded, ids)

    with pytest.raises(ValueError):
        binio.write_ids(path, np.array([1 << 40]))
    with pytest.raises(ValueError):
        binio.write_ids(path, np.zeros((2, 2), dtype=np.int32))


def test_groundtruth_round_trip_and_validation(tmp_path):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 1000, size=(8, 10)).astype(np.int32)
    dists = rng.random(size=(8, 10)).astype(np.float32)
    path = tmp_path / "gt.bin"
    binio.write_groundtruth(path, ids, dists)

    loaded = binio.read_groundtruth(path)
    assert np.array_equal(loaded.neighbour_ids, ids)
    assert np.array_equal(loaded.neighbour_dists, dists)

    with pytest.raises(ValueError):
        binio.write_groundtruth(path, ids.astype(np.int64), dists)
    with pytest.raises(ValueError):
        binio.write_groundtruth(path, ids, dists.astype(np.float64))
    with pytest.raises(ValueError):
        binio.write_groundtruth(path, ids, dists[:, :5])

    raw = path.read_bytes()
    path.write_bytes(raw[:-4])
    with pytest.raises(ValueError):
        binio.read_groundtruth(path)


# ---------------------------------------------------------------------------
# read_block: the same rows as read_range, read with large pread calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,suffix", [(np.uint8, ".u8bin"), (np.float32, ".fbin"), (np.int32, ".ibin")])
def test_read_block_returns_the_same_rows_as_read_range(tmp_path, dtype, suffix):
    matrix = (np.arange(37 * 5).reshape(37, 5) % 251).astype(dtype)
    path = tmp_path / f"m{suffix}"
    binio.write_array(path, matrix)
    with binio.BinReader(path) as reader:
        for start, count in ((0, 37), (0, 1), (36, 1), (5, 20), (10, 0), (37, 0)):
            block = reader.read_block(start, count)
            assert block.shape == (count, 5) and block.dtype == np.dtype(dtype)
            assert np.array_equal(block, matrix[start : start + count])
            assert np.array_equal(block, reader.read_range(start, count))


def test_read_block_checks_its_range(tmp_path):
    binio.write_array(tmp_path / "m.u8bin", np.zeros((10, 3), dtype=np.uint8))
    with binio.BinReader(tmp_path / "m.u8bin") as reader:
        for start, count in ((-1, 2), (0, 11), (9, 2), (11, 0), (0, -1)):
            with pytest.raises(IndexError):
                reader.read_block(start, count)


def test_read_block_is_a_copy_that_does_not_change_the_file(tmp_path):
    matrix = np.arange(60, dtype=np.uint8).reshape(10, 6)
    binio.write_array(tmp_path / "m.u8bin", matrix)
    with binio.BinReader(tmp_path / "m.u8bin") as reader:
        block = reader.read_block(0, 10)
        block[:] = 0
        assert np.array_equal(reader.read_range(0, 10), matrix)
