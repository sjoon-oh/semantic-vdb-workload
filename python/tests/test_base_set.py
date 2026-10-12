"""Tests for semantic_vdb.base_set."""

from pathlib import Path

import numpy as np
import pytest

from semantic_vdb import base_set as base_set_module
from semantic_vdb import binio
from semantic_vdb.base_set import BaseSetError
from semantic_vdb.config import BaseSetConfig

TOTAL_ROWS, DIM = 400, 6


def write_raw(tmp_path, dtype=np.uint8, name="raw.u8bin"):
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, 200, size=(TOTAL_ROWS, DIM)).astype(dtype)
    path = tmp_path / name
    binio.write_array(path, matrix)
    return path, matrix


def make_config(tmp_path, raw_path, **overrides):
    values = dict(
        raw_file=raw_path,
        metric="l2",
        seed=3,
        num_base=150,
        base_selection="random",
        num_query_rows=0,
        max_inserts=100,
        output_dir=tmp_path / "out",
        name="base_a",
    )
    values.update(overrides)
    return BaseSetConfig(**values)


def test_vector_files_contain_exactly_the_selected_raw_rows(tmp_path):
    raw_path, raw = write_raw(tmp_path)
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path, num_query_rows=20))

    base_ids = base_set_module.read_base_row_ids(built)
    insert_ids = base_set_module.read_insert_row_ids(built)
    query_ids = base_set_module.read_query_row_ids(built)

    assert (built.num_base, built.num_inserts, built.num_query_rows) == (150, 100, 20)
    with built.open_base() as base:
        assert np.array_equal(base.read_range(0, 150), raw[base_ids])
    with built.open_insert_pool() as pool:
        assert np.array_equal(pool.read_range(0, 100), raw[insert_ids])
    with built.open_query_rows() as queries:
        assert np.array_equal(queries.read_range(0, 20), raw[query_ids])

    everything = np.concatenate([base_ids, insert_ids, query_ids])
    assert len(np.unique(everything)) == len(everything)  # the three sets never overlap
    assert np.all(np.diff(base_ids) > 0)


def test_rest_inserts_use_every_remaining_row(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    built = base_set_module.build_base_set(
        make_config(tmp_path, raw_path, num_base=100, num_query_rows=0, max_inserts=None)
    )
    assert built.num_inserts == TOTAL_ROWS - 100


def test_no_query_rows_means_no_query_files(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path))
    assert built.num_query_rows == 0
    assert not (built.directory / "query_rows.u8bin").exists()
    assert base_set_module.read_query_row_ids(built) is None
    with pytest.raises(BaseSetError):
        built.open_query_rows()


def test_insert_vectors_are_prefix_slices_of_one_fixed_sequence(tmp_path):
    raw_path, raw = write_raw(tmp_path)
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path))
    insert_ids = base_set_module.read_insert_row_ids(built)

    # Two workloads with different insertion totals must see the SAME j-th vector.
    short = built.insert_vectors(0, 30)
    longer = built.insert_vectors(0, 80)
    assert np.array_equal(longer[:30], short)
    assert np.array_equal(built.insert_vectors(30, 20), raw[insert_ids[30:50]])
    assert np.array_equal(built.insert_vectors(0, 100), raw[insert_ids])
    with pytest.raises(BaseSetError):
        built.insert_vectors(90, 20)  # beyond the 100 vectors of this base set
    with pytest.raises(BaseSetError):
        built.insert_vectors(-1, 5)


def test_build_is_deterministic_and_refuses_to_overwrite(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    first = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="one"))
    second = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="two"))
    for name in ["base.u8bin", "insert_pool.u8bin", "base_row_ids.ibin", "insert_row_ids.ibin"]:
        assert (first.directory / name).read_bytes() == (second.directory / name).read_bytes()

    with pytest.raises(FileExistsError):
        base_set_module.build_base_set(make_config(tmp_path, raw_path, name="one"))
    # No half-built directory is left behind.
    assert not list((tmp_path / "out").glob("*.building"))


def test_different_seeds_give_different_base_sets(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    one = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="s1", seed=1))
    two = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="s2", seed=2))
    assert one.base_row_ids_sha256 != two.base_row_ids_sha256


def test_load_detects_modified_or_missing_files(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path))
    directory = built.directory
    assert base_set_module.load_base_set(directory).num_base == 150

    # An id file with different content (same size) must be rejected by the hash check.
    ids_path = directory / "base_row_ids.ibin"
    original = ids_path.read_bytes()
    tampered = bytearray(original)
    tampered[-1] ^= 0x01
    ids_path.write_bytes(bytes(tampered))
    with pytest.raises(BaseSetError, match="hash"):
        base_set_module.load_base_set(directory)
    ids_path.write_bytes(original)
    assert base_set_module.load_base_set(directory).num_base == 150

    # A vector file with the wrong number of rows is rejected too.
    pool_path = directory / "insert_pool.u8bin"
    pool_path.unlink()
    binio.write_array(pool_path, np.zeros((7, DIM), dtype=np.uint8))
    with pytest.raises(BaseSetError):
        base_set_module.load_base_set(directory)

    (directory / "base_set.json").unlink()
    with pytest.raises(BaseSetError):
        base_set_module.load_base_set(directory)


def test_float_vectors_keep_their_dtype(tmp_path):
    raw_path, raw = write_raw(tmp_path, np.float32, "raw.fbin")
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path))
    assert built.dtype == np.float32
    with built.open_base() as base:
        assert base.dtype == np.float32
        assert np.array_equal(base.read_range(0, 150), raw[base_set_module.read_base_row_ids(built)])


def test_oversized_requests_are_rejected_before_anything_is_written(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    with pytest.raises(ValueError):
        base_set_module.build_base_set(make_config(tmp_path, raw_path, num_base=350, max_inserts=100))
    assert not (tmp_path / "out" / "base_a").exists()
    # A failed build removes its work directory as well.
    assert not list((tmp_path / "out").glob("*.building"))


# ---------------------------------------------------------------------------
# the front-to-back sweep of the raw file
# ---------------------------------------------------------------------------


def vector_files(built):
    """The bytes of every file of a base set that the sweep writes."""
    names = [f"base{built.suffix}", f"insert_pool{built.suffix}", "base_row_ids.ibin", "insert_row_ids.ibin"]
    if built.num_query_rows:
        names += [f"query_rows{built.suffix}", "query_row_ids.ibin"]
    return {name: (built.directory / name).read_bytes() for name in names}


@pytest.mark.parametrize("selection", ["random", "prefix"])
@pytest.mark.parametrize("chunk_rows", [1, 7, 50, 399, 400, 10_000])
def test_the_files_do_not_depend_on_the_size_of_the_stretches_read(tmp_path, selection, chunk_rows):
    raw_path, raw = write_raw(tmp_path)
    settings = dict(base_selection=selection, num_query_rows=20)
    whole = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="whole", **settings))
    swept = base_set_module.build_base_set(
        make_config(tmp_path, raw_path, name="swept", **settings), batch_rows=chunk_rows
    )
    assert vector_files(swept) == vector_files(whole)
    # And they are the raw rows the id files name, with the insertions in the random order.
    insert_ids = base_set_module.read_insert_row_ids(swept)
    assert not np.all(np.diff(insert_ids) > 0)
    with swept.open_insert_pool() as pool:
        assert np.array_equal(pool.read_range(0, 100), raw[insert_ids])


def test_float_vectors_are_swept_too(tmp_path):
    raw_path, raw = write_raw(tmp_path, dtype=np.float32, name="raw.fbin")
    built = base_set_module.build_base_set(make_config(tmp_path, raw_path), batch_rows=13)
    with built.open_base() as base:
        assert np.array_equal(base.read_range(0, 150), raw[base_set_module.read_base_row_ids(built)])
    with built.open_insert_pool() as pool:
        assert np.array_equal(pool.read_range(0, 100), raw[base_set_module.read_insert_row_ids(built)])


def record_reads(monkeypatch):
    reads = []
    original = binio.BinReader.read_block

    def recording(self, start, count):
        reads.append((start, count))
        return original(self, start, count)

    monkeypatch.setattr(binio.BinReader, "read_block", recording)
    return reads


def test_stretches_without_wanted_rows_are_never_read(tmp_path, monkeypatch):
    raw_path, _ = write_raw(tmp_path)
    reads = record_reads(monkeypatch)
    # The base is the first 100 rows and nothing is inserted: the other 300 rows are not needed.
    base_set_module.build_base_set(
        make_config(tmp_path, raw_path, base_selection="prefix", num_base=100, max_inserts=0),
        batch_rows=50,
    )
    assert reads == [(0, 50), (50, 50)]


def test_only_the_span_between_the_first_and_the_last_wanted_row_is_read(tmp_path, monkeypatch):
    raw_path, _ = write_raw(tmp_path)
    reads = record_reads(monkeypatch)
    built = base_set_module.build_base_set(
        make_config(tmp_path, raw_path, base_selection="prefix", num_base=100, max_inserts=3),
        batch_rows=100,
    )
    wanted = np.sort(base_set_module.read_insert_row_ids(built))
    # Chunk 0 is the base (100 rows); every other chunk is read from its first to its last wanted row.
    assert reads[0] == (0, 100)
    spans = reads[1:]
    assert all(start >= 100 for start, _ in spans)
    assert sum(count for _, count in spans) < 300
    for row in wanted:
        assert any(start <= row < start + count for start, count in spans)
    assert all(count >= 1 for _, count in spans)


def test_every_raw_row_is_read_at_most_once(tmp_path, monkeypatch):
    raw_path, _ = write_raw(tmp_path)
    reads = record_reads(monkeypatch)
    base_set_module.build_base_set(make_config(tmp_path, raw_path, num_query_rows=20), batch_rows=17)
    covered = [row for start, count in reads for row in range(start, start + count)]
    assert len(covered) == len(set(covered)) and covered == sorted(covered)  # front to back, no overlap


def test_many_insertions_can_be_collected_in_a_temporary_file(tmp_path, monkeypatch):
    raw_path, _ = write_raw(tmp_path)
    in_memory = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="memory"))
    monkeypatch.setattr(base_set_module, "IN_MEMORY_INSERT_BYTES", 0)
    spilled = base_set_module.build_base_set(make_config(tmp_path, raw_path, name="spilled"), batch_rows=9)
    assert vector_files(spilled) == vector_files(in_memory)
    assert not list((tmp_path / "out").glob("**/insert_pool.tmp"))
    assert not list((tmp_path / "out").glob("*.building"))


def test_ids_that_are_not_ascending_are_refused(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    ascending, descending = np.arange(10, dtype=np.int64), np.arange(10, dtype=np.int64)[::-1]
    nothing = np.empty(0, dtype=np.int64)
    with binio.BinReader(raw_path) as raw:
        for base_ids, query_ids in ((descending, nothing), (ascending, descending.copy())):
            with pytest.raises(BaseSetError, match="ascending"):
                base_set_module._extract_rows(
                    raw, base_ids, query_ids, np.array([20, 15], dtype=np.int64), work, ".u8bin", 5,
                    lambda message: None,
                )


def test_progress_is_reported_up_to_one_hundred_percent(tmp_path):
    raw_path, _ = write_raw(tmp_path)
    messages = []
    base_set_module.build_base_set(make_config(tmp_path, raw_path), batch_rows=40, log=messages.append)
    assert "150 base + 100 insert + 0 query rows out of 400" in messages[0]
    assert messages[-1].startswith("[base-set] raw file sweep 100%")
    percents = [int(m.split("sweep ")[1].split("%")[0]) for m in messages[1:]]
    assert percents == sorted(percents) and len(percents) <= 100 // base_set_module.PROGRESS_PERCENT + 1
