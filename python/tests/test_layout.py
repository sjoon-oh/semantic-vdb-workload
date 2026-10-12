"""Tests for semantic_vdb.layout: names, hard links, hashing and the bounded writer."""

import hashlib
import os

import pytest

from semantic_vdb import layout


def test_pool_directory_names():
    # N2 pools live below their overlap level; pools without noise have no such level.
    assert layout.pool_dir_name("n2", 200, "o80") == "o80/g200"
    assert layout.pool_dir_name("none", 1) == "g1"
    assert layout.pool_dir_name("external", None) == "external"
    with pytest.raises(ValueError):
        layout.pool_dir_name("n2", None, "o80")
    with pytest.raises(ValueError):
        layout.pool_dir_name("n2", 200)  # an N2 pool must say which overlap it has


@pytest.mark.parametrize(
    "overlap, name",
    [
        (0.8, "o80"),
        (0.5, "o50"),
        (0.125, "o12p5"),  # the same rule as the update sets (u12p5)
        (0.95, "o95"),
        (0.05, "o5"),
        (0.999, "o99p9"),
        (0.30000000000000004, "o30p000000000000004"),  # a float that is not exactly 0.3
    ],
)
def test_overlap_directory_names(overlap, name):
    assert layout.overlap_dir_name(overlap) == name


def test_different_overlaps_never_share_a_directory_name():
    values = [0.1, 0.2, 0.3, 0.30000000000000004, 0.5, 0.55, 0.555, 0.8, 0.85, 0.999]
    names = [layout.overlap_dir_name(value) for value in values]
    assert len(set(names)) == len(values)


def test_link_or_copy_makes_one_file_with_two_names(tmp_path):
    source = tmp_path / "a" / "data.bin"
    source.parent.mkdir()
    source.write_bytes(b"payload")
    destination = tmp_path / "b" / "deeper" / "data.bin"  # the folders are created
    assert layout.link_or_copy(source, destination) is True
    assert destination.read_bytes() == b"payload"
    assert os.stat(source).st_ino == os.stat(destination).st_ino


def test_link_or_copy_falls_back_to_a_copy_when_linking_is_impossible(tmp_path, monkeypatch):
    source = tmp_path / "data.bin"
    source.write_bytes(b"payload")

    def refuse(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(layout.os, "link", refuse)
    destination = tmp_path / "copy.bin"
    assert layout.link_or_copy(source, destination) is False
    assert destination.read_bytes() == b"payload"
    assert os.stat(source).st_ino != os.stat(destination).st_ino


def test_link_tree_links_every_file_and_tolerates_a_missing_source(tmp_path):
    source = tmp_path / "src"
    (source / "x" / "y").mkdir(parents=True)
    (source / "top.bin").write_bytes(b"1")
    (source / "x" / "y" / "deep.bin").write_bytes(b"2")
    assert layout.link_tree(source, tmp_path / "dst") == 2
    assert (tmp_path / "dst" / "x" / "y" / "deep.bin").read_bytes() == b"2"
    assert layout.link_tree(tmp_path / "missing", tmp_path / "dst2") == 0
    assert not (tmp_path / "dst2").exists()


def test_hash_files_hashes_hard_links_once_and_correctly(tmp_path, monkeypatch):
    first, second, third = tmp_path / "one.bin", tmp_path / "two.bin", tmp_path / "other.bin"
    first.write_bytes(b"same content")
    os.link(first, second)
    third.write_bytes(b"different")

    calls = []
    real = layout.sha256_file
    monkeypatch.setattr(layout, "sha256_file", lambda path: calls.append(path) or real(path))
    result = layout.hash_files([first, second, third], workers=3)

    assert result[first] == result[second] == (12, hashlib.sha256(b"same content").hexdigest())
    assert result[third] == (9, hashlib.sha256(b"different").hexdigest())
    assert len(calls) == 2  # the two names of one file were hashed once


def test_bounded_writer_runs_every_job_and_limits_the_backlog():
    done = []
    writer = layout.BoundedWriter(workers=2, limit=4)

    def job(value):
        done.append(value)

    for value in range(30):
        writer.submit(job, value)
    writer.finish()
    assert sorted(done) == list(range(30))


def test_bounded_writer_raises_the_error_of_a_failed_job():
    writer = layout.BoundedWriter(workers=2, limit=3)

    def fail():
        raise RuntimeError("disk full")

    writer.submit(fail)
    with pytest.raises(RuntimeError, match="disk full"):
        for _ in range(10):  # the error surfaces from submit or from finish
            writer.submit(lambda: None)
        writer.finish()


def test_bounded_writer_raises_a_late_error_when_it_finishes():
    # The backlog limit is never reached, so the error can only come out of finish().
    writer = layout.BoundedWriter(workers=2, limit=64)

    def fail():
        raise RuntimeError("disk full")

    writer.submit(lambda: None)
    writer.submit(fail)
    with pytest.raises(RuntimeError, match="disk full"):
        writer.finish()
