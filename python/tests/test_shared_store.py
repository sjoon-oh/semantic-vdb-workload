"""The shared/ directory: what is stored, and that stale or damaged results are refused."""

import dataclasses

import numpy as np
import pytest

from semantic_vdb import config, n2, shared_store
from semantic_vdb.calibrate import CalibrationResult
from semantic_vdb.shared_store import N2_FILES, SharedStore, SharedStoreError


class FakeBaseSet:
    """The few attributes the fingerprint reads."""

    base_row_ids_sha256 = "a" * 64
    insert_row_ids_sha256 = "b" * 64
    num_base = 800
    dtype = np.dtype(np.uint8)
    dim = 12
    metric = "l2"


def groups_config(**overrides):
    groups = config.GroupsConfig(
        group_sizes=(3,),
        include_original=False,
        neighbours=60,
        axes=4,
        residual=False,
        target_overlaps=(0.7,),
        calibration_queries=20,
        overlap_tolerance=0.05,
    )
    return dataclasses.replace(groups, **overrides)


class FakeConfig:
    """A configuration with only the ``groups`` section, which is all the fingerprint reads."""

    def __init__(self, **overrides):
        self.groups = groups_config(**overrides)


def write_n2_files(store, size=64):
    """Five small arrays under the names of the real neighbour and basis files."""
    store.n2_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for name in N2_FILES:
        np.save(store.n2_dir / name, rng.normal(size=size))


SEEDS = np.arange(24, dtype=np.uint8).reshape(2, 12)


def fingerprint(base_set=None, seeds=SEEDS, **overrides):
    return shared_store.n2_fingerprint(FakeConfig(**overrides), base_set or FakeBaseSet(), seeds)


# ---------------------------------------------------------------------------
# the fingerprint
# ---------------------------------------------------------------------------


def test_the_fingerprint_follows_exactly_what_the_neighbours_and_bases_depend_on():
    base = fingerprint()
    # Settings of the noise (not of the PCA) and of the pools do not change it ...
    unrelated = (
        dict(target_overlaps=(0.5,)),
        dict(group_sizes=(9,)),
        dict(residual=True),
        dict(calibration_queries=5),
        dict(overlap_tolerance=0.01),
        dict(include_original=True),
    )
    for overrides in unrelated:
        assert fingerprint(**overrides) == base, overrides
    # ... the seeds, the base set and the two settings of the PCA do.
    assert fingerprint(neighbours=50) != base
    assert fingerprint(axes=3) != base
    assert fingerprint(seeds=SEEDS[::-1].copy()) != base
    other_base = FakeBaseSet()
    other_base.base_row_ids_sha256 = "c" * 64
    assert fingerprint(base_set=other_base) != base


def test_the_files_are_verified_against_their_fingerprint_and_hashes(tmp_path):
    store = SharedStore(tmp_path / "shared")
    write_n2_files(store)
    assert not store.has_n2()
    store.write_n2_manifest(fingerprint())
    assert store.has_n2()
    store.verify_n2(fingerprint())  # intact: no error

    with pytest.raises(SharedStoreError, match="other inputs.*axes"):
        store.verify_n2(fingerprint(axes=3))

    path = store.n2_dir / N2_FILES[2]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))  # same size, one wrong bit
    with pytest.raises(SharedStoreError, match="sha256"):
        store.verify_n2(fingerprint())
    path.unlink()
    with pytest.raises(SharedStoreError, match="missing"):
        store.verify_n2(fingerprint())


def test_a_store_without_neighbours_says_so(tmp_path):
    with pytest.raises(SharedStoreError, match="no stored neighbours"):
        SharedStore(tmp_path / "empty").verify_n2({})


# ---------------------------------------------------------------------------
# the noise scale of every overlap
# ---------------------------------------------------------------------------


def test_calibrations_accumulate_and_survive_a_rewritten_manifest(tmp_path):
    store = SharedStore(tmp_path / "shared")
    store.set_calibrations({"o80": {"alpha": 0.25}})  # cache.json may start with a calibration
    store.set_calibrations({"o50": {"alpha": 0.75}})
    store.set_calibrations({"o80": {"alpha": 0.25}})  # setting it again changes nothing
    expected = {"o80": {"alpha": 0.25}, "o50": {"alpha": 0.75}}
    assert store.calibrations() == expected
    assert store.calibration("o50") == {"alpha": 0.75}
    assert store.calibration("o99") is None

    write_n2_files(store)
    store.write_n2_manifest(fingerprint())
    assert store.calibrations() == expected  # writing the manifest keeps them


def test_the_locations_inside_shared(tmp_path):
    store = SharedStore(tmp_path / "shared")
    assert store.n2_dir == tmp_path / "shared" / "n2"
    assert store.common_dir == tmp_path / "shared" / "common"
    assert store.eval_step0_dir == tmp_path / "shared" / "eval_step0"
    assert store.mix_dir("u25") == tmp_path / "shared" / "mixes" / "u25"
    # The info of a mix is NOT below mixes/<name>: that folder is linked into every set.
    assert store.mix_info_path("u25") == tmp_path / "shared" / "mix_info" / "u25.json"


def stored_info(**overrides):
    """A noise scale as it is stored: its settings plus its result."""
    info = {
        "target_overlap": 0.7,
        "tolerance": 0.05,
        "calibration_queries": 20,
        "residual": False,
        "seed": 11,
        "alpha": 0.25,
        "achieved_overlap": 0.745,
        "iterations": 6,
        "num_calibration": 20,
    }
    info.update(overrides)
    return info


def settings(**overrides):
    values = {
        "target_overlap": 0.7,
        "tolerance": 0.05,
        "calibration_queries": 20,
        "residual": False,
        "seed": 11,
    }
    values.update(overrides)
    return values


def test_a_stored_calibration_is_rebuilt_exactly():
    calibration = n2.calibration_from_info("o70", stored_info(), settings())
    assert calibration.overlap_name == "o70"
    assert calibration.target_overlap == 0.7
    assert calibration.result == CalibrationResult(0.25, 0.745, 6)
    assert calibration.num_calibration == 20
    # And it describes itself with the same keys it was stored with.
    assert calibration.to_info(settings()) == stored_info()


@pytest.mark.parametrize(
    "changed",
    [
        dict(tolerance=0.01),
        dict(calibration_queries=10),
        dict(residual=True),
        dict(seed=12),
        dict(target_overlap=0.6),
    ],
)
def test_a_stored_calibration_made_with_other_settings_is_refused(changed):
    with pytest.raises(n2.N2Error, match="other settings"):
        n2.calibration_from_info("o70", stored_info(), settings(**changed))


def test_the_settings_a_noise_scale_depends_on_follow_the_configuration():
    class FakeWorkload:
        groups = groups_config(residual=True, calibration_queries=7, overlap_tolerance=0.03)
        seeds = config.SeedsConfig("public_queries", None, None, None, seed=5)

    assert n2.calibration_settings(FakeWorkload, 0.55) == {
        "target_overlap": 0.55,
        "tolerance": 0.03,
        "calibration_queries": 7,
        "residual": True,
        "seed": 5,
    }
