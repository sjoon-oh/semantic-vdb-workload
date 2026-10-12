"""The ``shared/`` directory of a workload family: results that later runs reuse.

A finished family keeps the expensive results that do not belong to a single pool, so that a
later run which adds pools (``[workload] existing = extend``, see ``extend.py``) neither
recomputes them nor depends on a particular set still being there:

    shared/
        cache.json            fingerprint and sha256 of the N2 files, alpha of every overlap
        n2/                   exact neighbours of the seeds and the PCA bases (the long part)
        common/               the seed queries and the evaluation queries
        eval_step0/           the evaluation ground truth of step 0 (the same for every set)
        mixes/<u..>/          per update percentage: the deleted ids and the evaluation
                              ground truth
        mix_info/<u..>.json   per update percentage: how many evaluation answers needed a scan

``common``, ``eval_step0`` and ``mixes`` are hard-linked into every set, so they cost no
extra space. The N2 files are only read by later runs. ``cache.json`` is checked before the
N2 files are used: a changed seed file, base set or setting is an error, never a silent
reuse of results made for something else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from semantic_vdb.manifest import read_json, sha256_file, write_json
from semantic_vdb.resume import array_digest, canonical, describe_differences

SHARED_FORMAT_VERSION = 1
CACHE_FILE = "cache.json"

# Files of the N2 part, in the order they are produced.
N2_FILES = (
    "seed_neighbour_ids.npy",
    "seed_neighbour_dists.npy",
    "basis_axes.npy",
    "basis_std.npy",
    "basis_residual.npy",
)


class SharedStoreError(RuntimeError):
    """The shared results are missing, damaged, or were made for something else."""


def n2_fingerprint(config, base_set, seeds: np.ndarray) -> dict:
    """What the neighbours and PCA bases depend on.

    They are a function of the seed vectors, the base set and the two settings of the PCA
    (``neighbours`` and ``axes``). The noise scale and everything after it is not part of
    the fingerprint: it is stored per overlap (``calibrations``).
    """
    return {
        "format_version": SHARED_FORMAT_VERSION,
        "seeds_sha256": array_digest(seeds),
        "base_set": {
            "base_row_ids_sha256": base_set.base_row_ids_sha256,
            "insert_row_ids_sha256": base_set.insert_row_ids_sha256,
            "num_base": base_set.num_base,
            "dtype": str(base_set.dtype),
            "dim": base_set.dim,
            "metric": base_set.metric,
        },
        "neighbours": config.groups.neighbours,
        "axes": config.groups.axes,
    }


class SharedStore:
    """Reads and writes one ``shared/`` directory (either the family's or a build's own)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    # ---- locations ---------------------------------------------------------------------

    @property
    def n2_dir(self) -> Path:
        return self.directory / "n2"

    @property
    def common_dir(self) -> Path:
        return self.directory / "common"

    @property
    def eval_step0_dir(self) -> Path:
        return self.directory / "eval_step0"

    def mix_dir(self, mix_name: str) -> Path:
        return self.directory / "mixes" / mix_name

    def mix_info_path(self, mix_name: str) -> Path:
        # Outside mixes/<name>: that folder is linked into every set, the info is not.
        return self.directory / "mix_info" / f"{mix_name}.json"

    @property
    def cache_path(self) -> Path:
        return self.directory / CACHE_FILE

    # ---- cache.json --------------------------------------------------------------------

    def read_cache(self) -> Optional[dict]:
        """The parsed ``cache.json``, or None when there is none."""
        if not self.cache_path.is_file():
            return None
        return read_json(self.cache_path)

    def has_n2(self) -> bool:
        cache = self.read_cache()
        return cache is not None and "n2" in cache

    def write_n2_manifest(self, fingerprint: dict) -> None:
        """Record the fingerprint and the sha256 of the N2 files (call when they are complete).

        Calibrations that are already in the cache are kept.
        """
        files = {}
        for name in N2_FILES:
            path = self.n2_dir / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        cache = self.read_cache() or {"format_version": SHARED_FORMAT_VERSION, "calibrations": {}}
        cache["n2"] = {"fingerprint": canonical(fingerprint), "files": files}
        write_json(self.cache_path, cache)

    def verify_n2(self, fingerprint: dict) -> None:
        """Raise ``SharedStoreError`` unless the N2 files fit ``fingerprint`` and are intact."""
        cache = self.read_cache()
        if cache is None or "n2" not in cache:
            raise SharedStoreError(f"{self.directory} has no stored neighbours and bases")
        stored = cache["n2"]["fingerprint"]
        if stored != canonical(fingerprint):
            raise SharedStoreError(
                f"the stored neighbours and bases in {self.directory} were made for other inputs "
                f"({describe_differences(stored, canonical(fingerprint))})"
            )
        for name, expected in cache["n2"]["files"].items():
            path = self.n2_dir / name
            if not path.is_file():
                raise SharedStoreError(f"{path} is missing")
            if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
                raise SharedStoreError(
                    f"{path} does not match the sha256 in {CACHE_FILE}; it was changed or damaged"
                )

    # ---- noise scale per overlap -------------------------------------------------------

    def calibrations(self) -> Dict[str, dict]:
        cache = self.read_cache()
        return {} if cache is None else dict(cache.get("calibrations", {}))

    def calibration(self, overlap_name: str) -> Optional[dict]:
        return self.calibrations().get(overlap_name)

    def set_calibrations(self, calibrations: Dict[str, dict]) -> None:
        """Add (or replace) calibrations in ``cache.json``; the file changes atomically."""
        cache = self.read_cache() or {"format_version": SHARED_FORMAT_VERSION}
        merged = dict(cache.get("calibrations", {}))
        merged.update(canonical(calibrations))
        cache["calibrations"] = merged
        write_json(self.cache_path, cache)
