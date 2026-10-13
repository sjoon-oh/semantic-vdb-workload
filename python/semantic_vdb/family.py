"""The index of a workload family (``family.json``) and what must stay equal when it grows.

A family is one directory with a ``family.json``. Its pools are the combinations of target
overlap and group size, each with one set per update percentage. The index records, besides
the pools and sets, a **common fingerprint**: everything that every pool of the family has
in common, i.e. all settings except the group sizes and the target overlaps, plus the base
set and the digests of the query inputs. A later run may add pools only when its common
fingerprint is equal (``extend.py``); the pools that exist are never changed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from semantic_vdb.layout import overlap_dir_name
from semantic_vdb.manifest import read_json
from semantic_vdb.resume import canonical, describe_differences
from semantic_vdb.shared_store import SharedStore

# Format 3: the pool directories are o<overlap>/g<size>, family.json has "common" and
# "overlaps", and the family keeps a shared/ directory (format 2 had g<size>/ at the top).
FAMILY_FORMAT_VERSION = 3

# Settings that may differ between a family and a run that extends it: the machine
# ([parallel]), what to do about an existing family, the name (the directory is the name) and
# where the base set is (what is in it is compared through its hashes).
_IGNORED_FOR_COMMON = ("parallel", "existing", "name", "base_set_dir")
# Paths of query files, per section. Where a file is does not matter (a data set may have been
# moved, or the family is extended on another machine); what it holds is compared through the
# digests of the inputs.
_PATH_SETTINGS = {"seeds": ("query_file", "group_id_file"), "evaluation": ("eval_file",)}
# The two lists that define the pools; a run is allowed to bring other values for them.
_POOL_AXES = ("group_sizes", "target_overlaps")
# [groundtruth] settings of the follow-up stage that answers the stream queries (stream_gt.py).
# They steer a stage that may run later, on a family that exists, so they are no part of what
# the pools have in common. Families made before the stage existed stored them in "common",
# which is why they are removed from the stored fingerprint as well (see ``check_common``).
_STREAM_STAGE_SETTINGS = ("stream_groundtruth", "stream_k", "stream_pools", "progress_seconds")


class FamilyError(RuntimeError):
    """The family directory cannot be used (wrong format) or does not fit the configuration."""


def common_fingerprint(config, base_set, inputs: dict) -> dict:
    """What every pool of the family has in common (plain JSON data).

    ``inputs`` are the digests of the query inputs (``workload._input_description``). The
    tool version is deliberately not part of it: a newer tool of the same family format may
    add pools; every set records the tool version it was made with. File paths are not part
    of it either, only what the files hold.
    """
    description = dataclasses.asdict(config)
    for key in _IGNORED_FOR_COMMON:
        description.pop(key, None)
    description["groups"] = {
        key: value for key, value in description["groups"].items() if key not in _POOL_AXES
    }
    description["groundtruth"] = _without_stream_stage(description["groundtruth"])
    for section, keys in _PATH_SETTINGS.items():
        description[section] = {
            key: value for key, value in description[section].items() if key not in keys
        }
    return canonical(
        {
            "format_version": FAMILY_FORMAT_VERSION,
            "config": description,
            "base_set": {
                "base_row_ids_sha256": base_set.base_row_ids_sha256,
                "insert_row_ids_sha256": base_set.insert_row_ids_sha256,
                "num_base": base_set.num_base,
                "num_inserts": base_set.num_inserts,
                "dtype": str(base_set.dtype),
                "dim": base_set.dim,
                "metric": base_set.metric,
            },
            "inputs": inputs,
        }
    )


@dataclass
class ExistingFamily:
    """A finished family that a run wants to extend."""

    directory: Path
    index: dict  # the parsed family.json
    store: SharedStore  # <family>/shared

    @property
    def pool_names(self) -> Set[str]:
        return {pool["name"] for pool in self.index["pools"]}


def read_existing_family(directory: Path) -> ExistingFamily:
    """Open ``family.json`` of a finished family; only format 3 can be extended."""
    index_path = directory / "family.json"
    if not index_path.is_file():
        raise FamilyError(f"{directory} has no family.json; it is not a finished family")
    index = read_json(index_path)
    version = index.get("format_version")
    if version != FAMILY_FORMAT_VERSION:
        raise FamilyError(
            f"{directory} is a family of format {version}; only format "
            f"{FAMILY_FORMAT_VERSION} can be extended. Build it again under a new name."
        )
    return ExistingFamily(directory, index, SharedStore(directory / "shared"))


def _without_stream_stage(groundtruth: dict) -> dict:
    """The ``[groundtruth]`` settings without those of the stream answering stage."""
    return {key: value for key, value in groundtruth.items() if key not in _STREAM_STAGE_SETTINGS}


def check_common(existing: ExistingFamily, current: dict) -> None:
    """Raise ``FamilyError`` unless the configuration fits the family it would extend."""
    stored = existing.index["common"]
    # An older family stored the stage settings too; they must not make it look different.
    if "groundtruth" in stored.get("config", {}):
        stored = dict(stored)
        stored["config"] = dict(stored["config"])
        stored["config"]["groundtruth"] = _without_stream_stage(stored["config"]["groundtruth"])
    if stored != current:
        raise FamilyError(
            f"the configuration differs from the family {existing.directory.name} in "
            f"settings that all its pools share ({describe_differences(stored, current)}). "
            "Only new target overlaps and new group sizes can be added to a family; build "
            "anything else as a new family with another name."
        )


def merge_family_index(old: dict, fragment: dict, config) -> dict:
    """The family index after adding the pools and sets of ``fragment``.

    Pools, sets and overlaps are put in the order of the configuration (overlap by overlap,
    group size by group size, update percentage by update percentage), so that an extended
    family has the same index as one that was built with all pools at once. Entries that the
    configuration no longer mentions keep their old relative order behind the others.
    """
    merged = dict(old)
    merged["pools"] = list(old["pools"]) + list(fragment["pools"])
    merged["sets"] = list(old["sets"]) + list(fragment["sets"])
    # A new pool at an existing overlap brings that overlap's entry again: the family's own
    # entry is kept (it is the one the existing pools were made with).
    known = {entry["name"] for entry in old["overlaps"]}
    merged["overlaps"] = list(old["overlaps"]) + [
        entry for entry in fragment["overlaps"] if entry["name"] not in known
    ]
    overlap_order = {
        overlap_dir_name(value): position
        for position, value in enumerate(config.groups.target_overlaps)
    }
    size_order = {size: position for position, size in enumerate(config.groups.group_sizes)}
    percent_order = {
        percent: position for position, percent in enumerate(config.updates.update_percents)
    }
    unknown = len(overlap_order) + len(size_order) + len(percent_order) + 1

    def pool_key(pool: dict):
        return (
            overlap_order.get(pool.get("overlap"), unknown),
            size_order.get(pool.get("group_size"), unknown),
        )

    pools_in_order = sorted(merged["pools"], key=pool_key)  # stable: unknown ones stay in order
    merged["pools"] = pools_in_order
    position_of_pool = {pool["name"]: position for position, pool in enumerate(pools_in_order)}
    merged["sets"] = sorted(
        merged["sets"],
        key=lambda entry: (
            position_of_pool[entry["pool"]],
            percent_order.get(entry["update_percent"], unknown),
        ),
    )
    merged["overlaps"] = sorted(
        merged["overlaps"], key=lambda entry: overlap_order.get(entry["name"], unknown)
    )
    return merged
