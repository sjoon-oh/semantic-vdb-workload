"""Checkpoints of a workload build, so that an interrupted build continues where it stopped.

A build works in ``<name>.building``. Before it renames that directory to its final name it
passes through stages (neighbours of the seeds, N2 bases and alpha, one pool after the other,
the shared files, every scan of the universe, one update set after the other). Whenever a
stage is complete, the build writes a marker file below ``<name>.building/_resume``. Starting
the same build again finds the directory, checks that it was made for the same configuration
and inputs, and skips every stage that has a marker.

Rules that keep this safe:

* A marker is written only AFTER the files of its stage are complete and flushed to disk. A
  stage that was cut in the middle has no marker and is redone from its start.
* ``values/fingerprint.json`` describes the configuration and the base set; other values
  describe the query inputs. A stored value that differs from the present one is an error
  (``ResumeError``). The finished stages are never discarded silently.
* Everything a stage skips is a pure function of the configuration and the inputs (all random
  numbers come from the seeds of the configuration), so a continued build writes the same
  files as an uninterrupted one. The tests compare them byte by byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from semantic_vdb.manifest import json_text

_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class ResumeError(RuntimeError):
    """The unfinished build in a directory was made for other inputs than the present ones."""


def sync_path(path: str | os.PathLike) -> None:
    """Flush a file (or a directory entry list) to disk."""
    path = Path(path)
    flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_durable(path: str | os.PathLike, text: str) -> None:
    """Write a small file so that it appears atomically and survives a crash."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())
    os.replace(partial, path)
    sync_path(path.parent)


def touch_durable(path: str | os.PathLike) -> None:
    write_text_durable(path, "")


def array_digest(*arrays: Optional[np.ndarray]) -> str:
    """SHA-256 over the dtype, shape and bytes of arrays (``None`` counts as an empty entry)."""
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"none;")
            continue
        contiguous = np.ascontiguousarray(array)
        digest.update(f"{contiguous.dtype.str}{contiguous.shape};".encode())
        if contiguous.size:
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    """``value`` as plain JSON data (tuples become lists, paths become text)."""
    return json.loads(json_text(value))


def _differences(old: Any, new: Any, where: str = "") -> Iterator[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                yield f"{where}{key}: not there before, now {_short(new[key])}"
            elif key not in new:
                yield f"{where}{key}: was {_short(old[key])}, now missing"
            else:
                yield from _differences(old[key], new[key], f"{where}{key}.")
    elif old != new:
        yield f"{where.rstrip('.') or 'value'}: was {_short(old)}, now {_short(new)}"


def describe_differences(old: Any, new: Any, shown: int = 5) -> str:
    """The first ``shown`` differences between two JSON-like values, as one line of text."""
    changes = list(_differences(old, new))
    text = "; ".join(changes[:shown])
    if len(changes) > shown:
        text += f"; and {len(changes) - shown} more"
    return text


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ResumeState:
    """Markers and stored values of one unfinished build (a directory of tiny JSON files)."""

    def __init__(self, directory: str | os.PathLike) -> None:
        self.directory = Path(directory)
        self._done = self.directory / "done"
        self._values = self.directory / "values"
        self._done.mkdir(parents=True, exist_ok=True)
        self._values.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def exists(directory: str | os.PathLike) -> bool:
        """True when ``directory`` is the state of an earlier build (it has a fingerprint)."""
        return (Path(directory) / "values" / "fingerprint.json").is_file()

    @staticmethod
    def _checked(name: str) -> str:
        if not _NAME.match(name):
            raise ValueError(f"a stage or value name must match {_NAME.pattern}, got {name!r}")
        return name

    # ---- stored values ---------------------------------------------------------------------

    def check_or_set(self, key: str, value: Any) -> bool:
        """Store ``value`` the first time; later it must be equal.

        Returns True when an equal value was already stored (this is a continued build) and
        False when it was stored just now. Raises ``ResumeError`` for a different value.
        """
        path = self._values / f"{self._checked(key)}.json"
        present = canonical(value)
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))["value"]
            if stored != present:
                raise ResumeError(
                    f"{key} differs from the unfinished build "
                    f"({describe_differences(stored, present)})"
                )
            return True
        write_text_durable(path, json_text({"value": present}))
        return False

    # ---- finished stages -------------------------------------------------------------------

    def done(self, stage: str) -> bool:
        return (self._done / f"{self._checked(stage)}.json").is_file()

    def info(self, stage: str) -> Dict[str, Any]:
        """The values stored when ``stage`` was marked as finished."""
        path = self._done / f"{self._checked(stage)}.json"
        return json.loads(path.read_text(encoding="utf-8"))["info"]

    def mark(
        self, stage: str, info: Optional[Dict[str, Any]] = None, files: Iterable[str | os.PathLike] = ()
    ) -> None:
        """Record that ``stage`` is complete; ``files`` are flushed to disk first."""
        path = self._done / f"{self._checked(stage)}.json"
        for file in files:
            sync_path(file)
        write_text_durable(path, json_text({"stage": stage, "info": info or {}}))

    def completed(self) -> List[str]:
        return sorted(path.stem for path in self._done.glob("*.json"))

    def any_progress(self) -> bool:
        return any(self._done.glob("*.json"))
