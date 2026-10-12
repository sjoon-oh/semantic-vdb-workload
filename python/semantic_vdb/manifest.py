"""JSON manifests and file hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class _ManifestEncoder(json.JSONEncoder):
    """JSON encoder that understands the numpy scalars, arrays and paths we store."""

    def default(self, value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def json_text(payload: Any) -> str:
    """``payload`` as the sorted, indented JSON text that ``write_json`` stores."""
    return json.dumps(payload, cls=_ManifestEncoder, indent=2, sort_keys=True) + "\n"


def write_json(path: str | os.PathLike, payload: dict) -> None:
    """Write ``payload`` as sorted, indented JSON; the file appears atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    with open(partial_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, cls=_ManifestEncoder, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(partial_path, path)


def read_json(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: str | os.PathLike, chunk_bytes: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in chunks so large files never sit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
