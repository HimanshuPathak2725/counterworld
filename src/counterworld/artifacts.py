"""Versioned, content-hashed research artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np


def json_ready(value):
    if is_dataclass(value):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def canonical_json(value) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.files: dict[str, Path] = {}

    def write_json(self, name: str, value) -> Path:
        path = self.root / name
        path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
        self.files[name] = path
        return path

    def write_jsonl(self, name: str, rows) -> Path:
        path = self.root / name
        with path.open("w") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        self.files[name] = path
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text)
        self.files[name] = path
        return path

    def finalize(self, configuration) -> Path:
        files = {
            name: {"sha256": file_hash(path), "bytes": path.stat().st_size}
            for name, path in sorted(self.files.items())
        }
        manifest = {
            "protocol": "CW-0.2-draft",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "configuration": json_ready(configuration),
            "configuration_sha256": sha256(canonical_json(configuration).encode()).hexdigest(),
            "files": files,
        }
        return self.write_json("manifest.json", manifest)
