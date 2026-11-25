"""Run identity and checkpoint provenance."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data_fingerprint(directory: Path) -> dict[str, str]:
    return {
        name: sha256(directory / name)
        for name in ("train.csv", "validation.csv", "test.csv", "catalog.csv")
    }


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def run_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def reserve_run(directory: Path) -> None:
    """Reserve a new run directory; an existing run is never overwritten."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run directory already exists: {directory}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
