"""Atomic, durable, create-once JSON run storage."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_identifier(value: str, kind: str = "identifier") -> str:
    """Return a conservative filename/Slurm-safe identifier or fail closed."""
    if value in {".", ".."} or not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe {kind}: use 1-128 ASCII letters, digits, '.', '_' or '-' and start alphanumeric")
    return value


def contained_path(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("path escapes the configured output directory")
    return candidate


def reserve_run(output: Path, run_id: str) -> Path:
    """Atomically reserve a never-before-used run directory."""
    validate_identifier(run_id, "run ID")
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = contained_path(root, run_id)
    try:
        run_dir.mkdir(mode=0o750)
    except FileExistsError as exc:
        raise ValueError(f"run ID already exists and cannot be reused: {run_id}") from exc
    (run_dir / "nodes").mkdir(mode=0o750)
    _fsync_directory(run_dir)
    _fsync_directory(root)
    return run_dir


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Serialize manifest transitions and evidence publication for one run."""
    lock_path = run_dir / ".clustercheck.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    """Durably publish JSON; exclusive mode refuses to replace evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path.name}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            try:
                os.link(temp, path)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite immutable artifact: {path.name}") from exc
            finally:
                temp.unlink(missing_ok=True)
        else:
            os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
