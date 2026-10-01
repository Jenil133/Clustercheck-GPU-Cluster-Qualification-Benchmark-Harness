"""Safe subprocess execution, process-group timeout cleanup, and tool discovery."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .models import CommandEvidence

_CAPTURE_LIMIT = 100_000


class ToolUnavailable(RuntimeError):
    """Raised when a required executable cannot be found."""


@dataclass(frozen=True)
class CommandRunner:
    timeout_seconds: int = 900
    termination_grace_seconds: float = 5.0

    def require(self, executable: str) -> str:
        if "/" in executable:
            path = Path(executable)
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        resolved = shutil.which(executable)
        if not resolved:
            raise ToolUnavailable(f"required executable not found: {executable}")
        return resolved

    def run(
        self, argv: list[str], timeout: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> CommandEvidence:
        if not argv:
            raise ValueError("argv cannot be empty")
        resolved = [self.require(argv[0]), *argv[1:]]
        started = time.monotonic()
        process = subprocess.Popen(
            resolved,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
            env={**os.environ, "LC_ALL": "C", **(extra_env or {})},
        )
        timed_out = False
        termination: str | None = None
        try:
            stdout, stderr = process.communicate(timeout=timeout or self.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = "SIGTERM"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=self.termination_grace_seconds)
            except subprocess.TimeoutExpired:
                termination = "SIGKILL"
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
        return _evidence(resolved, process.returncode, started, stdout, stderr, timed_out, termination)


def split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("configured command cannot be empty")
    return parts


def _evidence(
    argv: list[str], return_code: int | None, started: float, stdout: bytes, stderr: bytes,
    timed_out: bool, termination: str | None,
) -> CommandEvidence:
    return CommandEvidence(
        argv=argv,
        return_code=return_code,
        duration_seconds=round(time.monotonic() - started, 3),
        stdout=_bounded_decode(stdout),
        stderr=_bounded_decode(stderr),
        timed_out=timed_out,
        stdout_bytes=len(stdout),
        stderr_bytes=len(stderr),
        stdout_truncated=len(stdout) > _CAPTURE_LIMIT,
        stderr_truncated=len(stderr) > _CAPTURE_LIMIT,
        termination=termination,
    )


def _bounded_decode(value: bytes) -> str:
    if len(value) <= _CAPTURE_LIMIT:
        selected = value
    else:
        half = _CAPTURE_LIMIT // 2
        selected = value[:half] + b"\n... CLUSTERCHECK OUTPUT TRUNCATED ...\n" + value[-half:]
    return selected.decode("utf-8", errors="replace")
