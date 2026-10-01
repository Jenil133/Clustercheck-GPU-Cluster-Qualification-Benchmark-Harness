"""Typed records used by ClusterCheck's stable JSON output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["pass", "fail", "error", "skipped"]
Provenance = Literal["measured", "synthetic"]


@dataclass(frozen=True)
class CommandEvidence:
    argv: list[str]
    return_code: int | None
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    termination: str | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    summary: str
    metrics: dict[str, float] = field(default_factory=dict)
    evidence: CommandEvidence | None = None


@dataclass(frozen=True)
class Inventory:
    hostname: str
    driver_version: str
    cuda_version: str
    gpu_models: list[str]
    gpu_firmware: list[str]
    ib_devices: list[str]
    ib_firmware: dict[str, str]
    kernel: str
    gpus: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class NodeRun:
    schema_version: str
    run_id: str
    started_at: str
    completed_at: str
    provenance: Provenance
    synthetic_notice: str | None
    node: str
    inventory: Inventory
    checks: list[CheckResult]
    overall_status: Status
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateFinding:
    node: str
    metric: str
    observed: float | None
    baseline: float | None
    threshold: float | None
    status: Status
    reason: str


@dataclass(frozen=True)
class FleetReport:
    schema_version: str
    run_id: str
    created_at: str
    provenance: Provenance
    synthetic_notice: str | None
    baseline_id: str
    baseline_sha256: str
    baseline_source_run_id: str
    nodes: dict[str, Status]
    findings: list[GateFinding]
    admitted: list[str]
    rejected: list[str]
    drained: list[str] = field(default_factory=list)
    quarantine_actions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
