"""Qualification execution, immutable run binding, baselines, and admission decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .adapters import RealAdapter, SyntheticAdapter, adapter_for
from .command import CommandRunner, split_command
from .config import Config
from .models import CheckResult, FleetReport, GateFinding, Inventory, NodeRun, Provenance, Status
from .storage import contained_path, read_json, reserve_run, run_lock, validate_identifier, write_json

SYNTHETIC_NOTICE = "SYNTHETIC DATA: deterministic fixture only; no GPU, Slurm, DCGM, NCCL, or InfiniBand hardware execution occurred."
LOCAL_CHECKS = {"dcgm_diagnostic", "gpu_burn", "ib_bandwidth"}
REQUIRED_METRICS = {"gpu_burn_gflops", "nccl_bus_bw_GBps", "ib_bandwidth_Gbps"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex


def topology_sha256(nodes: list[str], gpus_per_node: int) -> str:
    value = json.dumps({"nodes": nodes, "gpus_per_node": gpus_per_node}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def initialize_run(config: Config, mode: str, run_id: str, output: Path, state: str = "running", synthetic_fixture: bool = False) -> dict[str, Any]:
    """Exclusively reserve a run ID and publish its authoritative manifest before work starts."""
    if mode not in {"real", "simulation"} or state not in {"running", "submitting"}:
        raise ValueError("invalid initial run mode or state")
    run_dir = reserve_run(output, run_id)
    provenance: Provenance = "synthetic" if mode == "simulation" else "measured"
    nodes = [validate_identifier(str(node), "node identifier") for node in config.cluster["nodes"]]
    gpus = int(config.cluster["gpus_per_node"])
    manifest: dict[str, Any] = {
        "schema_version": "1.1", "run_id": run_id, "created_at": now(), "updated_at": now(),
        "state": state, "provenance": provenance,
        "synthetic_notice": SYNTHETIC_NOTICE if provenance == "synthetic" else None,
        "synthetic_fixture": synthetic_fixture,
        "cluster": str(config.cluster["name"]), "requested_nodes": nodes,
        "gpus_per_node": gpus, "requested_gpu_count": len(nodes) * gpus,
        "config_name": config.path.name, "config_sha256": config.sha256,
        "topology_sha256": topology_sha256(nodes, gpus),
        "scheduler": {"worker_job_ids": {}, "fabric_job_id": None, "admission_job_id": None},
        "state_history": [{"state": state, "at": now()}],
    }
    write_json(run_dir / "run-manifest.json", manifest, exclusive=True)
    return manifest


def update_submission(run_dir: Path, state: str, *, worker_node: str | None = None, worker_job_id: str | None = None, fabric_job_id: str | None = None, admission_job_id: str | None = None, error: str | None = None) -> dict[str, Any]:
    if state not in {"submitting", "submitted", "failed"}:
        raise ValueError("invalid submission state")
    with run_lock(run_dir):
        manifest = _manifest(run_dir)
        current = str(manifest["state"])
        allowed = {"submitting": {"submitting", "submitted", "failed"}, "submitted": {"failed"}}
        if manifest["provenance"] != "measured" or state not in allowed.get(current, set()):
            raise ValueError(f"invalid terminal or backward submission transition: {current} -> {state}")
        scheduler = cast(dict[str, Any], manifest["scheduler"])
        if worker_job_id:
            _validate_job_id(worker_job_id)
            if not worker_node:
                raise ValueError("worker node is required with worker job ID")
            validate_identifier(worker_node, "worker node")
            if worker_node not in manifest["requested_nodes"]:
                raise ValueError("worker node is not in manifest topology")
            workers = cast(dict[str, str], scheduler["worker_job_ids"])
            if worker_node in workers:
                raise ValueError("worker job is already recorded for node")
            workers[worker_node] = worker_job_id
        if fabric_job_id:
            _validate_job_id(fabric_job_id)
            scheduler["fabric_job_id"] = fabric_job_id
        if admission_job_id:
            _validate_job_id(admission_job_id)
            scheduler["admission_job_id"] = admission_job_id
        manifest["state"] = state
        manifest["updated_at"] = now()
        if error:
            manifest["failure"] = error[:1000]
        cast(list[dict[str, str]], manifest["state_history"]).append({"state": state, "at": now()})
        write_json(run_dir / "run-manifest.json", manifest)
        return manifest


def run_node(config: Config, mode: str, node: str, run_id: str, output: Path, synthetic_fixture: bool = False) -> NodeRun:
    validate_identifier(run_id, "run ID")
    validate_identifier(node, "node identifier")
    run_dir = contained_path(output, run_id)
    manifest = _bound_manifest(config, run_dir, allowed_states={"running", "submitted"})
    expected_provenance = "synthetic" if mode == "simulation" else "measured"
    if manifest["provenance"] != expected_provenance or node not in manifest["requested_nodes"] or bool(manifest.get("synthetic_fixture")) != synthetic_fixture:
        raise ValueError("node execution does not match manifest provenance, fixture mode, or requested topology")
    _authorize_scheduler(manifest, "worker", node)
    started = now()
    try:
        adapter = SyntheticAdapter(config, force_healthy=synthetic_fixture) if mode == "simulation" else adapter_for(mode, config)
        inventory = adapter.inventory(node)
        checks = adapter.checks(node)
        status: Status = "pass" if all(check.status == "pass" for check in checks) else "fail"
    except (OSError, RuntimeError, ValueError) as exc:
        inventory = Inventory(node, "unknown", "unknown", [], [], [], {}, "unknown", [])
        checks = [CheckResult("node_execution", "error", f"{type(exc).__name__}: {exc}")]
        status = "error"
    record = NodeRun("1.1", run_id, started, now(), cast(Provenance, expected_provenance), SYNTHETIC_NOTICE if mode == "simulation" else None, node, inventory, checks, status, config.sha256)
    with run_lock(run_dir):
        current = _bound_manifest(config, run_dir, allowed_states={"running", "submitted"})
        _authorize_scheduler(current, "worker", node)
        write_json(run_dir / "nodes" / f"{node}.json", record.to_dict(), exclusive=True)
    return record


def attach_fabric_result(config: Config, run_dir: Path, mode: str = "real", synthetic_fixture: bool = False) -> dict[str, Any]:
    manifest = _bound_manifest(config, run_dir, allowed_states={"running", "submitted"})
    expected_provenance = "synthetic" if mode == "simulation" else "measured"
    if manifest["provenance"] != expected_provenance or bool(manifest.get("synthetic_fixture")) != synthetic_fixture:
        raise ValueError("fabric mode/fixture does not match manifest provenance")
    _authorize_scheduler(manifest, "fabric")
    requested = cast(list[str], manifest["requested_nodes"])
    allocated_raw = os.environ.get("CLUSTERCHECK_ALLOCATED_NODES")
    allocated_nodes = allocated_raw.split(",") if allocated_raw else requested
    allocation_count = int(os.environ.get("SLURM_JOB_NUM_NODES", len(allocated_nodes)))
    coverage_verified = mode == "simulation" or (allocation_count == len(requested) and set(allocated_nodes) == set(requested))
    if mode == "simulation":
        result = SyntheticAdapter(config, force_healthy=synthetic_fixture).nccl()
    else:
        result = RealAdapter(config, CommandRunner(int(config.tools.get("default_timeout", 900)))).nccl()
    if not coverage_verified and result.status == "pass":
        result = replace(result, status="fail", summary=f"{result.summary}; Slurm allocation count {allocation_count} != {len(requested)}")
    artifact = {
        "schema_version": "1.1", "run_id": manifest["run_id"], "created_at": now(),
        "provenance": manifest["provenance"], "synthetic_notice": manifest["synthetic_notice"],
        "config_sha256": manifest["config_sha256"], "topology_sha256": manifest["topology_sha256"],
        "scope": "allocation-wide", "participants": requested, "allocated_nodes": allocated_nodes,
        "allocation_node_count": allocation_count, "topology_count_verified": coverage_verified,
        "rank_gpu_mapping_verified": False,
        "limitation": "ClusterCheck verifies configured topology/allocation count, not rank-to-host/GPU mapping from nccl-tests output.",
        "check": result.__dict__ | {"evidence": result.evidence.__dict__ if result.evidence else None},
    }
    with run_lock(run_dir):
        current = _bound_manifest(config, run_dir, allowed_states={"running", "submitted"})
        _authorize_scheduler(current, "fabric")
        write_json(run_dir / "fabric-result.json", artifact, exclusive=True)
    return artifact


def finalize_run(config: Config, run_dir: Path) -> dict[str, Any]:
    with run_lock(run_dir):
        manifest = _bound_manifest(config, run_dir, allowed_states={"running", "submitted"})
        _authorize_scheduler(manifest, "admission")
        try:
            runs = _load_nodes(run_dir)
            _validate_record_coverage(manifest, runs)
            fabric = _bound_fabric(manifest, run_dir)
            if fabric["check"].get("status") not in {"pass", "fail", "error"}:
                raise ValueError("fabric result has invalid status")
        except (FileNotFoundError, ValueError) as exc:
            manifest["state"] = "failed"
            manifest["failure"] = str(exc)[:1000]
            cast(list[dict[str, str]], manifest["state_history"]).append({"state": "failed", "at": now()})
            manifest["updated_at"] = now()
            write_json(run_dir / "run-manifest.json", manifest)
            raise
        manifest["state"] = "completed"
        manifest["completed_at"] = now()
        manifest["updated_at"] = now()
        cast(list[dict[str, str]], manifest["state_history"]).append({"state": "completed", "at": now()})
        write_json(run_dir / "run-manifest.json", manifest)
        return manifest


def create_baseline(run_dir: Path, baseline_path: Path, baseline_id: str | None = None) -> dict[str, Any]:
    manifest = _manifest(run_dir, required_state="completed")
    runs = _load_nodes(run_dir)
    _validate_record_coverage(manifest, runs)
    fabric = _bound_fabric(manifest, run_dir)
    if manifest["provenance"] == "synthetic" and manifest.get("synthetic_fixture") is not True:
        raise ValueError("synthetic baselines require an explicit all-passing synthetic fixture run")
    if any(run.get("overall_status") != "pass" for run in runs) or fabric["check"].get("status") != "pass":
        raise ValueError("baseline requires every requested node and allocation-wide fabric check to pass")
    metrics: dict[str, list[float]] = {"nccl_bus_bw_GBps": [_metric_value(fabric["check"], "nccl_bus_bw_GBps")]}
    versions: dict[str, list[str]] = {"driver_version": [], "cuda_version": [], "gpu_firmware": [], "ib_firmware": [], "gpu_models": []}
    for run in runs:
        if {check.get("name") for check in run.get("checks", [])} != LOCAL_CHECKS:
            raise ValueError(f"cannot create baseline: {run['node']} lacks the complete node-local check set")
        metrics.setdefault("gpu_burn_gflops", []).append(_metric_value_from_run(run, "gpu_burn_gflops"))
        metrics.setdefault("ib_bandwidth_Gbps", []).append(_metric_value_from_run(run, "ib_bandwidth_Gbps"))
        inventory = cast(dict[str, Any], run["inventory"])
        versions["driver_version"].append(str(inventory["driver_version"]))
        versions["cuda_version"].append(str(inventory["cuda_version"]))
        versions["gpu_firmware"].extend(map(str, inventory["gpu_firmware"]))
        versions["ib_firmware"].extend(map(str, cast(dict[str, Any], inventory["ib_firmware"]).values()))
        versions["gpu_models"].extend(map(str, inventory["gpu_models"]))
    identifier = validate_identifier(baseline_id or f"baseline-{new_run_id()}", "baseline ID")
    provenance = cast(Provenance, manifest["provenance"])
    baseline = {
        "schema_version": "1.1", "baseline_id": identifier, "created_at": now(), "provenance": provenance,
        "synthetic_notice": SYNTHETIC_NOTICE if provenance == "synthetic" else None,
        "source_run_id": manifest["run_id"], "source_config_sha256": manifest["config_sha256"],
        "source_topology_sha256": manifest["topology_sha256"], "source_nodes": manifest["requested_nodes"],
        "source_manifest_state": "completed", "all_requested_nodes_passed": True,
        "metrics": {key: {"median": statistics.median(values), "samples": len(values), "scope": "allocation-wide" if key.startswith("nccl_") else "node-local"} for key, values in sorted(metrics.items())},
        "versions": {key: sorted(set(values)) for key, values in versions.items()},
    }
    write_json(baseline_path, baseline, exclusive=True)
    return baseline


def aggregate(config: Config, run_dir: Path, baseline_path: Path, apply_quarantine: bool = False) -> FleetReport:
    """Serialize admission and scheduler side effects for apply-once behavior."""
    with run_lock(run_dir):
        return _aggregate(config, run_dir, baseline_path, apply_quarantine)


def _aggregate(config: Config, run_dir: Path, baseline_path: Path, apply_quarantine: bool = False) -> FleetReport:
    manifest = _bound_manifest(config, run_dir, allowed_states={"completed"})
    runs = _load_nodes(run_dir)
    _validate_record_coverage(manifest, runs)
    fabric = _bound_fabric(manifest, run_dir)
    baseline_bytes = baseline_path.read_bytes()
    baseline_sha256 = hashlib.sha256(baseline_bytes).hexdigest()
    baseline = read_json(baseline_path)
    _validate_baseline(baseline, manifest)
    expected_nodes = cast(list[str], manifest["requested_nodes"])
    by_node = {str(run["node"]): run for run in runs}
    provenance = cast(Provenance, manifest["provenance"])
    floor = float(config.thresholds["relative_floor"])
    findings: list[GateFinding] = []
    states: dict[str, Status] = {}
    fabric_check = cast(dict[str, Any], fabric["check"])
    fabric_metrics = _metrics_from_check(fabric_check)
    for node in expected_nodes:
        run = by_node[node]
        state: Status = "pass" if run.get("overall_status") == "pass" else "fail"
        structural = _structural_findings(run, manifest)
        for passed, metric, reason in structural:
            findings.append(GateFinding(node, metric, None, None, None, "pass" if passed else "fail", reason))
            if not passed:
                state = "fail"
        observed_metrics = _metrics(run) | fabric_metrics
        for metric in sorted(REQUIRED_METRICS):
            baseline_metric = cast(dict[str, Any] | None, baseline.get("metrics", {}).get(metric))
            reference = _finite_metric(baseline_metric.get("median") if baseline_metric else None, f"baseline {metric}") if baseline_metric else None
            threshold = reference * floor if reference is not None else None
            if metric == "nccl_bus_bw_GBps":
                absolute = float(config.thresholds["expected_nccl_bus_bw_GBps"]) * float(config.thresholds["nccl_bus_efficiency_floor"])
                threshold = max(threshold or 0.0, absolute)
            observed = observed_metrics.get(metric)
            passed = threshold is not None and observed is not None and observed >= threshold
            scope = "allocation-wide; conservatively rejects every participant" if metric == "nccl_bus_bw_GBps" else "node-local"
            findings.append(GateFinding(node, metric, observed, reference, round(threshold, 3) if threshold is not None else None, "pass" if passed else "fail", f"{scope}: at or above regression floor" if passed else f"{scope}: missing evidence or below regression floor"))
            if not passed:
                state = "fail"
        if fabric_check.get("status") != "pass":
            state = "fail"
        _append_version_findings(config, baseline, run, node, findings)
        if any(item.node == node and item.status != "pass" for item in findings):
            state = "fail"
        states[node] = state
    rejected = sorted(node for node, state in states.items() if state != "pass")
    report = FleetReport("1.1", str(manifest["run_id"]), now(), provenance, SYNTHETIC_NOTICE if provenance == "synthetic" else None, str(baseline["baseline_id"]), baseline_sha256, str(baseline["source_run_id"]), states, findings, sorted(node for node, state in states.items() if state == "pass"), rejected)
    if apply_quarantine:
        actions = _quarantine(config, report, run_dir)
        report = replace(report, drained=sorted(node for node, result in actions.items() if result == "drained"), quarantine_actions=actions)
    write_json(run_dir / "fleet-report.json", report.to_dict(), exclusive=True)
    _write_markdown(run_dir / "fleet-report.md", report)
    return report


def _manifest(run_dir: Path, required_state: str | None = None) -> dict[str, Any]:
    manifest = read_json(run_dir / "run-manifest.json")
    if manifest.get("schema_version") != "1.1" or manifest.get("run_id") != run_dir.name:
        raise ValueError("manifest schema/run ID is not bound to its run directory")
    validate_identifier(str(manifest["run_id"]), "run ID")
    if required_state and manifest.get("state") != required_state:
        raise ValueError(f"run manifest must be {required_state}; found {manifest.get('state')}")
    return manifest


def _bound_manifest(config: Config, run_dir: Path, allowed_states: set[str]) -> dict[str, Any]:
    manifest = _manifest(run_dir)
    nodes = [str(node) for node in config.cluster["nodes"]]
    gpus = int(config.cluster["gpus_per_node"])
    if manifest.get("state") not in allowed_states:
        raise ValueError(f"run state {manifest.get('state')} does not permit this operation")
    if manifest.get("config_sha256") != config.sha256 or manifest.get("requested_nodes") != nodes or manifest.get("gpus_per_node") != gpus or manifest.get("topology_sha256") != topology_sha256(nodes, gpus):
        raise ValueError("configuration or topology does not match the immutable run manifest")
    return manifest


def _bound_fabric(manifest: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    fabric = read_json(run_dir / "fabric-result.json")
    for field in ("run_id", "provenance", "config_sha256", "topology_sha256"):
        if fabric.get(field) != manifest.get(field):
            raise ValueError(f"fabric result {field} does not match manifest")
    if fabric.get("participants") != manifest.get("requested_nodes") or fabric.get("scope") != "allocation-wide":
        raise ValueError("fabric participants/scope do not match manifest topology")
    return fabric


def _validate_record_coverage(manifest: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    expected = cast(list[str], manifest["requested_nodes"])
    names = [str(run.get("node")) for run in runs]
    if len(names) != len(set(names)) or set(names) != set(expected) or len(names) != len(expected):
        raise ValueError("node records must provide exact, unique current manifest coverage")
    for run in runs:
        if run.get("run_id") != manifest["run_id"] or run.get("provenance") != manifest["provenance"] or run.get("config_sha256") != manifest["config_sha256"]:
            raise ValueError(f"node record {run.get('node')} is not bound to this manifest")


def _load_nodes(run_dir: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((run_dir / "nodes").glob("*.json"))]


def _metrics(run: dict[str, Any]) -> dict[str, float]:
    return {key: _finite_metric(value, key) for check in run.get("checks", []) for key, value in check.get("metrics", {}).items()}


def _metrics_from_check(check: dict[str, Any]) -> dict[str, float]:
    return {key: _finite_metric(value, key) for key, value in check.get("metrics", {}).items()}


def _metric_value(check: dict[str, Any], metric: str) -> float:
    return _finite_metric(check.get("metrics", {}).get(metric), metric)


def _metric_value_from_run(run: dict[str, Any], metric: str) -> float:
    values = _metrics(run)
    if metric not in values:
        raise ValueError(f"missing required metric {metric} in {run.get('node')}")
    return values[metric]


def _finite_metric(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _structural_findings(run: dict[str, Any], manifest: dict[str, Any]) -> list[tuple[bool, str, str]]:
    inventory = cast(dict[str, Any], run.get("inventory", {}))
    gpus = cast(list[dict[str, Any]], inventory.get("gpus", []))
    expected = int(manifest["gpus_per_node"])
    unique = len({gpu.get("index") for gpu in gpus}) == expected and len({gpu.get("uuid") for gpu in gpus}) == expected and len({gpu.get("pci_bus_id") for gpu in gpus}) == expected
    return [
        (len(gpus) == expected and len(inventory.get("gpu_models", [])) == expected and len(inventory.get("gpu_firmware", [])) == expected and unique, "gpu_identity_cardinality", "exact unique per-GPU index/UUID/PCI/model/firmware coverage"),
        ({check.get("name") for check in run.get("checks", [])} == LOCAL_CHECKS, "required_node_checks", "all required node-local checks are present"),
        (inventory.get("hostname") == run.get("node") or str(inventory.get("hostname", "")).split(".")[0] == str(run.get("node", "")).split(".")[0], "hostname_binding", "inventory hostname is bound to requested node"),
    ]


def _validate_baseline(baseline: dict[str, Any], manifest: dict[str, Any]) -> None:
    if baseline.get("schema_version") != "1.1" or baseline.get("provenance") != manifest["provenance"]:
        raise ValueError("baseline schema/provenance does not match run")
    if baseline.get("source_config_sha256") != manifest["config_sha256"] or baseline.get("source_topology_sha256") != manifest["topology_sha256"] or baseline.get("source_nodes") != manifest["requested_nodes"]:
        raise ValueError("baseline configuration/topology does not match run manifest")
    if baseline.get("source_manifest_state") != "completed" or baseline.get("all_requested_nodes_passed") is not True:
        raise ValueError("baseline does not attest a complete all-passing source run")
    validate_identifier(str(baseline.get("baseline_id", "")), "baseline ID")
    for metric in REQUIRED_METRICS:
        item = baseline.get("metrics", {}).get(metric)
        if not isinstance(item, dict) or not isinstance(item.get("samples"), int) or item["samples"] < 1:
            raise ValueError(f"baseline metric {metric} lacks valid samples")
        _finite_metric(item.get("median"), f"baseline {metric}")
    expected_node_samples = len(cast(list[str], manifest["requested_nodes"]))
    for metric in ("gpu_burn_gflops", "ib_bandwidth_Gbps"):
        item = baseline["metrics"][metric]
        if item.get("samples") != expected_node_samples or item.get("scope") != "node-local":
            raise ValueError(f"baseline {metric} must contain one node-local sample per requested node")
    if baseline["metrics"]["nccl_bus_bw_GBps"].get("samples") != 1 or baseline["metrics"]["nccl_bus_bw_GBps"].get("scope") != "allocation-wide":
        raise ValueError("baseline NCCL evidence must be one allocation-wide sample")


def _append_version_findings(config: Config, baseline: dict[str, Any], run: dict[str, Any], node: str, findings: list[GateFinding]) -> None:
    fields = ["driver_version", "cuda_version"]
    if config.thresholds.get("require_firmware_match", False):
        fields += ["gpu_firmware", "ib_firmware", "gpu_models"]
    if not config.thresholds.get("require_version_match", True):
        return
    inventory = cast(dict[str, Any], run["inventory"])
    for field in fields:
        raw = inventory.get(field)
        observed = set(raw.values() if isinstance(raw, dict) else raw if isinstance(raw, list) else [raw])
        allowed = set(baseline.get("versions", {}).get(field, []))
        passed = bool(observed) and observed <= allowed
        findings.append(GateFinding(node, field, None, None, None, "pass" if passed else "fail", "matches baseline" if passed else f"{sorted(map(str, observed))} not allowed by baseline"))


def _authorize_scheduler(manifest: dict[str, Any], role: str, node: str | None = None) -> None:
    if manifest.get("provenance") != "measured":
        return
    job_id = os.environ.get("SLURM_JOB_ID")
    scheduler = cast(dict[str, Any], manifest.get("scheduler", {}))
    if role == "worker":
        expected = cast(dict[str, str], scheduler.get("worker_job_ids", {})).get(str(node))
    else:
        expected = scheduler.get(f"{role}_job_id")
    if not job_id or job_id != expected:
        raise ValueError(f"current Slurm job is not authorized for {role} evidence")


def _validate_job_id(value: str) -> None:
    if not value.isdigit() or len(value) > 20:
        raise ValueError(f"invalid Slurm job ID: {value!r}")


def _quarantine(config: Config, report: FleetReport, run_dir: Path) -> dict[str, str]:
    if report.provenance == "synthetic":
        raise ValueError("quarantine is forbidden for synthetic runs")
    if not config.quarantine.get("enabled", False):
        raise ValueError("quarantine requested but quarantine.enabled is false")
    command = split_command(str(config.quarantine.get("command", "scontrol")))
    reason = str(config.quarantine.get("reason", "ClusterCheck regression gate failed"))
    runner = CommandRunner(60)
    actions: dict[str, str] = {}
    evidence: dict[str, Any] = {}
    write_json(run_dir / "quarantine-intent.json", {"run_id": report.run_id, "created_at": now(), "nodes": report.rejected}, exclusive=True)
    for node in report.rejected:
        result = runner.run([*command, "update", f"NodeName={node}", "State=DRAIN", f"Reason={reason}"], 60)
        actions[node] = "drained" if result.return_code == 0 and not result.timed_out else "drain_failed"
        evidence[node] = result.__dict__
    write_json(run_dir / "quarantine-actions.json", {"run_id": report.run_id, "created_at": now(), "actions": actions, "evidence": evidence}, exclusive=True)
    return actions


def _write_markdown(path: Path, report: FleetReport) -> None:
    banner = f"> **{SYNTHETIC_NOTICE}**\n\n" if report.provenance == "synthetic" else ""
    rows = "\n".join(f"| `{node}` | **{status.upper()}** |" for node, status in sorted(report.nodes.items()))
    gates = "\n".join(f"| `{item.node}` | `{item.metric}` | {item.status.upper()} | {item.reason} |" for item in report.findings)
    text = f"# ClusterCheck fleet report\n\n{banner}- Run: `{report.run_id}`\n- Baseline: `{report.baseline_id}`\n- Baseline SHA-256: `{report.baseline_sha256}`\n- Baseline source run: `{report.baseline_source_run_id}`\n- Provenance: **{report.provenance.upper()}**\n\n| Node | Admission |\n|---|---|\n{rows}\n\nRejected: {', '.join(report.rejected) or 'none'}  \nScheduler-drained: {', '.join(report.drained) or 'none'}\n\n## Regression gates\n\n| Node | Gate | Status | Reason |\n|---|---|---|---|\n{gates}\n"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path.name}")
    path.write_text(text, encoding="utf-8")
