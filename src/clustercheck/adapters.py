"""Real hardware command adapters and deterministic synthetic fixtures."""

from __future__ import annotations

import math
import os
import platform
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .command import CommandRunner, ToolUnavailable, split_command
from .config import Config
from .models import CheckResult, Inventory


class Adapter(Protocol):
    def inventory(self, node: str) -> Inventory: ...
    def checks(self, node: str) -> list[CheckResult]: ...


@dataclass
class RealAdapter:
    config: Config
    runner: CommandRunner

    def inventory(self, node: str) -> Inventory:
        smi = self.runner.run([
            self.config.tools.get("nvidia_smi", "nvidia-smi"),
            "--query-gpu=index,uuid,name,pci.bus_id,driver_version,vbios_version,memory.total",
            "--format=csv,noheader,nounits",
        ], 60)
        if smi.return_code != 0 or smi.timed_out:
            raise RuntimeError(f"nvidia-smi inventory failed: {smi.stderr.strip()}")
        parsed = [[part.strip() for part in row.split(",")] for row in smi.stdout.splitlines() if row.strip()]
        expected = int(self.config.cluster["gpus_per_node"])
        if len(parsed) != expected or any(len(row) != 7 for row in parsed):
            raise RuntimeError(f"nvidia-smi returned {len(parsed)} valid rows; expected exactly {expected}")
        indices = [int(row[0]) for row in parsed]
        uuids = [row[1] for row in parsed]
        buses = [row[3] for row in parsed]
        if len(set(indices)) != expected or len(set(uuids)) != expected or len(set(buses)) != expected:
            raise RuntimeError("GPU inventory contains duplicate index, UUID, or PCI bus identity")
        cuda = self.runner.run([self.config.tools.get("nvidia_smi", "nvidia-smi")], 60)
        if cuda.return_code != 0 or cuda.timed_out:
            raise RuntimeError("nvidia-smi CUDA inventory failed")
        cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", cuda.stdout)
        ib = self._ib_inventory()
        hostname = socket.gethostname()
        if hostname != node and hostname.split(".")[0] != node.split(".")[0]:
            raise RuntimeError(f"allocated node identity {node!r} does not match local hostname {hostname!r}")
        return Inventory(
            hostname=hostname,
            driver_version=parsed[0][4],
            cuda_version=cuda_match.group(1) if cuda_match else "unknown",
            gpu_models=[row[2] for row in parsed],
            gpu_firmware=[row[5] for row in parsed],
            ib_devices=sorted(ib),
            ib_firmware=ib,
            kernel=platform.release(),
            gpus=[{"index": int(row[0]), "uuid": row[1], "model": row[2], "pci_bus_id": row[3], "vbios": row[5], "memory_mib": float(row[6])} for row in parsed],
        )

    def _ib_inventory(self) -> dict[str, str]:
        try:
            evidence = self.runner.run([self.config.tools.get("ibv_devinfo", "ibv_devinfo")], 60)
        except ToolUnavailable:
            return {}
        if evidence.return_code != 0 or evidence.timed_out:
            raise RuntimeError("ibv_devinfo failed")
        devices: dict[str, str] = {}
        current: str | None = None
        for line in evidence.stdout.splitlines():
            match = re.search(r"hca_id:\s*(\S+)", line)
            if match:
                current = match.group(1)
                devices[current] = "unknown"
            firmware = re.search(r"fw_ver:\s*(\S+)", line)
            if firmware and current:
                devices[current] = firmware.group(1)
        return devices

    def checks(self, node: str) -> list[CheckResult]:
        del node
        checks = [self._safe("dcgm_diagnostic", self._dcgm), self._safe("gpu_burn", self._gpu_burn), self._safe("ib_bandwidth", self._ib)]
        if os.environ.get("CLUSTERCHECK_SKIP_NCCL") != "1":
            checks.insert(2, self._safe("nccl_all_reduce", self.nccl))
        return checks

    @staticmethod
    def _safe(name: str, operation: Callable[[], CheckResult]) -> CheckResult:
        try:
            return operation()
        except (OSError, RuntimeError, ValueError) as exc:
            return CheckResult(name, "error", f"{type(exc).__name__}: {exc}")

    def _dcgm(self) -> CheckResult:
        evidence = self.runner.run(split_command(self.config.tools.get("dcgm_command", "dcgmi diag -r 3")), int(self.config.tools.get("dcgm_timeout", 1800)))
        text = f"{evidence.stdout}\n{evidence.stderr}"
        expected = int(self.config.cluster["gpus_per_node"])
        gpu_ids = {int(value) for value in re.findall(r"(?:GPU|gpuId)\s*[:#]?\s*(\d+)", text)}
        has_pass = bool(re.search(r"\bpass(?:ed)?\b", text, re.I))
        has_bad = bool(re.search(r"\b(?:fail(?:ed|ure)?|error|warn(?:ing)?|skip(?:ped)?)\b", text, re.I))
        coverage = len(gpu_ids) >= expected
        passed = evidence.return_code == 0 and not evidence.timed_out and has_pass and not has_bad and coverage
        summary = f"DCGM pass evidence; GPU entities observed={len(gpu_ids)}/{expected}" if passed else f"DCGM evidence incomplete or failing; GPU entities observed={len(gpu_ids)}/{expected}"
        return CheckResult("dcgm_diagnostic", "pass" if passed else "fail", summary, evidence=evidence)

    def _gpu_burn(self) -> CheckResult:
        seconds = int(self.config.tools.get("gpu_burn_seconds", 120))
        evidence = self.runner.run([*split_command(self.config.tools.get("gpu_burn_command", "gpu_burn")), str(seconds)], seconds + 120)
        text = f"{evidence.stdout}\n{evidence.stderr}"
        per_gpu = parse_gpu_burn_output(text)
        expected_ids = set(range(int(self.config.cluster["gpus_per_node"])))
        complete = set(per_gpu) == expected_ids and all(per_gpu[index] for index in expected_ids)
        has_bad = bool(re.search(r"\b(?:FAULTY|FAIL(?:ED|URE)?|ERROR)\b", text, re.I))
        passed = evidence.return_code == 0 and not evidence.timed_out and complete and not has_bad
        values = [value for gpu_values in per_gpu.values() for value in gpu_values]
        metrics = {"gpu_burn_gflops": min(values)} if complete and values else {}
        summary = f"GPU burn evidence covers {len(per_gpu)}/{len(expected_ids)} configured GPUs"
        return CheckResult("gpu_burn", "pass" if passed else "fail", summary, metrics, evidence)

    def nccl(self) -> CheckResult:
        gpus = int(self.config.cluster["gpus_per_node"])
        command = [*split_command(self.config.tools.get("nccl_command", "all_reduce_perf")), "-b", str(self.config.tools.get("nccl_min_bytes", "8M")), "-e", str(self.config.tools.get("nccl_max_bytes", "8G")), "-f", "2", "-g", str(gpus)]
        evidence = self.runner.run(command, int(self.config.tools.get("nccl_timeout", 1200)))
        bus_bw, wrong, sizes = parse_nccl_output(evidence.stdout)
        expected_sizes = _power_two_sizes(str(self.config.tools.get("nccl_min_bytes", "8M")), str(self.config.tools.get("nccl_max_bytes", "8G")))
        complete = sizes == expected_sizes and len(bus_bw) == 2 * len(expected_sizes)
        metrics = {"nccl_bus_bw_GBps": min(bus_bw), "nccl_sizes_parsed": float(len(sizes))} if complete else {}
        passed = evidence.return_code == 0 and not evidence.timed_out and complete and wrong == 0
        summary = f"allocation-wide NCCL sweep parsed {len(sizes)}/{len(expected_sizes)} sizes; wrong elements={wrong}; attribution is not node-local"
        return CheckResult("nccl_all_reduce", "pass" if passed else "fail", summary, metrics, evidence)

    def _ib(self) -> CheckResult:
        evidence = self.runner.run(split_command(self.config.tools.get("ib_command", "ib_write_bw --report_gbits")), int(self.config.tools.get("ib_timeout", 300)))
        values = parse_ib_output(evidence.stdout)
        minimum_samples = int(self.config.tools.get("ib_min_samples", 1))
        complete = len(values) >= minimum_samples
        metrics = {"ib_bandwidth_Gbps": min(values), "ib_samples_parsed": float(len(values))} if complete else {}
        passed = evidence.return_code == 0 and not evidence.timed_out and complete
        summary = f"IB minimum of {len(values)} parsed BW average samples in Gb/s; configured command determines rail coverage"
        return CheckResult("ib_bandwidth", "pass" if passed else "fail", summary, metrics, evidence)


def parse_gpu_burn_output(output: str) -> dict[int, list[float]]:
    """Extract GFLOP/s measurements explicitly associated with a GPU index."""
    values: dict[int, list[float]] = {}
    patterns = (
        r"GPU\s*(\d+)[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*(?:GFLOP/s|GFLOPS|GFLOP)",
        r"\[(\d+)\][^\n]*?([0-9]+(?:\.[0-9]+)?)\s*(?:GFLOP/s|GFLOPS|GFLOP)",
    )
    for pattern in patterns:
        for gpu, raw in re.findall(pattern, output, re.I):
            value = float(raw)
            if math.isfinite(value) and value >= 0:
                values.setdefault(int(gpu), []).append(value)
    return values


def parse_nccl_output(output: str) -> tuple[list[float], int, list[int]]:
    values: list[float] = []
    sizes: set[int] = set()
    wrong = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 12 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        try:
            tail = [float(value) for value in fields[-8:]]
        except ValueError:
            continue
        buses = (tail[2], tail[6])
        errors = (tail[3], tail[7])
        if all(math.isfinite(value) and value >= 0 for value in (*buses, *errors)):
            values.extend(buses)
            sizes.add(int(fields[0]))
            wrong += int(errors[0]) + int(errors[1])
    return values, wrong, sorted(sizes)


def _power_two_sizes(minimum: str, maximum: str) -> list[int]:
    def bytes_value(value: str) -> int:
        match = re.fullmatch(r"([0-9]+)([KMG]?)", value.strip(), re.I)
        if not match:
            raise ValueError(f"NCCL size must be an integer with optional K/M/G suffix: {value}")
        parsed = int(match.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
        if parsed < 1:
            raise ValueError("NCCL sizes must be positive")
        return parsed

    start, end = bytes_value(minimum), bytes_value(maximum)
    if start > end:
        raise ValueError("nccl_min_bytes cannot exceed nccl_max_bytes")
    values: list[int] = []
    current = start
    while current <= end:
        values.append(current)
        current *= 2
    if not values or values[-1] != end:
        raise ValueError("NCCL byte range must form an inclusive power-of-two sweep")
    return values


def parse_ib_output(output: str) -> list[float]:
    if "BW average[Gb/sec]" not in output and "BW average[Gb/s]" not in output:
        return []
    values: list[float] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        try:
            value = float(fields[-2])
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return values


@dataclass
class SyntheticAdapter:
    """Deterministic fixtures. This class never invokes hardware commands."""

    config: Config
    force_healthy: bool = False

    def inventory(self, node: str) -> Inventory:
        count = int(self.config.cluster["gpus_per_node"])
        return Inventory(node, "SYNTHETIC-550.54.15", "SYNTHETIC-12.4", ["SYNTHETIC NVIDIA H100 80GB HBM3"] * count, ["SYNTHETIC-96.00.5E.00.01"] * count, ["SYNTHETIC-mlx5_0"], {"SYNTHETIC-mlx5_0": "SYNTHETIC-28.39.1002"}, "SYNTHETIC-6.5.0", [{"index": index, "uuid": f"SYNTHETIC-GPU-{index:02d}", "model": "SYNTHETIC NVIDIA H100 80GB HBM3", "pci_bus_id": f"SYNTHETIC-{index:02x}:00.0", "vbios": "SYNTHETIC-96.00.5E.00.01", "memory_mib": 81920.0} for index in range(count)])

    def checks(self, node: str) -> list[CheckResult]:
        degraded = not self.force_healthy and node in set(self.config.simulation.get("degraded_nodes", []))
        degradation = float(self.config.simulation.get("degraded_link_factor", 0.65))
        ib = float(self.config.simulation.get("expected_ib_Gbps", 200.0)) * (degradation if degraded else 0.98)
        label = "SYNTHETIC fixture; no hardware command executed"
        return [
            CheckResult("dcgm_diagnostic", "pass", label),
            CheckResult("gpu_burn", "pass", label, {"gpu_burn_gflops": 58_000.0 if not degraded else 57_900.0}),
            CheckResult("ib_bandwidth", "pass", label, {"ib_bandwidth_Gbps": round(ib, 3), "ib_samples_parsed": 1.0}),
        ]

    def nccl(self) -> CheckResult:
        degraded = not self.force_healthy and bool(self.config.simulation.get("degraded_nodes", []))
        expected = float(self.config.simulation.get("expected_bus_bw_GBps", 400.0))
        efficiency = float(self.config.simulation.get("healthy_bus_efficiency", 0.92))
        factor = float(self.config.simulation.get("degraded_link_factor", 0.65)) if degraded else 1.0
        sizes = _power_two_sizes(str(self.config.tools.get("nccl_min_bytes", "8M")), str(self.config.tools.get("nccl_max_bytes", "8G")))
        label = "SYNTHETIC allocation-wide fixture; no hardware command executed; not node-local"
        return CheckResult("nccl_all_reduce", "pass", label, {"nccl_bus_bw_GBps": round(expected * efficiency * factor, 3), "nccl_sizes_parsed": float(len(sizes))})


def adapter_for(mode: str, config: Config) -> Adapter:
    if mode == "simulation":
        return SyntheticAdapter(config)
    if mode == "real":
        return RealAdapter(config, CommandRunner(int(config.tools.get("default_timeout", 900))))
    raise ValueError(f"unsupported mode: {mode}")
