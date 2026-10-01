"""Strict, finite, typed configuration loading and validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .command import split_command
from .storage import validate_identifier


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class Config:
    path: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def cluster(self) -> dict[str, Any]: return cast(dict[str, Any], self.raw["cluster"])
    @property
    def tools(self) -> dict[str, Any]: return cast(dict[str, Any], self.raw["tools"])
    @property
    def thresholds(self) -> dict[str, Any]: return cast(dict[str, Any], self.raw["thresholds"])
    @property
    def simulation(self) -> dict[str, Any]: return cast(dict[str, Any], self.raw.get("simulation", {}))
    @property
    def quarantine(self) -> dict[str, Any]: return cast(dict[str, Any], self.raw.get("quarantine", {}))


_ALLOWED = {
    "cluster": {"name", "nodes", "gpus_per_node", "slurm_partition"},
    "tools": {"default_timeout", "dcgm_command", "dcgm_timeout", "gpu_burn_command", "gpu_burn_seconds", "nccl_command", "nccl_min_bytes", "nccl_max_bytes", "nccl_timeout", "ib_command", "ib_timeout", "ib_min_samples", "nvidia_smi", "ibv_devinfo"},
    "thresholds": {"relative_floor", "expected_nccl_bus_bw_GBps", "nccl_bus_efficiency_floor", "require_version_match", "require_firmware_match"},
    "simulation": {"healthy_bus_efficiency", "degraded_link_factor", "degraded_nodes", "expected_bus_bw_GBps", "expected_ib_Gbps"},
    "quarantine": {"enabled", "command", "reason"},
}


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    data = config_path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
        _validate(raw)
    except (ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc
    return Config(config_path, raw, hashlib.sha256(data).hexdigest())


def _validate(raw: dict[str, Any]) -> None:
    unknown_sections = set(raw) - set(_ALLOWED)
    if unknown_sections:
        raise ConfigError(f"unknown configuration sections: {', '.join(sorted(unknown_sections))}")
    for section in ("cluster", "tools", "thresholds"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(f"missing required [{section}] section")
    for section, values in raw.items():
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a table")
        unknown = set(values) - _ALLOWED[section]
        if unknown:
            raise ConfigError(f"unknown [{section}] keys: {', '.join(sorted(unknown))}")
    cluster = raw["cluster"]
    nodes = cluster.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2 or len(set(_strings(nodes))) != len(nodes):
        raise ConfigError("cluster.nodes must contain at least two unique safe node identifiers")
    for node in _strings(nodes):
        validate_identifier(node, "node identifier")
    validate_identifier(_string(cluster.get("name"), "cluster.name"), "cluster name")
    partition = cluster.get("slurm_partition")
    if partition is not None:
        validate_identifier(_string(partition, "cluster.slurm_partition"), "Slurm partition")
    _positive_int(cluster.get("gpus_per_node"), "cluster.gpus_per_node")

    tools = raw["tools"]
    for key in ("default_timeout", "dcgm_timeout", "gpu_burn_seconds", "nccl_timeout", "ib_timeout", "ib_min_samples"):
        if key in tools:
            _positive_int(tools[key], f"tools.{key}")
    for key in ("dcgm_command", "gpu_burn_command", "nccl_command", "ib_command", "nvidia_smi", "ibv_devinfo"):
        if key in tools:
            split_command(_string(tools[key], f"tools.{key}"))
    _validate_power_two_sizes(_string(tools.get("nccl_min_bytes", "8M"), "tools.nccl_min_bytes"), _string(tools.get("nccl_max_bytes", "8G"), "tools.nccl_max_bytes"))

    thresholds = raw["thresholds"]
    for key in ("relative_floor", "nccl_bus_efficiency_floor"):
        value = _finite_number(thresholds.get(key), f"thresholds.{key}")
        if not 0 < value <= 1:
            raise ConfigError(f"thresholds.{key} must be in (0, 1]")
    if _finite_number(thresholds.get("expected_nccl_bus_bw_GBps"), "thresholds.expected_nccl_bus_bw_GBps") <= 0:
        raise ConfigError("thresholds.expected_nccl_bus_bw_GBps must be positive")
    for key in ("require_version_match", "require_firmware_match"):
        if key in thresholds and not isinstance(thresholds[key], bool):
            raise ConfigError(f"thresholds.{key} must be a boolean")

    simulation = raw.get("simulation", {})
    degraded = simulation.get("degraded_nodes", [])
    if not isinstance(degraded, list) or not set(_strings(degraded)) <= set(_strings(nodes)):
        raise ConfigError("simulation.degraded_nodes must be a subset of cluster.nodes")
    for key in ("healthy_bus_efficiency", "degraded_link_factor"):
        if key in simulation and not 0 < _finite_number(simulation[key], f"simulation.{key}") <= 1:
            raise ConfigError(f"simulation.{key} must be in (0, 1]")
    for key in ("expected_bus_bw_GBps", "expected_ib_Gbps"):
        if key in simulation and _finite_number(simulation[key], f"simulation.{key}") <= 0:
            raise ConfigError(f"simulation.{key} must be positive")

    quarantine = raw.get("quarantine", {})
    if "enabled" in quarantine and not isinstance(quarantine["enabled"], bool):
        raise ConfigError("quarantine.enabled must be a boolean")
    for key in ("command", "reason"):
        if key in quarantine:
            text_value = _string(quarantine[key], f"quarantine.{key}")
            if any(ord(char) < 32 for char in text_value):
                raise ConfigError(f"quarantine.{key} contains control characters")


def _strings(values: list[Any]) -> list[str]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ConfigError("expected non-empty strings")
    return cast(list[str], values)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ConfigError(f"{name} must be a finite number")
    return float(value)


def _validate_power_two_sizes(minimum: str, maximum: str) -> None:
    def parse(value: str) -> int:
        match = re.fullmatch(r"([0-9]+)([KMG]?)", value.strip(), re.I)
        if not match:
            raise ConfigError(f"invalid NCCL size: {value}")
        return int(match.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2).upper()]
    start, end = parse(minimum), parse(maximum)
    if start < 1 or start > end or end % start or (end // start) & ((end // start) - 1):
        raise ConfigError("NCCL byte range must be a positive inclusive power-of-two sweep")


def canonical_config(config: Config) -> str:
    return json.dumps(config.raw, sort_keys=True, separators=(",", ":"))
