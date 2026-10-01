# Architecture and operations

## Transaction and data flow

`submit` validates configuration/baseline paths, exclusively creates `<output>/<run-id>`, and writes a `submitting` manifest before calling `sbatch`. Jobs are submitted held. Each accepted numeric Slurm ID is persisted. After all node, fabric, and admission jobs exist, the manifest transitions to `submitted` and all jobs are released. A trap cancels every accepted job and records `failed` on any error, including partial release failure.

Node workers can only create their own safe `<node>.json` once under a manifest matching the exact configuration, node set, GPU count, run ID, and measured provenance. The fabric stage resolves the Slurm host list, compares it with requested nodes, runs one collective, and creates `fabric-result.json`. Admission first finalizes only exact complete evidence coverage, then evaluates once.

```text
<run-id>/
├── run-manifest.json   # authoritative identity/binding and lifecycle
├── nodes/*.json        # create-once node-local evidence
├── fabric-result.json  # create-once allocation-wide collective evidence
├── fleet-report.json   # create-once admission decision
└── fleet-report.md     # create-once operator rendering
```

The manifest lifecycle is intentionally updated atomically; identity/config/topology fields do not change. Evidence files and reports are create-once. Run-directory reuse is always refused.

## Trust boundaries

Configuration, command output, scheduler output, identifiers, and artifact files are untrusted. Configuration rejects unknown keys, wrong scalar types, booleans masquerading as numbers, non-finite values, invalid sweeps, unsafe identifiers, duplicate nodes, and fewer than two nodes. Commands run as argv without `shell=True`. Timeouts target the process group with TERM/KILL and always reap the direct child. Output records byte counts and explicit truncation and retains head plus tail.

The manifest excludes absolute local config paths; it records only the basename and digest. CLI errors may still show operator-supplied local paths. Wheel resources are materialized below the installed `clustercheck` package.

## Evidence scope

Node-local evidence includes inventory, DCGM, burn, and an IB command. Inventory requires exact configured GPU rows and unique index, UUID, and PCI bus identity. DCGM and burn use conservative text evidence because their configured tools do not share one guaranteed structured format. Unsupported output fails rather than being inferred.

NCCL is one allocation-wide observation. Its artifact records expected participants, resolved allocated nodes, topology count status, and `rank_gpu_mapping_verified: false`. A failed/missing collective rejects all participants but is never presented as proof that each node independently produced that result. `nccl_bus_bw_GBps` is the minimum in-place/out-of-place nccl-tests `busbw` across the complete configured sweep.

IB requires a report-gbits heading and at least `ib_min_samples` valid rows, then gates `ib_bandwidth_Gbps` at the minimum. This catches weak rows that maximum selection masked. It does not prove untested rails, switch paths, bidirectional symmetry, or remote endpoint identity beyond the reviewed command/wrapper configuration.

## Baseline invariant

A baseline source manifest must be `completed`; node records must exactly equal requested nodes; every node-local check and the fabric check must pass; and every artifact must match run ID, provenance, config digest, and topology digest. Baseline writes are exclusive. Node-local metrics have N samples for N nodes. The single allocation-wide NCCL result has one sample. Evaluation requires the same exact config and topology.

For a deliberately degraded synthetic scenario, `--baseline-fixture-output` creates a separate all-healthy run under the same config. Its manifest has `synthetic_fixture: true`, and all output retains synthetic provenance/notice.

## Scheduler terminology

`rejected` is an admission result. It does not imply scheduler mutation. `drained` is populated only when an opt-in measured-run DRAIN command returns success. Failed DRAIN attempts are recorded in `quarantine_actions` with command evidence.

## Hardware-only limitations

Operators must independently verify NCCL rank-to-host/GPU binding, fabric path and multi-rail coverage, tool-version-specific DCGM/burn formats, remote process cleanup guarantees, Slurm policy/site directives, baseline approval governance, and whether one perftest direction is sufficient. ClusterCheck records these limits and fails unsupported evidence; it does not claim to solve them.
