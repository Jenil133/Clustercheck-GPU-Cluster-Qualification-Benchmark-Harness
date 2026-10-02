# ClusterCheck fleet report

> **SYNTHETIC DATA: deterministic fixture only; no GPU, Slurm, DCGM, NCCL, or InfiniBand hardware execution occurred.**

- Run: `synthetic-demo`
- Baseline: `SYNTHETIC-DEMO-BASELINE`
- Baseline SHA-256: `b380de101901ee8cfd602c22f23ca514c08497cee1d5e2ed96c830fecda1be26`
- Baseline source run: `synthetic-demo-baseline-fixture`
- Provenance: **SYNTHETIC**

| Node | Admission |
|---|---|
| `synthetic-node-01` | **FAIL** |
| `synthetic-node-02` | **FAIL** |

Rejected: synthetic-node-01, synthetic-node-02  
Scheduler-drained: none

## Regression gates

| Node | Gate | Status | Reason |
|---|---|---|---|
| `synthetic-node-01` | `gpu_identity_cardinality` | PASS | exact unique per-GPU index/UUID/PCI/model/firmware coverage |
| `synthetic-node-01` | `required_node_checks` | PASS | all required node-local checks are present |
| `synthetic-node-01` | `hostname_binding` | PASS | inventory hostname is bound to requested node |
| `synthetic-node-01` | `gpu_burn_gflops` | PASS | node-local: at or above regression floor |
| `synthetic-node-01` | `ib_bandwidth_Gbps` | PASS | node-local: at or above regression floor |
| `synthetic-node-01` | `nccl_bus_bw_GBps` | FAIL | allocation-wide; conservatively rejects every participant: missing evidence or below regression floor |
| `synthetic-node-01` | `driver_version` | PASS | matches baseline |
| `synthetic-node-01` | `cuda_version` | PASS | matches baseline |
| `synthetic-node-02` | `gpu_identity_cardinality` | PASS | exact unique per-GPU index/UUID/PCI/model/firmware coverage |
| `synthetic-node-02` | `required_node_checks` | PASS | all required node-local checks are present |
| `synthetic-node-02` | `hostname_binding` | PASS | inventory hostname is bound to requested node |
| `synthetic-node-02` | `gpu_burn_gflops` | PASS | node-local: at or above regression floor |
| `synthetic-node-02` | `ib_bandwidth_Gbps` | FAIL | node-local: missing evidence or below regression floor |
| `synthetic-node-02` | `nccl_bus_bw_GBps` | FAIL | allocation-wide; conservatively rejects every participant: missing evidence or below regression floor |
| `synthetic-node-02` | `driver_version` | PASS | matches baseline |
| `synthetic-node-02` | `cuda_version` | PASS | matches baseline |
